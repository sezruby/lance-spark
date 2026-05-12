/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.spark.knn.internal.staged

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.{CatalystTypeConverters, InternalRow}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.types.{ArrayType, DataType, FloatType, LongType, StructField, StructType}
import org.lance.spark.knn.internal.{ProbedLeft, ScoredRowRef}

/**
 * Catalyst-level codec for `(leftId: Long, ProbedLeft)` tuples flowing between the staged
 * physical operators (`LanceProbeExec → LanceMergeExec → LanceMaterializeExec`). The single-
 * exec wrapper used by Phase 2 keeps the tuple as a typed Scala value through `RDD[(Long,
 * ProbedLeft)]`; once we split into three separate `SparkPlan` operators the inter-stage RDD
 * has to be `RDD[InternalRow]`, so each boundary needs encode + decode.
 *
 * == Schema ==
 *
 * The inter-stage row carries:
 *
 *   - `leftId: long`              — synthetic per-row id used as the merge `reduceByKey` key
 *   - `leftRow: struct<leftSchema>` — the original left DataFrame row, schema preserved so
 *     `LanceMaterializeStage` can reconstruct the join output without re-fetching the left
 *     side
 *   - `refs: array<struct<rowAddr: long, score: float>>` — the probe's top-K row references
 *     for this left row
 *
 * == Why Catalyst struct, not Kryo blob ==
 *
 * The microbench (`InterStagePayloadOverheadBench`) showed both encodings cost <1 % of total
 * SQL benchmark wall-clock at every realistic scale, so the choice is code-aesthetic, not
 * performance. Catalyst struct wins on aesthetics:
 *
 *   - `df.explain()` shows the inter-stage operators with their schema readable
 *     (`[leftId#0L, leftRow#1, refs#2]`) instead of an opaque `[leftId, blob: binary]`.
 *   - Spark's shuffle path is native to Catalyst's UnsafeRow encoding — UnsafeRow shuffle
 *     write/read is heavily optimised. Kryo blobs would still ride that path but with an
 *     extra serialize-into-binary hop on top.
 *
 * == Encoder / decoder lifecycle ==
 *
 * `ExpressionEncoder(leftSchema).resolveAndBind()` is constructed driver-side and shipped to
 * executors via task closure serialization. Each partition then calls `createSerializer()`
 * / `createDeserializer()` once and reuses the resulting function across the partition's
 * iterator — this matches Spark's standard encoder lifecycle.
 */
private[knn] object ProbedLeftCodec {

  /** Schema of a single ref struct inside the `refs` array column. */
  private val RefStructFields: Array[StructField] = Array(
    StructField("rowAddr", LongType, nullable = false),
    StructField("score", FloatType, nullable = false))
  private val RefStruct: StructType = StructType(RefStructFields)
  val RefsType: ArrayType = ArrayType(RefStruct, containsNull = false)

  /**
   * Inter-stage row schema parameterised by the left side's schema. The leftRow's fields
   * are FLATTENED into the top-level schema (rather than nested as a sub-struct) — earlier
   * iterations used a nested struct and triggered Spark's `UnsafeRowSerializer` + nested-
   * struct reuse semantics into JVM-level SIGSEGV / unsafe-fault crashes inside the
   * materialize stage's deserializer. Flattening sidesteps the nested-struct path entirely
   * and keeps the binary layout to plain top-level fields plus one array-of-struct.
   *
   * Schema:
   *   - `_leftId: long`           — synthetic per-row id
   *   - `<leftSchema fields>`     — every field of the user's left DataFrame, inlined
   *   - `_refs: array<struct<rowAddr: long, score: float>>`
   *
   * The leading underscore on `_leftId` and `_refs` keeps them out of the way of any
   * reasonable user column name.
   */
  def interStageSchema(leftSchema: StructType): StructType = {
    val leadField = StructField("_leftId", LongType, nullable = false)
    val refsField = StructField("_refs", RefsType, nullable = false)
    StructType(leadField +: leftSchema.fields :+ refsField)
  }

  /** Number of leading non-leftSchema columns at the top of `interStageSchema` (just `_leftId`). */
  private val LeftIdColIndex: Int = 0
  private val FirstLeftColIndex: Int = 1

  /** The `_refs` column lives at the very end. */
  private def refsColIndex(leftSchema: StructType): Int = 1 + leftSchema.length

  /**
   * Build the AttributeReference list that operators expose as `output`. Created once per
   * plan tree and shared between probe and merge so attribute exprIds line up across the
   * boundary — Catalyst attribute resolution rejects mid-tree exprId changes.
   */
  def interStageAttributes(leftSchema: StructType): Seq[AttributeReference] =
    interStageSchema(leftSchema).fields.map { f =>
      AttributeReference(f.name, f.dataType, f.nullable, f.metadata)()
    }

  /**
   * Per-partition encoder/decoder. Construct once at the top of each task's `mapPartitions`
   * closure and reuse for every row. The underlying serializer/deserializer functions are
   * stateful (they cache writers/readers) so DO NOT share across partitions / threads.
   *
   * The codec uses a single `ExpressionEncoder(interStageSchema)` to handle the whole row
   * end-to-end. Earlier iterations tried pre-serialising the `leftRow` to `UnsafeRow` and
   * then re-wrapping in a `GenericInternalRow + UnsafeProjection` — that produced JVM-level
   * unsafe-memory faults at the materialize stage's deserializer (the array length read
   * from the nested struct's binary layout was corrupt). Letting Catalyst's encoder handle
   * the entire `Row` → `InternalRow` conversion in one pass avoids the nesting-mismatch
   * pitfall and produces UnsafeRow output that the shuffle path can serialize directly.
   *
   * Output is `UnsafeRow` (Spark's shuffle path requires it — `UnsafeRowSerializer` casts
   * every shuffled row to `UnsafeRow`).
   */
  final class Encoder(leftSchema: StructType) extends Serializable {
    @transient private lazy val outerSer =
      ExpressionEncoder(interStageSchema(leftSchema)).resolveAndBind().createSerializer()
    private val leftFieldCount: Int = leftSchema.length

    def encode(leftId: Long, pl: ProbedLeft): InternalRow = {
      // Flatten: outer row is `[leftId, leftField0, leftField1, ..., refs]`.
      val cols = new Array[Any](2 + leftFieldCount)
      cols(0) = java.lang.Long.valueOf(leftId)
      var i = 0
      while (i < leftFieldCount) {
        cols(1 + i) = pl.leftRow.get(i)
        i += 1
      }
      cols(1 + leftFieldCount) = pl.refs.map(r => Row(r.rowAddr, r.score)).toSeq
      outerSer(Row.fromSeq(cols.toSeq)).copy()
    }
  }

  /**
   * Decoder reads fields directly from the input `InternalRow` without going through
   * `ExpressionEncoder.Deserializer`. Earlier iterations did go through the deserializer
   * and tripped a JIT-compiled SIGSEGV at `UnsafeRow.getArray` in generated `MapObjects`
   * code on the `_refs` array column under sustained load (after JIT C2 had compiled the
   * inner loop). The deserializer's generated code interacts poorly with UnsafeRow array
   * accessors in some Spark 3.5 setups — known fragility.
   *
   * Reading via `InternalRow.get(i, dataType)` + `CatalystTypeConverters.createToScalaConverter`
   * sidesteps the problematic generated code entirely. The Catalyst→Scala converters handle
   * primitive unwrapping, `UnsafeArrayData` → `Seq`, etc., which is what the encoder's
   * deserializer would have done — just via the direct converter API rather than codegen.
   */
  final class Decoder(leftSchema: StructType) extends Serializable {
    private val leftFieldCount: Int = leftSchema.length
    // Per-column converters: build once on driver, reused on executors. The `Any => Any`
    // function is what `CatalystTypeConverters` exposes for converting raw InternalRow
    // values into idiomatic Scala/Java values.
    private val leftConverters: Array[Any => Any] = leftSchema.fields.map { f =>
      CatalystTypeConverters.createToScalaConverter(f.dataType)
    }
    private val leftDataTypes: Array[DataType] = leftSchema.fields.map(_.dataType)
    private val refsIdx: Int = refsColIndex(leftSchema)

    def decode(ir: InternalRow): (Long, ProbedLeft) = {
      val leftId = ir.getLong(LeftIdColIndex)

      // Read each leftRow field directly from the InternalRow.
      val leftValues = new Array[Any](leftFieldCount)
      var i = 0
      while (i < leftFieldCount) {
        val colIdx = FirstLeftColIndex + i
        if (ir.isNullAt(colIdx)) {
          leftValues(i) = null
        } else {
          val raw = ir.get(colIdx, leftDataTypes(i))
          leftValues(i) = leftConverters(i)(raw)
        }
        i += 1
      }
      val leftRow = Row.fromSeq(leftValues.toSeq)

      // Refs array: ArrayData of struct<rowAddr: long, score: float>.
      val arr: ArrayData = ir.getArray(refsIdx)
      val n = arr.numElements()
      val refs = new Array[ScoredRowRef](n)
      var r = 0
      while (r < n) {
        val refStruct = arr.getStruct(r, 2)
        refs(r) = ScoredRowRef(refStruct.getLong(0), refStruct.getFloat(1))
        r += 1
      }

      (leftId, ProbedLeft(leftRow, refs))
    }
  }
}
