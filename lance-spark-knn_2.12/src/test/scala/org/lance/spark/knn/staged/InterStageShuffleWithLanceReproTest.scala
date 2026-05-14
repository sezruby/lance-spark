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
package org.lance.spark.knn.staged

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, RowFactory, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.io.TempDir

import java.nio.file.Path
import java.util.Random

import scala.collection.JavaConverters._

/**
 * Follow-up to [[InterStageShuffleReproTest]] — all six synthetic (driver-built) tests
 * there passed at benchmark scale on the JVM/arch the original crash fired on. This suite
 * adds the missing input provenance: left rows read from an actual Lance dataset via
 * `spark.read.format("lance")`, then run through the same encode → shuffle → decode path
 * that [[org.lance.spark.knn.internal.staged.ProbedLeftCodec]] exercises.
 *
 * If the synthetic tests pass but Lance-backed tests at the same scale crash, the fault is
 * at the Lance → Spark boundary, not in Spark/JVM generally. Candidate causes:
 *
 *   1. Arrow-backed `ColumnarBatch` reads leaving JVM refs into off-heap buffers that are
 *      freed when the scanner closes. Subsequent `UnsafeArrayData.getFloat` reads would
 *      hit unmapped memory → SIGSEGV in native.
 *   2. Double/triple encoder round-trip (Arrow columnar → InternalRow → Row → InternalRow
 *      → UnsafeRow → shuffle) corrupting the nested array length header.
 *   3. Thread-safety: `local[4]` runs four tasks concurrently in the same JVM; any shared
 *      state in the per-task encoder would corrupt UnsafeRow writes.
 *
 * Each test isolates one step of the staged-exec pipeline against a Lance-source left.
 */
class InterStageShuffleWithLanceReproTest {

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  // Match the synthetic test for direct comparison.
  private val LeftN: Int = 100000
  private val Dim: Int = 128
  private val K: Int = 10
  private val ShuffleParts: Int = 4
  private val Seed: Long = 1337L

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("inter-stage-shuffle-with-lance-repro")
      .master("local[4]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.driver.memory", "4g")
      .config("spark.sql.shuffle.partitions", ShuffleParts.toString)
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = if (spark != null) spark.stop()

  // ---- schemas ----------------------------------------------------------------------------

  /** Left-side Lance schema: (lid, qvec). Matches the benchmark's left shape. */
  private val LeftSchema: StructType = StructType(Array(
    StructField("lid", LongType, nullable = false),
    StructField(
      "qvec",
      ArrayType(FloatType, containsNull = false),
      nullable = false,
      new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))

  /** Inter-stage shape: same as `ProbedLeftCodec.interStageSchema(LeftSchema)`. */
  private val RefStruct: StructType = StructType(Array(
    StructField("rowAddr", LongType, nullable = false),
    StructField("score", FloatType, nullable = false)))

  private val InterStageSchema: StructType = StructType(
    StructField("_leftId", LongType, nullable = false) +:
      LeftSchema.fields :+
      StructField("_refs", ArrayType(RefStruct, containsNull = false), nullable = false))

  // ---- data generation --------------------------------------------------------------------

  private def randomFloats(rng: Random, n: Int): Array[Float] = {
    val v = new Array[Float](n)
    var i = 0
    while (i < n) { v(i) = rng.nextFloat(); i += 1 }
    v
  }

  /** Write a Lance dataset with `LeftN` rows at the benchmark's left-schema shape. */
  private def writeLanceLeft(): String = {
    val rng = new Random(Seed)
    val rows = new java.util.ArrayList[Row](LeftN)
    var i = 0
    while (i < LeftN) {
      val qvec = randomFloats(rng, Dim).toSeq.asJava
      rows.add(RowFactory.create(java.lang.Long.valueOf(i.toLong), qvec))
      i += 1
    }
    val df = spark.createDataFrame(rows, LeftSchema)
    val uri = tempDir.resolve(s"left_${System.nanoTime()}").toString
    df.write.format("lance").save(uri)
    uri
  }

  /** Build fake refs (no Lance probe — the crash candidates sit on the LEFT side's read). */
  private def fakeRefs(rng: Random): Seq[Row] = {
    val buf = new scala.collection.mutable.ArrayBuffer[Row](K)
    var r = 0
    while (r < K) {
      buf += Row(rng.nextLong() & 0x7FFFFFFFFFFFFFFFL, rng.nextFloat())
      r += 1
    }
    buf.toSeq
  }

  // ---- tests ------------------------------------------------------------------------------

  /**
   * Read left side from Lance, count. If the Lance columnar read path itself can't sustain
   * 100K × 128-dim, this is where that shows up. Doesn't exercise any of the codec.
   */
  @Test def test1_lanceReadThenCount(): Unit = {
    val uri = writeLanceLeft()
    val left = spark.read.format("lance").load(uri)
    val n = left.count()
    assertEquals(LeftN.toLong, n)
  }

  /**
   * Read left from Lance → `ExpressionEncoder(LeftSchema).createDeserializer()` → `Row`
   * in each partition. Mirrors what `LanceProbeExec.doExecute` does BEFORE adding refs:
   * it takes `child.execute()` (which for a Lance table is `RDD[InternalRow]` backed by
   * ColumnarBatch) and deserializes via encoder into `Row`.
   *
   * If off-heap lifetime is the bug, this is close to the root — the Arrow buffer may
   * still be live here, but the deserialized `Row` should no longer reference it.
   */
  @Test def test2_lanceReadThenEncoderRoundTrip(): Unit = {
    val uri = writeLanceLeft()
    val left = spark.read.format("lance").load(uri)
    // `.rdd` on a DataFrame triggers DeserializeToObject via the row encoder — same code
    // path the staged codec uses via `createDeserializer()`.
    val n = left.rdd.count()
    assertEquals(LeftN.toLong, n)
  }

  /**
   * The full staged-codec hot path against a Lance-source left:
   *   Lance scan → InternalRow child.execute()
   *     → mapPartitions { deserialize to Row via ExpressionEncoder(leftSchema) }
   *     → mapPartitions { zipWithUniqueId + attach fake refs → encode via ExpressionEncoder(InterStageSchema) }
   *     → repartition by _leftId (ShuffleExchange)
   *     → mapPartitions { direct InternalRow decode: getLong, getArray, getStruct }
   *     → count
   *
   * This is the closest non-trivial mirror of `LanceProbeExec.doExecute` feeding
   * `LanceMergeExec.doExecute` across a `ShuffleExchangeExec`. If the reverted codec's
   * crash is Lance-boundary-induced, this should SIGSEGV.
   */
  @Test def test3_lanceSourceFullCodecRoundTripThenShuffle(): Unit = {
    val leftSchema = LeftSchema
    val interStageSchema = InterStageSchema
    val shuffleParts = ShuffleParts
    val seed = Seed
    val k = K

    val uri = writeLanceLeft()
    val left = spark.read.format("lance").load(uri)

    // Step 1: Lance → InternalRow (Catalyst toRdd) → Row via encoder deserialize.
    // This is LanceProbeExec.doExecute lines "Decode user's left-side InternalRows into Rows".
    val leftInternal: RDD[InternalRow] = left.queryExecution.toRdd
    val rowRdd: RDD[Row] = leftInternal.mapPartitions { iter =>
      val enc = ExpressionEncoder(leftSchema).resolveAndBind()
      val deser = enc.createDeserializer()
      iter.map(ir => deser(ir.copy()))
    }

    // Step 2: attach a synthetic leftId + fake refs; encode to InterStageSchema via the
    // same single ExpressionEncoder path the codec uses (the fix from commit 6218b1c).
    val encodedRdd: RDD[InternalRow] = rowRdd
      .zipWithUniqueId()
      .map { case (row, id) => (id, row) }
      .mapPartitionsWithIndex { case (partIdx, iter) =>
        val interEnc = ExpressionEncoder(interStageSchema).resolveAndBind()
        val ser = interEnc.createSerializer()
        val rng = new Random(seed + partIdx.toLong)
        val leftFieldCount = leftSchema.length
        iter.map { case (leftId, leftRow) =>
          // Flatten: [_leftId, leftField0, ..., leftFieldN, _refs]
          val cols = new Array[Any](2 + leftFieldCount)
          cols(0) = java.lang.Long.valueOf(leftId)
          var i = 0
          while (i < leftFieldCount) {
            cols(1 + i) = leftRow.get(i)
            i += 1
          }
          val refs = new scala.collection.mutable.ArrayBuffer[Row](k)
          var r = 0
          while (r < k) {
            refs += Row(rng.nextLong() & 0x7FFFFFFFFFFFFFFFL, rng.nextFloat())
            r += 1
          }
          cols(1 + leftFieldCount) = refs.toSeq
          ser(Row.fromSeq(cols.toSeq)).copy()
        }
      }

    // Step 3: put into a DataFrame so repartition-by-column (which requires a Catalyst
    // shuffle) can consume it. Going RDD[InternalRow] → RDD[Row] → createDataFrame
    // mirrors the original production code path's `createDataFrame(rdd, schema)` shape.
    val backToRow: RDD[Row] = encodedRdd.mapPartitions { iter =>
      val enc = ExpressionEncoder(interStageSchema).resolveAndBind()
      val deser = enc.createDeserializer()
      iter.map(ir => deser(ir.copy()))
    }
    val df = spark.createDataFrame(backToRow, interStageSchema)

    // Step 4: shuffle by _leftId — this is what ClusteredDistribution(leftId) produces.
    val shuffled = df.repartition(shuffleParts, col("_leftId"))

    // Step 5: consume via direct InternalRow accessors (ProbedLeftCodec.Decoder shape).
    // Inter-stage schema has 4 cols: [_leftId:long, lid:long, qvec:array<float>, _refs:array<struct>]
    val shuffledInternal: RDD[InternalRow] = shuffled.queryExecution.toRdd
    val n = shuffledInternal.mapPartitions { iter =>
      var sum = 0L
      while (iter.hasNext) {
        val ir = iter.next().copy()
        val leftId = ir.getLong(0)
        val lid = ir.getLong(1)
        val qvec = ir.getArray(2)
        var j = 0
        var qsum = 0.0f
        while (j < qvec.numElements()) {
          qsum += qvec.getFloat(j)
          j += 1
        }
        val refs = ir.getArray(3)
        var r = 0
        var refSum = 0L
        while (r < refs.numElements()) {
          val s = refs.getStruct(r, 2)
          refSum += s.getLong(0)
          r += 1
        }
        sum += leftId + lid + qsum.toLong + refSum
      }
      Iterator.single(sum)
    }.count()

    // Count is per-partition: ShuffleParts partitions emit one Long each.
    assertEquals(shuffleParts.toLong, n)
  }

  /**
   * Same pipeline as test3 but with JIT warmup + 20 iterations, to give C2 time to compile
   * the hot loop over Lance-sourced UnsafeRows. The reverted crash fired at Spark stage
   * 79 — well past any first-pass interpreter/C1 execution.
   */
  @Test def test4_lanceSourceFullCodecJitStress(): Unit = {
    val leftSchema = LeftSchema
    val interStageSchema = InterStageSchema
    val shuffleParts = ShuffleParts
    val seed = Seed
    val k = K

    val uri = writeLanceLeft()
    val left = spark.read.format("lance").load(uri)

    // JIT warmup: small crossJoin shape (mirrors config A in the benchmark).
    val wA = spark.range(0, 500L).toDF("a")
    val wB = spark.range(0, 500L).toDF("b")
    wA.crossJoin(wB).groupBy(col("a")).count().count()

    // Build the pipeline once — we re-execute it per iteration.
    val leftInternal: RDD[InternalRow] = left.queryExecution.toRdd
    val rowRdd: RDD[Row] = leftInternal.mapPartitions { iter =>
      val enc = ExpressionEncoder(leftSchema).resolveAndBind()
      val deser = enc.createDeserializer()
      iter.map(ir => deser(ir.copy()))
    }
    val encodedRdd: RDD[InternalRow] = rowRdd
      .zipWithUniqueId()
      .map { case (row, id) => (id, row) }
      .mapPartitionsWithIndex { case (partIdx, iter) =>
        val interEnc = ExpressionEncoder(interStageSchema).resolveAndBind()
        val ser = interEnc.createSerializer()
        val rng = new Random(seed + partIdx.toLong)
        val leftFieldCount = leftSchema.length
        iter.map { case (leftId, leftRow) =>
          val cols = new Array[Any](2 + leftFieldCount)
          cols(0) = java.lang.Long.valueOf(leftId)
          var i = 0
          while (i < leftFieldCount) { cols(1 + i) = leftRow.get(i); i += 1 }
          val refs = new scala.collection.mutable.ArrayBuffer[Row](k)
          var r = 0
          while (r < k) {
            refs += Row(rng.nextLong() & 0x7FFFFFFFFFFFFFFFL, rng.nextFloat())
            r += 1
          }
          cols(1 + leftFieldCount) = refs.toSeq
          ser(Row.fromSeq(cols.toSeq)).copy()
        }
      }
    val backToRow: RDD[Row] = encodedRdd.mapPartitions { iter =>
      val enc = ExpressionEncoder(interStageSchema).resolveAndBind()
      val deser = enc.createDeserializer()
      iter.map(ir => deser(ir.copy()))
    }
    val df = spark.createDataFrame(backToRow, interStageSchema)
    val shuffled = df.repartition(shuffleParts, col("_leftId"))
    val shuffledInternal: RDD[InternalRow] = shuffled.queryExecution.toRdd

    var iter = 0
    val iterations = 100
    var observedSum = 0L
    while (iter < iterations) {
      val perIterSum = shuffledInternal.mapPartitions { it =>
        var sum = 0L
        while (it.hasNext) {
          val ir = it.next().copy()
          val leftId = ir.getLong(0)
          val lid = ir.getLong(1)
          val qvec = ir.getArray(2)
          var j = 0
          var qsum = 0.0f
          while (j < qvec.numElements()) {
            qsum += qvec.getFloat(j)
            j += 1
          }
          val refs = ir.getArray(3)
          var r = 0
          var refSum = 0L
          while (r < refs.numElements()) {
            val s = refs.getStruct(r, 2)
            refSum += s.getLong(0)
            r += 1
          }
          sum += leftId + lid + qsum.toLong + refSum
        }
        Iterator.single(sum)
      }.collect().sum
      observedSum += perIterSum
      iter += 1
    }
    assertTrue(observedSum != 0L, s"Expected non-zero observed sum after $iterations iterations")
  }
}
