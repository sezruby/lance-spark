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
package org.lance.spark.knn

import org.apache.spark.sql.{DataFrame, LanceKnnDatasetBridge}
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.types._
import org.lance.spark.knn.internal.{LanceFragments, LanceMaterializeStage, LanceMergeStage, LanceProbeStage, Metric}
import org.lance.spark.knn.internal.staged.{LanceKnnStagedStrategy, LanceMaterializeLogicalPlan, LanceMergeLogicalPlan, LanceProbeLogicalPlan, ProbedLeftCodec}

/**
 * Public entry point for the indexed nearest-by join over a Lance dataset.
 *
 * Phase 1 — staged RDD pipeline. The previous Phase 0 inline `mapPartitions` is split into a
 * three-stage pipeline that mirrors the IMPL_PLAN's eventual physical-operator design:
 *
 * {{{
 *   left.rdd
 *     -- leftKeyed (zipWithUniqueId)  --> leftId stable per row
 *     -- LanceProbeStage              --> (leftId, ProbedLeft)        per-task probe
 *     -- reduceByKey (Exchange)       --> shuffle by hash(leftId)     -- refs travel through
 *     -- LanceMergeStage              --> (leftId, ProbedLeft)        K-way bounded merge
 *     -- LanceMaterializeStage        --> Row                         point-fetch right rows
 * }}}
 *
 * Public API is unchanged from Phase 0 — this is a pure refactor. The shuffle is degenerate in
 * Phase 1 (single contributor per `leftId` because we still probe the whole dataset from each
 * task) but is structurally present, so plan-shape inspection sees the staged form.
 *
 * == Trade-offs vs. Phase 0 ==
 *
 *  - Phase 0 had no shuffle: probe + materialize ran in the same `mapPartitions`. Phase 1 adds an
 *    `Exchange` (`reduceByKey`) between the stages, shipping `(leftId, ProbedLeft)` pairs across
 *    the network. With single-task probing this is wasted bandwidth — the IMPL_PLAN's
 *    "refs only ~24B" bandwidth win lands when fragment-grouping is wired in (multiple probe
 *    tasks per right dataset). Until then Phase 1 is strictly slower than Phase 0 on small data.
 *  - The materialize stage now opens its own Lance dataset handle per task (separate task from
 *    the probe stage). One extra manifest read per task — cheap on Lance.
 *
 * Limitations carried forward unchanged from Phase 0:
 *  - No left broadcast / no per-fragment partitioning (Phase 1.5).
 *  - No filter pushdown into the probe (Phase 3).
 *  - Uses a synthetic `leftId` from `zipWithUniqueId`, not a user-supplied join key. Means we
 *    can't yet co-partition the left payload alongside `(leftId, refs)` to drop the leftRow
 *    from the shuffle.
 */
object IndexedNearestJoin {

  /**
   * Run an approximate-nearest-neighbor join.
   *
   * @param left          left DataFrame; one query vector per row in `leftVecCol`
   * @param rightLanceUri Lance dataset URI for the right side
   * @param leftVecCol    name of the vector column in `left`. Must be `ArrayType[Float]`.
   * @param rightVecCol   name of the indexed (or to-be-searched) vector column on the right
   * @param k             top-K rows per left row
   * @param metric        distance/similarity metric: "l2" / "cosine" / "dot" (and synonyms)
   * @param rightProjection columns to materialize from the right side. Defaults to all data
   *                      columns. The score column is added separately.
   * @param outerJoin     when true, left rows with zero matches are preserved with NULL right-
   *                      side columns. Defaults to false (inner-join semantics).
   * @param scoreCol      name of the synthesized score column added to the output. Defaults to
   *                      `__score`.
   * @param overfetch     ratio of internal candidates to k for indexed approximate metrics; with
   *                      no index this has no effect because Lance returns exact top-K. Defaults
   *                      to 1 because the merge stage's value comes from N-task aggregation, not
   *                      single-task overfetch.
   * @param nprobes       optional override of Lance's `nprobes` for IVF-PQ indexes
   * @param version       optional Lance version pin; if unset, latest version is used
   * @param refineFactor  IVF-PQ recall knob. When set, Lance fetches `k * refineFactor`
   *                      approximate candidates, re-ranks them with exact distance, and trims
   *                      back to k. Higher = better recall, more compute. `None` leaves Lance's
   *                      default (= 1, no re-rank). Ignored for non-IVF-PQ indexes / unindexed.
   * @param ef            HNSW search depth. Higher = better recall, more compute. `None` leaves
   *                      Lance's default (the index's build-time `ef_construction` value).
   *                      Ignored for non-HNSW indexes / unindexed.
   * @param balanceFragmentsByRowCount when true (Phase 1.5 + Phase 3 skew handling), fragment
   *                      groups are formed via LPT greedy bin-packing on per-fragment row
   *                      counts so groups have roughly equal total work. Default `false` =
   *                      round-robin (cheaper, fine when fragments are evenly sized).
   * @param probeParallelism Phase 1.5 fragment-grouping degree. `1` (default) keeps the Phase 1
   *                      path: one task probes the whole dataset per left row, with Lance doing
   *                      the cross-fragment merge internally. `> 1` enumerates Lance fragments
   *                      on the driver, splits them into N round-robin groups, and replicates
   *                      each left row across the N groups so the merge stage actually has work
   *                      to do. Capped at the number of Lance fragments — extra groups are
   *                      empty and skipped.
   */
  def apply(
      left: DataFrame,
      rightLanceUri: String,
      leftVecCol: String,
      rightVecCol: String,
      k: Int,
      metric: String = "l2",
      rightProjection: Option[Seq[String]] = None,
      outerJoin: Boolean = false,
      scoreCol: String = "__score",
      overfetch: Int = 1,
      nprobes: Option[Int] = None,
      version: Option[Long] = None,
      probeParallelism: Int = 1,
      refineFactor: Option[Int] = None,
      ef: Option[Int] = None,
      balanceFragmentsByRowCount: Boolean = false): DataFrame = {

    require(k > 0, "k must be positive")
    require(overfetch >= 1, "overfetch must be >= 1")
    require(probeParallelism >= 1, "probeParallelism must be >= 1")

    val spark = left.sparkSession
    val parsedMetric = Metric.fromName(metric)
    val internalK = k * overfetch

    // Snapshot right-side schema on the driver before any executor work happens.
    val rightSchema: StructType = {
      val reader = spark.read.format("lance")
      version.foreach(v => reader.option("version", v.toString))
      val raw = reader.load(rightLanceUri)
      val pruned = rightProjection match {
        case Some(cols) if cols.nonEmpty => raw.select(cols.head, cols.tail: _*)
        case _ => raw
      }
      pruned.schema
    }

    val outputSchema = buildOutputSchema(left.schema, rightSchema, scoreCol)
    val leftFieldCount = left.schema.fields.length
    val leftVecIdx = left.schema.fieldIndex(leftVecCol)
    val rightProjectionCols: Seq[String] =
      rightProjection.getOrElse(rightSchema.fieldNames.toSeq)

    val probeConf = LanceProbeStage.Conf(
      datasetUri = rightLanceUri,
      fragmentIds = None,
      vectorColumn = rightVecCol,
      version = version,
      metric = parsedMetric,
      k = internalK,
      nprobes = nprobes,
      leftVecIdx = leftVecIdx,
      refineFactor = refineFactor,
      ef = ef)

    val mergeConf = LanceMergeStage.Conf(
      finalK = k,
      smallerIsBetter = parsedMetric.smallerIsBetter)

    val materializeConf = LanceMaterializeStage.Conf(
      datasetUri = rightLanceUri,
      version = version,
      rightProjection = rightProjectionCols,
      rightFields = rightSchema.fields.toSeq,
      leftFieldCount = leftFieldCount,
      outerJoin = outerJoin)

    // Driver-side fragment-group enumeration for the Phase 1.5 path. Done here so the
    // probe operator doesn't have to talk to Lance's Java API during planning; the result
    // is carried in the logical plan as a serialisable field.
    val fragmentGroups: Option[Seq[Seq[Int]]] = if (probeParallelism > 1) {
      val rawGroups = if (balanceFragmentsByRowCount) {
        LanceFragments.enumerateGroupsByRowCount(rightLanceUri, version, probeParallelism)
      } else {
        LanceFragments.enumerateGroups(rightLanceUri, version, probeParallelism)
      }
      val nonEmpty = rawGroups.filter(_.nonEmpty)
      if (nonEmpty.size <= 1) None else Some(nonEmpty)
    } else {
      None
    }

    // Three-logical-plan tree:
    //   LanceMaterializeLogicalPlan
    //     ↳ LanceMergeLogicalPlan (requiredChildDistribution=ClusteredDistribution(_leftId)
    //        at the Exec level ⇒ Catalyst inserts `ShuffleExchangeExec` here, AQE engages)
    //       ↳ LanceProbeLogicalPlan
    //         ↳ user's left logical plan
    //
    // The `references = child.outputSet` override on Merge/Materialize (see
    // `StagedPlans.scala`) blocks Catalyst's `ColumnPruning` from inserting
    // `Project(Nil)` wrappers — that insertion was what caused an early 3-exec
    // iteration to crash with `AssertionError` / SIGSEGV in
    // `ProbedLeftCodec.Decoder.decode` reading 0-field UnsafeRows.
    LanceKnnStagedStrategy.ensureRegistered(spark)

    val leftSchema = left.schema
    val interStageAttrs = ProbedLeftCodec.interStageAttributes(leftSchema)
    val finalAttrs = outputSchema.fields.map { f =>
      AttributeReference(f.name, f.dataType, f.nullable, f.metadata)()
    }.toSeq

    val probeLogical = LanceProbeLogicalPlan(
      child = left.queryExecution.analyzed,
      stageConf = probeConf,
      fragmentGroups = fragmentGroups,
      leftSchema = leftSchema,
      interStageOutput = interStageAttrs)
    val mergeLogical = LanceMergeLogicalPlan(
      child = probeLogical,
      stageConf = mergeConf,
      leftSchema = leftSchema,
      interStageOutput = interStageAttrs)
    val materializeLogical = LanceMaterializeLogicalPlan(
      child = mergeLogical,
      stageConf = materializeConf,
      leftSchema = leftSchema,
      finalSchema = outputSchema,
      finalOutput = finalAttrs)

    LanceKnnDatasetBridge.asDataFrame(spark, materializeLogical)
  }

  private def buildOutputSchema(
      left: StructType,
      right: StructType,
      scoreCol: String): StructType = {
    val rightNullable = right.fields.map(f => f.copy(nullable = true))
    val score = StructField(scoreCol, FloatType, nullable = true)
    StructType(left.fields ++ rightNullable :+ score)
  }
}
