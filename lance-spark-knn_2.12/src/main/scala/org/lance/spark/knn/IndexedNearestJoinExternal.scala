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

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._
import org.lance.index.external.ExternalIvfPqIndexParams
import org.lance.spark.knn.internal.{ExternalFusedStage, ExternalIndexLifecycle, ExternalIndexProbe, Metric}

/**
 * Public entry point for the indexed nearest-by join when the right side is a set of
 * caller-supplied parquet files (no Lance dataset required). Sibling to [[IndexedNearestJoin]]
 * which targets a Lance dataset.
 *
 * == Pipeline ==
 *
 * Single Spark job, no SQL Catalyst integration (deferred to Phase 3). Probe + materialize
 * fused into one stage — no leftId shuffle, no inter-stage data ship:
 *
 * {{{
 *   left.rdd
 *     -- ExternalFusedStage.run              per task: open index once, probe each
 *                                            left row (Lance returns global top-K),
 *                                            then batched fetch_rows for the partition,
 *                                            emit final join Rows
 * }}}
 *
 * The shuffle in earlier iterations was inherited from the Lance-native pipeline where
 * a single left row could be probed by multiple tasks (fragment-grouped probe). Lance's
 * `idx.search()` already merges across partitions internally, so each leftId has
 * exactly one contributor — the merge stage was a passthrough and the shuffle was
 * vestigial. Removing it eliminates one Spark Exchange and the leftRow shuffle bytes.
 *
 * == Why no Catalyst integration in Phase 2 ==
 *
 * The Lance-dataset path uses three custom logical plans + a registered strategy so
 * `df.explain()` shows the staged pipeline as named operators. Replicating that pattern for
 * external-index would mean three more logical plans + three more execs + strategy entries —
 * substantial boilerplate before any benchmark proves the path is faster than temp-Lance for
 * SQL queries. Phase 2 keeps it imperative so we can ship and measure. Phase 3 promotes the
 * winning shape to Catalyst when the numbers warrant.
 *
 * == Index lifecycle ==
 *
 * The driver builds (or reuses, via the [[ExternalIndexLifecycle]] cache) an external IVF-PQ
 * index over the parquet files at job-submit time. The index URI is then broadcast through
 * the stage configurations. Cleanup is registered with [[org.lance.spark.knn.internal.LanceTempLifecycle]]
 * so the scratch directory is removed on application end / JVM shutdown — same machinery as
 * the temp-Lance path.
 */
object IndexedNearestJoinExternal {

  /**
   * Approximate nearest-neighbor join with the right side coming from caller-supplied
   * parquet files.
   *
   * @param left           left DataFrame; one query vector per row in `leftVecCol`
   * @param rightFilePaths parquet files that make up the right side. Order is significant —
   *                       reordering invalidates a cached index built earlier in the same
   *                       application.
   * @param leftVecCol     name of the vector column in `left`. `ArrayType[Float]`.
   * @param rightVecCol    name of the vector column on the parquet files. Must be a
   *                       `FixedSizeList<Float>` column in every file's schema (the build
   *                       step enforces this).
   * @param k              top-K rows per left row.
   * @param metric         "l2", "cosine", or "dot".
   * @param rightProjection columns to materialize from the right side. Defaults to all
   *                       columns of the parquet schema.
   * @param outerJoin      preserve left rows with zero matches.
   * @param scoreCol       output score column name.
   * @param nprobes        IVF probe width.
   * @param refineFactor   IVF-PQ refine multiplier.
   * @param indexParams    optional override for the index build (kmeans iterations, sample
   *                       rate, etc.). Defaults to [[ExternalIvfPqIndexParams.builder]] with
   *                       the metric set from `metric`.
   * @param mergeParallelism number of partitions for the hash-shuffle between probe and
   *                       merge. Defaults to `spark.sql.shuffle.partitions`.
   */
  def apply(
      left: DataFrame,
      rightFilePaths: Seq[String],
      leftVecCol: String,
      rightVecCol: String,
      k: Int,
      metric: String = "l2",
      rightProjection: Option[Seq[String]] = None,
      outerJoin: Boolean = false,
      scoreCol: String = "__score",
      nprobes: Int = 16,
      refineFactor: Int = 8,
      indexParams: Option[ExternalIvfPqIndexParams] = None,
      mergeParallelism: Option[Int] = None): DataFrame = {

    require(k > 0, "k must be positive")
    require(rightFilePaths.nonEmpty, "rightFilePaths must contain at least one path")
    require(nprobes > 0, "nprobes must be positive")
    require(refineFactor > 0, "refineFactor must be positive")

    val spark = left.sparkSession
    val parsedMetric = Metric.fromName(metric)
    val params = indexParams.getOrElse(ExternalIndexProbe.defaultParams(parsedMetric))

    // Driver-side: build (or reuse) the index. This also registers cleanup.
    val indexUri =
      ExternalIndexLifecycle.buildOrReuse(spark, rightFilePaths, rightVecCol, params)

    // Snapshot the parquet schema for the right side once on the driver. We need this for
    // both the output schema and the projection-column list.
    val rightSchema: StructType = {
      val raw = spark.read.parquet(rightFilePaths: _*)
      raw.schema
    }
    val rightProjectionCols: Seq[String] =
      rightProjection.getOrElse(rightSchema.fieldNames.toSeq)
    // Filter rightSchema to the projection in projection order so the output schema matches.
    val rightProjectedFields: Seq[StructField] = rightProjectionCols.map(rightSchema.apply)

    val outputSchema = buildOutputSchema(left.schema, rightProjectedFields, scoreCol)
    val leftFieldCount = left.schema.fields.length
    val leftVecIdx = left.schema.fieldIndex(leftVecCol)

    // Sort the file paths for deterministic file_id assignment. The lifecycle's cache key
    // already sorts, but the Conf carried into the stages must agree with whatever the
    // index was built over — so we sort here too and the resulting file_ids match the
    // index manifest.
    val sortedFilePaths = rightFilePaths.sorted.toArray

    val fusedConf = ExternalFusedStage.Conf(
      indexUri = indexUri,
      filePaths = sortedFilePaths,
      vectorColumn = rightVecCol,
      metric = parsedMetric,
      k = k,
      nprobes = nprobes,
      refineFactor = refineFactor,
      leftVecIdx = leftVecIdx,
      rightProjection = rightProjectionCols,
      rightFields = rightProjectedFields,
      leftFieldCount = leftFieldCount,
      outerJoin = outerJoin)
    val _ = mergeParallelism // unused after fusion; kept on the API for back-compat

    val leftRdd: RDD[Row] = left.rdd
    val joinedRows: RDD[Row] = ExternalFusedStage.run(leftRdd, fusedConf)

    spark.createDataFrame(joinedRows, outputSchema)
  }

  private def buildOutputSchema(
      left: StructType,
      rightFields: Seq[StructField],
      scoreCol: String): StructType = {
    val rightNullable = rightFields.map(f => f.copy(nullable = true))
    val score = StructField(scoreCol, FloatType, nullable = true)
    StructType(left.fields ++ rightNullable :+ score)
  }

  // unused stub kept to silence linter when the file is loaded standalone in tooling
  private[knn] def _unused: SparkSession => DataFrame = s => s.emptyDataFrame.select(col("*"))
}
