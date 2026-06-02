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

    // Repartition `left` before the fused stage so probe + materialize both run with
    // a configurable parallelism, independent of however the user partitioned `left`.
    // Without this, a small `|L|` (typical for KNN — hundreds to thousands of queries)
    // typically lands in 1-2 Spark partitions because Spark right-sizes for the row
    // count, leaving fetchRows running serially. With this repartition,
    // `mergeParallelism` (default = spark.sql.shuffle.partitions) tasks each probe a
    // slice of left rows and run their own fetchRows pass against the shared index.
    //
    // The `mergeParallelism` parameter name is back-compat from when the merge stage
    // existed. Today it controls the fused stage's parallelism. Renaming would break
    // the API without functional benefit.
    // Decide parallelism for the fused stage. The fused stage runs `left.rdd.partitions`
    // tasks; for KNN workloads `|L|` (number of query vectors) is often small relative
    // to cluster cores and Spark right-sizes left's partition count to row count, which
    // can leave the fused stage running 1-2 tasks regardless of how many cores are
    // available. We size parallelism from the source row count when the caller doesn't
    // specify, with these rules:
    //
    //   1. Explicit `mergeParallelism` from caller wins (caller knows their workload).
    //   2. Otherwise we estimate `numL` from Spark's optimizer stats. If a reliable
    //      estimate is available (cached DataFrame or upstream stats), use
    //      `targetTasks = ceil(numL / TargetRowsPerTask)`, capped at the cluster's
    //      `defaultParallelism` (so we don't over-shard tiny inputs into many empty
    //      tasks).
    //   3. Cap from below by `left.rdd.partitions.size` — we only repartition UP, never
    //      down. If the user already has good partitioning we leave it alone.
    //
    // `TargetRowsPerTask` is a heuristic. Each task processes its share serially
    // through Lance's idx.search(); one search() takes ~5-50ms at typical scales, so
    // ~100 rows/task gives ~0.5-5 sec per task — comfortable for Spark scheduling
    // overhead.
    //
    // The `mergeParallelism` parameter name is back-compat from when an explicit merge
    // stage existed. Today it controls the fused stage's parallelism.
    val targetRowsPerTask = TargetRowsPerTask
    val defaultPar = spark.sparkContext.defaultParallelism.max(1)
    val shufflePar = spark.conf.get("spark.sql.shuffle.partitions").toInt.max(defaultPar)
    val estimatedRows: Long = {
      val stats = left.queryExecution.optimizedPlan.stats
      // stats.rowCount is Optional in Spark ≥3.0; check via java.util.Optional API.
      val r = stats.rowCount
      if (r.isDefined) r.get.toLong else -1L
    }
    val sizedParallelism: Int =
      if (estimatedRows > 0) {
        val want = ((estimatedRows + targetRowsPerTask - 1) / targetRowsPerTask).toInt
        // Cap at shufflePar (= max(spark.sql.shuffle.partitions, defaultParallelism)).
        // This is what Spark uses for shuffle parallelism by convention; capping there
        // gives tasks finer-grained scheduling than capping at defaultParallelism, which
        // matters when |L| is large (1M+ queries) and the cluster has substantial cores.
        math.max(1, math.min(want, shufflePar))
      } else {
        // No stats available — fall back to shufflePar.
        shufflePar
      }
    val parallelism = mergeParallelism.getOrElse(sizedParallelism)
    val leftPreRdd: RDD[Row] = left.rdd
    val currentParts = leftPreRdd.getNumPartitions
    // scalastyle:off println
    println(
      s"[IndexedNearestJoinExternal] left.partitions = $currentParts, " +
        s"left.estimatedRows = $estimatedRows, " +
        s"defaultParallelism = $defaultPar, " +
        s"shuffle.partitions = $shufflePar, " +
        s"target parallelism = $parallelism")
    // scalastyle:on println
    val leftRdd: RDD[Row] =
      if (currentParts < parallelism) leftPreRdd.repartition(parallelism) else leftPreRdd
    val joinedRows: RDD[Row] = ExternalFusedStage.run(leftRdd, fusedConf)

    spark.createDataFrame(joinedRows, outputSchema)
  }

  /** Heuristic target rows per fused-stage task. See `apply`'s parallelism comment. */
  private val TargetRowsPerTask: Long = 100L

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
