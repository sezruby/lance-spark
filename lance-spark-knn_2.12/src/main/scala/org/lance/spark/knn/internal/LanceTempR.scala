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
package org.lance.spark.knn.internal

import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, monotonically_increasing_id}

import java.nio.file.{Files, Path, Paths}
import java.util.UUID

/**
 * Per-query temp Lance materialization for the right side of an indexed
 * `NearestByJoin` when R is not already a Lance scan (parquet, delta, in-memory,
 * arbitrary subplan). Materialization is eager: when called, the helper drives
 * `right.write.format("lance").save(tempUri)` and returns the URI. The caller
 * then passes that URI to the existing Lance-native probe pipeline; the rest of
 * the staged plan (probe / merge / materialize) is unchanged.
 *
 * Design: see [sezruby/lance-spark#2](https://github.com/sezruby/lance-spark/issues/2).
 *
 * == Why a synchronous helper rather than a Catalyst exec node ==
 *
 * The probe pipeline reads R via `LanceProbeStage.Conf.datasetUri`, which is captured
 * once at plan construction time. To plumb a temp-write through Catalyst as a separate
 * node we'd have to restructure `LanceProbeExec` (currently `UnaryExecNode` with the
 * left side as its child) into a multi-input shape that depends on both the left plan
 * AND a sibling temp-write — substantial blast radius. The simpler form: do the temp
 * write at plan-build time, hand the resulting URI to the probe like any other Lance
 * URI. Same data path on the wire; the only loss is `df.explain()` doesn't display the
 * temp write as its own Catalyst operator. A future PR can promote it to an exec node
 * if `df.explain()` visibility becomes load-bearing for users.
 *
 * == Why `monotonically_increasing_id` is the right rid ==
 *
 * Per-query temp doesn't need cross-execution rid stability — temp is built and consumed
 * in the same job, then deleted. `monotonically_increasing_id` is unique within a single
 * execution and zero-cost to compute, so it fits exactly. For a future cached / persistent
 * sidecar (different feature, not this issue), `_metadata.row_index` would be the natural
 * choice for parquet-backed R.
 */
private[knn] object LanceTempR {

  /**
   * Column name for the synthetic rid that the probe stage's `_rowid IN (...)` lookup
   * targets after the write. Stable identifier — referenced by callers when constructing
   * `rightProjection`.
   */
  val RidColumnName: String = "_rid"

  /**
   * User-tunable Spark conf key pointing at the directory under which per-query temp
   * Lance datasets are created. In cluster mode this MUST be set to a path every
   * executor (and the driver) can read+write — typically a shared object-store URI
   * (`s3://...`, `abfss://...`, `file:///shared-mount/...`) or HDFS. In local mode the
   * helper falls back to a subdirectory of `spark.local.dir` when this key is unset.
   */
  val ScratchDirConfKey: String = "spark.lance.knn.tempR.dir"

  /**
   * Materialize `right` to a temp Lance dataset suitable for use as the right side of
   * the existing indexed `NearestByJoin` pipeline.
   *
   * Steps:
   *   1. Compute the projection schema: rid (synthesised) + vec + any caller-requested
   *      additional columns the parent plan references.
   *   2. Build a projected DataFrame: `right.select(monotonically_increasing_id() as _rid,
   *      ...projection)`.
   *   3. Write it to `<scratchDir>/<unique-token>` as a Lance dataset.
   *   4. Return the URI string.
   *
   * The caller is responsible for:
   *   - Deleting the temp directory when the query finishes (see `LanceTempLifecycle`).
   *   - Telling the probe pipeline to project / materialize `RidColumnName` and `vecCol`
   *     (and any payload columns it should carry).
   *
   * @param right       The non-Lance DataFrame to materialize.
   * @param vecCol      Name of the FixedSizeList<f32, dim> vector column on `right`.
   * @param projection  Columns from `right` to carry into temp Lance, in addition to the
   *                    synthesised rid and the vector. Empty Seq = carry rid + vec only.
   *                    Use this to thread any payload columns the parent plan references.
   * @param scratchDir  Directory under which to create the temp Lance dataset. Must
   *                    be a path the executor processes can write to (local FS for
   *                    single-node, shared object store for cluster mode). Use
   *                    [[resolveScratchDir]] to pick this up from session config in
   *                    typical callers.
   * @return The URI of the materialized temp Lance dataset.
   */
  def materialize(
      right: DataFrame,
      vecCol: String,
      projection: Seq[String],
      scratchDir: String): String = {
    require(vecCol.nonEmpty, "vecCol must not be empty")
    require(scratchDir.nonEmpty, "scratchDir must not be empty")
    require(
      right.schema.fieldNames.contains(vecCol),
      s"right DataFrame schema does not contain vector column '$vecCol'; " +
        s"have [${right.schema.fieldNames.mkString(", ")}]")
    val unknownCols = projection.filterNot(right.schema.fieldNames.contains)
    require(
      unknownCols.isEmpty,
      s"projection columns not present in right DataFrame schema: " +
        s"[${unknownCols.mkString(", ")}]; have [${right.schema.fieldNames.mkString(", ")}]")
    require(
      !projection.contains(RidColumnName),
      s"projection must not include the reserved rid column name '$RidColumnName' — " +
        "the helper synthesises it. Pick a different name on `right` or rename before calling.")

    val tempUri = mintTempUri(scratchDir)
    val ridCol: Column = monotonically_increasing_id().as(RidColumnName)
    val payloadCols: Seq[Column] = (vecCol +: projection.filterNot(_ == vecCol)).distinct.map(col)
    val projected: DataFrame = right.select(ridCol +: payloadCols: _*)

    projected.write.format("lance").save(tempUri)

    // Register for query-scoped cleanup. Cleanup fires on SparkListenerApplicationEnd
    // (when the SparkSession stops cleanly) and on JVM shutdown via a shutdown hook
    // (covers crashes / hard kills). See LanceTempLifecycle.
    LanceTempLifecycle.register(right.sparkSession, tempUri)

    tempUri
  }

  /**
   * Resolve a writable scratch directory from session configuration, with a clear error
   * for cluster runs that haven't set [[ScratchDirConfKey]].
   *
   * Resolution order:
   *   1. `spark.lance.knn.tempR.dir` if set — used as-is. Caller is responsible for it
   *      being executor-readable.
   *   2. Local-mode-only fallback: `spark.local.dir` first entry + `/lance-temp-r`. Only
   *      acceptable in `local[*]` mode where the driver and executors share a JVM and
   *      hence the local FS.
   *
   * In a cluster (`spark.master` does not start with `local`), missing
   * [[ScratchDirConfKey]] throws [[IllegalStateException]] — failing here is much better
   * than failing later with a `FileNotFoundException` on an executor that can't see the
   * driver's local disk.
   */
  def resolveScratchDir(spark: SparkSession): String = {
    spark.conf.getOption(ScratchDirConfKey).filter(_.nonEmpty) match {
      case Some(p) => p
      case None =>
        val master = Option(spark.sparkContext.master).getOrElse("")
        if (!master.startsWith("local")) {
          throw new IllegalStateException(
            s"$ScratchDirConfKey is not set and Spark master is '$master' — per-query " +
              "temp Lance materialization needs a scratch path every executor can " +
              s"read+write. Set $ScratchDirConfKey to a shared URI (s3://..., abfss://..., " +
              "file:///shared-mount/..., hdfs://...).")
        }
        val localDir = Option(spark.sparkContext.getConf.get("spark.local.dir", null))
          .map(_.split(",").head.trim)
          .filter(_.nonEmpty)
          .getOrElse(System.getProperty("java.io.tmpdir"))
        s"$localDir/lance-temp-r"
    }
  }

  /**
   * Generate a unique scratch path under `scratchDir`. Caller must clean up. The path
   * deliberately does NOT include the substring "lance" — the V2 catalog's path-identifier
   * parser tokenises around it on writes (same workaround documented in
   * `LanceWriteBenchmark`).
   */
  def mintTempUri(scratchDir: String): String = {
    val token = "tempr-" + UUID.randomUUID().toString
    // For a local scratch dir, ensure the parent exists. Object-store URIs (s3://, etc.)
    // are passed through as-is — the lance writer creates the path.
    if (isLocalPath(scratchDir)) {
      val parent: Path = Paths.get(stripFileScheme(scratchDir))
      Files.createDirectories(parent)
      parent.resolve(token).toString
    } else {
      val sep = if (scratchDir.endsWith("/")) "" else "/"
      s"$scratchDir$sep$token"
    }
  }

  private def isLocalPath(uri: String): Boolean =
    uri.startsWith("/") || uri.startsWith("file://") || !uri.contains("://")

  private def stripFileScheme(uri: String): String =
    if (uri.startsWith("file://")) uri.substring("file://".length) else uri
}
