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

import org.apache.spark.sql.SparkSession
import org.lance.index.external.{ExternalIvfPqIndex, ExternalIvfPqIndexParams}

import java.nio.file.{Files, Paths}
import java.security.MessageDigest

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * Build + cache + clean-up management for query-time external Lance vector indexes over
 * direct parquet/Delta scans. Counterpart to [[LanceTempR]] for the new external-index path
 * (sezruby/lance-spark external-index).
 *
 * == Why a separate lifecycle ==
 *
 * Per-query temp-Lance ([[LanceTempR]]) writes ALL of R's columns into a temp Lance dataset
 * because Lance's standard probe path needs a Lance dataset on the right. The external-index
 * path is different: Lance only needs the parquet files + a vector column to build the index,
 * and the materialize stage fetches projection cols directly from those parquet files. So:
 *
 *   - Index files are MUCH smaller than a full temp-Lance dataset (only IVF + PQ + manifest)
 *   - Once built they're reusable across queries on the same source
 *
 * That makes a per-job-built-and-deleted lifecycle wasteful. We cache by content hash of
 * `(filePaths, vectorColumn, params)` so repeat queries on the same data reuse the index.
 *
 * == Cache key ==
 *
 * SHA-256 of: sorted file paths + ":" + vectorColumn + ":" + a stable string of the relevant
 * params. The hash becomes the directory name under [[ScratchDirConfKey]]. Hashing rather than
 * just concatenation keeps the directory name a fixed length even when the file list is huge.
 *
 * == Cleanup ==
 *
 * Indexes registered through [[register]] are deleted on `SparkListenerApplicationEnd` and
 * the JVM shutdown hook, same as [[LanceTempLifecycle]]. We delegate to the existing
 * `LanceTempLifecycle` since the cleanup mechanics are identical.
 *
 * == Conf ==
 *
 * `spark.lance.knn.externalIndex.dir` controls scratch root. In local mode this defaults to
 * `${java.io.tmpdir}/lance-knn-external-index`; in cluster mode the conf MUST be set to a
 * shared filesystem (s3://, abfss://, hdfs://...).
 */
private[knn] object ExternalIndexLifecycle {

  /** Conf for scratch root directory. */
  val ScratchDirConfKey: String = "spark.lance.knn.externalIndex.dir"

  /**
   * Driver-side cache: cacheKey -> (indexUri, params). When a job asks for an index over the
   * same parquet files + vector column + params, we hand back the same URI. Survives across
   * Spark jobs within one application; cleared on application end.
   */
  private val builtIndexes = new mutable.HashMap[String, String]

  /**
   * Build (or reuse) an external index over `filePaths` with the requested `vectorColumn` and
   * `params`. Returns the URI of the index directory (the `<uuid>` subdirectory under the
   * scratch root, suitable for `ExternalIvfPqIndex.open`).
   *
   * Idempotent: a second call with the same arguments inside the same Spark session returns
   * the same URI without rebuilding.
   */
  def buildOrReuse(
      spark: SparkSession,
      filePaths: Seq[String],
      vectorColumn: String,
      params: ExternalIvfPqIndexParams): String = synchronized {
    val key = cacheKey(filePaths, vectorColumn, params)
    builtIndexes.get(key) match {
      case Some(uri) =>
        uri
      case None =>
        val scratch = resolveScratchDir(spark)
        val outputUri = s"$scratch/$key"
        val sortedPaths = filePaths.sorted
        val uuid = ExternalIvfPqIndex.build(sortedPaths.asJava, vectorColumn, outputUri, params)
        val indexUri = s"$outputUri/$uuid"
        builtIndexes.put(key, indexUri)
        // Reuse the temp lifecycle's cleanup machinery — it deletes any registered URI on
        // application end / JVM shutdown via Hadoop FileSystem so it works for cloud paths too.
        LanceTempLifecycle.register(spark, outputUri)
        indexUri
    }
  }

  /**
   * Resolve scratch root from session conf, falling back to `${java.io.tmpdir}/lance-knn-
   * external-index` in local mode. Mirrors [[LanceTempR.resolveScratchDir]] but the conf key
   * is ours.
   */
  def resolveScratchDir(spark: SparkSession): String = {
    // Read from BOTH SparkSession's runtime conf (set via `spark.conf.set(...)`) and the
    // immutable SparkContext conf. The benchmark sets the dir via spark.conf.set; the
    // static SparkConf wouldn't see it.
    val sessionConfigured =
      try Option(spark.conf.get(ScratchDirConfKey)).map(_.trim).filter(_.nonEmpty)
      catch { case _: java.util.NoSuchElementException => None }
    val staticConf = spark.sparkContext.getConf
    val staticConfigured =
      Option(staticConf.get(ScratchDirConfKey, null)).map(_.trim).filter(_.nonEmpty)
    sessionConfigured.orElse(staticConfigured) match {
      case Some(dir) =>
        dir
      case None =>
        // Cluster-mode fail-fast guard: if the master URL doesn't look local, refuse to fall
        // back to local-fs default — that would write the index to driver-local disk and the
        // executors would fail to read it. Mirrors LanceTempR's behavior.
        val isLocal = Option(staticConf.get("spark.master", null)).exists(_.startsWith("local"))
        if (!isLocal) {
          throw new IllegalStateException(
            s"$ScratchDirConfKey is not set and spark.master is not local. " +
              "Cluster mode requires a shared filesystem path (s3://, abfss://, hdfs://...).")
        }
        val tmp = Paths.get(System.getProperty("java.io.tmpdir"), "lance-knn-external-index")
        Files.createDirectories(tmp)
        tmp.toAbsolutePath.toString
    }
  }

  /** Hash inputs into a stable directory name. */
  private def cacheKey(
      filePaths: Seq[String],
      vectorColumn: String,
      params: ExternalIvfPqIndexParams): String = {
    val md = MessageDigest.getInstance("SHA-256")
    filePaths.sorted.foreach(p => md.update((p + "\n").getBytes("UTF-8")))
    md.update(s"vec=$vectorColumn\n".getBytes("UTF-8"))
    md.update(s"np=${params.getNumPartitions}\n".getBytes("UTF-8"))
    md.update(s"sv=${params.getNumSubVectors}\n".getBytes("UTF-8"))
    md.update(s"nb=${params.getNumBitsPerSubVector}\n".getBytes("UTF-8"))
    md.update(s"m=${params.getMetric.toString}\n".getBytes("UTF-8"))
    md.update(s"mi=${params.getMaxIters}\n".getBytes("UTF-8"))
    md.update(s"sr=${params.getSampleRate}\n".getBytes("UTF-8"))
    md.update(s"sd=${params.getSeed}\n".getBytes("UTF-8"))
    val digest = md.digest()
    val hex = digest.map(b => f"$b%02x").mkString
    // Truncate for friendly directory names; 16 hex = 64 bits of entropy is plenty.
    hex.substring(0, 16)
  }

  /** For tests: drop all driver-side cache entries. Does not delete files on disk. */
  def clearCacheForTesting(): Unit = synchronized { builtIndexes.clear() }

  /** For tests: count of cached indexes. */
  def cacheSizeForTesting: Int = synchronized { builtIndexes.size }
}
