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

import org.apache.hadoop.fs.{FileSystem, Path => HadoopPath}
import org.apache.spark.SparkContext
import org.apache.spark.scheduler.{SparkListener, SparkListenerApplicationEnd}
import org.apache.spark.sql.SparkSession

import java.io.{File, IOException}
import java.net.URI

import scala.collection.mutable

/**
 * Query-scoped cleanup for the per-query temp Lance datasets created by
 * [[LanceTempR.materialize]]. Without it, every `kNearestJoin` against a non-Lance R
 * leaks a Lance dataset on whatever scratch storage `spark.lance.knn.tempR.dir`
 * points at — local FS, S3, HDFS — until the JVM dies.
 *
 * == Design ==
 *
 * Cleanup runs on `SparkListenerApplicationEnd` and at JVM-shutdown via a `Runtime`
 * shutdown hook. We deliberately do NOT clean up on `onJobEnd`: a single
 * `kNearestJoin` invocation can run multiple Spark jobs (the temp write itself,
 * the probe stage, the merge shuffle, the materialize stage). Tying cleanup to
 * `onJobEnd` would race the still-running probe and break correctness.
 *
 * `onApplicationEnd` covers the well-behaved case where the SparkSession stops
 * cleanly. The shutdown hook covers crashes / hard kills. Either way the temp
 * dirs registered up to that point are deleted on a best-effort basis (errors
 * are logged but never re-thrown — cleanup must not break the user's job tear-
 * down).
 *
 * == Why not call `Files.delete` directly ==
 *
 * Temp URIs may live on object stores (s3://...), HDFS, ABFS — non-local FS.
 * `java.nio.file.Files` only handles local FS. We dispatch through
 * `org.apache.hadoop.fs.FileSystem.get(uri, conf)` which routes via the standard
 * Spark/Hadoop FileSystem registry — same machinery `df.write.format("lance")`
 * already uses to write the temp.
 */
private[knn] object LanceTempLifecycle {

  // Logger via Spark's slf4j via -- log directly with println to stderr if we can't import.
  // Kept small to avoid pulling slf4j into the Lance-knn module's surface.
  private def logWarn(msg: String): Unit = System.err.println(s"[LanceTempLifecycle] $msg")
  private def logInfo(msg: String): Unit = {} // intentionally quiet at info level

  // Synchronised because Spark task threads, listener-bus threads, and the JVM shutdown
  // thread can all touch this. Per-application instances live forever in a static map;
  // cleanup is keyed on the application id so we can be sure not to drop a different
  // app's temps when a SparkContext stops within the same JVM.
  private val instances = new mutable.HashMap[String, ApplicationTempRegistry]

  // Single shutdown hook for the JVM, installed on first `register` call. Runs all
  // application registries' cleanup paths.
  private val shutdownHookInstalled = new java.util.concurrent.atomic.AtomicBoolean(false)

  /**
   * Track `tempUri` for cleanup when `spark`'s application ends or the JVM exits, whichever
   * comes first. Idempotent: if `tempUri` is already registered for this application, no-op.
   */
  def register(spark: SparkSession, tempUri: String): Unit = synchronized {
    val sc = spark.sparkContext
    val appId = sc.applicationId
    val registry = instances.getOrElseUpdate(
      appId, {
        val r = new ApplicationTempRegistry(sc, appId)
        sc.addSparkListener(r)
        r
      })
    registry.add(tempUri)
    ensureShutdownHook()
  }

  /** Drop all registered temp URIs for `appId` and clean up the listener. Public for tests. */
  private[knn] def stopForTesting(appId: String): Unit = synchronized {
    instances.remove(appId).foreach(_.cleanupAll())
  }

  /** Number of currently-registered temp URIs for an app — for assertions in tests. */
  private[knn] def registeredCount(appId: String): Int = synchronized {
    instances.get(appId).map(_.size).getOrElse(0)
  }

  private def ensureShutdownHook(): Unit = {
    if (shutdownHookInstalled.compareAndSet(false, true)) {
      Runtime.getRuntime.addShutdownHook(new Thread("lance-temp-r-cleanup") {
        override def run(): Unit = LanceTempLifecycle.synchronized {
          instances.values.foreach(_.cleanupAll())
          instances.clear()
        }
      })
    }
  }

  /**
   * Per-application registry of temp URIs. Subscribes to `SparkListenerApplicationEnd`
   * so cleanup fires as soon as the SparkContext starts shutting down — before the
   * scratch FS becomes unreachable in cluster-tear-down ordering.
   */
  final private class ApplicationTempRegistry(sc: SparkContext, appId: String)
    extends SparkListener {

    private val tempUris = new mutable.LinkedHashSet[String]
    private val hadoopConf = sc.hadoopConfiguration

    def add(uri: String): Unit = LanceTempLifecycle.synchronized {
      tempUris.add(uri)
    }

    def size: Int = LanceTempLifecycle.synchronized(tempUris.size)

    override def onApplicationEnd(end: SparkListenerApplicationEnd): Unit = {
      cleanupAll()
      LanceTempLifecycle.synchronized {
        instances.remove(appId)
      }
    }

    /**
     * Best-effort delete of every registered temp URI. Errors are logged and swallowed —
     * cleanup runs during shutdown / context-stop, where re-throwing would obscure the
     * actual reason the application is ending.
     */
    def cleanupAll(): Unit = LanceTempLifecycle.synchronized {
      val snapshot = tempUris.toSeq
      tempUris.clear()
      snapshot.foreach(deleteSilently)
    }

    private def deleteSilently(uri: String): Unit = {
      try {
        deleteUri(uri, hadoopConf)
        logInfo(s"deleted temp Lance dataset: $uri")
      } catch {
        case e: Throwable =>
          logWarn(s"failed to delete temp Lance dataset '$uri': ${e.getClass.getSimpleName}: ${e.getMessage}")
      }
    }
  }

  /**
   * Delete `uri` recursively. Routes through the same Hadoop FileSystem registry that
   * Spark uses for writes, so it handles local FS, S3, HDFS, ABFS, etc. uniformly.
   */
  private[knn] def deleteUri(
      uri: String,
      hadoopConf: org.apache.hadoop.conf.Configuration): Unit = {
    if (uri == null || uri.isEmpty) return
    if (looksLikeBareLocalPath(uri)) {
      // Hadoop FileSystem.get(uri) on a bare path can sometimes route through unexpected
      // FS implementations on YARN clusters. For unambiguous local paths use java.nio.
      val f = new File(uri)
      if (f.exists()) deleteRecursive(f)
    } else {
      val javaUri: URI = new URI(uri)
      val hPath = new HadoopPath(uri)
      val fs = FileSystem.get(javaUri, hadoopConf)
      if (fs.exists(hPath)) {
        if (!fs.delete(hPath, /* recursive = */ true)) {
          throw new IOException(s"FileSystem.delete returned false for $uri")
        }
      }
    }
  }

  private def looksLikeBareLocalPath(uri: String): Boolean =
    !uri.contains("://")

  private def deleteRecursive(f: File): Unit = {
    if (f.isDirectory) {
      val children = f.listFiles()
      if (children != null) children.foreach(deleteRecursive)
    }
    if (!f.delete() && f.exists()) {
      throw new IOException(s"failed to delete $f")
    }
  }
}
