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

import java.util.concurrent.ConcurrentHashMap

/**
 * Per-executor (per-JVM) cache of open [[ExternalIndexProbe]] handles, keyed by index URI.
 *
 * == Why ==
 *
 * The fused stage opens the index once per Spark task. On an executor running C task slots,
 * that's C independent opens of the same index — each re-reads the manifest + index footer and,
 * more expensively, warms its own copy of the native per-source-parquet metadata cache (footers +
 * page indexes) on first touch. Caching one handle per (executor, indexUri) amortizes both across
 * every task on that executor.
 *
 * == Why it's safe to share ==
 *
 * The underlying native handle is an `Arc<ExternalIvfPqIndex>`; `search` / `searchBatch` /
 * `fetchRows` are all `&self`; and the native parquet-metadata cache is `Mutex`-guarded. A single
 * opened index is therefore safe for concurrent calls from parallel task threads on the same
 * executor.
 *
 * == Lifecycle contract ==
 *
 * A handle obtained from [[acquire]] is owned by the cache, not the caller. Callers MUST NOT call
 * `close()` on it — doing so would free the native handle out from under sibling tasks (use-after-
 * free). The cache closes every handle once, on JVM shutdown. This is the one behavioral
 * difference from the per-task path (which opens and closes its own handle) and is why the feature
 * is opt-in.
 *
 * Correctness note on the key: index URIs already embed the content hash + build uuid
 * (`<scratch>/<cacheKey>/<uuid>`), so a given URI always names one immutable index. Reusing a
 * handle across jobs in the same application is therefore sound.
 */
private[knn] object ExternalIndexProbeCache {

  private val handles = new ConcurrentHashMap[String, ExternalIndexProbe]()

  @volatile private var shutdownHookInstalled = false

  /**
   * Return the cached probe for `indexUri`, opening (and caching) it on first request. Concurrent
   * requests for the same URI serialize inside `computeIfAbsent` so the index opens exactly once;
   * requests for different URIs do not block each other.
   */
  def acquire(indexUri: String): ExternalIndexProbe = {
    ensureShutdownHook()
    handles.computeIfAbsent(indexUri, uri => new ExternalIndexProbe(uri))
  }

  /** Number of cached handles. For tests / diagnostics. */
  def size: Int = handles.size

  /**
   * Close and drop all cached handles. Idempotent. Invoked by the JVM shutdown hook; also usable
   * directly from tests. Best-effort: a failure closing one handle does not prevent closing the
   * rest.
   */
  def closeAll(): Unit = {
    val it = handles.keySet().iterator()
    while (it.hasNext) {
      val uri = it.next()
      val probe = handles.remove(uri)
      if (probe != null) {
        try probe.close()
        catch { case _: Throwable => () }
      }
    }
  }

  private def ensureShutdownHook(): Unit = {
    if (!shutdownHookInstalled) {
      synchronized {
        if (!shutdownHookInstalled) {
          Runtime.getRuntime.addShutdownHook(new Thread(
            new Runnable {
              override def run(): Unit = closeAll()
            },
            "lance-external-index-cache-cleanup"))
          shutdownHookInstalled = true
        }
      }
    }
  }
}
