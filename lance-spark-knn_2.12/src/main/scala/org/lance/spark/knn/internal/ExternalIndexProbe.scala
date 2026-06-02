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

import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.ipc.ArrowStreamReader
import org.apache.spark.sql.Row
import org.lance.index.external.{ExternalIvfPqIndex, ExternalIvfPqIndexParams, ParquetRowKey, SearchResult}

import java.io.ByteArrayInputStream

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * Per-task wrapper around `ExternalIvfPqIndex` JNI. Opens the index handle once, runs many
 * `probe()` calls, materializes payload rows for surviving top-K via `fetchRows()`. Mirrors the
 * shape of [[LanceProbe]] but the underlying engine is the external IVF-PQ index over caller-
 * supplied parquet files (no Lance dataset required on the right side).
 *
 * The killer-feature payoff lives in [[materialize]]: it fetches projection columns for ONLY the
 * surviving top-K rows from source parquet, eliminating the per-query temp-Lance write that the
 * general-purpose path ([[LanceMaterializeStage]]) needs.
 *
 * Lifecycle: instantiate per task, call [[probe]] / [[materialize]] repeatedly, [[close]] at end.
 */
final class ExternalIndexProbe(indexUri: String) extends AutoCloseable {

  // Open the handle once. Cheap: mmaps the manifest + index header.
  private val index: ExternalIvfPqIndex = ExternalIvfPqIndex.open(indexUri)

  /**
   * Run a single nearest-neighbor query. Returns up to `k` `(filePath, rowIndex, distance)`
   * triples ordered best-first.
   *
   * The `filePath` is one of the parquet files registered with the index at build time. It's a
   * stable string that round-trips through subsequent [[materialize]] calls.
   *
   * `nprobes` controls IVF probe width; `refineFactor` controls re-rank candidate width
   * (`k * refineFactor` candidates fetched + refined exactly). Both are passed through to the
   * Rust impl unchanged.
   *
   * `deletedRids` is the optional row-deletion filter (Delta DV / Iceberg position deletes).
   * Pack via [[ExternalIvfPqIndex.packDeletedRids]] on the driver and broadcast.
   */
  def probe(
      query: Array[Float],
      k: Int,
      nprobes: Int,
      refineFactor: Int,
      deletedRids: Array[Byte] = null): Seq[SearchResult] = {
    require(query != null && query.nonEmpty, "query vector must be non-empty")
    require(k > 0, "k must be positive")
    index.search(query, k, nprobes, refineFactor, deletedRids).asScala.toSeq
  }

  /**
   * Materialize a list of `(filePath, rowIndex)` references with the requested projection
   * columns. Returns a `Seq[Map[String, Any]]` per row in the input order.
   *
   * Lance does the per-file batching internally (one parquet read per distinct file, page-
   * index-aware random access) and reassembles to caller order.
   *
   * Returns the row payloads as a `Seq[Map[colName -> value]]` — Spark-agnostic, mirroring the
   * shape that [[LanceProbe.materialize]] returns. The caller (`ExternalMaterializeStage`) maps
   * each row to a Spark `Row`.
   */
  def materialize(
      refs: Seq[ScoredFileRowRef],
      projection: Seq[String]): Seq[Map[String, Any]] = {
    if (refs.isEmpty) return Seq.empty
    val rowKeys = refs.map(r => new ParquetRowKey(r.filePath, r.rowIndex)).asJava
    val ipcBytes = index.fetchRows(rowKeys, projection.asJava)

    // Decode the Arrow IPC stream back into per-row maps. The batch is in input order so we can
    // walk rows index-by-index without rebuilding any reorder map.
    val allocator = new RootAllocator(Long.MaxValue)
    try {
      val reader = new ArrowStreamReader(new ByteArrayInputStream(ipcBytes), allocator)
      try {
        val out = mutable.ArrayBuffer.empty[Map[String, Any]]
        while (reader.loadNextBatch()) {
          val root = reader.getVectorSchemaRoot
          val rowCount = root.getRowCount
          val fields = root.getSchema.getFields.asScala
          var r = 0
          while (r < rowCount) {
            val map = mutable.LinkedHashMap.empty[String, Any]
            var f = 0
            while (f < fields.size) {
              val name = fields(f).getName
              val v = root.getVector(name)
              map(name) = if (v.isNull(r)) null else LanceProbe.toSparkValue(v.getObject(r))
              f += 1
            }
            out += map.toMap
            r += 1
          }
        }
        out.toSeq
      } finally {
        reader.close()
      }
    } finally {
      allocator.close()
    }
  }

  /** Number of registered parquet files. */
  def numFiles: Int = index.getNumFiles

  /** Vector column name. */
  def vectorColumn: String = index.getVectorColumn

  override def close(): Unit = index.close()
}

/**
 * `(filePath, rowIndex, distance)` produced by [[ExternalIndexProbe.probe]] and consumed by
 * [[ExternalIndexProbe.materialize]]. Counterpart to [[ScoredRowRef]] but carrying the parquet
 * file identity instead of an opaque Lance `_rowid`.
 */
final case class ScoredFileRowRef(filePath: String, rowIndex: Long, score: Float)
  extends Serializable

object ExternalIndexProbe {

  /** Convert a `SearchResult[]` to a `ScoredFileRowRef` array. */
  def toRefs(results: Seq[SearchResult]): Array[ScoredFileRowRef] =
    results.iterator
      .map(r => ScoredFileRowRef(r.getFilePath, r.getRowIndex, r.getDistance))
      .toArray

  /** Build params from runtime conf — wraps the Java builder for use by the lifecycle layer. */
  def defaultParams(metric: Metric): ExternalIvfPqIndexParams = {
    val javaMetric = metric match {
      case Metric.L2 => ExternalIvfPqIndexParams.Metric.L2
      case Metric.Cosine => ExternalIvfPqIndexParams.Metric.Cosine
      case Metric.Dot => ExternalIvfPqIndexParams.Metric.Dot
    }
    ExternalIvfPqIndexParams.builder().metric(javaMetric).build()
  }
}
