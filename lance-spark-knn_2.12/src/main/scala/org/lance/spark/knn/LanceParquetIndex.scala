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

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{FloatType, LongType, StringType, StructField, StructType}
import org.lance.index.external.{ExternalIvfPqIndex, ExternalIvfPqIndexParams, ParquetRowKey, SearchResult}
import org.lance.spark.knn.internal.{ExternalIndexLifecycle, ExternalIndexProbe, Metric}

import scala.collection.JavaConverters._

/**
 * Driver-side handle for the external IVF-PQ vector index over parquet files. Wraps the
 * `org.lance.index.external.ExternalIvfPqIndex` JNI handle with Scala-friendly returns and
 * adds optional `SparkSession`-aware variants that produce `DataFrame`s for pipeline use.
 *
 * == Three caller patterns ==
 *
 *  - '''Single-query, no Spark''': the underlying Java surface
 *    `org.lance.index.external.ExternalIvfPqIndex` is independently usable from any JVM
 *    (services, notebooks, Trino/Presto extensions). This wrapper is unnecessary in that case.
 *  - '''Single-query from Spark''': use `[[search]]` (returns `Seq[SearchResult]`) or
 *    `[[searchToDF]]` (returns a 1-partition `DataFrame` so the result composes with downstream
 *    Spark transforms). Both run on the driver — the cost of a single index probe is ~1-5 ms,
 *    not worth a Spark task launch.
 *  - '''Many independent queries''': use [[IndexedNearestJoinExternal]] directly
 *    (`queries.kNearestJoin(corpus, ...)`). That path distributes the probes across executors
 *    and reuses the same external index file (via [[ExternalIndexLifecycle]]'s build cache)
 *    if a prior call built it.
 *
 * == Build vs open ==
 *
 *  - [[build]]: eager, writes the index file on the driver. Use when the caller manages the
 *    index file's lifetime themselves (offline pipeline, scheduled job).
 *  - [[buildIfMissing]]: lazy, hashes inputs and reuses an existing index file if the same
 *    `(file paths, vector column, params)` was already built in this Spark application.
 *  - [[open]]: opens an index built earlier (by anyone). The caller is responsible for the
 *    lifetime of the URI.
 *
 * == Lifecycle ==
 *
 * `LanceParquetIndex` is `AutoCloseable`. Each instance owns a JNI handle that holds an
 * `mmap`'d index header. Close it when done; opening is cheap so opening once per query and
 * closing per query is fine. The index file on disk is not deleted on `close()` —
 * directory cleanup is independent (see [[buildIfMissing]] for application-scoped cleanup).
 *
 * == Example ==
 *
 * Driver-side single-query retrieval, returning a `DataFrame`:
 *
 * {{{
 *   import org.lance.spark.knn.LanceParquetIndex
 *
 *   val idx = LanceParquetIndex.buildIfMissing(
 *     spark,
 *     filePaths = Seq("/data/embeddings-0.parquet", "/data/embeddings-1.parquet"),
 *     vectorColumn = "vec",
 *     metric = "l2")
 *   try {
 *     // 10 nearest rows to `qvec`, projected to (doc_id, title)
 *     val topK: DataFrame = idx.searchToDF(qvec, k = 10, projection = Seq("doc_id", "title"))
 *     topK.show()
 *   } finally {
 *     idx.close()
 *   }
 * }}}
 */
final class LanceParquetIndex private[knn] (
    private val handle: ExternalIvfPqIndex,
    private val sourceFilePaths: Seq[String],
    private val sourceVectorColumn: String) extends AutoCloseable {

  // Cached parquet schema — populated lazily on first [[searchToDF]] / [[fetchRowsToDF]] call.
  // Reading the parquet footer is cheap but we don't want to do it eagerly because non-Spark
  // callers that only use [[search]] / [[fetchRows]] never need the Spark schema.
  @volatile private var cachedSparkSchema: Option[StructType] = None

  /** Number of registered parquet files. */
  def numFiles: Int = handle.getNumFiles

  /** Number of IVF partitions in the index. */
  def numPartitions: Int = handle.getNumPartitions

  /** Vector column the index was built over (matches [[sourceVectorColumn]]). */
  def vectorColumn: String = handle.getVectorColumn

  /**
   * Parquet files registered with the index, in the order they were registered. The index
   * encodes file_id by position; reordering invalidates the index. Returned as the same
   * `Seq` that was passed to [[build]] / [[buildIfMissing]] / [[open]].
   */
  def filePaths: Seq[String] = sourceFilePaths

  /**
   * Run a single nearest-neighbor query on the driver and return up to `k` `(filePath,
   * rowIndex, distance)` triples ordered best-first.
   *
   * @param query        query vector. Length must match the index's training dimension.
   * @param k            top-K rows to return.
   * @param nprobes      IVF probe width (default 16). Higher = better recall, more I/O.
   * @param refineFactor PQ-approx candidate multiplier (default 8). `k * refineFactor`
   *                     candidates are fetched + refined exactly.
   * @param deletedRids  optional packed `(file_id << 32) | row_index` array of deleted rids
   *                     (Delta DV / Iceberg position deletes). Pack with
   *                     `ExternalIvfPqIndex.packDeletedRids`. `null` means no filter.
   */
  def search(
      query: Array[Float],
      k: Int,
      nprobes: Int = 16,
      refineFactor: Int = 8,
      deletedRids: Array[Byte] = null): Seq[SearchResult] = {
    require(query != null && query.nonEmpty, "query vector must be non-empty")
    require(k > 0, "k must be positive")
    require(nprobes > 0, "nprobes must be positive")
    require(refineFactor > 0, "refineFactor must be positive")
    handle.search(query, k, nprobes, refineFactor, deletedRids).asScala.toSeq
  }

  /**
   * Driver-side single-query search returning a 1-partition `DataFrame`. Useful when the
   * caller wants to compose the top-K with downstream Spark transforms (joins, projections,
   * UDFs). For programmatic use without Spark composition, prefer [[search]].
   *
   * The returned schema is `(file_path STRING, row_index LONG, score FLOAT)` — keys plus
   * the exact distance. To get arbitrary payload columns alongside, pass `projection` and
   * the wrapper will issue a [[fetchRows]] per the topK keys and return the joined row.
   *
   * == Why driver-side ==
   *
   * A single probe takes ~1-5 ms (warm) to ~50 ms (cold mmap) on the index files used in our
   * benchmarks. Running it as a 1-partition Spark job is bounded below by Spark task launch
   * latency (~50-100 ms); the driver-local call is strictly faster. For batched queries (many
   * vectors at once), use [[IndexedNearestJoinExternal]] directly, which distributes the
   * probes across executors.
   */
  def searchToDF(
      query: Array[Float],
      k: Int,
      nprobes: Int = 16,
      refineFactor: Int = 8,
      deletedRids: Array[Byte] = null,
      projection: Seq[String] = Nil)(implicit spark: SparkSession): DataFrame = {
    val hits = search(query, k, nprobes, refineFactor, deletedRids)
    if (projection.isEmpty) {
      val rows = hits.map(h =>
        Row(
          h.getFilePath,
          java.lang.Long.valueOf(h.getRowIndex),
          java.lang.Float.valueOf(h.getDistance)))
      val schema = StructType(Seq(
        StructField("file_path", StringType, nullable = false),
        StructField("row_index", LongType, nullable = false),
        StructField("score", FloatType, nullable = false)))
      spark.createDataFrame(rows.asJava, schema)
    } else {
      // Materialize payload columns for the top-K. The result schema is the projection
      // schema (read from the parquet footer) extended with `score` so the caller has the
      // distance alongside the row.
      val refs =
        hits.map(h => internal.ScoredFileRowRef(h.getFilePath, h.getRowIndex, h.getDistance))
      fetchRowsToDF(refs, projection, includeScore = true)(spark)
    }
  }

  /**
   * Random-access fetch by `(filePath, rowIndex)` keys, returning the per-row payload as a
   * `Seq[Map[colName -> value]]` in the same order as `refs`. Driver-side.
   *
   * Lance batches reads by file internally and issues one page-index-aware parquet read per
   * file, then reassembles to caller order.
   */
  def fetchRows(
      refs: Seq[internal.ScoredFileRowRef],
      projection: Seq[String]): Seq[Map[String, Any]] = {
    if (refs.isEmpty) return Seq.empty
    require(projection.nonEmpty, "projection must contain at least one column")
    val keys = refs.map(r => new ParquetRowKey(r.filePath, r.rowIndex)).asJava
    val ipcBytes = handle.fetchRows(keys, projection.asJava)
    decodeArrowIpc(ipcBytes)
  }

  /**
   * Same as [[fetchRows]] but returns a 1-partition `DataFrame` whose schema mirrors the
   * source parquet schema for the projected columns (plus an optional `score` column). The
   * row order matches `refs`.
   */
  def fetchRowsToDF(
      refs: Seq[internal.ScoredFileRowRef],
      projection: Seq[String],
      includeScore: Boolean = true,
      scoreCol: String = "score")(implicit spark: SparkSession): DataFrame = {
    require(projection.nonEmpty, "projection must contain at least one column")
    val sparkSchema = parquetSchema(spark)
    val projFields: Seq[StructField] = projection.map(sparkSchema.apply).map(f =>
      f.copy(nullable = true))
    val outFields = if (includeScore) {
      projFields :+ StructField(scoreCol, FloatType, nullable = true)
    } else projFields
    val outSchema = StructType(outFields)
    val rowMaps = fetchRows(refs, projection)
    require(
      rowMaps.size == refs.size,
      s"fetchRows returned ${rowMaps.size} rows for ${refs.size} keys")
    val rows: Seq[Row] = refs.zip(rowMaps).map { case (ref, m) =>
      val cols = projection.map(c => m.getOrElse(c, null).asInstanceOf[Any])
      val all = if (includeScore) cols :+ java.lang.Float.valueOf(ref.score) else cols
      Row.fromSeq(all)
    }
    spark.createDataFrame(rows.asJava, outSchema)
  }

  override def close(): Unit = handle.close()

  // -- internal --------------------------------------------------------------

  /** Read the parquet footer once via `spark.read.parquet` to get a Spark `StructType`. */
  private def parquetSchema(spark: SparkSession): StructType = {
    cachedSparkSchema match {
      case Some(s) => s
      case None =>
        val s = spark.read.parquet(sourceFilePaths: _*).schema
        cachedSparkSchema = Some(s)
        s
    }
  }

  /** Decode an Arrow IPC stream returned by `ExternalIvfPqIndex.fetchRows`. */
  private def decodeArrowIpc(bytes: Array[Byte]): Seq[Map[String, Any]] = {
    import org.apache.arrow.memory.RootAllocator
    import org.apache.arrow.vector.ipc.ArrowStreamReader
    import java.io.ByteArrayInputStream

    val allocator = new RootAllocator(Long.MaxValue)
    try {
      val reader = new ArrowStreamReader(new ByteArrayInputStream(bytes), allocator)
      try {
        val out = scala.collection.mutable.ArrayBuffer.empty[Map[String, Any]]
        while (reader.loadNextBatch()) {
          val root = reader.getVectorSchemaRoot
          val rowCount = root.getRowCount
          val fields = root.getSchema.getFields.asScala
          var r = 0
          while (r < rowCount) {
            val map = scala.collection.mutable.LinkedHashMap.empty[String, Any]
            var f = 0
            while (f < fields.size) {
              val name = fields(f).getName
              val v = root.getVector(name)
              map(name) =
                if (v.isNull(r)) null else internal.LanceProbe.toSparkValue(v.getObject(r))
              f += 1
            }
            out += map.toMap
            r += 1
          }
        }
        out.toSeq
      } finally reader.close()
    } finally allocator.close()
  }
}

object LanceParquetIndex {

  /**
   * Open an index that was built earlier. Caller owns the URI and is responsible for
   * deleting the directory when done with it.
   *
   * The wrapper validates that the index's vector column matches `vectorColumn` and that
   * the manifest's file list matches `filePaths` (modulo order — the manifest's order is
   * authoritative; mismatches throw).
   *
   * @param indexUri full URI of the index directory (the `<uuid>` directory under the
   *                 build's `outputUri`).
   * @param filePaths registered parquet files. Must match the manifest's list and order.
   * @param vectorColumn vector column. Must match the manifest's value.
   */
  def open(
      indexUri: String,
      filePaths: Seq[String],
      vectorColumn: String): LanceParquetIndex = {
    val h = ExternalIvfPqIndex.open(indexUri)
    val actualVecCol = h.getVectorColumn
    if (actualVecCol != vectorColumn) {
      h.close()
      throw new IllegalArgumentException(
        s"index at $indexUri was built with vector column '$actualVecCol', " +
          s"caller passed '$vectorColumn'")
    }
    val actualFiles = h.getNumFiles
    if (actualFiles != filePaths.size) {
      h.close()
      throw new IllegalArgumentException(
        s"index at $indexUri was built over $actualFiles files, " +
          s"caller passed ${filePaths.size} (path order is significant)")
    }
    new LanceParquetIndex(h, filePaths, vectorColumn)
  }

  /**
   * Eagerly build an index over `filePaths` and return an open handle. Writes the index
   * directory under `outputUri / <uuid>`.
   *
   * For application-scoped builds (the index file is reused across queries in the same
   * Spark app and cleaned up on application end), prefer [[buildIfMissing]].
   *
   * @return an opened [[LanceParquetIndex]]. The caller owns the index directory's lifetime.
   */
  def build(
      filePaths: Seq[String],
      vectorColumn: String,
      outputUri: String,
      params: ExternalIvfPqIndexParams): LanceParquetIndex = {
    require(filePaths.nonEmpty, "filePaths must contain at least one path")
    val sortedPaths = filePaths.sorted
    val uuid = ExternalIvfPqIndex.build(sortedPaths.asJava, vectorColumn, outputUri, params)
    val indexUri = s"$outputUri/$uuid"
    val h = ExternalIvfPqIndex.open(indexUri)
    new LanceParquetIndex(h, sortedPaths, vectorColumn)
  }

  /**
   * Build (or reuse a previously-built) index over `filePaths`, with cleanup wired into
   * the Spark application's lifecycle. The index directory is hashed by
   * `(filePaths, vectorColumn, params)` so repeated calls within one Spark application
   * share the same on-disk file. The directory is deleted on `SparkListenerApplicationEnd`
   * and the JVM shutdown hook (same machinery as [[IndexedNearestJoinExternal]]).
   *
   * Requires `spark.lance.knn.externalIndex.dir` to be set when running on a non-local
   * `spark.master` (the index needs to live on a shared filesystem so executors can read it).
   *
   * @param metric "l2", "cosine", or "dot". Defaults to "l2".
   * @param params override for the index build (kmeans iterations, sample rate, etc.).
   *               Defaults to [[ExternalIvfPqIndexParams.builder]] with metric set from
   *               `metric`.
   */
  def buildIfMissing(
      spark: SparkSession,
      filePaths: Seq[String],
      vectorColumn: String,
      metric: String = "l2",
      params: Option[ExternalIvfPqIndexParams] = None): LanceParquetIndex = {
    require(filePaths.nonEmpty, "filePaths must contain at least one path")
    val parsedMetric = Metric.fromName(metric)
    val effective = params.getOrElse(ExternalIndexProbe.defaultParams(parsedMetric))
    val sortedPaths = filePaths.sorted
    val indexUri = ExternalIndexLifecycle.buildOrReuse(
      spark,
      sortedPaths,
      vectorColumn,
      effective)
    val h = ExternalIvfPqIndex.open(indexUri)
    new LanceParquetIndex(h, sortedPaths, vectorColumn)
  }
}
