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

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructField
import org.lance.index.external.ParquetRowKey

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * Fused probe + materialize stage for the external-index path. Replaces the
 * three-stage [[ExternalProbeStage]] → shuffle → merge → [[ExternalMaterializeStage]]
 * pipeline with a single per-task stage that:
 *
 *   1. Opens the [[ExternalIndexProbe]] handle once
 *   2. For each left row in the task, calls `idx.search(query, K)` — Lance returns the
 *      already-refined, already-global top-K
 *   3. Decodes refs into `(file_path, row_index)` keys
 *   4. Calls `idx.fetch_rows(rowKeys, projection)` to materialize payload columns from
 *      source parquet
 *   5. Emits final join `Row` values directly
 *
 * == Why this fuses correctly ==
 *
 * The shuffle in the staged path was inherited from the Lance-native pipeline where
 * `LanceProbeStage.runWithFragmentGroups` splits R fragments across tasks, so a single
 * left row can have multiple contributors. The external-index path doesn't split R —
 * Lance's IVF probe internally merges across all partitions inside one `idx.search()`
 * call, returning the global top-K for that one query. Each leftId has exactly ONE
 * contributor, so the merge stage collapses to a passthrough and the shuffle ships
 * data only to no-op on the other side.
 *
 * Removing the shuffle:
 *   - eliminates one Spark Exchange (one fewer stage in the DAG)
 *   - drops `(leftId, leftRow)` shuffling cost — leftRow can be wide; refs[K] are tiny
 *   - keeps the same parallelism: probe ran with `|left.partitions|` tasks; fused does
 *     too. Each task processes its slice of the left rows and emits final output rows
 *     in the same partition.
 *
 * == Batched fetch within a partition ==
 *
 * Per-leftId `idx.fetch_rows(K_keys)` calls are correct but inefficient when many left
 * rows in the same task hit the same parquet file — each call opens the file fresh.
 * The fused stage **batches per-partition** instead: collect all (leftId, search_refs)
 * pairs first, then issue one `fetch_rows` for the whole batch (file-grouped inside),
 * then assemble final rows. This keeps amortized parquet read cost low while still
 * eliminating the shuffle.
 *
 * Memory: holding all per-left-row refs + leftRows in a partition before the batched
 * fetch is bounded by the partition's row count × (leftRow bytes + K * 24 B). For a
 * partition of 1M left rows × K=10 that's ~240 MB just for refs. For typical KNN
 * workloads (|L| in thousands), it's negligible.
 */
object ExternalFusedStage {

  /**
   * Driver-side configuration. Combines the [[ExternalProbeStage.Conf]] and
   * [[ExternalMaterializeStage.Conf]] fields — same values, one carrier.
   */
  final case class Conf(
      indexUri: String,
      filePaths: Array[String],
      vectorColumn: String,
      metric: Metric,
      k: Int,
      nprobes: Int,
      refineFactor: Int,
      leftVecIdx: Int,
      rightProjection: Seq[String],
      rightFields: Seq[StructField],
      leftFieldCount: Int,
      outerJoin: Boolean,
      deletedRids: Array[Byte] = null)
    extends Serializable

  def run(left: RDD[Row], conf: Conf): RDD[Row] = {
    left.mapPartitions(iter => fusedPartition(iter, conf))
  }

  private def fusedPartition(iter: Iterator[Row], conf: Conf): Iterator[Row] = {
    if (iter.isEmpty) return Iterator.empty

    val probe = new ExternalIndexProbe(conf.indexUri)
    val pathToFileId: Map[String, Int] = conf.filePaths.zipWithIndex.toMap
    val out = mutable.ArrayBuffer.empty[Row]
    try {
      // Pass 1: probe every left row, collect (leftRow, refs).
      // Refs from Lance are already SearchResult(filePath, rowIndex, distance) — we
      // keep them as ScoredFileRowRef so the materialize batch step can group by file.
      val perLeft = mutable.ArrayBuffer.empty[(Row, Array[ScoredFileRowRef])]
      iter.foreach { leftRow =>
        val q = LanceProbeStage.extractVector(leftRow, conf.leftVecIdx)
        val refs: Array[ScoredFileRowRef] =
          if (q == null) Array.empty[ScoredFileRowRef]
          else {
            val results =
              probe.probe(q, conf.k, conf.nprobes, conf.refineFactor, conf.deletedRids)
            results.iterator.map { r =>
              val _ = pathToFileId // file-id sanity is already enforced by Lance
              ScoredFileRowRef(r.getFilePath, r.getRowIndex, r.getDistance)
            }.toArray
          }
        perLeft += ((leftRow, refs))
      }

      // Pass 2: batched fetch_rows for ALL surviving (file, row) keys across the
      // whole partition. One JNI call per partition (vs one per left row in the
      // staged path). Lance's fetchRows internally batches by file_path → one
      // page-index-aware parquet read per distinct file regardless of how many
      // left rows hit it.
      val allKeys = mutable.ArrayBuffer.empty[ParquetRowKey]
      val flatRanges = mutable.ArrayBuffer.empty[(Int, Int)] // (start, end) into allKeys
      perLeft.foreach { case (_, refs) =>
        val start = allKeys.size
        refs.foreach(r => allKeys += new ParquetRowKey(r.filePath, r.rowIndex))
        flatRanges += ((start, allKeys.size))
      }

      val materialized: Seq[Map[String, Any]] =
        if (allKeys.isEmpty) Seq.empty
        else
          probe.materialize(
            // Convert to ScoredFileRowRef for the Java helper signature; score is
            // unused on the materialize path so we pass a placeholder.
            allKeys.iterator.map(k => ScoredFileRowRef(k.getFilePath, k.getRowIndex, 0.0f)).toSeq,
            conf.rightProjection)

      // Pass 3: assemble final Rows in input order.
      perLeft.iterator.zipWithIndex.foreach {
        case ((leftRow, refs), liIdx) =>
          val (start, end) = flatRanges(liIdx)
          if (refs.isEmpty && conf.outerJoin) {
            out += assembleRow(
              leftRow, conf.leftFieldCount, conf.rightFields, rightValues = null, score = null)
          } else if (refs.nonEmpty) {
            var i = 0
            while (i < refs.length) {
              val rightMap = if (start + i < materialized.size) materialized(start + i) else null
              out += assembleRow(
                leftRow, conf.leftFieldCount, conf.rightFields, rightMap, refs(i).score)
              i += 1
            }
          }
      }
    } finally probe.close()
    out.iterator
  }

  private def assembleRow(
      leftRow: Row,
      leftFieldCount: Int,
      rightFields: Seq[StructField],
      rightValues: Map[String, Any],
      score: Any): Row = {
    val arr = new Array[Any](leftFieldCount + rightFields.size + 1)
    var i = 0
    while (i < leftFieldCount) { arr(i) = leftRow.get(i); i += 1 }
    var j = 0
    while (j < rightFields.size) {
      arr(leftFieldCount + j) =
        if (rightValues == null) null else rightValues.getOrElse(rightFields(j).name, null)
      j += 1
    }
    arr(leftFieldCount + rightFields.size) = score
    Row.fromSeq(arr.toSeq)
  }
}
