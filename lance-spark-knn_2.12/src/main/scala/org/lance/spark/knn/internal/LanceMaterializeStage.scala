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

import scala.collection.mutable

/**
 * Materialize stage. Per task, opens the Lance dataset and point-fetches right rows by
 * `_rowaddr`. Joins against the carried left payload to emit final join rows.
 *
 * Lance's row-address-IN filter is the natural index point-fetch path (validated by
 * `LanceProbeValidationTest`). Materialize re-opens the Lance dataset rather than reusing the
 * probe stage's open because the two stages are now in separate Spark tasks (across the shuffle).
 * The cost is one Lance manifest read per task — Lance's metadata is mmap-friendly and that's
 * the same trade-off lance-spark already accepts for fragment scans.
 *
 * The materialize-only `LanceProbe` instance never calls `probe()`, so `vectorColumn` is unused on
 * this path; we pass an empty string. Refactoring `LanceProbe`'s constructor to drop the param
 * for materialize-only use is left for a follow-up, since the current shape is the validated one.
 */
object LanceMaterializeStage {

  final case class Conf(
      datasetUri: String,
      version: Option[Long],
      rightProjection: Seq[String],
      rightFields: Seq[StructField],
      leftFieldCount: Int,
      outerJoin: Boolean)
    extends Serializable

  def run(merged: RDD[(Long, ProbedLeft)], conf: Conf): RDD[Row] = {
    merged.mapPartitions(iter => materializePartition(iter, conf))
  }

  private def materializePartition(
      iter: Iterator[(Long, ProbedLeft)],
      conf: Conf): Iterator[Row] = {
    if (iter.isEmpty) return Iterator.empty

    val probe =
      new LanceProbe(conf.datasetUri, fragmentIds = None, version = conf.version)
    val out = mutable.ArrayBuffer.empty[Row]
    try {
      iter.foreach { case (_, pl) =>
        if (pl.refs.isEmpty && conf.outerJoin) {
          out += assembleRow(
            pl.leftRow,
            conf.leftFieldCount,
            conf.rightFields,
            rightValues = null,
            score = null)
        } else if (pl.refs.nonEmpty) {
          // Build a `rowAddr -> materialized row` map. If `pl.refs` ever contains duplicate
          // `rowAddr`s (same row referenced by multiple probe contributions) the map collapses
          // them to one entry; the `pl.refs.foreach` loop below still emits one output row per
          // ref, all sharing that materialized payload. Phase 1.5's `TopKHeap.merge` does not
          // dedupe across contributions, so this collapse is intentional rather than a bug —
          // duplicate refs would mean the same right row is the K-th nearest along multiple
          // fragment-group paths, which is genuinely "the same hit" and should appear once
          // per ref in the output.
          val materialized: Map[Long, Map[String, Any]] = probe
            .materialize(pl.refs.iterator.map(_.rowAddr).toSeq, conf.rightProjection)
            .map(m => extractRowAddr(m) -> m)
            .toMap
          pl.refs.foreach { ref =>
            val rightMap = materialized.getOrElse(ref.rowAddr, null)
            out += assembleRow(
              pl.leftRow,
              conf.leftFieldCount,
              conf.rightFields,
              rightMap,
              ref.score)
          }
        }
      }
    } finally probe.close()
    out.iterator
  }

  private def extractRowAddr(m: Map[String, Any]): Long =
    m.get(LanceProbe.RowIdColumn) match {
      case Some(l: java.lang.Long) => l.longValue()
      case Some(l: Long) => l
      case Some(other) => other.toString.toLong
      case None =>
        throw new IllegalStateException(
          s"Materialized row missing ${LanceProbe.RowIdColumn}; " +
            s"got keys: ${m.keys.mkString(", ")}")
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
