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

import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.collection.mutable

/**
 * Probe stage of the indexed nearest-by pipeline. Per task, opens a Lance dataset once and runs
 * `LanceProbe.probe(...)` per left row, restricted to the task's assigned fragments. Emits
 * `(leftId, ProbedLeft)` keyed by `leftId` so the downstream Exchange can hash-shuffle by it.
 *
 * Map-side combine via [[TopKHeap]] only kicks in when a single task probes multiple fragments
 * directly. With `fragmentIds = None` Lance does the cross-fragment merge internally and a
 * single `probe` call already returns the right K — the per-row heap collapses to a passthrough.
 * Splitting fragments across tasks via `runWithFragmentGroups` (Phase 1.5) gives each task a
 * subset of the dataset's fragments and lets the downstream merge stage aggregate contributions.
 *
 * The stage materializes its output into an `ArrayBuffer` before closing the probe handle. This
 * is the same closing-iterator pattern used in Phase 0 — necessary because Spark's iterator
 * model lazily pulls from `mapPartitions`, which would otherwise let the consumer outlive the
 * `try`/`finally`.
 */
object LanceProbeStage {

  /**
   * Driver-side configuration shipped to every probe task. Kept minimal so adding new probe knobs
   * (filter pushdown, refine factor, etc.) stays local to this object.
   */
  final case class Conf(
      datasetUri: String,
      fragmentIds: Option[Seq[Int]],
      vectorColumn: String,
      version: Option[Long],
      metric: Metric,
      k: Int,
      nprobes: Option[Int],
      leftVecIdx: Int,
      refineFactor: Option[Int] = None,
      ef: Option[Int] = None,
      prefilter: Option[String] = None)
    extends Serializable

  def run(leftKeyed: RDD[(Long, Row)], conf: Conf): RDD[(Long, ProbedLeft)] = {
    leftKeyed.mapPartitions(iter => probePartition(iter, conf))
  }

  /**
   * Phase 1.5 — fragment-grouped probe. Replicates each left row across `fragmentGroups.size`
   * partitions, so each task probes a single group's fragments only and ALL left rows produce
   * `fragmentGroups.size` contributions per `leftId`. Downstream `LanceMergeStage` then has real
   * work to do — its `reduceByKey` aggregates across groups via `TopKHeap.merge`.
   *
   * Topology:
   *
   * {{{
   *   leftKeyed: RDD[(Long, Row)]
   *     -- flatMap (leftId, row) -> (groupIdx, (leftId, row))      replicate G times
   *     -- partitionBy(HashPartitioner(G))                          one partition per group
   *     -- mapPartitionsWithIndex { (idx, iter) =>
   *          openLanceProbe(fragmentGroups(idx))
   *          probe each row              }
   *     -- emits (leftId, ProbedLeft)                               G entries per leftId
   * }}}
   *
   * The `flatMap` + `partitionBy` together form one shuffle. The merge stage's `reduceByKey`
   * adds a second shuffle (since the output here is keyed by `leftId` but the partitioning is by
   * `groupIdx`). Two shuffles is the cost of fragment-grouping.
   *
   * Empty groups (when `fragmentGroups.size > numFragments`) are skipped — the partition's
   * iterator yields zero output. Callers don't need to special-case this.
   *
   * @param leftKeyed     left rows keyed by stable `leftId`
   * @param conf          probe config; `fragmentIds` on the conf is IGNORED — this method
   *                      overrides it per group
   * @param fragmentGroups fragment ID assignment; one entry per group
   */
  def runWithFragmentGroups(
      leftKeyed: RDD[(Long, Row)],
      conf: Conf,
      fragmentGroups: Seq[Seq[Int]]): RDD[(Long, ProbedLeft)] = {
    require(fragmentGroups.nonEmpty, "fragmentGroups must not be empty")
    val groupCount = fragmentGroups.size
    val groupsBcast = leftKeyed.context.broadcast(fragmentGroups)

    val replicated: RDD[(Int, (Long, Row))] = leftKeyed.flatMap {
      case (leftId, leftRow) =>
        (0 until groupCount).iterator.map(g => (g, (leftId, leftRow)))
    }
    val byGroup = replicated.partitionBy(new HashPartitioner(groupCount))

    byGroup.mapPartitionsWithIndex(
      { (partIdx, iter) =>
        if (!iter.hasNext) Iterator.empty
        else {
          val groups = groupsBcast.value
          // partIdx maps directly to groupIdx because HashPartitioner places key i in
          // partition `i % groupCount`, and our keys are 0..groupCount-1.
          val frags = groups(partIdx)
          if (frags.isEmpty) Iterator.empty
          else {
            val groupConf = conf.copy(fragmentIds = Some(frags))
            probePartition(iter.map(_._2), groupConf)
          }
        }
      },
      preservesPartitioning = false)
  }

  private def probePartition(
      iter: Iterator[(Long, Row)],
      conf: Conf): Iterator[(Long, ProbedLeft)] = {
    if (iter.isEmpty) return Iterator.empty

    val probe =
      new LanceProbe(conf.datasetUri, conf.fragmentIds, conf.version)
    val out = mutable.ArrayBuffer.empty[(Long, ProbedLeft)]
    try {
      iter.foreach { case (leftId, leftRow) =>
        val q = extractVector(leftRow, conf.leftVecIdx)
        val refs =
          if (q == null) Array.empty[ScoredRowRef]
          else probe.probe(
            conf.vectorColumn,
            q,
            conf.k,
            conf.metric,
            conf.nprobes,
            conf.refineFactor,
            conf.ef,
            conf.prefilter)
            .toArray
        out += ((leftId, ProbedLeft(leftRow, refs)))
      }
    } finally probe.close()
    out.iterator
  }

  /**
   * Pull a query vector out of a Spark `Row`'s ArrayType column. Mirrors the matching logic from
   * Phase 0 — the Scala 2.13 `Seq` gotcha is real: `Row.get` on `ArrayType` returns
   * `mutable.ArraySeq`, which `case s: Seq[_]` only matches against the root `scala.collection.Seq`
   * trait (the default `Seq` alias is `immutable.Seq` on 2.13).
   */
  private[knn] def extractVector(row: Row, idx: Int): Array[Float] = {
    if (row.isNullAt(idx)) return null
    row.get(idx) match {
      case s: scala.collection.Seq[_] =>
        s.iterator.map {
          case f: java.lang.Float => f.floatValue()
          case f: Float => f
          case d: java.lang.Double => d.doubleValue().toFloat
          case d: Double => d.toFloat
          case other =>
            throw new IllegalStateException(
              s"Unsupported vector element type: ${other.getClass.getName}")
        }.toArray
      case arr: Array[Float] => arr
      case arr: Array[java.lang.Float] => arr.map(_.floatValue())
      case other =>
        throw new IllegalStateException(
          s"Unsupported vector column representation: ${other.getClass.getName}")
    }
  }
}
