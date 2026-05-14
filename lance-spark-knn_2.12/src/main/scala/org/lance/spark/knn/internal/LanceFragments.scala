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

import org.lance.{Dataset, ReadOptions}
import org.lance.spark.LanceRuntime

import scala.collection.JavaConverters._

/**
 * Driver-side helper that enumerates Lance fragment IDs and partitions them into balanced groups.
 * Foundation of Phase 1.5 fragment-grouped probing — once we know all fragment IDs and how they
 * should be split across probe tasks, the probe stage's per-task `fragmentIds` becomes a function
 * of the partition index and the group count.
 *
 * Round-robin assignment is the simplest balanced split: fragments tend to be similarly sized in
 * a healthy Lance dataset, so straight round-robin gives groups within ~1 fragment of each other
 * in count. For uneven datasets, [[enumerateGroupsByRowCount]] uses LPT greedy bin-packing on
 * per-fragment row counts (Phase 3 skew handling).
 */
object LanceFragments {

  /**
   * Open the dataset, enumerate fragment IDs, and round-robin them into `groupCount` groups.
   * If `groupCount > numFragments`, the trailing groups will be empty — call sites must tolerate
   * an empty group rather than fail (the probe stage skips them, since an empty fragment list
   * means "no rows to probe").
   *
   * @return a list of length exactly `groupCount`. Each inner list is the fragment IDs assigned
   *         to that group. Concatenating all groups in order yields the full fragment-id set
   *         (each fragment appears in exactly one group).
   */
  def enumerateGroups(
      datasetUri: String,
      version: Option[Long],
      groupCount: Int): Seq[Seq[Int]] = {
    require(groupCount > 0, s"groupCount must be positive, got $groupCount")
    val dataset = openDataset(datasetUri, version)
    try {
      val ids = dataset.getFragments.asScala.iterator.map(_.getId).toIndexedSeq
      roundRobin(ids, groupCount)
    } finally dataset.close()
  }

  /**
   * Phase 3 skew handling — group fragments such that each group's total row count is balanced.
   * Falls back to round-robin if the row-count metadata isn't available for a fragment (defensive
   * default of 1 row per missing fragment so the assignment doesn't degenerate).
   *
   * Uses the classic "Longest Processing Time" greedy heuristic: sort fragments by row count
   * descending, then assign each to whichever group currently has the smallest total. Worst-case
   * makespan within 4/3 of optimal — sufficient for fragment-grouping where the goal is "no
   * task does dramatically more work than another", not perfect balance.
   *
   * Use this over `enumerateGroups` when fragments are known to be uneven (e.g., produced by an
   * unbalanced upstream write). For evenly-sized fragments the simpler round-robin gives the
   * same result with less overhead.
   */
  def enumerateGroupsByRowCount(
      datasetUri: String,
      version: Option[Long],
      groupCount: Int): Seq[Seq[Int]] = {
    require(groupCount > 0, s"groupCount must be positive, got $groupCount")
    val dataset = openDataset(datasetUri, version)
    try {
      val weighted = dataset.getFragments.asScala.iterator.map { f =>
        // Some fragment metadata implementations return -1 / 0 if not populated. Treat any
        // non-positive value as "1 row" so it still occupies a slot and gets assigned somewhere.
        val rows = scala.math.max(1L, f.metadata.getNumRows)
        (f.getId, rows)
      }.toIndexedSeq
      greedyBalance(weighted, groupCount)
    } finally dataset.close()
  }

  /**
   * Public for testing without spinning up a Lance dataset. Round-robins `ids` into `groupCount`
   * sub-sequences while preserving relative order within each group.
   */
  private[knn] def roundRobin(ids: Seq[Int], groupCount: Int): Seq[Seq[Int]] = {
    val groups = Array.fill(groupCount)(scala.collection.mutable.ArrayBuffer.empty[Int])
    var i = 0
    while (i < ids.size) {
      groups(i % groupCount) += ids(i)
      i += 1
    }
    groups.toSeq.map(_.toSeq)
  }

  /**
   * Public for testing. LPT (Longest Processing Time) greedy bin-packing: sort by weight desc,
   * assign each item to the currently lightest group. 4/3-approximation of optimal makespan.
   */
  private[knn] def greedyBalance(weighted: Seq[(Int, Long)], groupCount: Int): Seq[Seq[Int]] = {
    val groups = Array.fill(groupCount)(scala.collection.mutable.ArrayBuffer.empty[Int])
    val totals = Array.fill(groupCount)(0L)
    val sorted = weighted.sortBy { case (_, w) => -w } // descending by weight
    sorted.foreach { case (id, w) =>
      var minIdx = 0
      var i = 1
      while (i < groupCount) {
        if (totals(i) < totals(minIdx)) minIdx = i
        i += 1
      }
      groups(minIdx) += id
      totals(minIdx) += w
    }
    groups.toSeq.map(_.toSeq)
  }

  private def openDataset(datasetUri: String, version: Option[Long]): Dataset = {
    val readOpts = {
      val b = new ReadOptions.Builder()
      version.foreach(v => b.setVersion(v))
      b.build()
    }
    Dataset
      .open()
      .uri(datasetUri)
      .allocator(LanceRuntime.allocator())
      .readOptions(readOpts)
      .build()
  }
}
