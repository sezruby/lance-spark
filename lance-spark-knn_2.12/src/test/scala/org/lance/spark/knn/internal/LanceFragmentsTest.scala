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

import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.Test

/**
 * Unit tests for [[LanceFragments.roundRobin]]. The actual Lance-backed enumeration is exercised
 * indirectly by the Phase 1.5 oracle test (which writes a dataset and reads its fragment list).
 * Here we just check the partitioning math.
 */
class LanceFragmentsTest {

  @Test def testRoundRobinBalanced(): Unit = {
    val groups = LanceFragments.roundRobin(Seq(10, 11, 12, 13, 14, 15), 3)
    assertEquals(3, groups.size)
    assertEquals(Seq(10, 13), groups(0))
    assertEquals(Seq(11, 14), groups(1))
    assertEquals(Seq(12, 15), groups(2))
  }

  /**
   * `groupCount > numFragments` produces empty trailing groups. The probe stage must tolerate
   * empty groups (skip them) — this is the contract the empty result encodes.
   */
  @Test def testMoreGroupsThanFragments(): Unit = {
    val groups = LanceFragments.roundRobin(Seq(7, 8), 5)
    assertEquals(5, groups.size)
    assertEquals(Seq(7), groups(0))
    assertEquals(Seq(8), groups(1))
    assertTrue(groups(2).isEmpty)
    assertTrue(groups(3).isEmpty)
    assertTrue(groups(4).isEmpty)
  }

  @Test def testSingleGroupReturnsAll(): Unit = {
    val groups = LanceFragments.roundRobin(Seq(1, 2, 3, 4), 1)
    assertEquals(Seq(Seq(1, 2, 3, 4)), groups)
  }

  @Test def testEmptyInputProducesEmptyGroups(): Unit = {
    val groups = LanceFragments.roundRobin(Seq.empty, 3)
    assertEquals(3, groups.size)
    assertTrue(groups.forall(_.isEmpty))
  }

  // -- greedyBalance / Phase 3 skew handling -------------------------------------------------

  /**
   * LPT greedy: given imbalanced fragments, the worst group's total should be no more than
   * 4/3 of the optimal. With 4 frags of weights (10, 10, 10, 1) split into 2 groups, optimal
   * makespan = 16. LPT places 10 in g0, 10 in g1, 10 in g0 (now 20), 1 in g1 (now 11) — so
   * g0=20, g1=11. Best balance achievable is g0=11, g1=20 (or symmetric); LPT happens to
   * arrive at one of those orderings here. Either way, no group exceeds 21 which is well within
   * the 4/3 bound (~21.3).
   */
  @Test def testGreedyBalanceKeepsHeaviestGroupBoundedFor4_3OptOpt(): Unit = {
    val groups = LanceFragments.greedyBalance(
      Seq((1, 10L), (2, 10L), (3, 10L), (4, 1L)),
      groupCount = 2)
    assertEquals(2, groups.size)
    val totals = groups.map(g => g.map(id => Map(1 -> 10L, 2 -> 10L, 3 -> 10L, 4 -> 1L)(id)).sum)
    val maxTotal = totals.max
    val sumAll = 31L
    val optimal = math.ceil(sumAll.toDouble / 2).toInt // 16
    val bound = math.ceil(optimal * 4.0 / 3.0).toInt // 22
    assertTrue(maxTotal <= bound, s"LPT maxTotal=$maxTotal exceeded 4/3 bound $bound")
  }

  /**
   * LPT collapses to round-robin-style behavior when all weights are equal — every group ends
   * up with the same number of items.
   */
  @Test def testGreedyBalanceEqualWeightsBalancesItemCount(): Unit = {
    val groups = LanceFragments.greedyBalance(
      Seq((1, 5L), (2, 5L), (3, 5L), (4, 5L), (5, 5L), (6, 5L)),
      groupCount = 3)
    assertEquals(3, groups.size)
    groups.foreach(g => assertEquals(2, g.size, "equal weights should yield equal item counts"))
  }

  /**
   * One huge fragment + many small ones: the huge one anchors a group on its own, and the
   * smalls fill the others. This is the textbook skew case Phase 3 cares about.
   */
  @Test def testGreedyBalanceIsolatesSkewedFragment(): Unit = {
    val groups = LanceFragments.greedyBalance(
      Seq((10, 100L), (20, 5L), (30, 5L), (40, 5L), (50, 5L)),
      groupCount = 2)
    assertEquals(2, groups.size)
    val groupWith10 = groups.find(_.contains(10)).get
    assertEquals(Seq(10), groupWith10, "skewed fragment should land in its own group")
    val otherGroup = groups.find(!_.contains(10)).get
    assertEquals(Set(20, 30, 40, 50), otherGroup.toSet)
  }
}
