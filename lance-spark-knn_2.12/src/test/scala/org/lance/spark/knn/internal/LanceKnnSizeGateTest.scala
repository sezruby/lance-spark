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
 * Unit tests for [[LanceKnnSizeGate]]'s pure arithmetic — the footprint model, the executor-budget
 * parsing, and the dimension scaling. The live path (opening a Lance dataset, throwing/warning per
 * mode) is exercised by the integration tests; here we pin the constants and the decision math so a
 * regression in the published `a + b·|R|` model is caught without a cluster.
 */
class LanceKnnSizeGateTest {

  private val GiB: Long = 1L << 30
  private val MiB: Long = 1L << 20

  // ---- budget parsing (executor.memory + memoryOverhead) -----------------------------------

  @Test def testBudgetDefaultOverhead(): Unit = {
    // 8g heap + default overhead = max(384 MiB, 0.1 × 8 GiB) = 819.2 MiB.
    val expected = 8L * GiB + math.max(384L * MiB, (0.1 * (8L * GiB)).toLong)
    assertEquals(expected, LanceKnnSizeGate.budgetBytes("8g", None, None))
  }

  @Test def testBudgetExplicitOverhead(): Unit = {
    assertEquals(8L * GiB + 2L * GiB, LanceKnnSizeGate.budgetBytes("8g", Some("2g"), None))
  }

  @Test def testBudgetBareNumberIsMiB(): Unit = {
    // Bare numbers follow Spark's ByteUnit.MiB convention.
    assertEquals(512L * MiB + 128L * MiB, LanceKnnSizeGate.budgetBytes("512", Some("128"), None))
  }

  @Test def testBudgetSmallHeapClampsOverheadFloor(): Unit = {
    // 1g heap: 0.1 × 1 GiB = 102.4 MiB < 384 MiB floor, so overhead clamps to 384 MiB.
    assertEquals(1L * GiB + 384L * MiB, LanceKnnSizeGate.budgetBytes("1g", None, None))
  }

  // ---- per-row slope scales linearly with dimension ----------------------------------------

  @Test def testPerRowBytesReferenceDim(): Unit = {
    assertEquals(64.0 * MiB / 1e6, LanceKnnSizeGate.perRowBytesForDim(Some(128)), 1e-9)
  }

  @Test def testPerRowBytesScalesWithDim(): Unit = {
    val base = LanceKnnSizeGate.perRowBytesForDim(Some(128))
    assertEquals(2.0 * base, LanceKnnSizeGate.perRowBytesForDim(Some(256)), 1e-9)
  }

  @Test def testPerRowBytesUnknownDimUsesReference(): Unit = {
    assertEquals(
      LanceKnnSizeGate.perRowBytesForDim(Some(128)),
      LanceKnnSizeGate.perRowBytesForDim(None),
      1e-9)
  }

  // ---- footprint model vs. the published threshold table -----------------------------------

  private def estimateAt(numRows: Long, execRamBytes: Long) =
    LanceKnnSizeGate.estimateFor(
      numRows = numRows,
      dim = Some(128),
      safety = LanceKnnSizeGate.DefaultSafetyFraction,
      baselineBytes = LanceKnnSizeGate.DefaultBaselineBytes,
      bytesPerRowOverride = None,
      execRamBytes = execRamBytes,
      mode = "error")

  @Test def testThreshold8GiBAt78MFits(): Unit = {
    // Issue #11 table: 8 GiB @ SAFETY=0.7 → ~78M rows (dim=128).
    assertTrue(estimateAt(78000000L, 8L * GiB).fits, "78M rows should fit an 8 GiB budget")
    assertFalse(estimateAt(80000000L, 8L * GiB).fits, "80M rows should exceed an 8 GiB budget")
  }

  @Test def testThreshold16GiBAt167MFits(): Unit = {
    // Issue #11 table: 16 GiB @ SAFETY=0.7 → ~167M rows (dim=128).
    assertTrue(estimateAt(167000000L, 16L * GiB).fits, "167M rows should fit a 16 GiB budget")
    assertFalse(estimateAt(170000000L, 16L * GiB).fits, "170M rows should exceed a 16 GiB budget")
  }

  @Test def testBaselineDominatesForTinyR(): Unit = {
    // Small R: footprint ≈ baseline `a`, comfortably under any real budget.
    val est = estimateAt(1000L, 8L * GiB)
    assertTrue(est.fits)
    assertEquals(
      LanceKnnSizeGate.DefaultBaselineBytes + math.round(est.bytesPerRow * 1000L),
      est.footprintBytes)
  }

  @Test def testGiBFormattingIsHumanReadable(): Unit = {
    val est = estimateAt(80000000L, 8L * GiB)
    assertEquals("8.0", est.execRamGiB)
    assertEquals("5.6", est.budgetGiB) // 0.7 × 8 GiB
    // footprint carries the score through the `%.1f` GiB formatter used in the error message.
    assertTrue(est.footprintGiB.matches("""\d+\.\d"""), s"unexpected format: ${est.footprintGiB}")
  }
}
