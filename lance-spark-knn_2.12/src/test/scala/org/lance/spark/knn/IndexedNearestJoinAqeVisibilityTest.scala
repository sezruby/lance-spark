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

import org.apache.spark.sql.{DataFrame, RowFactory, SparkSession}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanExec
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec
import org.apache.spark.sql.types._
import org.junit.jupiter.api.{AfterEach, Test}
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.io.TempDir

import java.nio.file.Path
import java.util.Random

import scala.collection.JavaConverters._

/**
 * Proves that the 3-exec staged pipeline (LanceProbeExec → ShuffleExchangeExec →
 * LanceMergeExec → LanceMaterializeExec) is Catalyst-visible end-to-end, so AQE
 * engages on the merge-side shuffle.
 *
 * The `LanceMergeExec.requiredChildDistribution = ClusteredDistribution(leftId)` on the
 * physical exec is what makes `EnsureRequirements` insert a `ShuffleExchangeExec` between
 * probe and merge. That exchange is the AQE wrap point.
 *
 * Unlike the previous `InterStageShuffle` (`repartition`-based) approach — which called
 * `.rdd` on an intermediate DataFrame, collapsing the Catalyst plan to an RDD before the
 * final join wrapped it — here the three execs live in ONE `joined.queryExecution.executedPlan`
 * tree. The Exchange is directly inspectable in that tree and AQE's `AdaptiveSparkPlanExec`
 * wraps the whole thing when AQE is enabled.
 */
class IndexedNearestJoinAqeVisibilityTest {

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  private val Dim = 8

  @AfterEach def teardown(): Unit = if (spark != null) spark.stop()

  private def newSparkSession(aqeEnabled: Boolean): SparkSession = {
    spark = SparkSession.builder()
      .appName(s"aqe-visibility-aqe-$aqeEnabled")
      .master("local[2]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.sql.adaptive.enabled", aqeEnabled.toString)
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark
  }

  /**
   * The joined DataFrame's executed plan contains a `ShuffleExchangeExec` with
   * `hashpartitioning` on the `_leftId` attribute — inserted by `EnsureRequirements` in
   * response to `LanceMergeExec.requiredChildDistribution = ClusteredDistribution`.
   * This is the shape-level proof the exchange is Catalyst-planned.
   */
  @Test def testExchangeHashpartitioningOnLeftIdWithAqeEnabled(): Unit = {
    val s = newSparkSession(aqeEnabled = true)
    val joined = buildJoined(s)
    joined.collect()
    assertExchangeOnLeftId(joined)
  }

  @Test def testExchangeHashpartitioningOnLeftIdWithAqeDisabled(): Unit = {
    val s = newSparkSession(aqeEnabled = false)
    val joined = buildJoined(s)
    joined.collect()
    assertExchangeOnLeftId(joined)
  }

  /**
   * With AQE enabled, the executed plan is wrapped in an `AdaptiveSparkPlanExec` — AQE's
   * entry point. This is the end-to-end proof AQE engaged on the full tree (not just an
   * inner sub-plan like the previous `InterStageShuffle` approach produced).
   */
  @Test def testAqeAdaptivePlanWrapsEntireExecution(): Unit = {
    val s = newSparkSession(aqeEnabled = true)
    val joined = buildJoined(s)
    joined.collect()
    val plan = joined.queryExecution.executedPlan
    val aqeNodes = collectAqe(plan)
    assertTrue(
      aqeNodes.nonEmpty,
      s"Expected AdaptiveSparkPlanExec at the top of executed plan; got:\n${plan.treeString}")
  }

  /**
   * `LanceProbe`, `LanceMerge`, and `LanceMaterialize` are all named in the executed
   * plan's tree string. Confirms the three custom execs are actually wired in.
   */
  @Test def testAllThreeCustomExecsInTree(): Unit = {
    val s = newSparkSession(aqeEnabled = true)
    val joined = buildJoined(s)
    joined.collect()
    val tree = joined.queryExecution.executedPlan.treeString
    assertTrue(tree.contains("LanceProbe"), s"Expected LanceProbe in executed plan; got:\n$tree")
    assertTrue(tree.contains("LanceMerge"), s"Expected LanceMerge in executed plan; got:\n$tree")
    assertTrue(
      tree.contains("LanceMaterialize"),
      s"Expected LanceMaterialize in executed plan; got:\n$tree")
  }

  /**
   * Regression for the `missingInput` bug caught during the initial 3-exec split: if
   * `producedAttributes` is not set, Spark's tree-string prefixes each custom node with
   * `!` to flag "this node references attrs not in child.outputSet". A clean tree string
   * has no `!` prefix. Asserts we haven't regressed.
   */
  @Test def testNoMissingInputBangPrefix(): Unit = {
    val s = newSparkSession(aqeEnabled = false)
    val joined = buildJoined(s)
    joined.collect()
    val tree = joined.queryExecution.executedPlan.treeString
    // Every line that starts with optional whitespace + "!" is a missingInput warning.
    // Allow `!` elsewhere (e.g. inside parenthesized text). Check for the known pattern:
    //   "+- !LanceProbe"  or  "!LanceMerge"  etc.
    assertFalse(
      tree.contains("!LanceProbe") || tree.contains("!LanceMerge") ||
        tree.contains("!LanceMaterialize"),
      s"Found `!` missingInput prefix on a custom exec; got:\n$tree")
  }

  // -- helpers ------------------------------------------------------------------------------

  private def assertExchangeOnLeftId(joined: DataFrame): Unit = {
    val plan = joined.queryExecution.executedPlan
    val treeString = plan.treeString
    // Use string-match on the tree. With AQE enabled, the Exchange can be nested inside
    // `ShuffleQueryStageExec` / `AQEShuffleRead` wrappers whose `children` relationship
    // isn't a plain SparkPlan child — walking via `.children` misses them. The string
    // form is stable across AQE on/off and is what `df.explain()` prints, so matching
    // the same text the user would see is both simpler and correct.
    assertTrue(
      treeString.contains("Exchange hashpartitioning(_leftId") ||
        treeString.contains("hashpartitioning(_leftId"),
      s"Expected Exchange hashpartitioning on _leftId; executedPlan:\n$treeString")
  }

  private def collectAqe(plan: SparkPlan): Seq[AdaptiveSparkPlanExec] = {
    val hits = scala.collection.mutable.ArrayBuffer.empty[AdaptiveSparkPlanExec]
    def walk(p: SparkPlan): Unit = {
      p match {
        case aqe: AdaptiveSparkPlanExec =>
          hits += aqe
          walk(aqe.executedPlan)
        case _ =>
      }
      p.children.foreach(walk)
    }
    walk(plan)
    hits.toSeq
  }

  private def buildJoined(s: SparkSession): DataFrame = {
    val rng = new Random(17L)
    val leftDf = buildLeft(s, rng, n = 8, dim = Dim)
    val rightUri = writeRight(s, rng, n = 16, dim = Dim)
    IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = 2,
      metric = "l2",
      rightProjection = Some(Seq("rid")))
  }

  private def buildLeft(s: SparkSession, rng: Random, n: Int, dim: Int) = {
    val schema = new StructType(Array(
      StructField("lid", IntegerType, nullable = false),
      StructField(
        "lvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", dim.toLong).build())))
    val rows = (0 until n).map { i =>
      RowFactory.create(Integer.valueOf(i), randomVector(rng, dim))
    }
    s.createDataFrame(rows.asJava, schema)
  }

  private def writeRight(s: SparkSession, rng: Random, n: Int, dim: Int): String = {
    val schema = new StructType(Array(
      StructField("rid", IntegerType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", dim.toLong).build())))
    val rows = (0 until n).map { i =>
      RowFactory.create(Integer.valueOf(i + 1000), randomVector(rng, dim))
    }
    val df = s.createDataFrame(rows.asJava, schema)
    val out = tempDir.resolve(s"right_${System.nanoTime()}").toString
    df.write.format("lance").save(out)
    out
  }

  private def randomVector(rng: Random, dim: Int): Array[Float] = {
    val v = new Array[Float](dim)
    var i = 0
    while (i < dim) { v(i) = rng.nextFloat(); i += 1 }
    v
  }
}
