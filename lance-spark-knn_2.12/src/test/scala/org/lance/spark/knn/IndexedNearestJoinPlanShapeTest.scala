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

import org.apache.spark.sql.{RowFactory, SparkSession}
import org.apache.spark.sql.types._
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.io.TempDir

import java.nio.file.Path
import java.util.Random

import scala.collection.JavaConverters._

/**
 * Plan-shape assertions for the 3-exec staged pipeline.
 *
 * `IndexedNearestJoin.apply` builds a `LanceMaterializeLogicalPlan → LanceMergeLogicalPlan →
 * LanceProbeLogicalPlan` tree, which lowers to `LanceMaterializeExec → LanceMergeExec →
 * ShuffleExchangeExec(inserted by EnsureRequirements) → LanceProbeExec → user-plan`.
 *
 * This test asserts the shape at the executed-plan level. Deeper AQE / correctness checks
 * live in [[IndexedNearestJoinAqeVisibilityTest]] and [[IndexedNearestJoinCorrectnessTest]].
 */
class IndexedNearestJoinPlanShapeTest {

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  private val Dim = 8

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("indexed-nearest-join-plan-shape")
      .master("local[2]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = if (spark != null) spark.stop()

  /**
   * The executed plan's tree string must contain all three custom exec names. The
   * strategy (`LanceKnnStagedStrategy`) must have lowered each logical node to its exec.
   */
  @Test def testExecutedPlanContainsAllThreeCustomExecs(): Unit = {
    val rng = new Random(11L)
    val leftDf = buildLeft(rng, n = 4, dim = Dim)
    val rightUri = writeRight(rng, n = 8, dim = Dim)

    val joined = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = 2,
      metric = "l2",
      rightProjection = Some(Seq("rid")))

    joined.collect() // stabilise AQE plan
    val tree = joined.queryExecution.executedPlan.treeString
    assertTrue(tree.contains("LanceProbe"), s"Expected LanceProbe exec in plan; got:\n$tree")
    assertTrue(tree.contains("LanceMerge"), s"Expected LanceMerge exec in plan; got:\n$tree")
    assertTrue(
      tree.contains("LanceMaterialize"),
      s"Expected LanceMaterialize exec in plan; got:\n$tree")
    assertTrue(
      tree.contains("Exchange"),
      s"Expected Exchange (from ClusteredDistribution) in plan; got:\n$tree")
  }

  // -- helpers ------------------------------------------------------------------------------

  private def buildLeft(rng: Random, n: Int, dim: Int) = {
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
    spark.createDataFrame(rows.asJava, schema)
  }

  private def writeRight(rng: Random, n: Int, dim: Int): String = {
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
    val df = spark.createDataFrame(rows.asJava, schema)
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
