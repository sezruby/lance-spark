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

import org.apache.spark.sql.{Row, RowFactory, SparkSession}
import org.apache.spark.sql.types._
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.io.TempDir

import java.nio.file.Path
import java.util.Random

import scala.collection.JavaConverters._

/**
 * Phase 1.5 — fragment-grouped probing.
 *
 * The substantive change vs. Phase 1: with `probeParallelism > 1`, the rule splits Lance
 * fragments into N groups, replicates each left row across the groups, and lets the merge stage
 * aggregate N contributions per leftId via [[org.lance.spark.knn.internal.TopKHeap]].
 *
 * Coverage:
 *   - Oracle equivalence: top-K matches the brute-force oracle when probeParallelism = N. With
 *     no vector index, every per-fragment-group probe is exact (recall = 1.0), so the merged
 *     result must match exact brute force.
 *   - Plan-shape: the lineage contains TWO `ShuffledRDD`s — one from the replicate-and-
 *     partition-by-group step (probe), one from `reduceByKey` (merge). Phase 1 had only one.
 *   - Falls back gracefully when probeParallelism > numFragments (extra groups are empty).
 */
class IndexedNearestJoinFragmentGroupingTest {

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  private val Dim = 8
  private val NumRight = 64
  private val NumLeft = 16
  private val Seed = 31L

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("indexed-nearest-fragment-grouping-test")
      .master("local[2]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = if (spark != null) spark.stop()

  /**
   * Top-K matches the brute-force oracle when `probeParallelism = 4` and the right dataset has
   * at least 4 fragments. Confirms the merge function correctly combines per-fragment-group
   * contributions.
   */
  @Test def testOracleEquivalenceWithFragmentGrouping(): Unit = {
    val rng = new Random(Seed)
    val leftDf = buildLeft(rng, NumLeft, Dim)
    val rightUri = writeRight(rng, NumRight, Dim, fragments = 4)

    val k = 5
    val joined = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = k,
      metric = "l2",
      rightProjection = Some(Seq("rid", "rvec")),
      probeParallelism = 4)

    val result = joined.collect()
    assertEquals(NumLeft * k, result.length, "expected k results per left row")

    val byLid = result.groupBy(_.getAs[Int]("lid"))
    val rightVecs = readRightVectors(rightUri)
    val rightIds = readRightIds(rightUri)

    byLid.foreach { case (lid, rows) =>
      val sortedRows = rows.sortBy(_.getAs[Float]("__score"))
      val leftVec = leftVectorFor(leftDf, lid)
      val oracle = rightVecs.indices
        .map(idx => (rightIds(idx), l2(leftVec, rightVecs(idx))))
        .sortBy(_._2)
        .take(k)

      assertEquals(
        oracle.map(_._1).toSet,
        sortedRows.map(_.getAs[Int]("rid")).toSet,
        s"top-K right ids for lid=$lid mismatch oracle (probeParallelism = 4)")

      oracle.map(_._2).zip(sortedRows.map(_.getAs[Float]("__score"))).foreach {
        case (expected, actual) =>
          assertEquals(expected, actual, 1e-4f, s"score mismatch for lid=$lid")
      }
    }
  }

  /**
   * Plan-shape: with `probeParallelism > 1` the lineage gains a second `ShuffledRDD` (the
   * replicate-and-partition-by-group step). Phase 1's degenerate single-task path has only the
   * merge-side shuffle.
   */
  @Test def testFragmentGroupingAddsExtraShuffleToLineage(): Unit = {
    val rng = new Random(Seed + 1)
    val leftDf = buildLeft(rng, 4, Dim)
    val rightUri = writeRight(rng, 16, Dim, fragments = 2)

    val phase1 = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = 2,
      metric = "l2",
      rightProjection = Some(Seq("rid")),
      probeParallelism = 1)
    val phase15 = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = 2,
      metric = "l2",
      rightProjection = Some(Seq("rid")),
      probeParallelism = 2)

    val phase1Lineage = phase1.rdd.toDebugString
    val phase15Lineage = phase15.rdd.toDebugString

    val countShuffles = (s: String) => "ShuffledRDD".r.findAllIn(s).length
    val phase1Shuffles = countShuffles(phase1Lineage)
    val phase15Shuffles = countShuffles(phase15Lineage)
    assertTrue(
      phase15Shuffles > phase1Shuffles,
      s"Phase 1.5 should add at least one more ShuffledRDD; phase1=$phase1Shuffles, " +
        s"phase15=$phase15Shuffles\nlineage:\n$phase15Lineage")
  }

  /**
   * Phase 3 — skew handling. With `balanceFragmentsByRowCount = true`, fragment groups are
   * balanced via LPT bin-packing on per-fragment row counts. The oracle equivalence test still
   * holds — the ordering of fragments within a group doesn't change top-K results, only the
   * load distribution.
   */
  @Test def testOracleEquivalenceWithRowCountBalancing(): Unit = {
    val rng = new Random(Seed + 3)
    val leftDf = buildLeft(rng, NumLeft, Dim)
    val rightUri = writeRight(rng, NumRight, Dim, fragments = 4)

    val k = 5
    val joined = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = k,
      metric = "l2",
      rightProjection = Some(Seq("rid", "rvec")),
      probeParallelism = 4,
      balanceFragmentsByRowCount = true)

    val result = joined.collect()
    assertEquals(NumLeft * k, result.length)

    val byLid = result.groupBy(_.getAs[Int]("lid"))
    val rightVecs = readRightVectors(rightUri)
    val rightIds = readRightIds(rightUri)
    byLid.foreach { case (lid, rows) =>
      val sorted = rows.sortBy(_.getAs[Float]("__score"))
      val leftVec = leftVectorFor(leftDf, lid)
      val oracle = rightVecs.indices
        .map(idx => (rightIds(idx), l2(leftVec, rightVecs(idx))))
        .sortBy(_._2)
        .take(k)
      assertEquals(
        oracle.map(_._1).toSet,
        sorted.map(_.getAs[Int]("rid")).toSet,
        s"top-K mismatch with balanceFragmentsByRowCount = true (lid=$lid)")
    }
  }

  /**
   * `probeParallelism` > num fragments → extra groups are empty → result is still correct.
   * Specifically, the rule degenerates to the Phase 1 path when only one non-empty group exists.
   */
  @Test def testProbeParallelismExceedingFragmentsStillCorrect(): Unit = {
    val rng = new Random(Seed + 2)
    val leftDf = buildLeft(rng, 4, Dim)
    // Dataset with a single fragment — probeParallelism = 8 must still produce correct results.
    val rightUri = writeRight(rng, 16, Dim, fragments = 1)

    val joined = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = 3,
      metric = "l2",
      rightProjection = Some(Seq("rid")),
      probeParallelism = 8)
    assertEquals(4 * 3, joined.collect().length)
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

  /**
   * Write the right dataset, repartitioning the source DataFrame into `fragments` partitions
   * before save so that Lance produces approximately one fragment per Spark partition.
   */
  private def writeRight(rng: Random, n: Int, dim: Int, fragments: Int): String = {
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
    val df = spark.createDataFrame(rows.asJava, schema).repartition(fragments)
    val out = tempDir.resolve(s"right_${System.nanoTime()}").toString
    df.write.format("lance").save(out)
    out
  }

  private def readRightVectors(uri: String): Array[Array[Float]] =
    spark.read.format("lance").load(uri).orderBy("rid").collect().map { r =>
      r.getAs[Seq[Float]]("rvec").toArray
    }

  private def readRightIds(uri: String): Array[Int] =
    spark.read.format("lance").load(uri).orderBy("rid").collect().map(_.getAs[Int]("rid"))

  private def leftVectorFor(left: org.apache.spark.sql.DataFrame, lid: Int): Array[Float] = {
    val r = left.filter(s"lid = $lid").collect().head
    r.getAs[Seq[Float]]("lvec").toArray
  }

  private def randomVector(rng: Random, dim: Int): Array[Float] = {
    val v = new Array[Float](dim)
    var i = 0
    while (i < dim) { v(i) = rng.nextFloat(); i += 1 }
    v
  }

  private def l2(a: Array[Float], b: Array[Float]): Float = {
    var s = 0.0f
    var i = 0
    while (i < a.length) {
      val d = a(i) - b(i); s += d * d; i += 1
    }
    s
  }
}
