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
 * End-to-end correctness test for [[IndexedNearestJoin]].
 *
 * The right side is a Lance dataset written without a vector index, which means Lance falls back
 * to brute-force per-fragment search. That makes the result a recall = 1.0 oracle: the join's
 * top-K must equal the top-K we compute in plain Scala. Mismatches are real bugs, not recall
 * issues.
 *
 * These tests intentionally don't exercise indexed (approximate) search yet — that's the next
 * test class to add once we wire vector index DDL through Lance's Java API for the test setup.
 */
class IndexedNearestJoinTest {

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  private val Dim = 8
  private val NumRight = 64
  private val NumLeft = 16
  private val Seed = 7L

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("indexed-nearest-join-test")
      .master("local[2]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = if (spark != null) spark.stop()

  /**
   * Top-K result for every left row matches the brute-force oracle exactly. With no vector index
   * Lance does an exact scan, so this is the strictest correctness check we can write.
   */
  @Test def testInnerJoinMatchesBruteForceOracle(): Unit = {
    val rng = new Random(Seed)
    val leftDf = buildLeft(rng, NumLeft, Dim)
    val rightUri = writeRight(rng, NumRight, Dim)

    val k = 5

    val joined = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = k,
      metric = "l2",
      // Project a small subset of right columns to keep the test grounded in the expected shape.
      rightProjection = Some(Seq("rid", "rvec")))

    val result = joined.collect()
    // Every left row produced exactly k matches.
    assertEquals(NumLeft * k, result.length, "expected k results per left row")

    // Group by the left id (`lid`), compute oracle, compare.
    val byLid = result.groupBy(_.getAs[Int]("lid"))
    val rightVecs: Array[Array[Float]] = readRightVectors(rightUri)
    val rightIds: Array[Int] = readRightIds(rightUri)

    byLid.foreach { case (lid, rows) =>
      val sortedRows = rows.sortBy(_.getAs[Float]("__score"))
      val leftVec = leftVectorFor(leftDf, lid)
      val oracle = rightVecs.indices
        .map(idx => (rightIds(idx), l2(leftVec, rightVecs(idx))))
        .sortBy(_._2)
        .take(k)

      val actualIds = sortedRows.map(_.getAs[Int]("rid"))
      val actualScores = sortedRows.map(_.getAs[Float]("__score"))

      // Compare ids (set equality up to ties is enough — score equality below catches ordering).
      assertEquals(
        oracle.map(_._1).toSet,
        actualIds.toSet,
        s"top-K right ids for lid=$lid mismatch oracle")
      // Score values match within float tolerance.
      oracle.map(_._2).zip(actualScores).foreach { case (expected, actualScore) =>
        assertEquals(
          expected,
          actualScore,
          1e-4f,
          s"score mismatch for lid=$lid: oracle=${oracle.map(_._2)} actual=$actualScores")
      }
    }
  }

  /**
   * Output schema is `left.* ++ right.* ++ __score`. Verifies the column carry-through machinery,
   * including projection-driven right schema selection.
   */
  @Test def testOutputSchemaCarriesLeftThenRightThenScore(): Unit = {
    val leftDf = buildLeft(new Random(Seed), 4, Dim)
    val rightUri = writeRight(new Random(Seed + 1), 8, Dim)

    val joined = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = 2,
      metric = "l2",
      rightProjection = Some(Seq("rid")))

    val expectedFieldNames = Seq("lid", "lvec", "rid", "__score")
    assertEquals(expectedFieldNames, joined.schema.fieldNames.toSeq)
    // Right columns are widened to nullable to support left-outer; left fields keep their
    // declared nullability.
    assertTrue(joined.schema("rid").nullable, "right-side `rid` should be widened to nullable")
    assertTrue(joined.schema("__score").nullable, "score should be nullable")
  }

  /**
   * Phase 3 — `refineFactor` parameter passes through the pipeline without affecting correctness
   * on an unindexed dataset. Lance's brute-force scan is already exact, so any refine factor
   * yields the same result as without it. The point of this test is wiring: confirm
   * `IndexedNearestJoin.apply` plumbs the parameter to `LanceProbe.probe` via the stage Conf
   * without throwing. Real recall improvement only kicks in once an IVF-PQ index is built — that
   * test is a Phase 3.x follow-up.
   */
  @Test def testRefineFactorPassesThroughWithoutBreakingCorrectness(): Unit = {
    val rng = new Random(Seed)
    val leftDf = buildLeft(rng, NumLeft, Dim)
    val rightUri = writeRight(rng, NumRight, Dim)
    val k = 5

    val joined = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = k,
      metric = "l2",
      rightProjection = Some(Seq("rid", "rvec")),
      refineFactor = Some(3))

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
        s"top-K mismatch with refineFactor=3 (lid=$lid)")
    }
  }

  /**
   * `outerJoin = true` should preserve a left row when no right rows match — but with an unindexed
   * right side every probe always returns k results, so the no-match case can't happen
   * organically. We approximate it by passing a left row with a NULL vector, which `extractVector`
   * surfaces as zero-score "no result" and the outer path emits with NULL right columns.
   */
  @Test def testLeftOuterPreservesUnmatchedLeftRowsWithNullVector(): Unit = {
    val nullSchema = new StructType(Array(
      StructField("lid", IntegerType, nullable = false),
      StructField(
        "lvec",
        ArrayType(FloatType, containsNull = false),
        nullable = true,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val rows = Seq(
      RowFactory.create(Integer.valueOf(1), null) // null vector; should surface as a no-match left
    )
    val leftDf = spark.createDataFrame(rows.asJava, nullSchema)
    val rightUri = writeRight(new Random(Seed + 2), 8, Dim)

    val joined = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = 3,
      metric = "l2",
      rightProjection = Some(Seq("rid")),
      outerJoin = true)

    val rows2 = joined.collect()
    assertEquals(1, rows2.length, "outer join should preserve the single null-vector left row")
    val r = rows2.head
    assertEquals(1, r.getAs[Int]("lid"))
    assertTrue(
      r.isNullAt(joined.schema.fieldIndex("rid")),
      "rid should be NULL on outer-join no-match")
    assertTrue(
      r.isNullAt(joined.schema.fieldIndex("__score")),
      "score should be NULL on outer-join no-match")
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

  private def readRightVectors(uri: String): Array[Array[Float]] = {
    val df = spark.read.format("lance").load(uri).orderBy("rid")
    df.collect().map { r =>
      r.getAs[Seq[Float]]("rvec").toArray
    }
  }

  private def readRightIds(uri: String): Array[Int] = {
    val df = spark.read.format("lance").load(uri).orderBy("rid")
    df.collect().map(_.getAs[Int]("rid"))
  }

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
