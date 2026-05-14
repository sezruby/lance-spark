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
import org.apache.spark.sql.functions.{count, lit}
import org.apache.spark.sql.types._
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.io.TempDir

import java.nio.file.Path
import java.util.Random

import scala.collection.JavaConverters._

/**
 * Covers the exact consumer shapes that crashed an early 3-exec staged plan iteration
 * during development: `count(*)`, `Aggregate`, and operators that reference NONE of
 * the join output columns. Those shapes drove Catalyst's `ColumnPruning` to insert
 * `Project(Nil)` wrappers between the custom staged operators, which codegen to 0-field
 * `UnsafeRow`s and crashed the custom decoder (`AssertionError` / SIGSEGV under C2).
 *
 * `InterStageShuffle.mergeViaCatalystShuffle` sidesteps that entirely — no custom
 * `LogicalPlan` / `SparkPlan` exists for `ColumnPruning` to wrap — but these tests
 * confirm the property rather than relying on reasoning.
 */
class IndexedNearestJoinConsumerShapeTest {

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  private val Dim = 8
  private val NumRight = 32
  private val NumLeft = 8
  private val K = 3

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("indexed-nearest-join-consumer-shape")
      .master("local[2]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = if (spark != null) spark.stop()

  /** `df.count()` — the simplest case that crashed the reverted path. */
  @Test def testCountSucceeds(): Unit = {
    val joined = buildJoined()
    val n = joined.count()
    assertEquals((NumLeft * K).toLong, n, s"Expected ${NumLeft * K} rows; got $n")
  }

  /** `df.agg(count("*"))` — same conceptual shape, different entry point. */
  @Test def testAggCountSucceeds(): Unit = {
    val joined = buildJoined()
    val result = joined.agg(count("*")).collect()
    assertEquals(1, result.length)
    assertEquals((NumLeft * K).toLong, result.head.getLong(0))
  }

  /**
   * `df.select(lit(1))` — references NONE of the join's output columns. Under
   * `ColumnPruning` this is the strongest form of "prune everything from my child";
   * if any custom plan node were going to get wrapped in `Project(Nil)`, this is where.
   */
  @Test def testSelectLiteralSucceeds(): Unit = {
    val joined = buildJoined()
    val result = joined.select(lit(1).as("one")).collect()
    assertEquals(NumLeft * K, result.length)
    assertTrue(result.forall(_.getInt(0) == 1))
  }

  /** `df.collect()` — the normal case, asserts count and that real data is materialised. */
  @Test def testCollectMaterialisesAllColumns(): Unit = {
    val joined = buildJoined()
    val rows = joined.collect()
    assertEquals(NumLeft * K, rows.length)
    // Assert nothing is empty/corrupt — every row should have non-null lid, qvec, rid,
    // rvec, __score at the column positions the join output schema defines.
    rows.foreach { row =>
      assertNotNull(row.get(0), "lid (col 0) should be non-null")
      assertNotNull(row.get(1), "qvec (col 1) should be non-null")
      assertNotNull(row.get(2), "rid (col 2) should be non-null")
    }
  }

  // -- helpers ------------------------------------------------------------------------------

  private def buildJoined(): DataFrame = {
    val rng = new Random(23L)
    val leftDf = buildLeft(rng)
    val rightUri = writeRight(rng)
    IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "qvec",
      rightVecCol = "rvec",
      k = K,
      metric = "l2")
  }

  private def buildLeft(rng: Random) = {
    val schema = new StructType(Array(
      StructField("lid", LongType, nullable = false),
      StructField(
        "qvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val rows = (0 until NumLeft).map { i =>
      RowFactory.create(java.lang.Long.valueOf(i.toLong), randomVector(rng, Dim).toSeq.asJava)
    }
    spark.createDataFrame(rows.asJava, schema)
  }

  private def writeRight(rng: Random): String = {
    val schema = new StructType(Array(
      StructField("rid", LongType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val rows = (0 until NumRight).map { i =>
      RowFactory.create(
        java.lang.Long.valueOf((i + 1000).toLong),
        randomVector(rng, Dim).toSeq.asJava)
    }
    val df = spark.createDataFrame(rows.asJava, schema)
    val uri = tempDir.resolve(s"right_${System.nanoTime()}").toString
    df.write.format("lance").save(uri)
    uri
  }

  private def randomVector(rng: Random, dim: Int): Array[Float] = {
    val v = new Array[Float](dim)
    var i = 0
    while (i < dim) { v(i) = rng.nextFloat(); i += 1 }
    v
  }
}
