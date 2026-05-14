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
 * End-to-end tests for the `df.kNearestJoin(rightDf, ...)` extension. Three things to
 * verify:
 *
 *  1. The extension returns the same rows as `IndexedNearestJoin.apply(uri, ...)` — the
 *     wrapper just changes the call site, not the semantics.
 *  2. URI extraction handles a plain Lance `spark.read.load` — the common case.
 *  3. URI extraction throws cleanly when the right DataFrame isn't backed by a Lance scan
 *     (e.g. created from in-memory rows). Bad input must fail fast with a helpful message,
 *     not surface as a confusing runtime error inside the probe.
 *
 * The Phase 0 oracle test in `LanceProbeValidationTest` covers correctness of the underlying
 * probe; we can keep these tests light and not re-validate that.
 */
class LanceKnnImplicitsTest {

  import LanceKnnImplicits._

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  private val Dim = 8
  private val NumRight = 64
  private val NumLeft = 8
  private val Seed = 17L

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("lance-knn-implicits-test")
      .master("local[2]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = if (spark != null) spark.stop()

  /**
   * Happy path: a Lance-backed right DataFrame. Extension returns the same row-count and
   * rid set per left row as the URI-string `IndexedNearestJoin.apply` form.
   */
  @Test def testKNearestJoinAgainstLanceScanMatchesUriForm(): Unit = {
    val (leftDf, _, _) = buildLeft()
    val (rightUri, _, _) = writeRight()
    val rightDf = spark.read.format("lance").load(rightUri)

    val viaExtension = leftDf
      .kNearestJoin(
        right = rightDf,
        leftVecCol = "lvec",
        rightVecCol = "rvec",
        k = 5,
        metric = "l2",
        rightProjection = Some(Seq("rid")))
      .collect()
    val viaUri = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = 5,
      metric = "l2",
      rightProjection = Some(Seq("rid")))
      .collect()

    assertEquals(viaUri.length, viaExtension.length)
    val byLid = (rs: Array[org.apache.spark.sql.Row]) =>
      rs.groupBy(_.getAs[Int]("lid")).map { case (lid, group) =>
        lid -> group.map(_.getAs[Int]("rid")).toSet
      }
    assertEquals(byLid(viaUri), byLid(viaExtension))
  }

  /**
   * The right DataFrame is wrapped in a `Filter` (e.g. user wrote `docs.filter("rid > 0")`).
   * The URI extractor must walk past it to the underlying Lance relation.
   */
  @Test def testFilterOnRightStillExtractsUri(): Unit = {
    val (leftDf, _, _) = buildLeft()
    val (rightUri, _, _) = writeRight()
    val rightDf = spark.read.format("lance").load(rightUri).filter("rid > 0")

    val joined = leftDf
      .kNearestJoin(
        right = rightDf,
        leftVecCol = "lvec",
        rightVecCol = "rvec",
        k = 3,
        metric = "l2",
        rightProjection = Some(Seq("rid")))
      .collect()
    assertEquals(NumLeft * 3, joined.length, "expected k results per left row")
  }

  /**
   * A DataFrame backed by Parquet (or any non-Lance format) must also fail the Lance-only
   * guard — the API contract is `format("lance").load(...)` specifically, not "any DataFrame
   * Spark can read." Catches the case where a user wires in the wrong reader by mistake.
   */
  @Test def testParquetRightThrowsClearError(): Unit = {
    val (leftDf, _, _) = buildLeft()
    val parquetSchema = new StructType(Array(
      StructField("rid", IntegerType, nullable = false),
      StructField("rvec", ArrayType(FloatType, containsNull = false), nullable = false)))
    val rows = Seq(
      RowFactory.create(Integer.valueOf(1), Array.fill(Dim)(0.0f)),
      RowFactory.create(Integer.valueOf(2), Array.fill(Dim)(0.5f)))
    val parquetPath = tempDir.resolve(s"docs_${System.nanoTime()}.parquet").toString
    spark.createDataFrame(rows.asJava, parquetSchema).write.parquet(parquetPath)
    val parquetDf = spark.read.parquet(parquetPath)

    val ex = assertThrows(
      classOf[IllegalArgumentException],
      () =>
        leftDf.kNearestJoin(
          right = parquetDf,
          leftVecCol = "lvec",
          rightVecCol = "rvec",
          k = 3,
          metric = "l2"))
    assertTrue(
      ex.getMessage.contains("Lance scan"),
      s"expected error message to mention Lance scan for parquet input; got: ${ex.getMessage}")
  }

  /**
   * Non-Lance DataFrame wrapped in a `SubqueryAlias` (via `as("d")`) must still fail. The
   * URI extractor walks `SubqueryAlias` to find the underlying relation; if the underlying
   * is not Lance, alias unwrapping must NOT silently accept it.
   */
  @Test def testNonLanceUnderAliasThrowsClearError(): Unit = {
    val (leftDf, _, _) = buildLeft()
    val rows = Seq(RowFactory.create(Integer.valueOf(1), Array.fill(Dim)(0.0f)))
    val schema = new StructType(Array(
      StructField("rid", IntegerType, nullable = false),
      StructField("rvec", ArrayType(FloatType, containsNull = false), nullable = false)))
    val notLance = spark.createDataFrame(rows.asJava, schema).as("d")

    val ex = assertThrows(
      classOf[IllegalArgumentException],
      () =>
        leftDf.kNearestJoin(
          right = notLance,
          leftVecCol = "lvec",
          rightVecCol = "rvec",
          k = 3,
          metric = "l2"))
    assertTrue(
      ex.getMessage.contains("Lance scan"),
      s"alias-wrapped non-Lance must still fail; got: ${ex.getMessage}")
  }

  /**
   * A DataFrame built from in-memory rows is NOT a Lance scan — the extension should throw
   * an `IllegalArgumentException` with a message naming the constraint, so the user knows
   * to hand a real Lance DataFrame instead.
   */
  @Test def testNonLanceRightThrowsClearError(): Unit = {
    val (leftDf, _, _) = buildLeft()
    val ridSchema = new StructType(Array(
      StructField("rid", IntegerType, nullable = false),
      StructField("rvec", ArrayType(FloatType, containsNull = false), nullable = false)))
    val notLance = spark.createDataFrame(
      Seq(RowFactory.create(Integer.valueOf(1), Array.fill(Dim)(0.0f))).asJava,
      ridSchema)

    val ex = assertThrows(
      classOf[IllegalArgumentException],
      () =>
        leftDf.kNearestJoin(
          right = notLance,
          leftVecCol = "lvec",
          rightVecCol = "rvec",
          k = 3,
          metric = "l2"))
    assertTrue(
      ex.getMessage.contains("Lance scan"),
      s"expected error message to mention Lance scan; got: ${ex.getMessage}")
  }

  // -- helpers ------------------------------------------------------------------------------

  private def buildLeft(): (org.apache.spark.sql.DataFrame, Array[Int], Array[Array[Float]]) = {
    val rng = new Random(Seed)
    val schema = new StructType(Array(
      StructField("lid", IntegerType, nullable = false),
      StructField(
        "lvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val ids = (0 until NumLeft).toArray
    val vecs = ids.map(_ => randomVector(rng, Dim))
    val rows = ids.zip(vecs).map { case (id, v) => RowFactory.create(Integer.valueOf(id), v) }
    val df = spark.createDataFrame(rows.toSeq.asJava, schema)
    (df, ids, vecs)
  }

  private def writeRight(): (String, Array[Int], Array[Array[Float]]) = {
    val rng = new Random(Seed + 1)
    val schema = new StructType(Array(
      StructField("rid", IntegerType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val ids = (0 until NumRight).map(_ + 1000).toArray
    val vecs = ids.map(_ => randomVector(rng, Dim))
    val rows = ids.zip(vecs).map { case (id, v) => RowFactory.create(Integer.valueOf(id), v) }
    val df = spark.createDataFrame(rows.toSeq.asJava, schema)
    val out = tempDir.resolve(s"right_${System.nanoTime()}").toString
    df.write.format("lance").save(out)
    (out, ids, vecs)
  }

  private def randomVector(rng: Random, dim: Int): Array[Float] = {
    val v = new Array[Float](dim)
    var i = 0
    while (i < dim) { v(i) = rng.nextFloat(); i += 1 }
    v
  }
}
