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

import org.apache.spark.sql.{Row, RowFactory, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.io.TempDir

import java.nio.file.{Files, Path}
import java.util.Random

import scala.collection.JavaConverters._

/**
 * Behavioural tests for the per-query temp-Lance helper [[LanceTempR]]. Validates the
 * properties the design relies on:
 *
 *   - Round-trip: rows materialized to temp Lance read back with the same row count and
 *     payload as the source DataFrame.
 *   - Synthesised rid is unique within a single materialization.
 *   - FixedSizeList<f32, dim> vector columns survive the write+read.
 *   - Caller-requested projection columns are present in the temp; non-requested are
 *     dropped (column pruning).
 *   - Subplan-backed sources (Filter / Project chains over parquet) work the same as
 *     a flat parquet read — the helper only sees a `DataFrame`, not its source.
 *   - Empty source produces an empty (but readable) temp.
 *   - Validation: missing vec col, unknown projection col, reserved rid name → fail fast.
 */
class LanceTempRTest {

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  private val Dim: Int = 8
  private val NumRows: Int = 32
  private val Seed: Long = 11L

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("lance-temp-r-test")
      .master("local[2]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = {
    if (spark != null) spark.stop()
  }

  // -- core round-trip ------------------------------------------------------------------------

  /** Parquet-backed source: write via the helper, read back, verify row count + uniqueness of rid. */
  @Test def testParquetRoundTripRowCountAndRidUnique(): Unit = {
    val parquetUri = writeRandomParquet(NumRows, Dim)
    val rightDf = spark.read.parquet(parquetUri)

    val tempUri =
      LanceTempR.materialize(
        rightDf,
        vecCol = "vec",
        projection = Seq.empty,
        scratchDir = scratch())

    val readBack = spark.read.format("lance").load(tempUri)
    val rows = readBack.collect()
    assertEquals(NumRows, rows.length, "row count round-trips")

    val rids = readBack.select(col(LanceTempR.RidColumnName)).collect().map(_.getLong(0))
    assertEquals(NumRows, rids.distinct.length, "rid column is unique within one materialization")
  }

  /** Vec column survives unchanged: per-row equality with the source. */
  @Test def testVectorColumnPreserved(): Unit = {
    val parquetUri = writeRandomParquet(NumRows, Dim)
    val rightDf = spark.read.parquet(parquetUri)
    val tempUri =
      LanceTempR.materialize(
        rightDf,
        vecCol = "vec",
        projection = Seq("id"),
        scratchDir = scratch())

    val srcRows = rightDf.orderBy("id").collect()
    val tempRows = spark.read.format("lance").load(tempUri).orderBy("id").collect()
    assertEquals(srcRows.length, tempRows.length)
    srcRows.zip(tempRows).foreach { case (s, t) =>
      val sv = s.getAs[Seq[Float]]("vec").toArray
      val tv = t.getAs[Seq[Float]]("vec").toArray
      assertArrayEquals(sv, tv, 1e-6f, s"vector mismatch for id=${s.getInt(s.fieldIndex("id"))}")
    }
  }

  // -- projection (column pruning at write time) ---------------------------------------------

  /** When projection is specified, only requested cols (plus rid + vec) appear in temp. */
  @Test def testProjectionTrimsExtraColumns(): Unit = {
    val parquetUri = writeWideParquet(NumRows, Dim)
    val rightDf = spark.read.parquet(parquetUri)
    // Source has id, vec, label, payload, untouched. Project only id + label.
    val tempUri = LanceTempR.materialize(
      rightDf,
      vecCol = "vec",
      projection = Seq("id", "label"),
      scratchDir = scratch())

    val tempSchema = spark.read.format("lance").load(tempUri).schema.fieldNames.toSet
    assertEquals(
      Set(LanceTempR.RidColumnName, "vec", "id", "label"),
      tempSchema,
      "temp Lance schema should be exactly rid + vec + projection cols")
  }

  /** Empty projection still gets rid + vec. */
  @Test def testEmptyProjectionGivesRidPlusVec(): Unit = {
    val parquetUri = writeRandomParquet(NumRows, Dim)
    val rightDf = spark.read.parquet(parquetUri)
    val tempUri =
      LanceTempR.materialize(
        rightDf,
        vecCol = "vec",
        projection = Seq.empty,
        scratchDir = scratch())

    val tempSchema = spark.read.format("lance").load(tempUri).schema.fieldNames.toSet
    assertEquals(Set(LanceTempR.RidColumnName, "vec"), tempSchema)
  }

  // -- subplan source (the load-bearing case) -------------------------------------------------

  /**
   * Source is a subplan: parquet → Filter → Project. Helper handles it the same as flat parquet,
   * because it only consumes a DataFrame, not knowledge of the source.
   */
  @Test def testSubplanSourceFilterPlusProject(): Unit = {
    val parquetUri = writeWideParquet(NumRows, Dim)
    val raw = spark.read.parquet(parquetUri)
    // Keep only label = "even" rows (ids 0,2,4,...) and drop the payload column.
    val subplanRight = raw.filter(col("label") === "even").select("id", "vec")

    val expectedCount = subplanRight.count()
    assertTrue(expectedCount > 0, "test setup: subplan should have rows")

    val tempUri = LanceTempR.materialize(
      subplanRight,
      vecCol = "vec",
      projection = Seq("id"),
      scratchDir = scratch())
    val readBack = spark.read.format("lance").load(tempUri)
    assertEquals(
      expectedCount,
      readBack.count(),
      "row count of temp Lance equals row count of subplan-evaluated source")
  }

  // -- empty input ----------------------------------------------------------------------------

  /** Empty source DataFrame: temp dataset is created and reads back as 0 rows. */
  @Test def testEmptyDataFrame(): Unit = {
    val parquetUri = writeRandomParquet(0, Dim)
    val rightDf = spark.read.parquet(parquetUri)
    val tempUri =
      LanceTempR.materialize(
        rightDf,
        vecCol = "vec",
        projection = Seq.empty,
        scratchDir = scratch())

    val readBack = spark.read.format("lance").load(tempUri)
    assertEquals(0L, readBack.count(), "empty source produces empty Lance dataset")
  }

  // -- validation -----------------------------------------------------------------------------

  @Test def testRejectsMissingVecColumn(): Unit = {
    val parquetUri = writeRandomParquet(2, Dim)
    val rightDf = spark.read.parquet(parquetUri)
    val ex = assertThrows(
      classOf[IllegalArgumentException],
      () =>
        LanceTempR.materialize(
          rightDf,
          vecCol = "no_such_col",
          projection = Seq.empty,
          scratchDir = scratch()))
    assertTrue(ex.getMessage.contains("no_such_col"))
  }

  @Test def testRejectsUnknownProjectionColumn(): Unit = {
    val parquetUri = writeRandomParquet(2, Dim)
    val rightDf = spark.read.parquet(parquetUri)
    val ex = assertThrows(
      classOf[IllegalArgumentException],
      () =>
        LanceTempR.materialize(
          rightDf,
          vecCol = "vec",
          projection = Seq("id", "ghost"),
          scratchDir = scratch()))
    assertTrue(ex.getMessage.contains("ghost"))
  }

  @Test def testRejectsReservedRidName(): Unit = {
    val parquetUri = writeRandomParquet(2, Dim)
    val rightDf = spark.read.parquet(parquetUri)
    val ex = assertThrows(
      classOf[IllegalArgumentException],
      () =>
        LanceTempR.materialize(
          rightDf,
          vecCol = "vec",
          projection = Seq(LanceTempR.RidColumnName),
          scratchDir = scratch()))
    assertTrue(ex.getMessage.contains(LanceTempR.RidColumnName))
  }

  // -- resolveScratchDir ---------------------------------------------------------------------

  /** Conf key set: returned as-is. */
  @Test def testResolveScratchDirHonoursConfKey(): Unit = {
    val explicit = scratch()
    spark.conf.set(LanceTempR.ScratchDirConfKey, explicit)
    try {
      assertEquals(explicit, LanceTempR.resolveScratchDir(spark))
    } finally spark.conf.unset(LanceTempR.ScratchDirConfKey)
  }

  /**
   * Conf key unset in local mode: falls back to a local-FS path under spark.local.dir
   * (or system tmp). The exact path is implementation-detail; the contract is "non-empty
   * and works locally", verified by an actual round-trip below.
   */
  @Test def testResolveScratchDirLocalFallbackWorks(): Unit = {
    spark.conf.unset(LanceTempR.ScratchDirConfKey)
    val resolved = LanceTempR.resolveScratchDir(spark)
    assertTrue(resolved.nonEmpty, "local fallback must produce a non-empty path")
    // Confirm it actually works as a scratchDir argument: round-trip a tiny dataset.
    val parquetUri = writeRandomParquet(4, Dim)
    val rightDf = spark.read.parquet(parquetUri)
    val tempUri = LanceTempR.materialize(
      rightDf,
      vecCol = "vec",
      projection = Seq.empty,
      scratchDir = resolved)
    assertEquals(4L, spark.read.format("lance").load(tempUri).count())
  }

  // -- helpers --------------------------------------------------------------------------------

  /** Build (id, vec) parquet under tempDir; returns its URI. */
  private def writeRandomParquet(n: Int, dim: Int): String = {
    val schema = new StructType(Array(
      StructField("id", IntegerType, nullable = false),
      StructField(
        "vec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", dim.toLong).build())))
    val rng = new Random(Seed)
    val rows: Seq[Row] = (0 until n).map { i =>
      RowFactory.create(Integer.valueOf(i), randomVector(rng, dim))
    }
    val df = spark.createDataFrame(rows.asJava, schema)
    val uri = subPath("right_parquet").toString
    df.write.parquet(uri)
    uri
  }

  /** Wider source with label + payload + untouched, for projection-trim and subplan tests. */
  private def writeWideParquet(n: Int, dim: Int): String = {
    val schema = new StructType(Array(
      StructField("id", IntegerType, nullable = false),
      StructField(
        "vec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", dim.toLong).build()),
      StructField("label", StringType, nullable = false),
      StructField("payload", StringType, nullable = false),
      StructField("untouched", IntegerType, nullable = false)))
    val rng = new Random(Seed)
    val rows: Seq[Row] = (0 until n).map { i =>
      RowFactory.create(
        Integer.valueOf(i),
        randomVector(rng, dim),
        if (i % 2 == 0) "even" else "odd",
        s"p$i",
        Integer.valueOf(i * 17))
    }
    val df = spark.createDataFrame(rows.asJava, schema)
    val uri = subPath("wide_parquet").toString
    df.write.parquet(uri)
    uri
  }

  /** Scratch directory exists (the helper writes a child of it) — created if needed. */
  private def scratch(): String = {
    val p = tempDir.resolve("scratch_" + System.nanoTime())
    Files.createDirectories(p)
    p.toString
  }

  /**
   * Path that does NOT pre-exist — used as a Spark write target. Spark refuses to write
   * to an existing path without overwrite mode.
   */
  private def subPath(name: String): Path =
    tempDir.resolve(name + "_" + System.nanoTime())

  private def randomVector(rng: Random, dim: Int): Array[Float] = {
    val v = new Array[Float](dim)
    var i = 0
    while (i < dim) { v(i) = rng.nextFloat(); i += 1 }
    v
  }
}
