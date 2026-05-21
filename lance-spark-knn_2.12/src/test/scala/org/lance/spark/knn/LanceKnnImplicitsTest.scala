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
 * End-to-end tests for the `df.kNearestJoin(rightDf, ...)` extension. Verifies:
 *
 *  1. The extension returns the same rows as `IndexedNearestJoin.apply(uri, ...)` — the
 *     wrapper just changes the call site, not the semantics, when R is a Lance scan.
 *  2. URI extraction handles a plain Lance `spark.read.load` — the common case.
 *  3. **Non-Lance R is supported** via per-query temp Lance materialization (sezruby/lance-spark#2):
 *     parquet, in-memory, alias-wrapped, and subplan-backed sources all produce the same
 *     top-K as the equivalent Lance-native R.
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
   * Parquet R: kNearestJoin transparently materializes a temp Lance dataset and returns
   * the same top-K row IDs as the equivalent Lance-native R run. Validates the oracle
   * equivalence end-to-end.
   *
   * Per-issue #2 design: non-Lance R is materialized once via [[LanceTempR.materialize]],
   * then the existing probe pipeline runs on the temp URI.
   */
  @Test def testKNearestJoinAgainstParquetRMatchesLanceR(): Unit = {
    val (leftDf, _, _) = buildLeft()
    val (rightLanceUri, rightIds, rightVecs) = writeRight()

    // Same R data as parquet, with rid + rvec only (matches the projection rightDf-Lance uses)
    val parquetPath = tempDir.resolve(s"docs_${System.nanoTime()}.parquet").toString
    rightIds.zip(rightVecs).map { case (rid, vec) => (rid, vec) } // sanity unused
    val parquetSchema = new StructType(Array(
      StructField("rid", IntegerType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val rows = rightIds.zip(rightVecs).map { case (rid, vec) =>
      RowFactory.create(Integer.valueOf(rid), vec)
    }
    spark.createDataFrame(rows.toSeq.asJava, parquetSchema).write.parquet(parquetPath)
    val parquetDf = spark.read.parquet(parquetPath)

    val viaParquet = leftDf
      .kNearestJoin(
        right = parquetDf,
        leftVecCol = "lvec",
        rightVecCol = "rvec",
        k = 5,
        metric = "l2",
        rightProjection = Some(Seq("rid")))
      .collect()

    val viaLance = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightLanceUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = 5,
      metric = "l2",
      rightProjection = Some(Seq("rid")))
      .collect()

    val byLid = (rs: Array[org.apache.spark.sql.Row]) =>
      rs.groupBy(_.getAs[Int]("lid")).map { case (lid, group) =>
        lid -> group.map(_.getAs[Int]("rid")).toSet
      }
    assertEquals(
      byLid(viaLance),
      byLid(viaParquet),
      "parquet R via temp materialization must produce the same top-K rid set as Lance R")
  }

  /**
   * Subplan-backed R: parquet → Filter → Project. The kNearestJoin extension only sees a
   * DataFrame; the temp-Lance materialization driver-evaluates whatever subplan is
   * underneath. Tests that this load-bearing case (issue #2 primary motivation) works.
   */
  @Test def testKNearestJoinAgainstSubplanR(): Unit = {
    val (leftDf, _, _) = buildLeft()
    val (_, rightIds, rightVecs) = writeRight()

    // Wider source so we have something to Filter / Project away. Add a `kept` boolean
    // column; the subplan will keep only rows where kept=true.
    val wideSchema = new StructType(Array(
      StructField("rid", IntegerType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build()),
      StructField("kept", BooleanType, nullable = false),
      StructField("payload", StringType, nullable = false)))
    val rows = rightIds.zip(rightVecs).zipWithIndex.map {
      case ((rid, vec), idx) =>
        RowFactory.create(
          Integer.valueOf(rid),
          vec,
          java.lang.Boolean.valueOf(idx % 2 == 0),
          s"p$idx")
    }
    val widePath = tempDir.resolve(s"wide_${System.nanoTime()}.parquet").toString
    spark.createDataFrame(rows.toSeq.asJava, wideSchema).write.parquet(widePath)

    import org.apache.spark.sql.functions.col
    val subplan = spark.read.parquet(widePath)
      .filter(col("kept") === true)
      .select("rid", "rvec")

    val expectedKeptIds = rightIds.zipWithIndex.collect { case (rid, idx) if idx % 2 == 0 => rid }
      .toSet

    val joined = leftDf
      .kNearestJoin(
        right = subplan,
        leftVecCol = "lvec",
        rightVecCol = "rvec",
        k = 3,
        metric = "l2",
        rightProjection = Some(Seq("rid")))
      .collect()

    // Every returned rid must come from the kept subset — proves the subplan was actually
    // evaluated before materialization (not just a reference to the underlying parquet).
    val joinedRids = joined.map(_.getAs[Int]("rid")).toSet
    val leakedRids = joinedRids -- expectedKeptIds
    assertTrue(
      leakedRids.isEmpty,
      s"top-K must be drawn only from the kept subset; got leaks: $leakedRids")
    assertEquals(NumLeft * 3, joined.length, "expected k=3 results per left row")
  }

  /**
   * Pin the within-query "same Lance dataset path" property: both the probe and the
   * materialize stages of the staged plan must reference the same temp Lance URI. If
   * the wiring ever drifts (e.g. helper produces URI A but probe pipeline gets URI B),
   * the staged pipeline reads from the wrong dataset and produces silent wrong results.
   * This is the structural pin that prevents that.
   */
  @Test def testProbeAndMaterializeShareSameTempUri(): Unit = {
    import org.lance.spark.knn.internal.staged.{LanceMaterializeLogicalPlan, LanceProbeLogicalPlan}
    val (leftDf, _, _) = buildLeft()
    val (_, rightIds, rightVecs) = writeRight()

    val schema = new StructType(Array(
      StructField("rid", IntegerType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val rows = rightIds.zip(rightVecs).map { case (rid, vec) =>
      RowFactory.create(Integer.valueOf(rid), vec)
    }
    val inMemoryR = spark.createDataFrame(rows.toSeq.asJava, schema)

    val joined = leftDf.kNearestJoin(
      right = inMemoryR,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = 3,
      metric = "l2",
      rightProjection = Some(Seq("rid")))

    // Walk the analyzed plan looking for the probe and materialize logical nodes; both
    // must carry the same `datasetUri` in their stage configs.
    val plan = joined.queryExecution.analyzed
    val probeNodes = plan.collect { case p: LanceProbeLogicalPlan => p.stageConf.datasetUri }
    val materializeNodes = plan.collect { case m: LanceMaterializeLogicalPlan =>
      m.stageConf.datasetUri
    }
    assertEquals(
      1,
      probeNodes.size,
      s"expected exactly one LanceProbeLogicalPlan; got: $probeNodes")
    assertEquals(
      1,
      materializeNodes.size,
      s"expected exactly one LanceMaterializeLogicalPlan; got: $materializeNodes")
    assertEquals(
      probeNodes.head,
      materializeNodes.head,
      "probe and materialize must reference the SAME temp Lance dataset URI")
  }

  /**
   * Right side has a column type Lance can't write (MapType). The DataFrame API path
   * is explicit — the user called `kNearestJoin` directly — so it must throw with a
   * helpful message rather than silently fall through. (The SQL rule path in the 4.2
   * module makes the opposite choice — it falls through to Spark's brute-force rewrite
   * because the user wrote a generic `APPROX NEAREST` query.)
   */
  @Test def testKNearestJoinRejectsUnsupportedColumnType(): Unit = {
    import org.apache.spark.sql.functions.{lit, map}
    val (leftDf, _, _) = buildLeft()
    val (_, rightIds, rightVecs) = writeRight()
    val schema = new StructType(Array(
      StructField("rid", IntegerType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val rows = rightIds.zip(rightVecs).map { case (rid, vec) =>
      RowFactory.create(Integer.valueOf(rid), vec)
    }
    val withMap = spark.createDataFrame(rows.toSeq.asJava, schema)
      .withColumn("attrs", map(lit("k"), lit("v")))

    val ex = assertThrows(
      classOf[IllegalArgumentException],
      () =>
        leftDf.kNearestJoin(
          right = withMap,
          leftVecCol = "lvec",
          rightVecCol = "rvec",
          k = 3,
          metric = "l2",
          rightProjection = Some(Seq("rid", "attrs"))))
    val msg = ex.getMessage.toLowerCase
    assertTrue(
      msg.contains("attrs") || msg.contains("map"),
      s"error should mention the offending column or type; got: ${ex.getMessage}")
  }

  /**
   * In-memory R (no underlying source — `createDataFrame(rows.asJava, schema)`). Same
   * temp-materialization path; just exercises the case where the rid synthesis and write
   * have no parquet/delta to come from.
   */
  @Test def testKNearestJoinAgainstInMemoryR(): Unit = {
    val (leftDf, _, _) = buildLeft()
    val (_, rightIds, rightVecs) = writeRight()

    val schema = new StructType(Array(
      StructField("rid", IntegerType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val rows = rightIds.zip(rightVecs).map { case (rid, vec) =>
      RowFactory.create(Integer.valueOf(rid), vec)
    }
    val inMemoryR = spark.createDataFrame(rows.toSeq.asJava, schema)

    // Verify even alias-wrapped works (SubqueryAlias unwrap → not a Lance scan → temp).
    val aliased = inMemoryR.as("docs")
    val joined = leftDf.kNearestJoin(
      right = aliased,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = 3,
      metric = "l2",
      rightProjection = Some(Seq("rid")))
      .collect()
    assertEquals(NumLeft * 3, joined.length, "expected k=3 results per left row")
    // All returned rids must come from the actual right side — sanity check.
    val joinedRids = joined.map(_.getAs[Int]("rid")).toSet
    val leaks = joinedRids -- rightIds.toSet
    assertTrue(leaks.isEmpty, s"rids should be drawn from the input set; leaks: $leaks")
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
