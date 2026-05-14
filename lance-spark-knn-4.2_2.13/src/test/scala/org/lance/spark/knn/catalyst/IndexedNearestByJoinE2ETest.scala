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
package org.lance.spark.knn.catalyst

import org.apache.spark.sql.{RowFactory, SparkSession}
import org.apache.spark.sql.types._
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.io.TempDir
import org.lance.spark.knn.internal.staged.LanceProbeLogicalPlan

import java.nio.file.Path
import java.util.Random

import scala.collection.JavaConverters._

/**
 * End-to-end SQL test for the Phase 2 Catalyst integration. Drives the full path:
 *
 *   ANTLR parser  ─▶  Analyzer  ─▶  IndexedNearestByJoinRule (our postHoc)  ─▶
 *      Optimizer  ─▶  LanceKnnStagedStrategy  ─▶
 *      LanceProbeExec ─▶ ShuffleExchangeExec ─▶ LanceMergeExec ─▶ LanceMaterializeExec ─▶
 *      Lance brute-force per-fragment scan  ─▶  Rows
 *
 * Requires Spark 4.2-SNAPSHOT (the only release where `NearestByJoin` exists today, added by
 * SPARK-56395) AND the lance-spark-4.1_2.13 connector built against the same Spark version. To
 * set up:
 *
 * {{{
 *   cd /path/to/spark/master
 *   ./build/mvn install -DskipTests -DskipChecks -pl sql/core -am
 *   cd /path/to/lance-spark
 *   ./mvnw install -pl lance-spark-4.1_2.13 -am -DskipTests \
 *     -Dspark41.version=4.2.0-SNAPSHOT -Darrow183.version=19.0.0
 *   ./mvnw install -pl lance-spark-knn_2.13 -am -DskipTests
 *   ./mvnw -pl lance-spark-knn-4.2_2.13 test
 * }}}
 *
 * Coverage:
 *   - SQL `APPROX NEAREST k BY DISTANCE vector_l2_distance(...)` parses, the rule rewrites
 *     to the 3-logical-plan tree (probe/merge/materialize), the strategy lowers it to the
 *     matching exec chain, which executes against a real Lance dataset; results match the
 *     brute-force oracle.
 *   - With the gating config disabled, the same SQL falls through to Spark's
 *     `RewriteNearestByJoin` (cross-product + `MaxMinByK`) — proves the rule's opt-in
 *     behavior at the SQL level.
 */
class IndexedNearestByJoinE2ETest {

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  private val Dim = 16
  private val NumRight = 64
  private val NumLeft = 8
  private val Seed = 4242L

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("indexed-nearest-by-join-e2e")
      .master("local[2]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .config(
        "spark.sql.extensions",
        "org.lance.spark.knn.extensions.LanceKnnSparkSessionExtensions")
      .config("spark.sql.crossJoin.enabled", "true")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  @AfterEach def teardown(): Unit = if (spark != null) spark.stop()

  /**
   * Full SQL path with the rule enabled. The physical plan must contain
   * [[LanceProbeExec]] AND the result must match the brute-force oracle on every left row.
   * With no vector index built, Lance does an exact per-fragment scan, so any disagreement
   * with brute force is a bug.
   */
  @Test def testSqlApproxNearestRoutesThroughIndexedPathAndMatchesOracle(): Unit = {
    spark.conf.set(IndexedNearestByJoinRule.EnabledConfKey, "true")

    val (leftRows, leftVectors, leftIds) = generateLeft(NumLeft, Dim, Seed)
    val (rightRows, rightVectors, rightIds) = generateRight(NumRight, Dim, Seed + 1)
    val rightUri = writeRightLance(rightRows)

    spark.createDataFrame(leftRows.asJava, leftSchema()).createOrReplaceTempView("queries")
    spark.read.format("lance").load(rightUri).createOrReplaceTempView("docs")

    val k = 5
    val sql =
      s"""SELECT q.lid, d.rid
         |FROM queries q INNER JOIN docs d
         |APPROX NEAREST $k BY DISTANCE vector_l2_distance(q.lvec, d.rvec)""".stripMargin
    val df = spark.sql(sql)

    // Plan-shape: confirm the rule fired AND the strategy lowered all three execs with a
    // Catalyst-inserted Exchange between probe and merge. Use `optimizedPlan` for the
    // logical assertion (AQE-independent) and the physical `treeString` for the exec
    // assertion. Checking all three nodes + the hashpartitioning exchange is the SQL-side
    // analogue of `IndexedNearestJoinAqeVisibilityTest.testAllThreeCustomExecsInTree` on
    // the DataFrame path — the strategy is shared, but the rule wiring is SQL-specific.
    val optimized = df.queryExecution.optimizedPlan
    val probeLogicals = optimized.collect { case p: LanceProbeLogicalPlan => p }
    assertTrue(
      probeLogicals.nonEmpty,
      s"expected LanceProbeLogicalPlan in optimized plan; got:\n$optimized")
    val executed = df.queryExecution.executedPlan
    val tree = executed.treeString
    assertTrue(tree.contains("LanceProbe"), s"expected LanceProbe exec in tree:\n$tree")
    assertTrue(tree.contains("LanceMerge"), s"expected LanceMerge exec in tree:\n$tree")
    assertTrue(
      tree.contains("LanceMaterialize"),
      s"expected LanceMaterialize exec in tree:\n$tree")
    assertTrue(
      tree.contains("hashpartitioning(_leftId"),
      s"expected hashpartitioning(_leftId) Exchange in tree:\n$tree")

    // Correctness: oracle equivalence.
    val rows = df.collect()
    assertEquals(NumLeft * k, rows.length, "expected k results per left row")
    val byLid = rows.groupBy(_.getAs[Int]("lid"))
    leftIds.zip(leftVectors).foreach { case (lid, lvec) =>
      val oracle = rightVectors.indices
        .map(i => (rightIds(i), l2(lvec, rightVectors(i))))
        .sortBy(_._2)
        .take(k)
        .map(_._1)
        .toSet
      val actual = byLid(lid).map(_.getAs[Int]("rid")).toSet
      assertEquals(
        oracle,
        actual,
        s"top-K mismatch for lid=$lid (rule on, brute-force oracle)")
    }
  }

  /**
   * Right-side `WHERE` clause must round-trip through the prefilter pushdown — Lance computes
   * top-K only over rows matching the filter, so the result must equal the brute-force oracle
   * computed AFTER applying the same filter. If the rule pushed the filter wrong (or dropped
   * it), this test would diverge from the oracle.
   *
   * Two right-side rows in this test share each `category` value, so a `WHERE category = 'A'`
   * shrinks the candidate pool meaningfully without zeroing it out.
   */
  @Test def testSqlWherePushdownMatchesFilteredOracle(): Unit = {
    spark.conf.set(IndexedNearestByJoinRule.EnabledConfKey, "true")

    val (leftRows, leftVectors, leftIds) = generateLeft(NumLeft, Dim, Seed + 200)
    val (rightRows, rightVectors, rightIds, rightCategories) =
      generateRightWithCategories(NumRight, Dim, Seed + 201)
    val rightUri = writeRightLanceWithCategories(rightRows)

    spark.createDataFrame(leftRows.asJava, leftSchema()).createOrReplaceTempView("queries")
    spark.read.format("lance").load(rightUri).createOrReplaceTempView("docs")

    val k = 4
    val targetCat = "A"
    val sql =
      s"""SELECT q.lid, d.rid
         |FROM queries q INNER JOIN (SELECT * FROM docs WHERE category = '$targetCat') d
         |APPROX NEAREST $k BY DISTANCE vector_l2_distance(q.lvec, d.rvec)""".stripMargin
    val df = spark.sql(sql)

    val optimized = df.queryExecution.optimizedPlan
    val probeLogicals = optimized.collect { case p: LanceProbeLogicalPlan => p }
    assertTrue(
      probeLogicals.nonEmpty,
      s"expected LanceProbeLogicalPlan; optimized plan was:\n$optimized")
    val prefilter = probeLogicals.head.stageConf.prefilter
    assertTrue(
      prefilter.exists(_.contains(s"'$targetCat'")),
      s"expected prefilter to carry category='$targetCat'; got: $prefilter")

    // Oracle: brute-force top-K computed AFTER applying the same filter on the right side.
    val filteredIdxs = rightCategories.indices.filter(rightCategories(_) == targetCat)
    val rows = df.collect()
    val byLid = rows.groupBy(_.getAs[Int]("lid"))
    leftIds.zip(leftVectors).foreach { case (lid, lvec) =>
      val oracle = filteredIdxs
        .map(i => (rightIds(i), l2(lvec, rightVectors(i))))
        .sortBy(_._2)
        .take(k)
        .map(_._1)
        .toSet
      assertTrue(
        oracle.nonEmpty,
        s"oracle is empty for lid=$lid — test setup didn't produce filterable rows")
      val actual = byLid(lid).map(_.getAs[Int]("rid")).toSet
      assertEquals(
        oracle,
        actual,
        s"top-K mismatch under WHERE pushdown for lid=$lid (filtered brute-force oracle)")
    }
  }

  /**
   * With the gating config disabled, the SAME SQL falls through to Spark's
   * `RewriteNearestByJoin` (cross-product + `MaxMinByK`). The plan contains NO
   * `LanceProbeLogicalPlan` and (importantly) results still match the oracle. This proves
   * the rule's opt-in behavior at the SQL level: turning it off doesn't break correctness.
   */
  @Test def testSqlFallsThroughToBruteForceWhenRuleDisabled(): Unit = {
    spark.conf.set(IndexedNearestByJoinRule.EnabledConfKey, "false")

    val (leftRows, leftVectors, leftIds) = generateLeft(NumLeft, Dim, Seed + 100)
    val (rightRows, rightVectors, rightIds) = generateRight(NumRight, Dim, Seed + 101)
    val rightUri = writeRightLance(rightRows)

    spark.createDataFrame(leftRows.asJava, leftSchema()).createOrReplaceTempView("queries")
    spark.read.format("lance").load(rightUri).createOrReplaceTempView("docs")

    val k = 4
    val df = spark.sql(
      s"""SELECT q.lid, d.rid
         |FROM queries q INNER JOIN docs d
         |APPROX NEAREST $k BY DISTANCE vector_l2_distance(q.lvec, d.rvec)""".stripMargin)

    val optimized = df.queryExecution.optimizedPlan
    val probeLogicals = optimized.collect { case p: LanceProbeLogicalPlan => p }
    assertTrue(
      probeLogicals.isEmpty,
      s"rule disabled — expected NO LanceProbeLogicalPlan; got:\n$optimized")

    val rows = df.collect()
    assertEquals(NumLeft * k, rows.length)
    val byLid = rows.groupBy(_.getAs[Int]("lid"))
    leftIds.zip(leftVectors).foreach { case (lid, lvec) =>
      val oracle = rightVectors.indices
        .map(i => (rightIds(i), l2(lvec, rightVectors(i))))
        .sortBy(_._2)
        .take(k)
        .map(_._1)
        .toSet
      val actual = byLid(lid).map(_.getAs[Int]("rid")).toSet
      assertEquals(oracle, actual, s"top-K mismatch for lid=$lid (rule off, brute-force fallback)")
    }
  }

  // -- helpers ------------------------------------------------------------------------------

  private def leftSchema(): StructType = new StructType(Array(
    StructField("lid", IntegerType, nullable = false),
    StructField(
      "lvec",
      ArrayType(FloatType, containsNull = false),
      nullable = false,
      new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))

  private def rightSchema(): StructType = new StructType(Array(
    StructField("rid", IntegerType, nullable = false),
    StructField(
      "rvec",
      ArrayType(FloatType, containsNull = false),
      nullable = false,
      new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))

  private def generateLeft(
      n: Int,
      dim: Int,
      seed: Long): (Seq[org.apache.spark.sql.Row], Array[Array[Float]], Array[Int]) = {
    val rng = new Random(seed)
    val vectors = (0 until n).map(_ => randomVector(rng, dim)).toArray
    val ids = (0 until n).toArray
    val rows = ids.zip(vectors).map { case (id, v) => RowFactory.create(Integer.valueOf(id), v) }
    (rows.toSeq, vectors, ids)
  }

  private def generateRight(
      n: Int,
      dim: Int,
      seed: Long): (Seq[org.apache.spark.sql.Row], Array[Array[Float]], Array[Int]) = {
    val rng = new Random(seed)
    val vectors = (0 until n).map(_ => randomVector(rng, dim)).toArray
    val ids = (0 until n).map(_ + 1000).toArray
    val rows = ids.zip(vectors).map { case (id, v) => RowFactory.create(Integer.valueOf(id), v) }
    (rows.toSeq, vectors, ids)
  }

  private def writeRightLance(rows: Seq[org.apache.spark.sql.Row]): String = {
    val df = spark.createDataFrame(rows.asJava, rightSchema())
    val out = tempDir.resolve(s"right_${System.nanoTime()}").toString
    df.write.format("lance").save(out)
    out
  }

  private def rightSchemaWithCategories(): StructType = new StructType(Array(
    StructField("rid", IntegerType, nullable = false),
    StructField("category", StringType, nullable = false),
    StructField(
      "rvec",
      ArrayType(FloatType, containsNull = false),
      nullable = false,
      new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))

  /**
   * Like `generateRight`, but every row also carries a category drawn from a small alphabet so
   * the e2e WHERE-pushdown test has a non-trivial filter to apply.
   */
  private def generateRightWithCategories(n: Int, dim: Int, seed: Long): (
      Seq[org.apache.spark.sql.Row],
      Array[Array[Float]],
      Array[Int],
      Array[String]) = {
    val rng = new Random(seed)
    val vectors = (0 until n).map(_ => randomVector(rng, dim)).toArray
    val ids = (0 until n).map(_ + 2000).toArray
    val alphabet = Array("A", "B", "C", "D")
    val categories = (0 until n).map(i => alphabet(i % alphabet.length)).toArray
    val rows = ids.zip(vectors).zip(categories).map { case ((id, v), cat) =>
      RowFactory.create(Integer.valueOf(id), cat, v)
    }
    (rows.toSeq, vectors, ids, categories)
  }

  private def writeRightLanceWithCategories(rows: Seq[org.apache.spark.sql.Row]): String = {
    val df = spark.createDataFrame(rows.asJava, rightSchemaWithCategories())
    val out = tempDir.resolve(s"right_cat_${System.nanoTime()}").toString
    df.write.format("lance").save(out)
    out
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
    while (i < a.length) { val d = a(i) - b(i); s += d * d; i += 1 }
    s
  }
}
