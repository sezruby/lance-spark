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
import org.lance.spark.knn.internal.LanceVectorIndexBuilder
import org.lance.spark.knn.testutil.ClusteredEmbeddings

import java.nio.file.Path
import java.util.Random

import scala.collection.JavaConverters._

/**
 * Phase 3 — real-recall validation against an IVF-PQ-indexed Lance dataset.
 *
 * Builds an IVF-PQ vector index via Lance's `Dataset.createIndex` Java binding, then runs
 * `IndexedNearestJoin` and measures recall@K vs. the brute-force ground truth. With an index
 * Lance returns *approximate* top-K, so recall is < 1.0 — the point of this test is to verify:
 *
 *   1. The indexed path actually engages (Lance's `useIndex` defaults to true on a Query
 *      against an indexed column; our `LanceProbe.probe` doesn't override it).
 *   2. Recall at the default settings is in a sane range — our small synthetic dataset is
 *      small enough that recall should be high (most rows survive the IVF cluster cut).
 *   3. `refineFactor > 1` improves recall by re-ranking more candidates with exact distance.
 *
 * Until this test, the 608x / 17.4x benchmark headlines were on a NO-INDEX Lance dataset where
 * Lance's brute-force per-fragment scan made everything exact (recall = 1.0). The
 * approximate-vs-exact recall trade-off that an indexed connector exposes was unmeasured. This
 * test closes that gap.
 *
 * Setup specifics: 1024 right rows, dim 32, 4 IVF partitions, 8 PQ sub-vectors. The dataset
 * is intentionally tiny so the test runs in a few seconds — production-realistic dataset
 * sizes would need much larger N.
 */
class IndexedNearestJoinIvfPqRecallTest {

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  private val Dim = 32
  private val NumRight = 1024
  private val NumLeft = 32
  private val K = 10
  private val Seed = 0xCAFEL

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("indexed-nearest-ivfpq-recall")
      .master("local[2]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = if (spark != null) spark.stop()

  /**
   * The headline test: build IVF-PQ, run IndexedNearestJoin, measure recall@10 against the
   * brute-force oracle. With 1024 rows × 4 IVF partitions, each partition holds ~256 rows;
   * a default-`nprobes` query hits ~1 partition, so we expect recall to be lower than 1.0
   * but still substantial.
   */
  @Test def testIvfPqRecallReasonableAtDefaults(): Unit = {
    val (leftDf, leftIds, leftVecs) = buildLeft()
    val (rightUri, rightIds, rightVecs) = writeRight()
    LanceVectorIndexBuilder.buildIvfPq(
      datasetUri = rightUri,
      vectorColumn = "rvec",
      numPartitions = 4,
      numSubVectors = 8,
      numBits = 8)
    assertEquals(
      1,
      LanceVectorIndexBuilder.listIndexCount(rightUri),
      "expected exactly one index after build")

    val joined = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = K,
      metric = "l2",
      rightProjection = Some(Seq("rid")))

    val rows = joined.collect()
    val recall = computeRecallAtK(rows, leftIds, leftVecs, rightIds, rightVecs, K)
    println(s"  IVF-PQ recall@$K (no refine, default nprobes): $recall")
    // With 1024 rows and 4 IVF partitions, default nprobes = 1, the index returns ~256
    // candidates per query. Recall should be substantially > 0; our acceptance threshold
    // is loose because IVF-PQ recall depends on the random data layout — anything < 0.3
    // would suggest a real bug, not an inherent IVF limitation.
    assertTrue(recall > 0.3, s"recall@$K=$recall too low; index path probably not engaging")
  }

  /**
   * Production-realistic distribution: clustered Gaussian mixture, unit-sphere-normalized —
   * the geometry of typical sentence-transformer / image-feature embeddings. The benchmark and
   * the recall test elsewhere use uniform-random vectors over [0, 1]^d, which is the WORST
   * case for IVF (k-means has no natural cluster structure to latch onto). This test exercises
   * the indexed path on a more realistic distribution and asserts:
   *
   *   1. Recall@K on clustered data >= 0.5 at default IVF-PQ settings. If realistic data
   *      collapsed to coin-flip recall, the indexed path wouldn't be useful in production.
   *   2. Both uniform and clustered recall numbers are printed, so a reviewer can see whether
   *      the realistic case actually helps in practice (it should — see the file's preamble).
   *
   * Why we don't `assert(clustered >= uniform)`: Lance's IVF training (k-means initialization)
   * is non-deterministic across JVM sessions, so on a tiny 1024-row dataset the run-to-run
   * noise in either recall number routinely exceeds the structural advantage of realistic
   * data. A reliable comparison would need either (a) averaging over many seeds, which is
   * slow and fragile in CI, or (b) much larger N where the structural effect dominates noise.
   * We chose (c): print both, assert only the realistic-floor invariant.
   */
  @Test def testClusteredEmbeddingsRecallSurvives(): Unit = {
    val (uniformDf, uniformIds, uniformVecs) =
      buildLeftFromVectors(generateUniform(NumLeft, Dim, Seed))
    val (uniformUri, uniformRightIds, uniformRightVecs) =
      writeRightFromVectors(generateUniform(NumRight, Dim, Seed + 1))
    LanceVectorIndexBuilder.buildIvfPq(uniformUri, "rvec", numPartitions = 4, numSubVectors = 8)

    val (clusteredDf, clusteredIds, clusteredVecs) = buildLeftFromVectors(
      ClusteredEmbeddings.generate(NumLeft, Dim, numClusters = 4, seed = Seed + 2))
    val (clusteredUri, clusteredRightIds, clusteredRightVecs) = writeRightFromVectors(
      ClusteredEmbeddings.generate(NumRight, Dim, numClusters = 16, seed = Seed + 3))
    LanceVectorIndexBuilder.buildIvfPq(
      clusteredUri,
      "rvec",
      numPartitions = 4,
      numSubVectors = 8)

    val uniformRecall = recallAgainst(
      uniformDf,
      uniformUri,
      uniformIds,
      uniformVecs,
      uniformRightIds,
      uniformRightVecs)
    val clusteredRecall = recallAgainst(
      clusteredDf,
      clusteredUri,
      clusteredIds,
      clusteredVecs,
      clusteredRightIds,
      clusteredRightVecs)
    println(
      s"  IVF-PQ recall@$K: uniform=$uniformRecall, clustered=$clusteredRecall " +
        "(uniform = IVF worst case; clustered = production-shaped)")

    assertTrue(
      clusteredRecall >= 0.5,
      s"clustered-data recall@$K=$clusteredRecall is unexpectedly low; " +
        "defaults should comfortably exceed 0.5 on production-shaped embeddings — " +
        "if this fails, suspect a regression in Lance's index path or in our probe wiring")
  }

  /**
   * `refineFactor > 1` engages Lance's exact-distance re-rank: fetch `K * refineFactor`
   * approximate candidates, re-rank, trim back to K. Strictly improves (or matches) recall
   * vs. no refine. We assert the >= relation rather than a strict > so the test isn't flaky
   * on tiny datasets where both paths happen to find the same K rows.
   */
  @Test def testRefineFactorImprovesRecall(): Unit = {
    val (leftDf, leftIds, leftVecs) = buildLeft()
    val (rightUri, rightIds, rightVecs) = writeRight()
    LanceVectorIndexBuilder.buildIvfPq(rightUri, "rvec", numPartitions = 4)

    val baseline = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = K,
      metric = "l2",
      rightProjection = Some(Seq("rid")))
    val refined = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = K,
      metric = "l2",
      rightProjection = Some(Seq("rid")),
      refineFactor = Some(8))

    val recallBaseline =
      computeRecallAtK(baseline.collect(), leftIds, leftVecs, rightIds, rightVecs, K)
    val recallRefined =
      computeRecallAtK(refined.collect(), leftIds, leftVecs, rightIds, rightVecs, K)
    println(s"  IVF-PQ recall@$K: no refine = $recallBaseline, refineFactor=8 = $recallRefined")
    assertTrue(
      recallRefined >= recallBaseline,
      s"refineFactor should not hurt recall: baseline=$recallBaseline, refined=$recallRefined")
  }

  // -- helpers ------------------------------------------------------------------------------

  private def buildLeft(): (org.apache.spark.sql.DataFrame, Array[Int], Array[Array[Float]]) =
    buildLeftFromVectors(generateUniform(NumLeft, Dim, Seed))

  private def writeRight(): (String, Array[Int], Array[Array[Float]]) =
    writeRightFromVectors(generateUniform(NumRight, Dim, Seed + 1))

  /** Build a left-side DataFrame from a pre-generated vector array. */
  private def buildLeftFromVectors(
      vectors: Array[Array[Float]])
      : (org.apache.spark.sql.DataFrame, Array[Int], Array[Array[Float]]) = {
    val schema = new StructType(Array(
      StructField("lid", IntegerType, nullable = false),
      StructField(
        "lvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val ids = (0 until vectors.length).toArray
    val rows = ids.zip(vectors).map { case (id, v) =>
      RowFactory.create(Integer.valueOf(id), v)
    }
    val df = spark.createDataFrame(rows.toSeq.asJava, schema)
    (df, ids, vectors)
  }

  /** Write a right-side Lance dataset from a pre-generated vector array. */
  private def writeRightFromVectors(
      vectors: Array[Array[Float]]): (String, Array[Int], Array[Array[Float]]) = {
    val schema = new StructType(Array(
      StructField("rid", IntegerType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val ids = (0 until vectors.length).map(_ + 100000).toArray
    val rows = ids.zip(vectors).map { case (id, v) =>
      RowFactory.create(Integer.valueOf(id), v)
    }
    val df = spark.createDataFrame(rows.toSeq.asJava, schema)
    val out = tempDir.resolve(s"right_${System.nanoTime()}").toString
    df.write.format("lance").save(out)
    (out, ids, vectors)
  }

  /** Run an indexed nearest join against the given right dataset and compute recall@K. */
  private def recallAgainst(
      leftDf: org.apache.spark.sql.DataFrame,
      rightUri: String,
      leftIds: Array[Int],
      leftVecs: Array[Array[Float]],
      rightIds: Array[Int],
      rightVecs: Array[Array[Float]]): Double = {
    val joined = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = K,
      metric = "l2",
      rightProjection = Some(Seq("rid")))
    computeRecallAtK(joined.collect(), leftIds, leftVecs, rightIds, rightVecs, K)
  }

  /** Uniform-random vectors over the unit hypercube — the IVF-worst-case data distribution. */
  private def generateUniform(n: Int, dim: Int, seed: Long): Array[Array[Float]] = {
    val rng = new Random(seed)
    Array.fill(n)(randomVector(rng, dim))
  }

  /**
   * Mean recall@K across all left rows: |intersection of indexed top-K with brute-force
   * top-K| divided by K. A value of 1.0 means the indexed path returned the same K rows as
   * brute force; lower values mean the IVF cluster cut excluded some true neighbors.
   */
  private def computeRecallAtK(
      joinedRows: Array[org.apache.spark.sql.Row],
      leftIds: Array[Int],
      leftVecs: Array[Array[Float]],
      rightIds: Array[Int],
      rightVecs: Array[Array[Float]],
      k: Int): Double = {
    val byLid = joinedRows.groupBy(_.getAs[Int]("lid"))
    val perLidRecall = leftIds.zip(leftVecs).map { case (lid, lvec) =>
      val oracle = rightVecs.indices
        .map(i => (rightIds(i), l2(lvec, rightVecs(i))))
        .sortBy(_._2)
        .take(k)
        .map(_._1)
        .toSet
      val actual = byLid.getOrElse(lid, Array.empty).map(_.getAs[Int]("rid")).toSet
      val hit = (oracle intersect actual).size.toDouble
      hit / k
    }
    perLidRecall.sum / perLidRecall.length
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
