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
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.io.TempDir

import java.nio.file.Path
import java.util.Random

import scala.collection.JavaConverters._

/**
 * Runs the join at the exact scale that SIGSEGV'd the reverted 3-exec staged code:
 * 10K right × 100 left × dim=128, K=10, with a crossJoin JIT-warmup preceding the
 * join iterations to force C2 to compile all the hot UnsafeRow accessors.
 *
 * The originally-reported crash signature (`UnsafeRow.getLong(I)J` SIGSEGV in C2-compiled
 * code, hs_err from early-development reproducer) fires on this exact shape. A clean pass here is the strongest
 * available evidence that `InterStageShuffle.mergeViaCatalystShuffle` doesn't inherit the
 * fragility.
 *
 * Right side kept at 10K (not the reverted benchmark's 100K) because we also run
 * correctness + count tests in the same module and don't need the extra scan cost —
 * the crash was codec-fragility, not a scale-dependent race. The revert commit's repro
 * fired at 100K too.
 */
class IndexedNearestJoinJitStressTest {

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  private val NumRight = 10000
  private val NumLeft = 100
  private val Dim = 128
  private val K = 10
  private val Seed = 1337L

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("indexed-nearest-join-jit-stress")
      .master("local[4]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.driver.memory", "4g")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = if (spark != null) spark.stop()

  /**
   * JIT warmup via crossJoin + group-by (mirrors the benchmark's baseline config A
   * which ran first and produced the JIT state the staged config B/C/D/E then crashed
   * in), followed by 20 iterations of the join at benchmark scale. Each iteration
   * collects the full result set to force the whole pipeline to run end-to-end.
   */
  @Test def testRepeatedJoinAtBenchmarkScale(): Unit = {
    warmupJit()

    val rng = new Random(Seed)
    val rightUri = writeRight(rng)
    val leftDf = buildLeft(rng)

    var iter = 0
    val iterations = 20
    while (iter < iterations) {
      val joined = IndexedNearestJoin(
        left = leftDf,
        rightLanceUri = rightUri,
        leftVecCol = "qvec",
        rightVecCol = "rvec",
        k = K,
        metric = "l2",
        rightProjection = Some(Seq("rid")))
      val rows = joined.collect()
      assertEquals(NumLeft * K, rows.length, s"iteration $iter wrong row count")
      iter += 1
    }
  }

  /**
   * Count-based variant at the same scale. This is what exercises `ColumnPruning` and
   * was the proximate cause of the revert's crash (pruning inserted Project(Nil) that
   * emitted 0-field UnsafeRows). Running count() 20 times at this scale is the tightest
   * analog of the reverted repro.
   */
  @Test def testRepeatedCountAtBenchmarkScale(): Unit = {
    warmupJit()

    val rng = new Random(Seed + 1L)
    val rightUri = writeRight(rng)
    val leftDf = buildLeft(rng)

    var iter = 0
    val iterations = 20
    while (iter < iterations) {
      val joined = IndexedNearestJoin(
        left = leftDf,
        rightLanceUri = rightUri,
        leftVecCol = "qvec",
        rightVecCol = "rvec",
        k = K,
        metric = "l2",
        rightProjection = Some(Seq("rid")))
      val n = joined.count()
      assertEquals((NumLeft * K).toLong, n, s"iteration $iter wrong count")
      iter += 1
    }
  }

  // -- helpers ------------------------------------------------------------------------------

  private def warmupJit(): Unit = {
    // ~250K-row crossJoin-groupBy to build JIT state on UnsafeRow accessors, Exchange,
    // HashAggregate. Mirrors what `IndexedNearestJoinBenchmark.runSparkCrossJoinBaseline`
    // runs as config A immediately before the staged configs that crashed.
    val a = spark.range(0, 500L).toDF("a")
    val b = spark.range(0, 500L).toDF("b")
    a.crossJoin(b).groupBy(col("a")).count().count()
  }

  private def randomVector(rng: Random, dim: Int): Array[Float] = {
    val v = new Array[Float](dim)
    var i = 0
    while (i < dim) { v(i) = rng.nextFloat(); i += 1 }
    v
  }

  private def writeRight(rng: Random): String = {
    val schema = new StructType(Array(
      StructField("rid", LongType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val rows = new java.util.ArrayList[Row](NumRight)
    var i = 0
    while (i < NumRight) {
      rows.add(RowFactory.create(
        java.lang.Long.valueOf(i.toLong),
        randomVector(rng, Dim).toSeq.asJava))
      i += 1
    }
    val df = spark.createDataFrame(rows, schema)
    val uri = tempDir.resolve(s"right_${System.nanoTime()}").toString
    df.write.format("lance").save(uri)
    uri
  }

  private def buildLeft(rng: Random) = {
    val schema = new StructType(Array(
      StructField("lid", LongType, nullable = false),
      StructField(
        "qvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val rows = new java.util.ArrayList[Row](NumLeft)
    var i = 0
    while (i < NumLeft) {
      rows.add(RowFactory.create(
        java.lang.Long.valueOf(i.toLong),
        randomVector(rng, Dim).toSeq.asJava))
      i += 1
    }
    spark.createDataFrame(rows, schema)
  }
}
