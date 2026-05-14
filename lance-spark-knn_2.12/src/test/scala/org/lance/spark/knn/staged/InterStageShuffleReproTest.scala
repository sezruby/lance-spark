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
package org.lance.spark.knn.staged

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, RowFactory, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}
import org.junit.jupiter.api.Assertions._

import java.util.Random

import scala.collection.JavaConverters._

/**
 * Repro harness for the JVM SIGSEGV observed by the reverted 3-exec staged split
 * (commits 882fcdb / 4b68ee3 / 6218b1c, reverted in 2e2ba94). Goal: narrow the fault to
 * either (a) Spark/JVM, or (b) the staged-exec + Lance interaction.
 *
 * Each test isolates one variable of the staged-exec pipeline:
 *
 *   test1  — baseline: no shuffle, schema with ArrayType(FloatType, 128) + array<struct>
 *   test2  — encoder round-trip at the same benchmark scale, no shuffle
 *   test3  — shuffle (repartition by leftId) + encoder round-trip, same schema
 *   test4  — same as test3 but payload comes from Row → InternalRow via ExpressionEncoder,
 *            mirroring the staged-exec codec's hot path
 *   test5  — test4 + a downstream mapPartitions that decodes via direct InternalRow
 *            accessors, mirroring ProbedLeftCodec.Decoder
 *
 * The benchmark reported the SIGSEGV at "stage 79 task 0" at 100K rows × 128-dim. We run at
 * 100K here — the reverted codec consistently crashed at that scale on M5 Max + Temurin 17.
 * If any of these tests SIGSEGV, it's a Spark/JVM bug. If they all pass, the fault is
 * specific to how the staged execs wire themselves into Lance's output.
 */
class InterStageShuffleReproTest {

  private var spark: SparkSession = _

  // Benchmark-scale knobs. The original crash fired at leftN = 100K with a 128-dim qvec on
  // the left side. K = 10 refs per row matches the join's top-K. Shuffle over 4 partitions
  // to force cross-partition movement of every row.
  private val LeftN: Int = 100000
  private val Dim: Int = 128
  private val K: Int = 10
  private val ShuffleParts: Int = 4
  private val Seed: Long = 1337L

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("inter-stage-shuffle-repro")
      .master("local[4]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.driver.memory", "4g")
      .config("spark.sql.shuffle.partitions", ShuffleParts.toString)
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = if (spark != null) spark.stop()

  // ---- schemas ----------------------------------------------------------------------------

  /**
   * Matches `ProbedLeftCodec.interStageSchema` shape:
   *   - leading `_leftId: long`
   *   - user left-schema fields flattened — here: one `ArrayType(FloatType)` vec column
   *   - trailing `_refs: array<struct<rowAddr: long, score: float>>`
   */
  private val RefStruct: StructType = StructType(Array(
    StructField("rowAddr", LongType, nullable = false),
    StructField("score", FloatType, nullable = false)))

  private val InterStageSchema: StructType = StructType(Array(
    StructField("_leftId", LongType, nullable = false),
    StructField(
      "qvec",
      ArrayType(FloatType, containsNull = false),
      nullable = false,
      new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build()),
    StructField(
      "_refs",
      ArrayType(RefStruct, containsNull = false),
      nullable = false)))

  // ---- data generation --------------------------------------------------------------------

  private def randomFloats(rng: Random, n: Int): Array[Float] = {
    val v = new Array[Float](n)
    var i = 0
    while (i < n) { v(i) = rng.nextFloat(); i += 1 }
    v
  }

  /** Build `LeftN` rows driver-side. Values are Java-shaped to match `RowFactory.create`. */
  private def buildInterStageRows(): java.util.List[Row] = {
    val rng = new Random(Seed)
    val rows = new java.util.ArrayList[Row](LeftN)
    var i = 0
    while (i < LeftN) {
      val qvec = randomFloats(rng, Dim).toSeq.asJava
      val refs = new java.util.ArrayList[Row](K)
      var r = 0
      while (r < K) {
        refs.add(RowFactory.create(
          java.lang.Long.valueOf(rng.nextLong() & 0x7FFFFFFFFFFFFFFFL),
          java.lang.Float.valueOf(rng.nextFloat())))
        r += 1
      }
      rows.add(RowFactory.create(
        java.lang.Long.valueOf(i.toLong),
        qvec,
        refs))
      i += 1
    }
    rows
  }

  // ---- tests ------------------------------------------------------------------------------

  /** Baseline: no shuffle. Just build a DF and `count()`. If this crashes, schema alone is broken. */
  @Test def test1_buildAndCountNoShuffle(): Unit = {
    val df = spark.createDataFrame(buildInterStageRows(), InterStageSchema)
    val n = df.count()
    assertEquals(LeftN.toLong, n)
  }

  /**
   * Force an `ExpressionEncoder(InterStageSchema)` round-trip. Row → InternalRow → Row via
   * `as[Row]` on a Dataset forces encoder codegen. No shuffle. If this crashes, encoder
   * codegen + this specific schema (array<float>, array<struct>) is what trips the JIT.
   */
  @Test def test2_encoderRoundTripNoShuffle(): Unit = {
    val df = spark.createDataFrame(buildInterStageRows(), InterStageSchema)
    // Force encoder deserialization by collecting via rdd → row. This is the simplest
    // mirror of what ProbedLeftCodec.Decoder does on each task.
    val collected = df.rdd.count()
    assertEquals(LeftN.toLong, collected)
  }

  /**
   * Shuffle by `_leftId`, then materialize. Mimics `EnsureRequirements` inserting a
   * `ShuffleExchangeExec` when `LanceMergeExec` declares `ClusteredDistribution(_leftId)`.
   * Encoder round-trips on both sides of the shuffle. If this crashes, Spark's UnsafeRow
   * shuffle of `(long, array<float>, array<struct>)` is the bug.
   */
  @Test def test3_shufflePlusCount(): Unit = {
    val df = spark.createDataFrame(buildInterStageRows(), InterStageSchema)
    val shuffled = df.repartition(ShuffleParts, col("_leftId"))
    val n = shuffled.count()
    assertEquals(LeftN.toLong, n)
  }

  /**
   * Same shuffle shape as test3 but the payload starts life as `RDD[InternalRow]` produced
   * by an `ExpressionEncoder(InterStageSchema).createSerializer()` in a `mapPartitions` —
   * exactly what `LanceProbeExec.doExecute` does. This is the closest non-Lance mirror of
   * the reverted staged probe stage's output.
   *
   * We then read back through `df.rdd` which triggers the decoder. If this crashes, the
   * encoder-driven inter-stage row path is the bug independent of Lance.
   */
  @Test def test4_encodeToInternalRowThenShuffle(): Unit = {
    val schemaCaptured = InterStageSchema
    val leftN = LeftN
    val dim = Dim
    val k = K
    val seed = Seed

    // Build an RDD[Row] driver-side (small synthetic gen, no Lance). Each partition gets
    // ~leftN/parallelism rows. Parallelism is local[4] ⇒ 4 partitions upstream of the shuffle.
    val rows: Seq[Row] = {
      val rng = new Random(seed)
      (0 until leftN).map { i =>
        val qvec = randomFloats(rng, dim).toSeq
        val refs = (0 until k).map { _ =>
          Row(rng.nextLong() & 0x7FFFFFFFFFFFFFFFL, rng.nextFloat())
        }
        Row(i.toLong, qvec, refs)
      }
    }
    val rowRdd: RDD[Row] = spark.sparkContext.parallelize(rows, 4)

    // Encode Row → InternalRow in a mapPartitions, mirroring LanceProbeExec.doExecute.
    val internalRdd: RDD[InternalRow] = rowRdd.mapPartitions { iter =>
      val enc = ExpressionEncoder(schemaCaptured).resolveAndBind()
      val ser = enc.createSerializer()
      iter.map(r => ser(r).copy())
    }

    // Wrap back into a DataFrame via internalCreateDataFrame-style path. This is the
    // public equivalent: go RDD[InternalRow] → RDD[Row] → createDataFrame.
    val backToRow: RDD[Row] = internalRdd.mapPartitions { iter =>
      val enc = ExpressionEncoder(schemaCaptured).resolveAndBind()
      val deser = enc.createDeserializer()
      iter.map(ir => deser(ir.copy()))
    }

    val df = spark.createDataFrame(backToRow, schemaCaptured)
    val shuffled = df.repartition(ShuffleParts, col("_leftId"))
    val n = shuffled.count()
    assertEquals(leftN.toLong, n)
  }

  /**
   * Adds a post-shuffle consumer that decodes each `InternalRow` via direct accessors —
   * `ir.getLong(0)`, `ir.getArray(1)`, `ir.getArray(2).getStruct(i, 2)`. This is what
   * `ProbedLeftCodec.Decoder` does, and what the revert commit said was SIGSEGV-ing in
   * `UnsafeRow.getArray` under C2.
   *
   * We iterate through the shuffle's output InternalRows directly (via `queryExecution.toRdd`,
   * which is the Catalyst-internal `RDD[InternalRow]` — same shape LanceMergeExec sees from
   * its upstream ShuffleExchangeExec). Then sum the leftIds + refs length to force JIT to
   * compile the hot loop over the shuffled UnsafeRows.
   *
   * Runs the inner loop multiple times to give C2 a chance to compile and mis-speculate.
   */
  @Test def test5_directInternalRowAccessorsPostShuffle(): Unit = {
    val schemaCaptured = InterStageSchema
    val leftN = LeftN

    val df = spark.createDataFrame(buildInterStageRows(), schemaCaptured)
    val shuffled = df.repartition(ShuffleParts, col("_leftId"))

    // Catalyst-internal RDD[InternalRow] — what a physical child's execute() returns.
    val shuffledInternal: RDD[InternalRow] = shuffled.queryExecution.toRdd

    // Direct-accessor consumer, matching ProbedLeftCodec.Decoder's hot loop. Run it a few
    // times so JIT C2 has a chance to compile + speculate on UnsafeRow.getArray.
    var totalRows = 0L
    var trial = 0
    while (trial < 3) {
      val count = shuffledInternal.mapPartitions { iter =>
        var sum = 0L
        while (iter.hasNext) {
          val ir = iter.next().copy()
          val leftId = ir.getLong(0)
          // qvec: ArrayType(FloatType). getArray returns UnsafeArrayData.
          val qvec = ir.getArray(1)
          var qsum = 0.0f
          var j = 0
          while (j < qvec.numElements()) {
            qsum += qvec.getFloat(j)
            j += 1
          }
          // refs: ArrayType(StructType(...)). Iterate via getArray + getStruct.
          val refs = ir.getArray(2)
          var refSum = 0L
          var r = 0
          while (r < refs.numElements()) {
            val s = refs.getStruct(r, 2)
            refSum += s.getLong(0)
            r += 1
          }
          sum += leftId + qsum.toLong + refSum
        }
        Iterator.single(sum)
      }.count()
      totalRows += count
      trial += 1
    }

    // Each of 3 trials visits ShuffleParts partitions ⇒ 3 * ShuffleParts rows of count output.
    assertEquals((3 * ShuffleParts).toLong, totalRows)
  }

  /**
   * JIT-warmup stress. The reverted PR reported the SIGSEGV at Spark's "stage 79 task 0.0"
   * right after a crossJoin baseline finished — i.e. after the JVM had been running hot for
   * minutes and C2 had compiled essentially everything in sight. Tests 1-5 each only execute
   * the hot loop a handful of times before asserting; that's not enough for C2 to compile +
   * mis-speculate. This test runs:
   *
   *   1. An upstream crossJoin-esque warmup (generates JIT pressure on encoder / shuffle /
   *      UnsafeRow paths, similar to what the benchmark's config A does first).
   *   2. 50 iterations of encode → shuffle → direct-accessor-decode at benchmark scale.
   *
   * If the reverted codec's fault is reproducible on this machine via Spark's codegen path
   * alone, this is where it will surface.
   */
  @Test def test6_jitWarmupStress(): Unit = {
    val schemaCaptured = InterStageSchema
    val leftN = LeftN

    // --- Warmup: produce JIT pressure on the shuffle/UnsafeRow paths. -------------------
    // A small crossJoin-ish workload whose shape touches UnsafeRow getters repeatedly.
    val warmupA = spark.range(0, 500L).toDF("a")
    val warmupB = spark.range(0, 500L).toDF("b")
    // 500 × 500 = 250K rows; filter + group, crossJoin-shaped, touches codegen paths.
    warmupA.crossJoin(warmupB)
      .groupBy(col("a"))
      .count()
      .count()

    // --- Main loop: 50 iterations of the staged-codec hot path. -------------------------
    val df = spark.createDataFrame(buildInterStageRows(), schemaCaptured)
    val shuffled = df.repartition(ShuffleParts, col("_leftId"))
    val shuffledInternal: RDD[InternalRow] = shuffled.queryExecution.toRdd

    var iter = 0
    val iterations = 50
    var observedSum = 0L
    while (iter < iterations) {
      val perIterSum = shuffledInternal.mapPartitions { it =>
        var sum = 0L
        while (it.hasNext) {
          val ir = it.next().copy()
          val leftId = ir.getLong(0)
          val qvec = ir.getArray(1)
          var j = 0
          var qsum = 0.0f
          while (j < qvec.numElements()) {
            qsum += qvec.getFloat(j)
            j += 1
          }
          val refs = ir.getArray(2)
          var r = 0
          var refSum = 0L
          while (r < refs.numElements()) {
            val s = refs.getStruct(r, 2)
            refSum += s.getLong(0)
            r += 1
          }
          sum += leftId + qsum.toLong + refSum
        }
        Iterator.single(sum)
      }.collect().sum
      observedSum += perIterSum
      iter += 1
    }

    // Weak check: every iteration sees the same `leftN` rows, so observedSum should be
    // nonzero and equal across iterations. A meaningful crash would prevent reaching here.
    assertTrue(observedSum != 0L, s"Expected non-zero observed sum after $iterations iters")
  }
}
