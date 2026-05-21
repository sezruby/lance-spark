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
package org.lance.spark.knn.benchmark

import org.apache.spark.sql.{DataFrame, Row, RowFactory, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.lance.spark.knn.LanceKnnImplicits._

import java.nio.file.Files
import java.util.{Locale, Random}
import java.util.concurrent.TimeUnit

import scala.collection.JavaConverters._

/**
 * Validates the per-query temp-Lance design from
 * [sezruby/lance-spark#2](https://github.com/sezruby/lance-spark/issues/2): when R lives in
 * parquet (or any non-Lance source), the indexed `NearestByJoin` path can still apply by
 * writing R to a temp Lance dataset before the probe. Three configs run on the SAME data
 * in the SAME job (no cross-run noise):
 *
 *   A: vanilla Spark crossJoin + L2 UDF + min_by_k on parquet R — the brute-force baseline
 *      a user would write today; matches what Spark 4.2's `RewriteNearestByJoin` lowers to.
 *   B: per-query temp Lance write + `kNearestJoin` against the temp URI — the per-query
 *      temp design under test.
 *   C: Lance-native R + `kNearestJoin` — same probe pipeline as B but R was pre-written to
 *      Lance once outside the timing loop. The "already-Lance" reference; (B - C) is the
 *      pure temp-write overhead per query.
 *
 * Three configs answer:
 *   1. Does B beat A on parquet R? (B vs A speedup — is the per-query temp story faster
 *      than the brute-force baseline at all?)
 *   2. How much overhead does the temp write add vs. the existing Lance-already path?
 *      (B - C, on the same hardware in the same run, no cross-run noise)
 *   3. Sanity: probe-only median of B (excluding temp write) should match C closely; if
 *      not, something's off in the pipeline.
 *
 * == Local run ==
 * {{{
 *   MAVEN_OPTS="-Xmx12g <JDK 17 add-opens flags>" \
 *     ./mvnw -pl lance-spark-knn_2.12 -q exec:java -Pbenchmark \
 *     -Dexec.mainClass="org.lance.spark.knn.benchmark.IndexedNearestJoinTempRBenchmark"
 * }}}
 *
 * Local laptop runs are noisy enough that single-run point estimates are unreliable
 * (multi-tenant CPU contention on macOS, single-machine no-real-parallelism). Headline
 * numbers should come from a real distributed cluster — see `BENCHMARK_RESULTS.md`
 * § "Variance / multi-tenant noise" for the discussion in the existing benchmark.
 *
 * == Cluster run ==
 * Build the fat JAR (`./mvnw -pl lance-spark-knn_2.12 package -Pbenchmark -DskipTests`),
 * upload, submit with `BENCH_CLUSTER_MODE=true` and `BENCH_DATA_PATH=<shared-uri>`.
 *
 * Environment:
 *   BENCH_SCALES        — comma-separated subset of {tiny, small, medium, fat};
 *                         default "tiny,small".
 *                         tiny    = |R|=100K, |L|=100, dim=128
 *                         small   = |R|=1M,   |L|=1000, dim=128
 *                         medium  = |R|=100K, |L|=100, dim=1024
 *                         fat     = |R|=1M,   |L|=1000, dim=1024
 *   BENCH_REPEATS       — measured iterations per config; default 3. 1 warmup.
 *   BENCH_CLUSTER_MODE  — `true` to skip `.master()` and bind-address configs;
 *                         must be set when submitting to a real Spark cluster.
 *   BENCH_DATA_PATH     — shared scratch URI (file://, s3://, hdfs://, etc.);
 *                         required in cluster mode, optional locally.
 *
 * Reports timing breakdown for B: (temp_write_ms + probe_ms) so the temp-write cost is
 * visible relative to the probe itself.
 */
object IndexedNearestJoinTempRBenchmark {

  private val K: Int = 10
  private val Seed: Long = 1337L

  private case class Scale(name: String, numR: Int, numL: Int, dim: Int) {
    def vectorBytesR: Long = numR.toLong * dim.toLong * 4L
    override def toString: String =
      f"$name (|R|=$numR%,d, |L|=$numL%,d, dim=$dim, R-vec=${vectorBytesR / (1024.0 * 1024.0)}%.0f MB)"
  }

  private val Scales: Map[String, Scale] = Seq(
    Scale("tiny", numR = 100000, numL = 100, dim = 128),
    Scale("small", numR = 1000000, numL = 1000, dim = 128),
    Scale("medium", numR = 100000, numL = 100, dim = 1024),
    Scale("fat", numR = 1000000, numL = 1000, dim = 1024)).map(s => s.name -> s).toMap

  private case class Result(
      scale: String,
      config: String,
      tempWriteMs: Option[Long],
      totalMs: Long,
      runs: Seq[Long]) {
    def probeMs: Option[Long] = tempWriteMs.map(w => totalMs - w)
  }

  def main(args: Array[String]): Unit = {
    val scaleNames = sys.env
      .getOrElse("BENCH_SCALES", "tiny,small")
      .toLowerCase(Locale.ROOT)
      .split(",")
      .map(_.trim)
      .filter(_.nonEmpty)
    val scales = scaleNames.map { n =>
      Scales.getOrElse(
        n,
        sys.error(s"unknown scale '$n'; valid: ${Scales.keys.toSeq.sorted.mkString(", ")}"))
    }
    val repeats = sys.env.get("BENCH_REPEATS").map(_.toInt).getOrElse(3)
    val clusterMode = sys.env.get("BENCH_CLUSTER_MODE").exists(_.equalsIgnoreCase("true"))
    val dataRootOpt = sys.env.get("BENCH_DATA_PATH").orElse(sys.env.get("BENCH_DATA"))

    // Use user-supplied path in cluster mode, otherwise a local temp dir.
    // Avoid "lance" in the path token; the V2 catalog's path-identifier parser tokenises
    // around it on writes. Same workaround as LanceWriteBenchmark.
    val dataRoot =
      dataRootOpt.getOrElse(Files.createTempDirectory("knn-tempr-bench-").toString)

    val builder = SparkSession
      .builder()
      .appName("indexed-nearest-temp-r-benchmark")
      .config("spark.sql.crossJoin.enabled", "true")
    if (!clusterMode) {
      builder
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "16")
    }
    val spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    println("=" * 76)
    println("Per-query temp Lance benchmark — non-Lance R via temp write")
    println("=" * 76)
    val masterDesc = if (clusterMode) "cluster (BENCH_CLUSTER_MODE=true)" else "local[*]"
    println(f"Spark master:   $masterDesc (cores=${spark.sparkContext.defaultParallelism})")
    println(f"Repeats:        $repeats (median reported); 1 warmup")
    println(f"Data root:      $dataRoot")
    println(f"K:              $K")
    println(f"Scales:         ${scales.map(_.name).mkString(", ")}")
    println()

    val results = scala.collection.mutable.ArrayBuffer.empty[Result]
    try {
      scales.foreach { scale =>
        println("-" * 76)
        println(s"Scale: $scale")
        println("-" * 76)

        // Build left in-memory and right as parquet on disk. This is the realistic shape:
        // R isn't in Lance yet; we want to time the cost of materializing it.
        val leftDf = buildLeft(spark, scale).cache()
        leftDf.count()
        val rightParquetUri = s"$dataRoot/${scale.name}_right.parquet"
        writeRightParquet(spark, scale, rightParquetUri)
        val rightDfParquet = spark.read.parquet(rightParquetUri)

        // Pre-materialize a Lance-native R once (outside the timing loop) for config C.
        // This is the apples-to-apples reference: same data, already-Lance — what users get
        // when they store R in Lance natively. The kNearestJoin call against this URI uses
        // exactly the same probe pipeline as B; the only difference is no temp write.
        val rightLanceUri = s"$dataRoot/${scale.name}_right_native.lance"
        writeTempLance(rightDfParquet, rightLanceUri)
        val rightDfLance = spark.read.format("lance").load(rightLanceUri)

        // Sanity-check on a 16-row subset that the per-query-temp path agrees with the
        // crossJoin baseline. Bail early if not.
        verifyOracle(spark, scale, leftDf, rightParquetUri, dataRoot)

        // ---- Config A: vanilla Spark crossJoin + L2 UDF + min_by_k baseline ----
        // Use fewer repeats at higher scales since each A run is O(|L| × |R|) pair
        // evaluations and quickly becomes minutes per run.
        val baselineRepeats = if (scale.numL.toLong * scale.numR > 100000000L) 1 else repeats
        val resultA =
          timeIt(scale.name, "A: Spark crossJoin + min_by_k (parquet R)", baselineRepeats) {
            () => crossJoinMinByK(leftDf, rightDfParquet, K)
          }
        results += resultA

        // ---- Config B: per-query temp Lance write + existing kNearestJoin ----
        val resultB = timeWithBreakdown(scale.name, "B: temp Lance write + kNearestJoin", repeats) {
          () =>
            val tempUri = s"$dataRoot/${scale.name}_temp_${System.nanoTime()}"
            // Step 1: temp write (timed)
            val twStart = System.nanoTime()
            writeTempLance(rightDfParquet, tempUri)
            val twMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - twStart)
            // Step 2: KNN against temp Lance (timed via runFull at outer level)
            val tempLanceDf = spark.read.format("lance").load(tempUri)
            val joined = leftDf.kNearestJoin(
              right = tempLanceDf,
              leftVecCol = "lvec",
              rightVecCol = "rvec",
              k = K,
              metric = "l2",
              rightProjection = Some(Seq("rid")),
              probeParallelism = 1)
            (twMs, joined)
        }
        results += resultB

        // ---- Config C: Lance-native R (already-Lance reference; no temp write) ----
        val resultC = timeIt(scale.name, "C: Lance-native R + kNearestJoin", repeats) { () =>
          leftDf.kNearestJoin(
            right = rightDfLance,
            leftVecCol = "lvec",
            rightVecCol = "rvec",
            k = K,
            metric = "l2",
            rightProjection = Some(Seq("rid")),
            probeParallelism = 1)
        }
        results += resultC

        leftDf.unpersist()
        println()
      }

      println("=" * 76)
      println("Summary")
      println("=" * 76)
      printSummary(results.toSeq)
    } finally {
      spark.stop()
    }
  }

  // -- workload setup ----------------------------------------------------------------------

  private def buildLeft(spark: SparkSession, scale: Scale): DataFrame = {
    val schema = leftSchema(scale.dim)
    val rng = new Random(Seed ^ 1L)
    val rows = (0 until scale.numL).map { i =>
      RowFactory.create(Integer.valueOf(i), randomVector(rng, scale.dim))
    }
    spark.createDataFrame(rows.asJava, schema)
  }

  private def writeRightParquet(spark: SparkSession, scale: Scale, uri: String): Unit = {
    val schema = rightSchema(scale.dim)
    val parts = math.max(spark.sparkContext.defaultParallelism, 8)
    val numR = scale.numR
    val dim = scale.dim
    val rdd = spark.sparkContext
      .range(0L, numR.toLong, 1L, parts)
      .mapPartitionsWithIndex { (idx, iter) =>
        val rng = new Random(0xCAFEBABEL ^ idx.toLong)
        iter.map { i =>
          val v = new Array[Float](dim)
          var k = 0
          while (k < dim) { v(k) = rng.nextFloat(); k += 1 }
          RowFactory.create(Integer.valueOf(i.toInt), v): Row
        }
      }
    spark.createDataFrame(rdd, schema).write.parquet(uri)
  }

  private def writeTempLance(rightDf: DataFrame, tempUri: String): Unit = {
    // Per-query temp: project rid + rvec only (the columns the existing kNearestJoin path
    // needs). Carrying additional payload columns is a follow-up; this benchmark validates
    // the minimal shape.
    rightDf.select("rid", "rvec").write.format("lance").save(tempUri)
  }

  private def leftSchema(dim: Int): StructType = new StructType(
    Array(
      StructField("lid", IntegerType, nullable = false),
      StructField(
        "lvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", dim.toLong).build())))

  private def rightSchema(dim: Int): StructType = new StructType(
    Array(
      StructField("rid", IntegerType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", dim.toLong).build())))

  private def randomVector(rng: Random, dim: Int): Array[Float] = {
    val v = new Array[Float](dim)
    var i = 0
    while (i < dim) { v(i) = rng.nextFloat(); i += 1 }
    v
  }

  // -- baseline ----------------------------------------------------------------------------

  private def l2Udf =
    udf((a: Seq[Float], b: Seq[Float]) => {
      var s = 0.0f
      var i = 0
      while (i < a.length) { val d = a(i) - b(i); s += d * d; i += 1 }
      s
    })

  /**
   * Same shape as `IndexedNearestJoinBenchmark.crossProductMinByK`. The realistic
   * baseline a user would write today on parquet R (and what Spark 4.2's
   * RewriteNearestByJoin lowers to).
   */
  private def crossJoinMinByK(left: DataFrame, right: DataFrame, k: Int): DataFrame = {
    val l2 = l2Udf
    val r = right.select("rid", "rvec")
    val crossed = left.crossJoin(r).withColumn("__dist", l2(col("lvec"), col("rvec")))
    crossed.groupBy("lid")
      .agg(
        slice(
          sort_array(collect_list(struct(col("__dist"), col("rid"))), asc = true),
          1,
          k).as("__matches"))
      .select(col("lid"), inline(col("__matches")).as(Seq("__dist", "rid")))
      .select("lid", "rid", "__dist")
  }

  // -- timing harness ----------------------------------------------------------------------

  private def runFull(df: DataFrame): Unit =
    df.write.format("noop").mode("overwrite").save()

  private def timeIt(scale: String, config: String, repeats: Int)(f: () => DataFrame): Result = {
    print(s"  $config ... ")
    System.out.flush()
    runFull(f()) // warmup
    val runs = (0 until repeats).map { _ =>
      val t0 = System.nanoTime()
      runFull(f())
      TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0)
    }
    val sortedRuns = runs.sorted
    val median = sortedRuns(sortedRuns.length / 2)
    println(s"runs=${runs.mkString("[", ",", "]")} ms, median=$median ms")
    Result(scale, config, tempWriteMs = None, totalMs = median, runs = runs)
  }

  /**
   * Like timeIt but the runnable returns (temp_write_ms, df). The full timing includes
   * the temp write; we report it separately so the probe-only cost is visible.
   */
  private def timeWithBreakdown(scale: String, config: String, repeats: Int)(
      f: () => (Long, DataFrame)): Result = {
    print(s"  $config ... ")
    System.out.flush()
    val (_, warmDf) = f()
    runFull(warmDf) // warmup
    val records = (0 until repeats).map { _ =>
      val t0 = System.nanoTime()
      val (twMs, df) = f()
      runFull(df)
      val total = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0)
      (twMs, total)
    }
    val sortedTotals = records.map(_._2).sorted
    val medianTotal = sortedTotals(sortedTotals.length / 2)
    val medianTw = records.map(_._1).sorted.apply(records.length / 2)
    println(
      f"runs(total)=${records.map(_._2).mkString("[", ",", "]")} ms, median=$medianTotal%d ms " +
        f"(temp write median=$medianTw%d ms, probe median=${medianTotal - medianTw}%d ms)")
    Result(
      scale,
      config,
      tempWriteMs = Some(medianTw),
      totalMs = medianTotal,
      runs = records.map(_._2))
  }

  // -- oracle equivalence ------------------------------------------------------------------

  private def verifyOracle(
      spark: SparkSession,
      scale: Scale,
      leftDf: DataFrame,
      rightParquetUri: String,
      dataRoot: String): Unit = {
    println("  Sanity: oracle check on 16-row left subset ...")
    val left16 = leftDf.limit(16).cache()
    left16.count()
    val rightDfParquet = spark.read.parquet(rightParquetUri)
    val tempUri = s"$dataRoot/${scale.name}_oracle_temp_${System.nanoTime()}"
    writeTempLance(rightDfParquet, tempUri)
    val tempLance = spark.read.format("lance").load(tempUri)

    val viaTemp = left16.kNearestJoin(
      right = tempLance,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = K,
      metric = "l2",
      rightProjection = Some(Seq("rid")),
      probeParallelism = 1)
    val viaBaseline = crossJoinMinByK(left16, rightDfParquet, K)

    val tempByLid = viaTemp.collect().groupBy(_.getAs[Int]("lid")).map {
      case (lid, rows) => lid -> rows.map(_.getAs[Int]("rid")).toSet
    }
    val baseByLid = viaBaseline.collect().groupBy(_.getAs[Int]("lid")).map {
      case (lid, rows) => lid -> rows.map(_.getAs[Int]("rid")).toSet
    }
    val mismatches = tempByLid.toSeq.flatMap {
      case (lid, ids) =>
        baseByLid.get(lid) match {
          case Some(b) if b == ids => None
          case Some(b) => Some(s"lid=$lid: temp=$ids baseline=$b")
          case None => Some(s"lid=$lid: missing from baseline")
        }
    }
    if (mismatches.nonEmpty) {
      sys.error(s"Oracle mismatch:\n  ${mismatches.mkString("\n  ")}")
    }
    left16.unpersist()
    println(f"  ... oracle equivalence holds (${tempByLid.size} left rows × K=$K).")
  }

  // -- reporting ---------------------------------------------------------------------------

  private def printSummary(results: Seq[Result]): Unit = {
    val byScale = results.groupBy(_.scale)
    println(
      f"${"scale"}%-8s  ${"config"}%-46s  ${"med ms"}%8s  ${"tw ms"}%8s  ${"probe ms"}%9s  ${"vs A"}%6s")
    println("-" * 95)
    val scaleOrder = Seq("tiny", "small", "medium", "fat").filter(byScale.contains)
    scaleOrder.foreach { sc =>
      val rs = byScale(sc)
      val baseline = rs.find(_.config.startsWith("A:")).map(_.totalMs).getOrElse(0L)
      rs.foreach { r =>
        val twStr = r.tempWriteMs.map(_.toString).getOrElse("—")
        val probeStr = r.probeMs.map(_.toString).getOrElse("—")
        val speedup =
          if (baseline > 0 && r.totalMs > 0) f"${baseline.toDouble / r.totalMs}%.1fx" else "—"
        println(
          f"${r.scale}%-8s  ${r.config}%-46s  ${r.totalMs}%8d  $twStr%8s  $probeStr%9s  $speedup%6s")
      }
      println()
    }
  }
}
