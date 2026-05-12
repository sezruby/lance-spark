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
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.lance.spark.knn.IndexedNearestJoin
import org.lance.spark.knn.LanceKnnImplicits._

import java.nio.file.{Files, Paths}
import java.util.{Locale, Random}
import java.util.concurrent.TimeUnit

import scala.collection.JavaConverters._

/**
 * Benchmark comparing the indexed nearest-by-join paths against vanilla Spark's cross-product
 * baseline. Works in both local and cluster mode.
 *
 * == Local run ==
 *
 * {{{
 *   cd /path/to/lance-spark
 *   ./mvnw -pl lance-spark-knn_2.12 install -DskipTests -Pbenchmark   # build fat JAR
 *   MAVEN_OPTS="-Xmx12g" ./mvnw -pl lance-spark-knn_2.12 \
 *     exec:java -Pbenchmark \
 *     -Dexec.mainClass="org.lance.spark.knn.benchmark.IndexedNearestJoinBenchmark"
 * }}}
 *
 * == Cluster run (YARN / K8s / managed-Spark distributions) ==
 *
 * Build the benchmark fat JAR first:
 * {{{
 *   ./mvnw -pl lance-spark-knn_2.12 package -Pbenchmark -DskipTests
 *   # → target/lance-spark-knn_2.12-<version>-benchmark.jar
 * }}}
 *
 * Then upload and submit via your cluster's job API. Set environment variables in the job:
 *   - `BENCH_CLUSTER_MODE=true`  — skips setting `.master()` and bind-address configs
 *   - `BENCH_DATA_PATH=<uri>`    — shared path for synthetic datasets (s3://, hdfs://, etc.)
 *   - `BENCHMARK_SCALE=small`    — `small`, `medium`, or `both` (default)
 *
 * The baseline Spark crossJoin is only run at small scale (O(|L|×|R|) is impractical at medium).
 *
 * == Environment variables ==
 *
 * | Variable           | Default           | Description                                      |
 * |--------------------|-------------------|--------------------------------------------------|
 * | `BENCH_CLUSTER_MODE` | `false`         | Set to `true` to skip `.master()` + bind addrs  |
 * | `BENCH_DATA_PATH`  | tmp dir (local)   | URI for synthetic Lance datasets                 |
 * | `BENCHMARK_SCALE`  | `both`            | `small`, `medium`, or `both`                     |
 *
 * == What this measures ==
 *
 * Five configurations, run at two scales (small: 100K×100, medium: 1M×1000):
 *
 *   A) Vanilla Spark cross-product   — `crossJoin` + custom L2 UDF + `row_number` window.
 *      The baseline a user would write today without our extension. This is what
 *      Spark's `RewriteNearestByJoin` rule lowers to.
 *   B) Phase 0/1 single-task probe   — `df.kNearestJoin(probeParallelism = 1)`. One task
 *      probes the whole right dataset per partition; Lance does the cross-fragment merge
 *      internally.
 *   C) Phase 1.5 with 4 groups       — `probeParallelism = 4`. Four parallel probe tasks,
 *      each handling a quarter of the right dataset's fragments. The merge stage actually
 *      aggregates contributions for the first time.
 *   D) Phase 1.5 with 8 groups       — `probeParallelism = 8`.
 *   E) Phase 1.5 with 8 + skew bal   — `probeParallelism = 8`, `balanceFragments = true`.
 *      LPT bin-packing on per-fragment row counts. With evenly-sized synthetic fragments this
 *      should match (D); the win lands on real-world skewed data.
 *
 * The B–E configs use the `df.kNearestJoin(rightDf, ...)` extension method, which is the
 * idiomatic DataFrame API form (the URI-based `IndexedNearestJoin.apply` still works and
 * does the same thing). The pipeline lowers to the 3-exec Catalyst-visible staged plan
 * (`LanceProbeExec → ShuffleExchangeExec → LanceMergeExec → LanceMaterializeExec` under
 * `AdaptiveSparkPlanExec`); `df.explain()` shows all four nodes and AQE coalesces the
 * merge shuffle (`AQEShuffleRead coalesced`). See `IMPL_PLAN.md` "3-exec staged split
 * — root cause and fix" for the ColumnPruning + `references = child.outputSet` detail
 * that makes this shape safe against `count()`-style consumers.
 *
 * == What this does NOT measure ==
 *
 * IVF-PQ approximate vs. exact recall trade-off — requires building a vector index via Lance
 * Java DDL, which lance-spark-knn's test setup doesn't yet do. Without an index Lance
 * brute-force-scans each fragment, so all our paths return exact (recall = 1.0) results. The
 * "X-x faster than vanilla Spark" headline is real; the additional 10-100x speedup from index
 * lookups is a Phase 3.x demo.
 */
object IndexedNearestJoinBenchmark {

  private val Dim: Int = 128
  private val K: Int = 10
  private val Seed: Long = 1337L

  /**
   * Each scale: (numRight, numLeft, numFragments, runBaseline). The vanilla-Spark crossJoin
   * baseline is `O(|L|×|R|)` and gets impractical fast — at medium scale (1M × 1000 = 1B
   * pairs) it's measured in tens of minutes per run, which would dominate the benchmark with
   * a number we already established at small scale. So we only run baseline at small.
   */
  private case class Scale(
      name: String,
      numRight: Int,
      numLeft: Int,
      numFragments: Int,
      runBaseline: Boolean) {
    override def toString: String = s"$name (|R|=$numRight, |L|=$numLeft, frags=$numFragments)"
  }
  private val Small =
    Scale("small", numRight = 100000, numLeft = 100, numFragments = 4, runBaseline = true)
  private val Medium =
    Scale("medium", numRight = 1000000, numLeft = 1000, numFragments = 8, runBaseline = false)

  /** A single timing result. */
  private case class Result(scale: String, config: String, medianMs: Long, runs: Seq[Long]) {
    def speedupVs(baseline: Long): Double =
      if (medianMs <= 0) Double.NaN else baseline.toDouble / medianMs
  }

  def main(args: Array[String]): Unit = {
    val scales = sys.env.getOrElse("BENCHMARK_SCALE", "both").toLowerCase(Locale.ROOT) match {
      case "small" => Seq(Small)
      case "medium" => Seq(Medium)
      case _ => Seq(Small, Medium)
    }
    val clusterMode = sys.env.get("BENCH_CLUSTER_MODE").exists(_.equalsIgnoreCase("true"))
    val dataDirOpt = sys.env.get("BENCH_DATA_PATH")

    println(banner("Indexed Nearest-By-Join Benchmark"))
    val masterDesc = if (clusterMode) "cluster (BENCH_CLUSTER_MODE=true)" else "local[*]"
    println(s"Spark master: $masterDesc   Dim: $Dim   K: $K   Seed: $Seed")
    println(s"Scales:       ${scales.map(_.name).mkString(", ")}")
    dataDirOpt.foreach(p => println(s"Data root:    $p"))
    println()

    val builder = SparkSession
      .builder()
      .appName("indexed-nearest-by-join-benchmark")
      .config("spark.sql.crossJoin.enabled", "true")
      .config("spark.sql.shuffle.partitions", "32")
    if (!clusterMode) {
      builder
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
    }
    val spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    // Use user-supplied shared path in cluster mode, otherwise a local temp dir.
    val (ownedTmpDir, dataRoot) = dataDirOpt match {
      case Some(p) => (None, p)
      case None =>
        val tmp = Files.createTempDirectory("knn-bench-")
        (Some(tmp.toFile), tmp.toString)
    }

    val results = scala.collection.mutable.ArrayBuffer.empty[Result]
    try {
      scales.foreach { scale =>
        println(banner(s"Scale: $scale"))
        val (leftDf, rightUri) = setupScale(spark, scale, dataRoot)
        val configs = makeConfigs(leftDf, rightUri, scale.runBaseline)

        // Sanity check: every config — INCLUDING the Spark crossJoin baseline — returns the
        // same top-K row IDs as the in-memory brute-force oracle on a 16-row left subset.
        // This is what makes the timing comparison meaningful: an 18×/608× number is hollow if
        // the paths disagree on output. Run at every scale on a small subset so the baseline's
        // O(|L|×|R|) crossJoin only does 16 × |R| work — sub-second even at medium scale.
        verifyAllConfigsAgainstOracle(spark, leftDf, rightUri)

        configs.foreach { case (name, run) =>
          val r = timeIt(scale.name, name, run)
          results += r
          println(formatResult(r))
        }
        println()
      }

      println(banner("Summary"))
      printSummaryTable(results.toSeq)
    } finally {
      spark.stop()
      // Only clean up a locally-created temp dir; leave user-supplied paths untouched.
      ownedTmpDir.foreach(deleteRecursively)
    }
  }

  // -- workload setup ----------------------------------------------------------------------

  private def setupScale(
      spark: SparkSession,
      scale: Scale,
      tmpRoot: String): (DataFrame, String) = {
    val rng = new Random(Seed)
    println(s"  Generating ${scale.numLeft} left rows × dim $Dim ...")
    val leftDf = buildLeft(spark, rng, scale.numLeft).cache()
    leftDf.count()

    val rightUri = Paths.get(tmpRoot, s"right_${scale.name}").toString
    val t0 = System.nanoTime()
    println(s"  Writing ${scale.numRight} right rows × dim $Dim across ${scale.numFragments} " +
      s"Spark partitions to $rightUri ...")
    writeRight(spark, rng, scale.numRight, scale.numFragments, rightUri)
    println(s"  ... done in ${TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - t0)}s")
    (leftDf, rightUri)
  }

  private def buildLeft(spark: SparkSession, rng: Random, n: Int): DataFrame = {
    val schema = new StructType(Array(
      StructField("lid", IntegerType, nullable = false),
      StructField(
        "lvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val rows = (0 until n).map { i =>
      RowFactory.create(Integer.valueOf(i), randomVector(rng, Dim))
    }
    spark.createDataFrame(rows.asJava, schema)
  }

  private def writeRight(
      spark: SparkSession,
      rng: Random,
      n: Int,
      fragments: Int,
      uri: String): Unit = {
    val schema = new StructType(Array(
      StructField("rid", IntegerType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    // Build rows on the driver in a streaming-ish pattern to keep memory bounded for medium scale.
    val rows = (0 until n).map { i =>
      RowFactory.create(Integer.valueOf(i + 1000000), randomVector(rng, Dim))
    }
    val df = spark.createDataFrame(rows.asJava, schema).repartition(fragments)
    df.write.format("lance").save(uri)
  }

  private def randomVector(rng: Random, dim: Int): Array[Float] = {
    val v = new Array[Float](dim)
    var i = 0
    while (i < dim) { v(i) = rng.nextFloat(); i += 1 }
    v
  }

  // -- configurations ----------------------------------------------------------------------

  private type Runnable = () => DataFrame

  private def makeConfigs(
      left: DataFrame,
      rightUri: String,
      runBaseline: Boolean): Seq[(String, Runnable)] = {
    val spark = left.sparkSession
    // Lance-backed right DataFrame for the new `df.kNearestJoin` extension. The extension
    // pulls the URI back out of the right DataFrame's analyzed plan internally — same probe
    // pipeline, just a more idiomatic call site that mirrors `df.join(other, ...)`.
    val rightDf = spark.read.format("lance").load(rightUri)

    val baseline: Runnable = () => crossProductTopK(spark, left, rightUri, K)
    val phase01: Runnable = () =>
      left.kNearestJoin(
        right = rightDf,
        leftVecCol = "lvec",
        rightVecCol = "rvec",
        k = K,
        metric = "l2",
        rightProjection = Some(Seq("rid")),
        probeParallelism = 1)
    val phase15_4: Runnable = () =>
      left.kNearestJoin(
        right = rightDf,
        leftVecCol = "lvec",
        rightVecCol = "rvec",
        k = K,
        metric = "l2",
        rightProjection = Some(Seq("rid")),
        probeParallelism = 4)
    val phase15_8: Runnable = () =>
      left.kNearestJoin(
        right = rightDf,
        leftVecCol = "lvec",
        rightVecCol = "rvec",
        k = K,
        metric = "l2",
        rightProjection = Some(Seq("rid")),
        probeParallelism = 8)
    val phase15_8_skew: Runnable = () =>
      left.kNearestJoin(
        right = rightDf,
        leftVecCol = "lvec",
        rightVecCol = "rvec",
        k = K,
        metric = "l2",
        rightProjection = Some(Seq("rid")),
        probeParallelism = 8,
        balanceFragments = true)

    val baseSeq = Seq(
      "B: Phase 0/1 (probeParallelism=1)" -> phase01,
      "C: Phase 1.5 (probeParallelism=4)" -> phase15_4,
      "D: Phase 1.5 (probeParallelism=8)" -> phase15_8,
      "E: Phase 1.5 (G=8, skew-balanced)" -> phase15_8_skew)
    if (runBaseline) ("A: Spark crossJoin (baseline)" -> baseline) +: baseSeq else baseSeq
  }

  /**
   * Vanilla-Spark baseline: cross product + custom L2 UDF + `row_number` window per `lid`. This
   * is the textbook way to express nearest-by-join in Spark today (Spark 3.5 doesn't have
   * vector_l2_distance; that's a 4.2 addition). It's also what `RewriteNearestByJoin` lowers
   * a `NearestByJoin` operator to under the hood — the apples-to-apples comparison.
   */
  private def crossProductTopK(
      spark: SparkSession,
      left: DataFrame,
      rightUri: String,
      k: Int): DataFrame = {
    val l2 = udf((a: Seq[Float], b: Seq[Float]) => {
      var s = 0.0f
      var i = 0
      while (i < a.length) { val d = a(i) - b(i); s += d * d; i += 1 }
      s
    })
    val right = spark.read.format("lance").load(rightUri).select("rid", "rvec")
    val crossed = left.crossJoin(right).withColumn("__dist", l2(col("lvec"), col("rvec")))
    val w = Window.partitionBy("lid").orderBy(col("__dist"))
    crossed.withColumn("__rank", row_number().over(w)).filter(col("__rank") <= k).select(
      "lid",
      "rid",
      "__dist")
  }

  // -- timing harness ----------------------------------------------------------------------

  private val WarmupRuns = 1
  private val MeasurementRuns = 3

  /**
   * Execute the plan fully and discard output — Spark's canonical benchmark sink, same
   * shape as `WikipediaKnnPerfBenchmark.runFull`. Avoids the `count()`-bias where the
   * crossJoin baseline skips result-row assembly while the indexed path runs
   * `LanceMaterialize` in full (the `references = child.outputSet` override on the
   * materialize logical plan blocks ColumnPruning from removing the join-row columns).
   */
  private def runFull(df: DataFrame): Unit =
    df.write.format("noop").mode("overwrite").save()

  /** Run `f` once for warmup, then 3x for measurement. Median wall-clock in ms. */
  private def timeIt(scale: String, config: String, f: Runnable): Result = {
    print(s"  $config ... ")
    System.out.flush()
    var i = 0
    while (i < WarmupRuns) { runFull(f()); i += 1 }
    val runs = (0 until MeasurementRuns).map { _ =>
      val t0 = System.nanoTime()
      runFull(f())
      TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0)
    }
    val sortedRuns = runs.sorted
    val median = sortedRuns(sortedRuns.length / 2)
    println(s"runs=${runs.mkString("[", ",", "]")} ms, median=$median ms")
    Result(scale, config, median, runs)
  }

  // -- oracle equivalence -----------------------------------------------------------------

  /**
   * Sanity check: confirm Phase 0/1 and Phase 1.5 paths agree with a brute-force oracle on a
   * 16-row subset of the left side. If this disagrees with the indexed path the benchmark
   * numbers are meaningless, so we bail before spending minutes on incorrect timings.
   */
  private def verifyOracleEquivalence(
      spark: SparkSession,
      leftDf: DataFrame,
      rightUri: String): Unit = {
    println("  Sanity check: indexed-path top-K matches brute-force oracle on a 16-row subset ...")
    val leftSubset = leftDf.limit(16).cache()
    leftSubset.count()
    val rightDf = spark.read.format("lance").load(rightUri)
    val joined = leftSubset.kNearestJoin(
      right = rightDf,
      leftVecCol = "lvec",
      rightVecCol = "rvec",
      k = K,
      metric = "l2",
      rightProjection = Some(Seq("rid", "rvec")),
      probeParallelism = 4)

    val byLid = joined.collect().groupBy(_.getAs[Int]("lid"))
    val rightVecs = readRightVectors(spark, rightUri)
    val rightIds = readRightIds(spark, rightUri)
    val leftRows = leftSubset.collect()

    leftRows.foreach { lr =>
      val lid = lr.getAs[Int]("lid")
      val leftVec = lr.getAs[Seq[Float]]("lvec").toArray
      val oracleIds = rightVecs.indices
        .map(i => (rightIds(i), l2(leftVec, rightVecs(i))))
        .sortBy(_._2)
        .take(K)
        .map(_._1)
        .toSet
      val actualIds = byLid(lid).map(_.getAs[Int]("rid")).toSet
      if (oracleIds != actualIds) {
        sys.error(
          s"ORACLE MISMATCH at lid=$lid:\n  oracle: $oracleIds\n  actual: $actualIds")
      }
    }
    leftSubset.unpersist()
    println("  ... oracle equivalence holds.")
  }

  /**
   * Run EVERY config (including the Spark crossJoin baseline) on a 16-row left subset and
   * compare each result against an in-memory brute-force oracle. Running on a subset keeps
   * the slow baseline tractable (16 × |R| pair evaluations is sub-second even at medium scale)
   * while still validating that all paths produce the SAME top-K as the ground truth.
   *
   * Compared as Sets to tolerate tied-distance ordering. Random data makes exact ties rare in
   * practice, but the comparison is robust either way.
   */
  private def verifyAllConfigsAgainstOracle(
      spark: SparkSession,
      leftDf: DataFrame,
      rightUri: String): Unit = {
    println("  Sanity check: indexed-path configs match brute-force oracle on a 16-row subset ...")
    val left16 = leftDf.limit(16).cache()
    left16.count()
    val leftIds = left16.select("lid").collect().map(_.getInt(0)).toSet

    // Brute-force oracle in plain Scala — the ground truth.
    val rightVecs = readRightVectors(spark, rightUri)
    val rightIds = readRightIds(spark, rightUri)
    val leftRows = left16.collect()
    val oracleByLid: Map[Int, Set[Int]] = leftRows.map { r =>
      val lid = r.getAs[Int]("lid")
      val lvec = r.getAs[Seq[Float]]("lvec").toArray
      val topKRids = rightVecs.indices
        .map(i => (rightIds(i), l2(lvec, rightVecs(i))))
        .sortBy(_._2)
        .take(K)
        .map(_._1)
        .toSet
      lid -> topKRids
    }.toMap

    // Build mini-configs that close over `left16`, not the full left.
    // Validate B/C/D/E against the in-memory brute-force oracle. We deliberately do NOT run
    // config A (Spark crossJoin) here — its output IS brute force by construction, so
    // comparing Spark's crossJoin to the in-memory brute force is a tautology, while the
    // window-function pipeline can be slow enough on 16 × |R| pairs to dominate the
    // benchmark's wall-clock. The semantic question "is the indexed path correct" is fully
    // answered by checking B/C/D/E against the oracle directly.
    val miniConfigs = makeConfigs(left16, rightUri, runBaseline = false)

    miniConfigs.foreach { case (name, run) =>
      val rows = run().collect()
      val byLid = rows.groupBy(_.getAs[Int]("lid"))
        .map { case (lid, rs) => lid -> rs.map(_.getAs[Int]("rid")).toSet }
      leftIds.foreach { lid =>
        val expected = oracleByLid(lid)
        val actual = byLid.getOrElse(lid, Set.empty[Int])
        if (expected != actual) {
          sys.error(
            s"ORACLE MISMATCH for $name at lid=$lid:\n  oracle: $expected\n  actual: $actual")
        }
      }
    }
    left16.unpersist()
    println(
      s"  ... all ${miniConfigs.size} indexed configs match the oracle " +
        s"(sample size: ${leftIds.size}).")
  }

  private def readRightVectors(spark: SparkSession, uri: String): Array[Array[Float]] =
    spark.read.format("lance").load(uri).orderBy("rid").collect().map { r =>
      r.getAs[Seq[Float]]("rvec").toArray
    }

  private def readRightIds(spark: SparkSession, uri: String): Array[Int] =
    spark.read.format("lance").load(uri).orderBy("rid").collect().map(_.getAs[Int]("rid"))

  private def l2(a: Array[Float], b: Array[Float]): Float = {
    var s = 0.0f
    var i = 0
    while (i < a.length) { val d = a(i) - b(i); s += d * d; i += 1 }
    s
  }

  // -- output formatting ------------------------------------------------------------------

  private def banner(s: String): String = s"\n=== $s " + ("=" * (76 - s.length - 5))

  private def formatResult(r: Result): String =
    f"    -> ${r.config}%-40s  median=${r.medianMs}%6d ms"

  private def printSummaryTable(results: Seq[Result]): Unit = {
    val byScale = results.groupBy(_.scale).map { case (k, vs) => k -> vs.sortBy(_.config) }
    val scaleOrder = Seq("small", "medium").filter(byScale.contains)
    println()
    val configWidth = 38
    val numWidth = 13
    val header = "%-".concat(s"${configWidth}s") +
      scaleOrder.map(_ => s"%${numWidth}s").mkString
    val divider = "-" * (configWidth + scaleOrder.size * numWidth)
    println(divider)
    val args1 = "Configuration" +: scaleOrder.map(s => s"$s (ms)")
    println(header.format(args1: _*))
    val args2 = "" +: scaleOrder.map(s => s"speedup ×")
    println(header.format(args2: _*))
    println(divider)

    val configs = scaleOrder.flatMap(byScale(_).map(_.config)).distinct
    val baselineByScale = scaleOrder.flatMap { s =>
      byScale(s).find(_.config.startsWith("A:")).map(b => s -> b.medianMs)
    }.toMap
    configs.foreach { config =>
      val cellsMs = scaleOrder.map { s =>
        byScale(s).find(_.config == config).map(_.medianMs.toString).getOrElse("-")
      }
      println(header.format((config +: cellsMs): _*))
      val cellsSpeedup = scaleOrder.map { s =>
        val mineMs = byScale(s).find(_.config == config).map(_.medianMs).getOrElse(0L)
        val baseMs = baselineByScale.getOrElse(s, 0L)
        if (config.startsWith("A:")) "1.00x"
        else if (mineMs <= 0 || baseMs <= 0) "(no base)"
        else f"${baseMs.toDouble / mineMs}%.2fx"
      }
      println(header.format(("" +: cellsSpeedup): _*))
    }
    println(divider)
    println("Speedup is `baseline / config`. Higher = faster than vanilla Spark crossJoin.")
    println("Medium scale skips the crossJoin baseline (1B pairs is impractical to bench locally);")
    println("compare B vs. C/D/E within the medium column for fragment-grouping speedup.")
  }

  private def deleteRecursively(f: java.io.File): Unit = {
    if (f.isDirectory) f.listFiles().foreach(deleteRecursively)
    f.delete()
  }
}
