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
import org.lance.spark.knn.LanceKnnImplicits._

import java.util.concurrent.TimeUnit

/**
 * Performance benchmark: indexed kNN-join vs vanilla Spark crossJoin on Cohere Wikipedia
 * embeddings (dim=1024). Production-shape counterpart to
 * [[IndexedNearestJoinBenchmark]], which uses synthetic random vectors at dim=128.
 *
 * Shape matches `IndexedNearestJoinBenchmark`: 5 configs, median of 3 runs + 1 warmup,
 * same 4 `probeParallelism` flavours on the indexed side:
 *
 *   - A: Vanilla Spark cross-product + L2 UDF + row_number window (baseline).
 *   - B: Phase 0/1 single-task probe (probeParallelism=1).
 *   - C: Phase 1.5 fragment-grouped probe (probeParallelism=4).
 *   - D: Phase 1.5 fragment-grouped probe (probeParallelism=8).
 *   - E: Phase 1.5 + skew-balanced fragment grouping.
 *
 * Unlike [[CohereWikiRecallBenchmark]], this does NOT build a vector index on the right
 * side. Lance therefore brute-force-scans each fragment for every probe call — the
 * speedup over Spark's crossJoin is entirely from the native SIMD distance kernel +
 * columnar Arrow iteration, not from an ANN index. Adding an IVF-PQ/IVF-FLAT index would
 * produce a further 10-100× speedup on top of what this measures.
 *
 * == Why dim=1024 matters ==
 *
 * The existing `IndexedNearestJoinBenchmark` runs at dim=128. At dim=128 the per-pair
 * distance kernel is tiny (~8-16 cycles), so the speedup is dominated by JVM-vs-native
 * constant overhead. At dim=1024 the kernel cost is 8× higher in absolute terms, so:
 *   - Spark's crossJoin baseline gets dramatically slower in absolute wall-clock
 *   - Lance's SIMD kernel advantage widens (AVX2/AVX-512 process 8-16 floats/cycle)
 *   - The speedup factor gets larger, not smaller
 * Which is the opposite of what one might naively expect. The production-shape number
 * this benchmark produces is more credible than the SIFT-style dim=128 result.
 *
 * == Cluster run ==
 *
 * {{{
 *   ./mvnw -pl lance-spark-knn_2.12 package -Pbenchmark -DskipTests
 *   # upload the fat jar + the Cohere parquet shard(s)
 *
 *   BENCH_CLUSTER_MODE=true \
 *   BENCH_DATA_PATH=file:///valve-binaries/wiki-perf-data \
 *   WIKI_PARQUET=/valve-binaries/wiki-*.parquet \
 *   WIKI_NUM_RIGHT=100000 \
 *   WIKI_NUM_LEFT=100 \
 *   WIKI_NUM_FRAGMENTS=8 \
 *   WIKI_RUN_BASELINE=true \
 *   spark-submit --class org.lance.spark.knn.benchmark.WikipediaKnnPerfBenchmark <jar>
 * }}}
 *
 * == Env knobs ==
 *
 *   - `WIKI_PARQUET=<path>`        -- Cohere parquet glob (required). Same source as
 *                                     `CohereWikiRecallBenchmark`. Only the `emb` column
 *                                     is used.
 *   - `WIKI_EMB_COL=emb`           -- embedding column name (default `emb`).
 *   - `WIKI_NUM_RIGHT=100000`      -- base-set size (default 100K). The remaining rows in
 *                                     the parquet are ignored. Larger values make the
 *                                     crossJoin baseline increasingly impractical.
 *   - `WIKI_NUM_LEFT=100`          -- query rows held out from the base (default 100).
 *                                     The crossJoin baseline is O(|L|×|R|); at |L|=100,
 *                                     |R|=100K that's 10M pair evaluations.
 *   - `WIKI_NUM_FRAGMENTS=8`       -- number of Lance fragments to split base into. Drives
 *                                     `probeParallelism` caps on the indexed side.
 *   - `WIKI_K=10`                  -- top-K neighbours per query (default 10).
 *   - `WIKI_RUN_BASELINE=true`     -- set false to skip the crossJoin baseline. Useful for
 *                                     medium/large scales where O(|L|×|R|) is > 30 minutes
 *                                     per run.
 *   - `WIKI_WARMUP_RUNS=1` / `WIKI_MEASURE_RUNS=3`  -- timing iterations.
 *   - `WIKI_SKIP_SETUP=false`      -- if "true", reuse the Lance dataset at
 *                                     `BENCH_DATA_PATH/right` written by a prior run.
 *   - `BENCH_CLUSTER_MODE`, `BENCH_DATA_PATH` -- same semantics as other benchmarks.
 *
 * == What this does NOT measure ==
 *
 *   - ANN-index speedup. The right side is written without a vector index; Lance does
 *     brute-force distance computation. [[CohereWikiRecallBenchmark]] is the right tool
 *     for IVF-FLAT / IVF-PQ recall × latency tradeoffs.
 *   - Warm vs cold cache effects. Warmup runs prime the JVM / native code caches; the
 *     reported median is a steady-state number.
 *   - Driver-side latency (left-row creation, result materialization). These are inside
 *     the timing loop but dominated by the probe at any non-trivial |L|.
 *
 * == Known: Cohere parquet → Lance fixed-size-list ==
 *
 * The Cohere parquet emits `emb` as variable-length `list<float>`; Lance's nearest-scan
 * needs `FixedSizeList<Float>`. `DataFrameWriter.option("vec.arrow.fixed-size-list.size",
 * dim)` does NOT propagate through the writer path — the option is only honoured by the
 * `CREATE TABLE` + TBLPROPERTIES route. Working shape (same as
 * [[CohereWikiRecallBenchmark]]'s fix): collect to driver, rebuild fresh rows with a
 * `StructType` that has `arrow.fixed-size-list.size` metadata on the vec StructField,
 * `createDataFrame` + write. Costs ~400 MB driver heap per 100K rows × dim=1024.
 */
object WikipediaKnnPerfBenchmark {

  private val K: Int = sys.env.get("WIKI_K").map(_.toInt).getOrElse(10)
  private val NumRight: Int = sys.env.get("WIKI_NUM_RIGHT").map(_.toInt).getOrElse(100000)
  private val NumLeft: Int = sys.env.get("WIKI_NUM_LEFT").map(_.toInt).getOrElse(100)
  private val NumFragments: Int =
    sys.env.get("WIKI_NUM_FRAGMENTS").map(_.toInt).getOrElse(8)
  private val RunBaseline: Boolean = sys.env.get("WIKI_RUN_BASELINE")
    .map(_.equalsIgnoreCase("true")).getOrElse(true)
  private val WarmupRuns: Int = sys.env.get("WIKI_WARMUP_RUNS").map(_.toInt).getOrElse(1)
  private val MeasurementRuns: Int =
    sys.env.get("WIKI_MEASURE_RUNS").map(_.toInt).getOrElse(3)
  private val SkipSetup: Boolean =
    sys.env.get("WIKI_SKIP_SETUP").exists(_.equalsIgnoreCase("true"))
  private val ParquetPath: String = sys.env.getOrElse(
    "WIKI_PARQUET",
    sys.error("WIKI_PARQUET is required (path to Cohere wiki parquet files)"))
  private val EmbCol: String = sys.env.getOrElse("WIKI_EMB_COL", "emb")
  private val ClusterMode: Boolean =
    sys.env.get("BENCH_CLUSTER_MODE").exists(_.equalsIgnoreCase("true"))
  private val DataPath: String = sys.env.getOrElse(
    "BENCH_DATA_PATH",
    "file:///tmp/wiki-perf-lance")

  private case class Result(config: String, medianMs: Long, runs: Seq[Long])
  private type RunFn = () => DataFrame

  def main(args: Array[String]): Unit = {
    val spark = buildSparkSession()
    try {
      println(banner("Wikipedia KNN-Join Perf Benchmark"))
      val masterDesc = if (ClusterMode) "cluster (BENCH_CLUSTER_MODE=true)" else "local[*]"
      println(s"  master:          $masterDesc")
      println(s"  parquet:         $ParquetPath")
      println(s"  data path:       $DataPath")
      println(s"  |R|=$NumRight   |L|=$NumLeft   fragments=$NumFragments   K=$K")
      println(s"  warmup/measure:  $WarmupRuns/$MeasurementRuns   runBaseline=$RunBaseline")
      println()

      val rightUri = s"$DataPath/right"
      val (leftDf, dim) = if (SkipSetup) {
        val d = detectDim(spark, rightUri)
        println(s"[wiki-perf] WIKI_SKIP_SETUP=true -> reusing $rightUri (dim=$d)")
        buildLeftFromLanceBase(spark, rightUri, d)
      } else {
        setupDatasets(spark, rightUri)
      }
      println(s"[wiki-perf] left=${leftDf.count()} right=$NumRight dim=$dim")
      println()

      val configs = makeConfigs(leftDf, rightUri)

      // Correctness gate: every config — including the crossJoin baseline — must agree
      // with a brute-force oracle on a 16-row left subset before we quote any speedup.
      // Without this, the `count()`/`noop` timing loop only checks cardinality, not
      // content — a bug that emits |L|×K garbage rows would still "validate." This is
      // the same check `IndexedNearestJoinBenchmark.verifyAllConfigsAgainstOracle`
      // runs on synthetic data; here it runs on real Cohere Wikipedia embeddings so we
      // also catch any dim=1024-specific correctness regressions.
      verifyAllConfigsAgainstOracle(spark, leftDf, rightUri)

      val results = scala.collection.mutable.ArrayBuffer.empty[Result]
      configs.foreach { case (name, run) =>
        val r = timeIt(name, run)
        results += r
        println(formatResult(r))
      }
      println()
      println(banner("Summary"))
      printSummary(results.toSeq)
    } finally {
      spark.stop()
    }
  }

  // -- Spark session --------------------------------------------------------------------------

  private def buildSparkSession(): SparkSession = {
    val b = SparkSession.builder().appName("wikipedia-knn-perf")
      .config("spark.sql.crossJoin.enabled", "true")
      .config("spark.sql.shuffle.partitions", "32")
    if (!ClusterMode) {
      b.master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
    }
    val s = b.getOrCreate()
    s.sparkContext.setLogLevel("WARN")
    s
  }

  // -- setup: Cohere parquet -> Lance right + sampled left ---------------------------------

  /**
   * Load the Cohere parquet, pull NumRight+NumLeft rows to the driver, tag the schema
   * with `arrow.fixed-size-list.size`, write the base set to Lance, keep the query rows
   * as an in-memory DataFrame. Returns (leftDf, dim).
   *
   * Driver-side collect is required because `DataFrame.write.format("lance")` does NOT
   * honour the `vec.arrow.fixed-size-list.size` option on the writer path — without
   * proper field metadata, Lance writes the column as variable-length `list<float>` and
   * the nearest-scan path returns `_rowid` only (no `_distance`), tripping
   * `LanceProbe.readScored`'s "did not return a score column" check. See
   * `CohereWikiRecallBenchmark` for the same workaround.
   */
  private def setupDatasets(spark: SparkSession, rightUri: String): (DataFrame, Int) = {
    println(s"[wiki-perf] reading source parquet: $ParquetPath")
    val raw = spark.read.parquet(ParquetPath)
    require(
      raw.schema.fieldNames.contains(EmbCol),
      s"Source parquet does not contain $EmbCol; fields: ${raw.schema.fieldNames.mkString(",")}")

    val total = NumRight + NumLeft
    // Stable: select ordered, then limit. Avoids rand()-driven non-determinism across runs.
    val embColExpr = col(EmbCol).cast(ArrayType(FloatType))
    val sliced = raw.select(embColExpr.as("vec")).limit(total)

    val allRows = sliced.collect()
    require(
      allRows.length >= NumLeft + 1,
      s"Parquet yielded ${allRows.length} rows; need at least ${NumLeft + 1} for a " +
        s"left/right split.")
    if (allRows.length < total) {
      // Shard smaller than requested |R|+|L|. Shrink the right side to whatever's left
      // after holding out NumLeft queries. Emit a warning so the user sees the scale
      // downshift (otherwise speedup numbers can look too good at a smaller |R|).
      println(f"[wiki-perf] WARN: parquet only yielded ${allRows.length}%,d rows; " +
        f"needed $total%,d. Right side shrunk from $NumRight%,d to " +
        f"${allRows.length - NumLeft}%,d.")
    }
    val effectiveRight = math.min(NumRight, allRows.length - NumLeft)
    val dim = allRows(0).getAs[scala.collection.Seq[Float]]("vec").length
    println(f"[wiki-perf]   collected ${allRows.length}%,d rows at dim=$dim; " +
      f"using |R|=$effectiveRight%,d |L|=$NumLeft%,d")

    // Split first NumRight rows -> right (base), remaining NumLeft -> left (queries).
    val embMeta = new MetadataBuilder()
      .putLong("arrow.fixed-size-list.size", dim.toLong)
      .build()

    // Right side: (rid: Long, rvec: Array<Float> [fixed-size]).
    val rightSchema = new StructType(Array(
      StructField("rid", LongType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        embMeta)))
    val rightRows = new java.util.ArrayList[Row](effectiveRight)
    var i = 0
    while (i < effectiveRight) {
      val s = allRows(i).getAs[scala.collection.Seq[Float]]("vec")
      val arr = new Array[Float](s.length)
      var j = 0
      while (j < s.length) { arr(j) = s(j); j += 1 }
      rightRows.add(RowFactory.create(java.lang.Long.valueOf(i.toLong), arr))
      i += 1
    }
    println(s"[wiki-perf] writing right (base) to Lance: $rightUri")
    val t0 = System.nanoTime()
    spark.createDataFrame(rightRows, rightSchema)
      .repartition(NumFragments)
      .write.format("lance").save(rightUri)
    println(f"[wiki-perf]   wrote in ${TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - t0)}%d s")

    // Left side: (lid: Long, lvec: Array<Float> [fixed-size]).
    val leftSchema = new StructType(Array(
      StructField("lid", LongType, nullable = false),
      StructField(
        "lvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        embMeta)))
    val leftRows = new java.util.ArrayList[Row](NumLeft)
    var li = 0
    while (li < NumLeft) {
      val r = allRows(effectiveRight + li)
      val s = r.getAs[scala.collection.Seq[Float]]("vec")
      val arr = new Array[Float](s.length)
      var j = 0
      while (j < s.length) { arr(j) = s(j); j += 1 }
      leftRows.add(RowFactory.create(java.lang.Long.valueOf(li.toLong), arr))
      li += 1
    }
    val leftDf = spark.createDataFrame(leftRows, leftSchema).cache()
    leftDf.count() // force cache materialization

    (leftDf, dim)
  }

  /**
   * When `WIKI_SKIP_SETUP=true`, synthesise a left side from the already-written Lance
   * base dataset. Takes the first `NumLeft` rows by `rid` (deterministic across reruns).
   * The base set keeps those rows — the query rows are "in" the base — which means the
   * nearest-neighbour for each query is the query itself at distance 0. Fine for a
   * pure perf benchmark (we're not measuring recall), but note in your write-up.
   */
  private def buildLeftFromLanceBase(
      spark: SparkSession,
      rightUri: String,
      dim: Int): (DataFrame, Int) = {
    val embMeta = new MetadataBuilder()
      .putLong("arrow.fixed-size-list.size", dim.toLong)
      .build()
    val leftSchema = new StructType(Array(
      StructField("lid", LongType, nullable = false),
      StructField(
        "lvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        embMeta)))
    val baseRows = spark.read.format("lance").load(rightUri)
      .orderBy("rid").limit(NumLeft).collect()
    val javaRows = new java.util.ArrayList[Row](baseRows.length)
    var i = 0
    while (i < baseRows.length) {
      val s = baseRows(i).getAs[scala.collection.Seq[Float]]("rvec")
      val arr = new Array[Float](s.length)
      var j = 0
      while (j < s.length) { arr(j) = s(j); j += 1 }
      javaRows.add(RowFactory.create(java.lang.Long.valueOf(i.toLong), arr))
      i += 1
    }
    val leftDf = spark.createDataFrame(javaRows, leftSchema).cache()
    leftDf.count()
    (leftDf, dim)
  }

  private def detectDim(spark: SparkSession, rightUri: String): Int = {
    val r = spark.read.format("lance").load(rightUri).select("rvec").limit(1).collect()
    require(r.nonEmpty, s"Empty Lance dataset at $rightUri")
    r(0).getAs[scala.collection.Seq[Float]]("rvec").length
  }

  // -- configs --------------------------------------------------------------------------------

  private def makeConfigs(
      leftDf: DataFrame,
      rightUri: String,
      runBaseline: Boolean = RunBaseline): Seq[(String, RunFn)] = {
    val spark = leftDf.sparkSession
    val rightDf = spark.read.format("lance").load(rightUri)

    val baseline: RunFn = () => crossProductTopK(spark, leftDf, rightUri, K)
    val phase01: RunFn = () =>
      leftDf.kNearestJoin(
        right = rightDf,
        leftVecCol = "lvec",
        rightVecCol = "rvec",
        k = K,
        metric = "l2",
        rightProjection = Some(Seq("rid")),
        probeParallelism = 1)
    val phase15_4: RunFn = () =>
      leftDf.kNearestJoin(
        right = rightDf,
        leftVecCol = "lvec",
        rightVecCol = "rvec",
        k = K,
        metric = "l2",
        rightProjection = Some(Seq("rid")),
        probeParallelism = 4)
    val phase15_8: RunFn = () =>
      leftDf.kNearestJoin(
        right = rightDf,
        leftVecCol = "lvec",
        rightVecCol = "rvec",
        k = K,
        metric = "l2",
        rightProjection = Some(Seq("rid")),
        probeParallelism = 8)
    val phase15_8_skew: RunFn = () =>
      leftDf.kNearestJoin(
        right = rightDf,
        leftVecCol = "lvec",
        rightVecCol = "rvec",
        k = K,
        metric = "l2",
        rightProjection = Some(Seq("rid")),
        probeParallelism = 8,
        balanceFragments = true)

    val indexed = Seq(
      "B: Phase 0/1 (probeParallelism=1)" -> phase01,
      "C: Phase 1.5 (probeParallelism=4)" -> phase15_4,
      "D: Phase 1.5 (probeParallelism=8)" -> phase15_8,
      "E: Phase 1.5 (G=8, skew-balanced)" -> phase15_8_skew)
    if (runBaseline) ("A: Spark crossJoin (baseline)" -> baseline) +: indexed else indexed
  }

  /**
   * Vanilla-Spark baseline: cross product + L2 UDF + `row_number` window per `lid`. Same
   * shape as `IndexedNearestJoinBenchmark.crossProductTopK`; what `RewriteNearestByJoin`
   * lowers to if the indexed-path rule is disabled. The only difference is dim=1024 vs
   * 128 in the synthetic benchmark.
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
    crossed.withColumn("__rank", row_number().over(w))
      .filter(col("__rank") <= k)
      .select("lid", "rid", "__dist")
  }

  // -- oracle equivalence --------------------------------------------------------------------

  /**
   * Run EVERY config (including the crossJoin baseline) on a 16-row left subset and compare
   * each result against an in-memory brute-force oracle. Running on a subset keeps the
   * baseline tractable (16 × |R| pair evaluations is sub-second even at dim=1024 × 100K)
   * while still validating that all paths agree on top-K row IDs.
   *
   * Compared as `Set[Long]` per `lid` to tolerate tied-distance ordering. Real embeddings
   * rarely produce exact ties but the comparison is robust either way.
   *
   * Why this matters: the `timeIt` harness uses `write.format("noop")` which materializes
   * every row but discards output. Cardinality alone isn't a correctness proof — a bug
   * that emits `|L|×K` garbage rows would still produce the expected count. This oracle
   * check closes that loop on real Cohere data at dim=1024.
   */
  private def verifyAllConfigsAgainstOracle(
      spark: SparkSession,
      leftDf: DataFrame,
      rightUri: String): Unit = {
    println("  Sanity check: all configs match brute-force oracle on a 16-row subset ...")
    val left16 = leftDf.limit(16).cache()
    left16.count()
    val leftRows = left16.collect()

    // Brute-force oracle in plain Scala — the ground truth.
    val rightDf = spark.read.format("lance").load(rightUri).select("rid", "rvec").collect()
    val rightVecs = rightDf.map(r => r.getAs[scala.collection.Seq[Float]]("rvec").toArray)
    val rightIds = rightDf.map(_.getAs[Long]("rid"))
    val oracleByLid: Map[Long, Set[Long]] = leftRows.map { r =>
      val lid = r.getAs[Long]("lid")
      val lvec = r.getAs[scala.collection.Seq[Float]]("lvec").toArray
      val topKRids = rightVecs.indices
        .map(i => (rightIds(i), l2(lvec, rightVecs(i))))
        .sortBy(_._2)
        .take(K)
        .map(_._1)
        .toSet
      lid -> topKRids
    }.toMap

    // Validate B/C/D/E against the oracle (A is brute force by construction — comparing
    // Spark's crossJoin to in-memory brute force would be a tautology, and the window
    // pipeline is slow enough on 16 × |R| dim=1024 pairs to add minutes per run).
    val miniConfigs = makeConfigs(left16, rightUri, runBaseline = false)
    miniConfigs.foreach { case (name, run) =>
      val rows = run().collect()
      val byLid = rows.groupBy(_.getAs[Long]("lid"))
        .map { case (lid, rs) => lid -> rs.map(_.getAs[Long]("rid")).toSet }
      leftRows.map(_.getAs[Long]("lid")).foreach { lid =>
        val expected = oracleByLid(lid)
        val actual = byLid.getOrElse(lid, Set.empty[Long])
        if (expected != actual) {
          sys.error(
            s"ORACLE MISMATCH for $name at lid=$lid:\n  oracle: $expected\n  actual: $actual")
        }
      }
    }
    left16.unpersist()
    println(
      s"  ... all ${miniConfigs.size} indexed configs match the oracle " +
        s"(sample size: ${leftRows.length}, K=$K).")
  }

  private def l2(a: Array[Float], b: Array[Float]): Float = {
    var s = 0.0f
    var i = 0
    while (i < a.length) { val d = a(i) - b(i); s += d * d; i += 1 }
    s
  }

  // -- timing ---------------------------------------------------------------------------------

  /**
   * Execute the plan fully and discard the result rows — Spark's canonical benchmark
   * sink. Unlike `count()`, this forces both paths to materialize every join row through
   * the projected columns, closing the `count()`-bias gap where the crossJoin baseline
   * skips result-row assembly while the indexed path runs `LanceMaterialize` in full
   * (due to the `references = child.outputSet` override on `LanceMaterializeLogicalPlan`).
   * No network round-trip to driver either, so the measurement is pure pipeline wall-clock.
   */
  private def runFull(df: DataFrame): Unit =
    df.write.format("noop").mode("overwrite").save()

  private def timeIt(config: String, f: RunFn): Result = {
    print(s"  $config ... ")
    System.out.flush()
    var i = 0
    while (i < WarmupRuns) { runFull(f()); i += 1 }
    val runs = (0 until MeasurementRuns).map { _ =>
      val t0 = System.nanoTime()
      runFull(f())
      TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0)
    }
    val sorted = runs.sorted
    val median = sorted(sorted.length / 2)
    println(s"runs=${runs.mkString("[", ",", "]")} ms, median=$median ms")
    Result(config, median, runs)
  }

  // -- output ---------------------------------------------------------------------------------

  private def banner(s: String): String = s"\n=== $s " + ("=" * math.max(0, 76 - s.length - 5))

  private def formatResult(r: Result): String =
    f"    -> ${r.config}%-40s  median=${r.medianMs}%8d ms"

  private def printSummary(results: Seq[Result]): Unit = {
    val configWidth = 40
    val numWidth = 14
    val divider = "-" * (configWidth + numWidth * 2)
    println(divider)
    println(("%-" + configWidth + "s%" + numWidth + "s%" + numWidth + "s")
      .format("Configuration", "median (ms)", "speedup ×"))
    println(divider)
    val baselineMs = results.find(_.config.startsWith("A:")).map(_.medianMs).getOrElse(0L)
    results.foreach { r =>
      val speedup =
        if (r.config.startsWith("A:")) "1.00x"
        else if (r.medianMs <= 0 || baselineMs <= 0) "(no base)"
        else f"${baselineMs.toDouble / r.medianMs}%.2fx"
      println(("%-" + configWidth + "s%" + numWidth + "d%" + numWidth + "s")
        .format(r.config, r.medianMs, speedup))
    }
    println(divider)
    println(
      "Speedup = baseline(A) / config. Higher = faster. Baseline is vanilla Spark " +
        "crossJoin + L2 UDF + row_number window, the lowering Spark's RewriteNearestByJoin " +
        "applies to SQL APPROX NEAREST when the indexed rule is off.")
  }
}
