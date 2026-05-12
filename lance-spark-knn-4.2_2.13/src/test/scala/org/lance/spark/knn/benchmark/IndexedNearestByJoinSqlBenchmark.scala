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

import org.apache.spark.sql.{DataFrame, RowFactory, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._
import org.lance.spark.knn.catalyst.IndexedNearestByJoinRule
import org.lance.spark.knn.internal.LanceVectorIndexBuilder

import java.nio.file.{Files, Paths}
import java.util.{Locale, Random}
import java.util.concurrent.TimeUnit

import scala.collection.JavaConverters._

/**
 * SQL-level benchmark for the Phase 2 Catalyst integration. Same `APPROX NEAREST k BY DISTANCE
 * vector_l2_distance(...)` SQL run with the rule ON vs OFF — measuring the speedup at the SQL
 * level a user would observe.
 *
 * Rule OFF lowers to Spark's built-in `RewriteNearestByJoin`, which is the optimizer rule that
 * rewrites `NearestByJoin` to `Generate(Inline(Aggregate(min_by_k(...))))` over a
 * `BroadcastNestedLoopJoin`. Cross-product semantics. This is what every user would get today
 * out of vanilla Spark when they write `APPROX NEAREST` against any data source.
 *
 * Rule ON intercepts the same SQL pre-optimizer and emits the same 3-plan staged tree
 * (`LanceProbe → LanceMerge → LanceMaterialize`, with a Catalyst-inserted Exchange on the
 * merge side) that the DataFrame API path builds — single-task probe per partition (Phase
 * 0/1 default) or fragment-grouped if `probeParallelism > 1`. The SQL path can't expose
 * `probeParallelism` directly via SQL — that's a session-config or rule-side choice (Phase
 * 3.x). Defaults to 1.
 *
 * Requires Spark 4.2-SNAPSHOT runtime AND lance-spark-4.1 connector recompiled against it. See
 * `IndexedNearestByJoinE2ETest` class doc for the setup commands.
 *
 * Invocation:
 *
 * {{{
 *   MAVEN_OPTS="-Xmx12g <Spark JDK17 --add-opens flags>" \
 *     ./mvnw -pl lance-spark-knn-4.2_2.13 -q exec:java \
 *       -Dexec.classpathScope=test \
 *       -Dexec.mainClass=org.lance.spark.knn.benchmark.IndexedNearestByJoinSqlBenchmark
 * }}}
 *
 * Override scale via `BENCHMARK_SCALE`: `small`, `medium`, or `both` (default).
 */
object IndexedNearestByJoinSqlBenchmark {

  private val Dim: Int = 128
  private val K: Int = 10
  private val Seed: Long = 1337L

  /**
   * Data distribution selector. `uniform` = independent floats over [0, 1]^Dim — the IVF worst
   * case (k-means has no cluster structure to latch onto, recall on indexed paths is poor).
   * `clustered` = unit-sphere-normalized Gaussian-mixture, the geometry of typical
   * sentence-transformer / image-feature embeddings. Override via `BENCHMARK_DATA` env var.
   */
  sealed private trait DataMode { def label: String }
  private object DataUniform extends DataMode { val label = "uniform" }
  private object DataClustered extends DataMode { val label = "clustered" }
  private val NumClusters: Int = 64

  /**
   * Index-type selector. `flat` = IVF_FLAT, exact distances within visited clusters (the
   * workaround we used to get >0.9 recall on uniform-random dim-128 data, where PQ noise
   * dominates). `pq` = IVF-PQ with `Dim / 16` sub-vectors and 8 bits — the production-realistic
   * compressed index. PQ is what actually gets used at scale in production deployments because
   * IVF_FLAT's per-cluster full-vector storage doesn't fit, but PQ's recall is highly sensitive
   * to data distribution. Override via `BENCHMARK_INDEX` env var.
   */
  sealed private trait IndexMode { def label: String }
  private object IndexFlat extends IndexMode { val label = "ivf_flat" }
  private object IndexPq extends IndexMode { val label = "ivf_pq" }

  /** runRuleOff: skip the brute-force baseline at scales where it'd take >10 min. */
  private case class Scale(
      name: String,
      numRight: Int,
      numLeft: Int,
      numFragments: Int,
      runRuleOff: Boolean) {
    override def toString: String = s"$name (|R|=$numRight, |L|=$numLeft, frags=$numFragments)"
  }

  private val Small =
    Scale("small", numRight = 100000, numLeft = 100, numFragments = 4, runRuleOff = true)
  private val Medium =
    Scale("medium", numRight = 1000000, numLeft = 1000, numFragments = 8, runRuleOff = false)

  private case class Result(scale: String, config: String, medianMs: Long, runs: Seq[Long])

  def main(args: Array[String]): Unit = {
    val scales = sys.env.getOrElse("BENCHMARK_SCALE", "both").toLowerCase(Locale.ROOT) match {
      case "small" => Seq(Small)
      case "medium" => Seq(Medium)
      case _ => Seq(Small, Medium)
    }
    val dataMode: DataMode =
      sys.env.getOrElse("BENCHMARK_DATA", "uniform").toLowerCase(Locale.ROOT) match {
        case "clustered" => DataClustered
        case _ => DataUniform
      }
    val indexMode: IndexMode =
      sys.env.getOrElse("BENCHMARK_INDEX", "flat").toLowerCase(Locale.ROOT) match {
        case "pq" | "ivf_pq" => IndexPq
        case _ => IndexFlat
      }

    println(banner("Phase 2 SQL Benchmark — APPROX NEAREST with rule ON vs OFF"))
    println(s"Spark: 4.2-SNAPSHOT, master=local[*]   Dim: $Dim   K: $K   Seed: $Seed")
    println(
      s"Scales: ${scales.map(_.name).mkString(", ")}   Data: ${dataMode.label}   " +
        s"Index: ${indexMode.label}")
    println()

    val spark = SparkSession
      .builder()
      .appName("indexed-nearest-by-join-sql-benchmark")
      .master("local[*]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .config(
        "spark.sql.extensions",
        "org.lance.spark.knn.extensions.LanceKnnSparkSessionExtensions")
      .config("spark.sql.crossJoin.enabled", "true")
      .config("spark.sql.shuffle.partitions", "32")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val tmpRoot = Files.createTempDirectory("knn-sql-bench-")
    val results = scala.collection.mutable.ArrayBuffer.empty[Result]
    try {
      scales.foreach { scale =>
        println(banner(s"Scale: $scale   Data: ${dataMode.label}"))
        val rng = new Random(Seed)
        println(s"  Generating ${scale.numLeft} left rows × dim $Dim (${dataMode.label}) ...")
        val leftVecs = generateVectors(dataMode, scale.numLeft, Dim, Seed)
        val leftRows = (0 until scale.numLeft).map { i =>
          RowFactory.create(Integer.valueOf(i), leftVecs(i))
        }
        spark.createDataFrame(leftRows.asJava, leftSchema()).createOrReplaceTempView("queries")

        val rightUri = Paths.get(tmpRoot.toString, s"right_${scale.name}").toString
        println(s"  Writing ${scale.numRight} right rows × dim $Dim across ${scale.numFragments} " +
          s"Spark partitions to $rightUri (${dataMode.label}) ...")
        val rightVecs = generateVectors(dataMode, scale.numRight, Dim, Seed + 1)
        val rightRows = (0 until scale.numRight).map { i =>
          RowFactory.create(Integer.valueOf(i + 1000000), rightVecs(i))
        }
        spark.createDataFrame(rightRows.asJava, rightSchema())
          .repartition(scale.numFragments)
          .write.format("lance").save(rightUri)
        spark.read.format("lance").load(rightUri).createOrReplaceTempView("docs")

        val sql =
          s"""SELECT q.lid, d.rid
             |FROM queries q INNER JOIN docs d
             |APPROX NEAREST $K BY DISTANCE vector_l2_distance(q.lvec, d.rvec)""".stripMargin

        // Cross-config validation: confirm rule ON returns the same top-K row IDs as rule OFF
        // (= Spark's RewriteNearestByJoin = brute-force ground truth) on a 16-row left
        // subset. Run before timing so the measured speedup is on equivalent results, not
        // on two paths that disagree on output. The check uses a separate `queries_small`
        // view (16 rows) so the rule-OFF cross-product is 16 × |R| (sub-second), not |L| × |R|.
        verifyRuleOnMatchesRuleOff(spark, leftRows)

        if (scale.runRuleOff) {
          spark.conf.set(IndexedNearestByJoinRule.EnabledConfKey, "false")
          val r =
            timeIt(scale.name, "A: rule OFF (Spark RewriteNearestByJoin)", () => spark.sql(sql))
          results += r
          println(formatResult(r))
        }

        spark.conf.set(IndexedNearestByJoinRule.EnabledConfKey, "true")
        val r =
          timeIt(scale.name, "B: rule ON  (no index, Lance brute-force scan)", () => spark.sql(sql))
        results += r
        println(formatResult(r))

        // Build the chosen vector index on the right dataset and time the rule-ON path again.
        // Lance's nearest-search auto-detects the index and switches to the indexed-scan code
        // path. Recall < 1.0 in general (approximate index), so we report recall@K rather than
        // strict-equality validation. numFragments is used as numPartitions so each partition
        // roughly maps to one Spark fragment.
        //
        // IVF_FLAT vs. IVF-PQ:
        //   - IVF_FLAT stores full vectors per cluster — exact distances within visited
        //     clusters, no PQ noise. Higher disk/memory cost but recovers high recall on
        //     high-dim or random workloads. The workaround when PQ noise dominates.
        //   - IVF-PQ compresses each vector into `Dim / 16` 8-bit sub-vectors. Smaller index
        //     and faster scan, but recall is highly sensitive to data distribution. On
        //     uniform-random high-dim data PQ collapses (~3% recall at defaults); on
        //     production-shaped clustered embeddings PQ codebook training latches onto natural
        //     structure and recall recovers. This is what most production deployments use.
        println(
          s"  Building ${indexMode.label} index (numPartitions=${scale.numFragments}, " +
            s"dim=$Dim) ...")
        val tIdx = System.nanoTime()
        indexMode match {
          case IndexFlat =>
            LanceVectorIndexBuilder.buildIvfFlat(
              datasetUri = rightUri,
              vectorColumn = "rvec",
              numPartitions = scale.numFragments)
          case IndexPq =>
            // numSubVectors trades index size + scan speed for code precision:
            //   - Dim/16 (8 at Dim=128): coarse PQ, 16 dims per sub-vector. Compact but very
            //     lossy — recall ~5–10% on dim-128 data even with clustered distribution.
            //     This is the default because it's the only setting Lance can train at our
            //     test scales (100K–1M rows). At 1M rows uniform, Lance rejects
            //     `numSubVectors=32` with "needs 4.3B training samples" — production
            //     deployments at much larger N can train fine PQ but we can't here.
            //   - Dim/4 (32 at Dim=128): fine PQ, 4 dims per sub-vector. The production-
            //     realistic setting; needs > a few B rows to train. Override via
            //     BENCHMARK_PQ_SUBVEC if you have a real dataset that supports it.
            val pqSubVec = sys.env
              .getOrElse("BENCHMARK_PQ_SUBVEC", math.max(1, Dim / 16).toString)
              .toInt
            println(s"  (PQ: numSubVectors=$pqSubVec, numBits=8 — ${Dim / pqSubVec} dims/code)")
            LanceVectorIndexBuilder.buildIvfPq(
              datasetUri = rightUri,
              vectorColumn = "rvec",
              numPartitions = scale.numFragments,
              numSubVectors = pqSubVec,
              numBits = 8)
        }
        println(s"  ... done in ${TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - tIdx)}s")

        // Default-tuning indexed run: nprobes default (1), refineFactor default (no re-rank).
        // For 100K rows × dim 128 with 4 IVF partitions this gives terrible recall (~3%) — PQ
        // compression noise dominates and only 1/4 of the dataset is visited. Demonstrates the
        // raw indexed-scan speedup but is unusable for real workloads.
        spark.conf.unset(IndexedNearestByJoinRule.NprobesConfKey)
        spark.conf.unset(IndexedNearestByJoinRule.RefineFactorConfKey)
        val rIndexed =
          timeIt(
            scale.name,
            s"C: rule ON  (${indexMode.label} indexed, defaults)",
            () => spark.sql(sql))
        results += rIndexed
        println(formatResult(rIndexed))
        val recallC = computeIndexedRecall(spark, leftRows)
        println(f"    -> recall@$K (indexed defaults, sample 16): $recallC%.3f")

        // Tuned indexed run #1: refineFactor = 8 alone. Fetches K * 8 PQ candidates *from the
        // probed clusters*, re-ranks with exact distance, returns top K. Sidesteps PQ noise but
        // can't recover true neighbors that live in unprobed clusters.
        spark.conf.set(IndexedNearestByJoinRule.RefineFactorConfKey, "8")
        val rTuned =
          timeIt(
            scale.name,
            s"D: rule ON  (${indexMode.label} + refineFactor=64)",
            () => spark.sql(sql))
        results += rTuned
        println(formatResult(rTuned))
        val recallD = computeIndexedRecall(spark, leftRows)
        println(f"    -> recall@$K (refineFactor=64, sample 16): $recallD%.3f")
        spark.conf.unset(IndexedNearestByJoinRule.RefineFactorConfKey)

        // Tuned indexed run #2: nprobes = numFragments. Visits every IVF cluster, recovering
        // true neighbors that the default nprobes=1 cuts away. Speedup degrades because we're
        // back to scanning roughly the whole dataset (just with extra IVF overhead), but
        // recall should approach 1.0.
        spark.conf.set(IndexedNearestByJoinRule.NprobesConfKey, scale.numFragments.toString)
        val rNprobes =
          timeIt(
            scale.name,
            s"E: rule ON  (${indexMode.label} + nprobes=${scale.numFragments})",
            () => spark.sql(sql))
        results += rNprobes
        println(formatResult(rNprobes))
        val recallE = computeIndexedRecall(spark, leftRows)
        println(f"    -> recall@$K (nprobes=${scale.numFragments}, sample 16): $recallE%.3f")

        // Tuned indexed run #3: nprobes = full + refineFactor = 8. Visits every cluster AND
        // re-ranks with exact distance. The high-recall configuration; the most expensive of
        // the indexed paths but still typically faster than rule OFF (Spark's brute-force
        // crossJoin) because Lance's native scan beats Catalyst per-pair overhead even when
        // the data volume is the same.
        spark.conf.set(IndexedNearestByJoinRule.RefineFactorConfKey, "64")
        val rFull = timeIt(
          scale.name,
          s"F: rule ON  (${indexMode.label} + nprobes=${scale.numFragments} + refineFactor=64)",
          () => spark.sql(sql))
        results += rFull
        println(formatResult(rFull))
        val recallF = computeIndexedRecall(spark, leftRows)
        println(
          f"    -> recall@$K (nprobes=${scale.numFragments} + refineFactor=64, sample 16): $recallF%.3f")
        spark.conf.unset(IndexedNearestByJoinRule.NprobesConfKey)
        spark.conf.unset(IndexedNearestByJoinRule.RefineFactorConfKey)

        // Drop the temp views so the next scale starts clean.
        spark.catalog.dropTempView("queries")
        spark.catalog.dropTempView("docs")
        println()
      }

      println(banner("Summary"))
      printSummaryTable(results.toSeq)
    } finally {
      spark.stop()
      deleteRecursively(tmpRoot.toFile)
    }
  }

  // -- timing harness ----------------------------------------------------------------------

  private val WarmupRuns = 1
  private val MeasurementRuns = 3

  /**
   * Execute the plan fully and discard output — Spark's canonical benchmark sink. Same
   * shape as the other two benchmarks. `count()` would skip result-row materialization
   * unequally (the crossJoin path skips it entirely; the indexed path still runs
   * `LanceMaterialize` due to the `references = child.outputSet` override), biasing the
   * speedup comparison. `noop` sink closes that gap.
   */
  private def runFull(df: DataFrame): Unit =
    df.write.format("noop").mode("overwrite").save()

  private def timeIt(scale: String, config: String, f: () => DataFrame): Result = {
    print(s"  $config ... ")
    System.out.flush()
    var i = 0
    while (i < WarmupRuns) { runFull(f()); i += 1 }
    val runs = (0 until MeasurementRuns).map { _ =>
      val t0 = System.nanoTime()
      runFull(f())
      TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0)
    }
    val median = runs.sorted.apply(runs.size / 2)
    println(s"runs=${runs.mkString("[", ",", "]")} ms, median=$median ms")
    Result(scale, config, median, runs)
  }

  // -- recall measurement -------------------------------------------------------------

  /**
   * Measure recall@K of the indexed rule-ON path on a 16-row left subset, with rule-OFF
   * (Spark `RewriteNearestByJoin` = brute-force) as the ground truth. Recall is the average
   * fraction of the brute-force top-K row IDs that the indexed path also returned.
   */
  private def computeIndexedRecall(
      spark: SparkSession,
      allLeftRows: Seq[org.apache.spark.sql.Row]): Double = {
    val sample = allLeftRows.take(16)
    spark.createDataFrame(sample.asJava, leftSchema()).createOrReplaceTempView("queries_recall")
    val sampleLids = sample.map(_.getInt(0)).toSet
    val sql =
      s"""SELECT q.lid, d.rid
         |FROM queries_recall q INNER JOIN docs d
         |APPROX NEAREST $K BY DISTANCE vector_l2_distance(q.lvec, d.rvec)""".stripMargin

    spark.conf.set(IndexedNearestByJoinRule.EnabledConfKey, "false")
    val truthRows = spark.sql(sql).collect()
    val truth = truthRows.groupBy(_.getAs[Int]("lid"))
      .map { case (lid, rs) => lid -> rs.map(_.getAs[Int]("rid")).toSet }

    spark.conf.set(IndexedNearestByJoinRule.EnabledConfKey, "true")
    val indexedRows = spark.sql(sql).collect()
    val indexed = indexedRows.groupBy(_.getAs[Int]("lid"))
      .map { case (lid, rs) => lid -> rs.map(_.getAs[Int]("rid")).toSet }

    val perLidRecall = sampleLids.toSeq.map { lid =>
      val truthSet = truth.getOrElse(lid, Set.empty[Int])
      val indexedSet = indexed.getOrElse(lid, Set.empty[Int])
      if (truthSet.isEmpty) 0.0
      else (truthSet intersect indexedSet).size.toDouble / truthSet.size
    }
    spark.catalog.dropTempView("queries_recall")
    perLidRecall.sum / perLidRecall.size
  }

  // -- cross-config validation -------------------------------------------------------------

  /**
   * Run the same SQL twice on a 16-row left subset — once with rule OFF (Spark's
   * RewriteNearestByJoin = brute-force cross-product + min_by_k = exact ground truth on
   * no-index Lance), once with rule ON (our 3-exec staged chain). Compare top-K row IDs
   * per left row. Bail if they disagree.
   *
   * Uses a separate `queries_small` view so the rule-OFF cross-product is 16 × |R|
   * (sub-second), not |L| × |R| which would dominate wall-clock at medium scale.
   * Compared as Sets to tolerate tied-distance ordering.
   */
  private def verifyRuleOnMatchesRuleOff(
      spark: SparkSession,
      allLeftRows: Seq[org.apache.spark.sql.Row]): Unit = {
    println("  Sanity check: rule ON top-K matches rule OFF on a 16-row left subset ...")
    val sample = allLeftRows.take(16)
    spark.createDataFrame(sample.asJava, leftSchema()).createOrReplaceTempView("queries_small")
    // RowFactory.create() makes schema-less Rows, so getAs[String] doesn't work; use positional.
    val sampleLids = sample.map(_.getInt(0)).toSet
    val verifySql =
      s"""SELECT q.lid, d.rid
         |FROM queries_small q INNER JOIN docs d
         |APPROX NEAREST $K BY DISTANCE vector_l2_distance(q.lvec, d.rvec)""".stripMargin

    spark.conf.set(IndexedNearestByJoinRule.EnabledConfKey, "false")
    val offRows = spark.sql(verifySql).collect()
    val offMap = offRows.groupBy(_.getAs[Int]("lid"))
      .map { case (lid, rs) => lid -> rs.map(_.getAs[Int]("rid")).toSet }

    spark.conf.set(IndexedNearestByJoinRule.EnabledConfKey, "true")
    val onRows = spark.sql(verifySql).collect()
    val onMap = onRows.groupBy(_.getAs[Int]("lid"))
      .map { case (lid, rs) => lid -> rs.map(_.getAs[Int]("rid")).toSet }

    sampleLids.foreach { lid =>
      val offSet = offMap.getOrElse(lid, Set.empty[Int])
      val onSet = onMap.getOrElse(lid, Set.empty[Int])
      if (offSet != onSet) {
        sys.error(
          s"RULE ON/OFF MISMATCH at lid=$lid:\n  rule OFF: $offSet\n  rule ON:  $onSet")
      }
    }
    spark.catalog.dropTempView("queries_small")
    println(s"  ... rule ON and rule OFF agree on top-K (sample size: ${sampleLids.size}).")
  }

  // -- output formatting ------------------------------------------------------------------

  private def banner(s: String): String = s"\n=== $s " + ("=" * (76 - s.length - 5))

  private def formatResult(r: Result): String =
    f"    -> ${r.config}%-44s  median=${r.medianMs}%6d ms"

  private def printSummaryTable(results: Seq[Result]): Unit = {
    val byScale = results.groupBy(_.scale)
    val scaleOrder = Seq("small", "medium").filter(byScale.contains)
    val configWidth = 44
    val numWidth = 13
    val divider = "-" * (configWidth + scaleOrder.size * numWidth)
    val header = s"%-${configWidth}s" + scaleOrder.map(_ => s"%${numWidth}s").mkString
    println(divider)
    println(header.format(("Configuration" +: scaleOrder.map(s => s"$s (ms)")): _*))
    println(header.format(("" +: scaleOrder.map(_ => "speedup ×")): _*))
    println(divider)
    val configs = results.map(_.config).distinct
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
    println("Same SQL: APPROX NEAREST K BY DISTANCE vector_l2_distance(q.lvec, d.rvec).")
    println("Rule OFF lowers to Spark's RewriteNearestByJoin (cross-product + min_by_k).")
    println("Rule ON  routes through the 3-exec staged pipeline (LanceProbe → LanceMerge → LanceMaterialize).")
  }

  // -- schemas + helpers ------------------------------------------------------------------

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

  private def randomVector(rng: Random, dim: Int): Array[Float] = {
    val v = new Array[Float](dim)
    var i = 0
    while (i < dim) { v(i) = rng.nextFloat(); i += 1 }
    v
  }

  /**
   * Vector generator dispatch. Uniform mode keeps the existing baseline (independent
   * `nextFloat`s — IVF's worst case at high dim because pairwise distances cluster around
   * a narrow range). Clustered mode draws unit-sphere-normalized Gaussian-mixture vectors
   * around `NumClusters` centers — the geometry of typical sentence-transformer / image-
   * feature embeddings that IVF was actually designed for.
   */
  private def generateVectors(
      mode: DataMode,
      n: Int,
      dim: Int,
      seed: Long): Array[Array[Float]] = mode match {
    case DataUniform =>
      val rng = new Random(seed)
      Array.fill(n)(randomVector(rng, dim))
    case DataClustered =>
      // sigma is in units of inter-cluster spacing. Override via BENCHMARK_SIGMA — tighter
      // clusters (e.g. 0.05) approximate real semantic embeddings where intra-cluster variance
      // is small relative to inter-cluster separation. Default 0.15 is the moderate setting.
      val sigma = sys.env.getOrElse("BENCHMARK_SIGMA", "0.15").toDouble
      generateClusteredVectors(n, dim, NumClusters, sigma = sigma, seed = seed)
  }

  /**
   * Inlined clustered-Gaussian-mixture generator. Lives here (rather than reused from
   * `lance-spark-knn_2.12`'s test util) because that helper is in test scope of a different
   * module and isn't visible across module test scopes. Logic mirrors
   * `org.lance.spark.knn.testutil.ClusteredEmbeddings`:
   *
   *   1. Pick `numClusters` centers uniformly on [0, 1]^dim.
   *   2. For each row, round-robin a cluster, sample N(center, sigma * sep) where `sep`
   *      is the median pairwise distance between centers (so sigma is in units of
   *      inter-cluster spacing — stable across (dim, numClusters) settings).
   *   3. L2-normalize. Production embeddings live on the unit sphere; that's the geometry
   *      IVF expects.
   */
  private def generateClusteredVectors(
      n: Int,
      dim: Int,
      numClusters: Int,
      sigma: Double,
      seed: Long): Array[Array[Float]] = {
    val rng = new Random(seed)
    val centers = Array.fill(numClusters)(Array.fill(dim)(rng.nextDouble()))
    val sep = medianPairwiseDistance(centers)
    val scaledSigma = sigma * sep
    val out = new Array[Array[Float]](n)
    var i = 0
    while (i < n) {
      val center = centers(i % numClusters)
      val v = new Array[Float](dim)
      var d = 0
      while (d < dim) {
        v(d) = (center(d) + rng.nextGaussian() * scaledSigma).toFloat
        d += 1
      }
      l2Normalize(v)
      out(i) = v
      i += 1
    }
    out
  }

  private def medianPairwiseDistance(centers: Array[Array[Double]]): Double = {
    val k = centers.length
    if (k < 2) return 1.0
    val rng = new Random(0L)
    val numPairs = math.min(1024, k * (k - 1) / 2)
    val dists = new Array[Double](numPairs)
    var p = 0
    while (p < numPairs) {
      var i = rng.nextInt(k)
      var j = rng.nextInt(k)
      while (j == i) j = rng.nextInt(k)
      var s = 0.0
      var d = 0
      while (d < centers(i).length) {
        val diff = centers(i)(d) - centers(j)(d)
        s += diff * diff
        d += 1
      }
      dists(p) = math.sqrt(s)
      p += 1
    }
    java.util.Arrays.sort(dists)
    dists(dists.length / 2)
  }

  private def l2Normalize(v: Array[Float]): Unit = {
    var s = 0.0
    var i = 0
    while (i < v.length) { s += v(i) * v(i); i += 1 }
    val norm = math.sqrt(s).toFloat
    if (norm > 0f) {
      i = 0
      while (i < v.length) { v(i) = v(i) / norm; i += 1 }
    }
  }

  private def deleteRecursively(f: java.io.File): Unit = {
    if (f.isDirectory) f.listFiles().foreach(deleteRecursively)
    f.delete()
  }
}
