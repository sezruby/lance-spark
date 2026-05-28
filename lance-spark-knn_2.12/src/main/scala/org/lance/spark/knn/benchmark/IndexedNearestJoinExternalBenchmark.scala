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

import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.linalg.{Vector => MLVector, Vectors}
import org.apache.spark.sql.{DataFrame, Row, RowFactory, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.lance.index.external.ExternalIvfPqIndexParams
import org.lance.spark.knn.IndexedNearestJoinExternal
import org.lance.spark.knn.LanceKnnImplicits._
import org.lance.spark.knn.internal.{LanceVectorIndexBuilder, Metric => InternalMetric}

import java.nio.file.{Files, Paths}
import java.util.{Locale, Random}
import java.util.concurrent.TimeUnit

import scala.collection.JavaConverters._

/**
 * Compares three paths for indexed kNN-by-join when R is parquet on disk:
 *
 *   A: vanilla Spark crossJoin + L2 UDF + min_by_k — the brute-force baseline a user
 *      would write today on parquet R.
 *   B: per-query temp Lance write + `kNearestJoin` against the temp URI (PR #3 path,
 *      [[IndexedNearestJoinTempRBenchmark]] config B). The general-purpose path that
 *      handles arbitrary subqueries.
 *   E: external Lance vector index over the same parquet files +
 *      [[IndexedNearestJoinExternal]] (the new path under test). Build the index once
 *      per data shape (cached across runs in this benchmark via [[org.lance.spark.knn.internal.ExternalIndexLifecycle]]),
 *      then probe + refine + post-topK fetchRows from the source parquet — no temp write.
 *
 * Three configs answer:
 *
 *   1. Does E beat A on parquet R? (E vs A speedup — the RFC's user-facing case for
 *      external-index existing at all)
 *   2. Does E beat B at any scale? (E vs B — when does avoiding the temp write +
 *      using post-topK fetchRows pay off vs. the general path that copies all R cols)
 *   3. How does the index build cost amortize? Reported as a separate first-run number;
 *      subsequent runs hit the cache and approximate "warm" steady-state.
 *
 * == Local run ==
 * {{{
 *   MAVEN_OPTS="-Xmx12g <JDK 17 add-opens flags>" \
 *     ./mvnw -pl lance-spark-knn_2.12 -q exec:java -Pbenchmark \
 *     -Dexec.mainClass="org.lance.spark.knn.benchmark.IndexedNearestJoinExternalBenchmark"
 * }}}
 *
 * Same env vars as [[IndexedNearestJoinTempRBenchmark]]: BENCH_SCALES, BENCH_REPEATS,
 * BENCH_CLUSTER_MODE, BENCH_DATA_PATH.
 */
object IndexedNearestJoinExternalBenchmark {

  private val K: Int = 10
  private val Seed: Long = 1337L

  /**
   * `numPayloadCols` controls the WIDTH of R beyond the `rid` and `rvec` columns. Each
   * payload column is a 64-byte UTF-8 string, so wide R has substantial column-copy cost.
   *
   * The join only projects `rid` for the materialize step; B (temp-Lance) still writes
   * ALL columns to its temp dataset (that's how it's designed — the temp is a generic
   * Lance dataset that downstream stages re-read), while E (external-index) reads only
   * the projection columns from source parquet via `fetchRows`. The gap (B - E) is the
   * cost of the temp-write column copy that external-index avoids.
   */
  private case class Scale(
      name: String,
      numR: Int,
      numL: Int,
      dim: Int,
      numPayloadCols: Int) {
    def vectorBytesR: Long = numR.toLong * dim.toLong * 4L
    def payloadBytesR: Long = numR.toLong * numPayloadCols.toLong * 64L
    override def toString: String =
      f"$name (|R|=$numR%,d, |L|=$numL%,d, dim=$dim, payload=${numPayloadCols} cols, " +
        f"R-vec=${vectorBytesR / (1024.0 * 1024.0)}%.0f MB, " +
        f"R-payload=${payloadBytesR / (1024.0 * 1024.0)}%.0f MB)"
  }

  // Wide-R scales designed to expose the temp-Lance vs external-index materialize cost gap.
  // For each scale the join asks for ONLY `rid` post-topK; B copies all columns to temp
  // anyway, E reads only `rid` from the source parquet.
  private val Scales: Map[String, Scale] = Seq(
    // narrow / fast: matches the original temp-R bench shape for sanity comparison
    Scale("narrow-tiny", numR = 100000, numL = 100, dim = 128, numPayloadCols = 0),
    // wide payload starts surfacing the gap: 16 string cols × 100K rows = ~100 MB extra
    Scale("wide-tiny", numR = 100000, numL = 100, dim = 128, numPayloadCols = 16),
    // 1M rows × 16 wide cols = ~1 GB extra payload — temp write becomes the dominant cost
    Scale("wide-medium", numR = 1000000, numL = 100, dim = 128, numPayloadCols = 16),
    // 1M rows × 64 wide cols ≈ 4 GB extra — realistic enterprise shape
    Scale("wide-large", numR = 1000000, numL = 100, dim = 128, numPayloadCols = 64),
    // Long-run target — slowest config aimed at ~2-3 min so the gap is unambiguous
    // even under cluster noise. 10M rows × 16 cols ≈ 15 GB total R; 1000 queries.
    // CAUTION: requires substantial scratch volume (≥30 GB free across temp Lance +
    // native Lance narrow + native Lance wide + external index). Cluster runs have
    // hit "Disk quota exceeded" on shared 100 GB volumes. Opt in only when scratch
    // capacity is verified. Default scales below skip this.
    Scale("mega-medium", numR = 10000000, numL = 1000, dim = 128, numPayloadCols = 16))
    .map(s => s.name -> s)
    .toMap

  private case class Result(
      scale: String,
      config: String,
      indexBuildMs: Option[Long],
      totalMs: Long,
      runs: Seq[Long]) {
    def queryMs: Option[Long] = indexBuildMs.map(b => totalMs - b)
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
    val dataRoot =
      dataRootOpt.getOrElse(Files.createTempDirectory("knn-external-bench-").toString)
    // Optional config gate: BENCH_CONFIGS=b-narrow,e (or any subset). If unset, all
    // configs run. Useful for narrow comparisons at large scale where running the full
    // suite would take 30+ min.
    val activeConfigs: Set[String] = sys.env.get("BENCH_CONFIGS") match {
      case Some(s) if s.nonEmpty =>
        s.toLowerCase(Locale.ROOT).split(",").map(_.trim).filter(_.nonEmpty).toSet
      case _ => Set("a", "b-narrow", "b-wide", "c-narrow", "c-wide", "e", "f")
    }

    val builder = SparkSession
      .builder()
      .appName("indexed-nearest-external-benchmark")
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

    // Conf for both the temp-R path (PR #3) and the external-index lifecycle. Re-use
    // the bench's BENCH_DATA_PATH for both scratch roots in cluster mode so a single
    // env var configures everything.
    if (clusterMode) {
      spark.conf.set("spark.lance.knn.tempR.dir", s"$dataRoot/temp-r-scratch")
      spark.conf.set("spark.lance.knn.externalIndex.dir", s"$dataRoot/external-idx-scratch")
    }

    // Cluster scratch volumes have limited quotas and the in-process LanceTempLifecycle
    // cleanup only fires for the CURRENT run. Old runs from prior submissions leave
    // multi-GB scratch dirs behind, eventually triggering "Disk quota exceeded" at
    // write time. Sweep sibling `knn-bench-data-*` directories before this run starts
    // to keep the volume clean. Only sweep when (a) clusterMode and (b) dataRoot
    // matches the cpd-submit-bench naming pattern (so we never accidentally delete
    // something else).
    if (clusterMode) {
      cleanupSiblingScratchDirs(dataRoot)
    }

    println("=" * 96)
    println("External-index benchmark — parquet R: crossJoin vs temp-Lance vs external Lance index")
    println("=" * 96)
    val masterDesc = if (clusterMode) "cluster (BENCH_CLUSTER_MODE=true)" else "local[*]"
    println(f"Spark master:   $masterDesc (cores=${spark.sparkContext.defaultParallelism})")
    println(f"Repeats:        $repeats (median reported); 1 warmup")
    println(f"Data root:      $dataRoot")
    println(f"K:              $K")
    println(f"Scales:         ${scales.map(_.name).mkString(", ")}")
    println()

    // Cluster health gate: probe every task slot with a fixed-cost CPU loop and print
    // per-executor timings. Outliers indicate noisy neighbors / pinned cores / thermal
    // throttling and make config-vs-config medians unreliable. With
    // BENCH_CPU_CHECK_FAIL_RATIO set, refuses to proceed when max/median exceeds the
    // ratio. With BENCH_CPU_CHECK_SKIP=true, skips entirely.
    if (!sys.env.get("BENCH_CPU_CHECK_SKIP").exists(_.equalsIgnoreCase("true"))) {
      val failRatio = sys.env.get("BENCH_CPU_CHECK_FAIL_RATIO").map(_.toDouble)
      ExecutorCpuCheck.run(spark, failRatio)
    }

    val results = scala.collection.mutable.ArrayBuffer.empty[Result]
    try {
      scales.foreach { scale =>
        println("-" * 96)
        println(s"Scale: $scale")
        println("-" * 96)

        // Build left in memory, right as a single-file parquet on disk. The external-index
        // path takes explicit file paths, so we coalesce(1) to control the file count.
        val leftDf = buildLeft(spark, scale).cache()
        leftDf.count()
        val rightParquetDir = s"$dataRoot/${scale.name}_right_parquet_dir"
        writeRightParquet(spark, scale, rightParquetDir, parts = 1)
        val rightDfParquet = spark.read.parquet(rightParquetDir)
        val rightFilePaths: Seq[String] = listParquetFiles(rightParquetDir)

        // Pre-write Lance-native R once (outside the timing loop) for configs C-*.
        // Skipped when neither C-narrow nor C-wide is active — the writes alone take
        // ~30s at wide-medium and we don't want to pay for them when running B vs E only.
        val needCNarrow = activeConfigs.contains("c-narrow")
        val needCWide = activeConfigs.contains("c-wide") && scale.numPayloadCols > 0
        val cIndexNumPartitions = math.min(256, math.max(8, scale.numR / 1024))
        val cIndexNumSubVectors = math.min(scale.dim / 4, 16)

        val rightDfLanceNarrow: Option[DataFrame] = if (needCNarrow) {
          val uri = s"$dataRoot/${scale.name}_right_native_narrow_lnc"
          spark.read.parquet(rightParquetDir)
            .select("rid", "rvec")
            .write
            .format("lance")
            .save(uri)
          val cIndexBuildStart = System.nanoTime()
          LanceVectorIndexBuilder.buildIvfPq(
            datasetUri = uri,
            vectorColumn = "rvec",
            numPartitions = cIndexNumPartitions,
            numSubVectors = cIndexNumSubVectors,
            numBits = 8,
            metric = InternalMetric.L2,
            maxIters = 50)
          val ms = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - cIndexBuildStart)
          println(f"  C-indexed-narrow: index built in $ms%d ms")
          Some(spark.read.format("lance").load(uri))
        } else None

        val rightDfLanceWide: Option[DataFrame] = if (needCWide) {
          val uri = s"$dataRoot/${scale.name}_right_native_wide_lnc"
          val payloadCols = (0 until scale.numPayloadCols).map(i => s"payload_$i")
          val widePayloadCols = ("rid" +: "rvec" +: payloadCols).map(col)
          spark.read.parquet(rightParquetDir)
            .select(widePayloadCols: _*)
            .write
            .format("lance")
            .save(uri)
          val t0 = System.nanoTime()
          LanceVectorIndexBuilder.buildIvfPq(
            datasetUri = uri,
            vectorColumn = "rvec",
            numPartitions = cIndexNumPartitions,
            numSubVectors = cIndexNumSubVectors,
            numBits = 8,
            metric = InternalMetric.L2,
            maxIters = 50)
          val ms = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0)
          println(f"  C-indexed-wide:   index built in $ms%d ms")
          Some(spark.read.format("lance").load(uri))
        } else None

        // ---- Config A: crossJoin baseline (skipped at large scales / wide payload) ----
        // The brute-force baseline is O(|L| × |R|) pair evaluations regardless of payload
        // width, but the parquet read still materializes payload cols across the crossJoin
        // so wide scales make it very slow without adding info. Skip when payload > 0
        // unless explicitly forced via BENCH_INCLUDE_BASELINE=true.
        val includeBaseline = scale.numPayloadCols == 0 ||
          sys.env.get("BENCH_INCLUDE_BASELINE").exists(_.equalsIgnoreCase("true"))
        if (includeBaseline) {
          val baselineRepeats = if (scale.numL.toLong * scale.numR > 100000000L) 1 else repeats
          val resultA =
            timeIt(scale.name, "A: Spark crossJoin + min_by_k (parquet R)", baselineRepeats) {
              () => crossJoinMinByK(leftDf, rightDfParquet, K)
            }
          results += resultA
        } else {
          println(s"  A: Spark crossJoin baseline ... skipped (wide payload; set BENCH_INCLUDE_BASELINE=true to force)")
        }

        // ---- Config B-narrow: PR #3 per-query temp Lance write, narrow projection ----
        if (activeConfigs.contains("b-narrow")) {
          val resultBNarrow =
            timeIt(scale.name, "B-narrow: temp-Lance + kNJ (project rid only)", repeats) { () =>
              leftDf.kNearestJoin(
                right = rightDfParquet,
                leftVecCol = "lvec",
                rightVecCol = "rvec",
                k = K,
                metric = "l2",
                rightProjection = Some(Seq("rid")),
                probeParallelism = 1)
            }
          results += resultBNarrow
        }

        // ---- Config B-wide: PR #3 with WIDE projection — all payload columns -------
        if (activeConfigs.contains("b-wide") && scale.numPayloadCols > 0) {
          val widePayload =
            "rid" +: (0 until scale.numPayloadCols).map(i => s"payload_$i")
          val resultBWide = timeIt(
            scale.name,
            s"B-wide: temp-Lance + kNJ (project rid+${scale.numPayloadCols} payload)",
            repeats) { () =>
            leftDf.kNearestJoin(
              right = rightDfParquet,
              leftVecCol = "lvec",
              rightVecCol = "rvec",
              k = K,
              metric = "l2",
              rightProjection = Some(widePayload),
              probeParallelism = 1)
          }
          results += resultBWide
        }

        // ---- Config C-indexed-narrow: Lance-native R + IVF-PQ index, project rid -----
        rightDfLanceNarrow.foreach { lanceNarrowDf =>
          val resultCNarrow =
            timeIt(
              scale.name,
              "C-indexed-narrow: Lance-native R (indexed) + kNJ (project rid)",
              repeats) { () =>
              leftDf.kNearestJoin(
                right = lanceNarrowDf,
                leftVecCol = "lvec",
                rightVecCol = "rvec",
                k = K,
                metric = "l2",
                rightProjection = Some(Seq("rid")),
                probeParallelism = 1)
            }
          results += resultCNarrow
        }

        // ---- Config C-indexed-wide: Lance-native R + IVF-PQ index, project rid + payload
        rightDfLanceWide.foreach { lanceWideDf =>
          val widePayload = "rid" +: (0 until scale.numPayloadCols).map(i => s"payload_$i")
          val resultCWide = timeIt(
            scale.name,
            s"C-indexed-wide: Lance-native R (indexed) + kNJ (project rid+${scale.numPayloadCols})",
            repeats) { () =>
            leftDf.kNearestJoin(
              right = lanceWideDf,
              leftVecCol = "lvec",
              rightVecCol = "rvec",
              k = K,
              metric = "l2",
              rightProjection = Some(widePayload),
              probeParallelism = 1)
          }
          results += resultCWide
        }

        // ---- Config E: external Lance index over parquet ----------------------------
        // ---- Config E: external Lance index over parquet ----------------------------
        if (activeConfigs.contains("e")) {
          val params = ExternalIvfPqIndexParams.builder()
            .numPartitions(math.min(256, math.max(8, scale.numR / 1024)))
            .numSubVectors(math.min(scale.dim / 4, 16))
            .numBitsPerSubVector(8)
            .metric(ExternalIvfPqIndexParams.Metric.L2)
            .build()
          val resultE =
            timeWithBuild(scale.name, "E: external Lance index + kNearestJoinExternal", repeats) {
              () =>
                IndexedNearestJoinExternal(
                  left = leftDf,
                  rightFilePaths = rightFilePaths,
                  leftVecCol = "lvec",
                  rightVecCol = "rvec",
                  k = K,
                  metric = "l2",
                  rightProjection = Some(Seq("rid")),
                  indexParams = Some(params))
            }
          results += resultE
        }

        // ---- Config F: Spark MLlib BucketedRandomProjectionLSH (L2) -----------------
        val skipLsh = sys.env.get("BENCH_SKIP_LSH").exists(_.equalsIgnoreCase("true"))
        if (activeConfigs.contains("f") && !skipLsh) {
          val resultF =
            timeWithBuild(scale.name, "F: MLlib BucketedRandomProjectionLSH + topK", repeats) {
              () => lshKnnJoin(spark, leftDf, rightDfParquet, scale.dim, K)
            }
          results += resultF
        } else if (activeConfigs.contains("f")) {
          println(s"  F: MLlib LSH ... skipped (BENCH_SKIP_LSH=true)")
        }

        leftDf.unpersist()
        // Clear the lifecycle cache between scales so each scale's first run includes
        // an honest build cost — different file paths anyway, but be explicit.
        org.lance.spark.knn.internal.ExternalIndexLifecycle.clearCacheForTesting()
        println()
      }

      println("=" * 96)
      println("Summary")
      println("=" * 96)
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

  private def writeRightParquet(
      spark: SparkSession,
      scale: Scale,
      uri: String,
      parts: Int): Unit = {
    val schema = rightSchema(scale.dim, scale.numPayloadCols)
    val effectiveParts = math.max(1, parts)
    val numR = scale.numR
    val dim = scale.dim
    val numPayloadCols = scale.numPayloadCols
    val rdd = spark.sparkContext
      .range(0L, numR.toLong, 1L, math.max(spark.sparkContext.defaultParallelism, 8))
      .mapPartitionsWithIndex { (idx, iter) =>
        val rng = new Random(0xCAFEBABEL ^ idx.toLong)
        iter.map { i =>
          val v = new Array[Float](dim)
          var k = 0
          while (k < dim) { v(k) = rng.nextFloat(); k += 1 }
          // Each payload column is a deterministic 64-byte string. Same length per row so
          // the temp Lance write has predictable cost per column.
          val payloads: Array[AnyRef] = (0 until numPayloadCols).map { col =>
            val seed = (i.toLong << 16) | col.toLong
            val s = f"$seed%016x" + "x" * 48 // 64 chars total
            s: AnyRef
          }.toArray
          val cells: Array[AnyRef] = Array(Integer.valueOf(i.toInt), v) ++ payloads
          RowFactory.create(cells: _*): Row
        }
      }
    spark
      .createDataFrame(rdd, schema)
      .coalesce(effectiveParts)
      .write
      .mode("overwrite")
      .parquet(uri)
  }

  /**
   * Sweep stale sibling `knn-bench-data-*` directories from `dataRoot`'s parent dir
   * before this run starts. Defends against the "Disk quota exceeded" failure mode
   * on cluster scratch volumes when prior bench runs left their scratch behind.
   *
   * Strict matching: only deletes siblings whose name starts with `knn-bench-data-`
   * (the cpd-submit-bench.sh naming pattern). Skips this run's own dataRoot. If the
   * parent dir doesn't exist or this run's path doesn't fit the pattern, no-op.
   */
  private def cleanupSiblingScratchDirs(dataRoot: String): Unit = {
    val localPath =
      if (dataRoot.startsWith("file://")) dataRoot.stripPrefix("file://") else dataRoot
    val rootPath = Paths.get(localPath)
    val name = Option(rootPath.getFileName).map(_.toString).getOrElse("")
    if (!name.startsWith("knn-bench-data-")) {
      println(
        s"[cleanup] dataRoot $dataRoot doesn't match knn-bench-data-* pattern; " +
          "skipping sibling sweep")
      return
    }
    val parent = rootPath.getParent
    if (parent == null || !Files.exists(parent)) {
      return
    }
    val it = Files.list(parent)
    try {
      val deleted = scala.collection.mutable.ArrayBuffer.empty[String]
      val errors = scala.collection.mutable.ArrayBuffer.empty[String]
      it.iterator().asScala.foreach { p =>
        val pname = Option(p.getFileName).map(_.toString).getOrElse("")
        if (pname.startsWith("knn-bench-data-") && p != rootPath) {
          try {
            // Recursive delete via Files.walk + reverse order.
            val walk = Files.walk(p)
            try {
              walk
                .iterator()
                .asScala
                .toSeq
                .reverse
                .foreach { q =>
                  try Files.deleteIfExists(q)
                  catch { case _: Throwable => /* best effort */ }
                }
            } finally walk.close()
            deleted += pname
          } catch {
            case e: Throwable =>
              errors += s"$pname: ${e.getMessage}"
          }
        }
      }
      if (deleted.nonEmpty) {
        println(s"[cleanup] swept ${deleted.size} stale scratch dirs: ${deleted.mkString(", ")}")
      }
      if (errors.nonEmpty) {
        println(
          s"[cleanup] errors during sweep (best-effort, continuing): ${errors.mkString("; ")}")
      }
    } finally it.close()
  }

  private def listParquetFiles(dir: String): Seq[String] = {
    // Strip the "file://" scheme so java.nio.file.Paths can read the directory.
    // For non-file schemes (s3://, abfss://, hdfs://) this would need a Hadoop FileSystem
    // listing — but the external-index API takes plain file paths anyway, and the cluster
    // runs use `file:///valve-binaries/...` (a shared mount). If we ever support cloud
    // sources directly, switch to `org.apache.hadoop.fs.FileSystem.get(uri).listStatus`.
    val localDir = if (dir.startsWith("file://")) dir.stripPrefix("file://") else dir
    val p = Paths.get(localDir)
    val it = Files.list(p)
    try {
      it.iterator().asScala.toSeq
        .filter(f => f.toString.endsWith(".parquet"))
        .map(_.toString)
        .sorted
    } finally it.close()
  }

  private def leftSchema(dim: Int): StructType = new StructType(
    Array(
      StructField("lid", IntegerType, nullable = false),
      StructField(
        "lvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", dim.toLong).build())))

  private def rightSchema(dim: Int, numPayloadCols: Int): StructType = {
    val core = Array(
      StructField("rid", IntegerType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", dim.toLong).build()))
    val payload = (0 until numPayloadCols).map { i =>
      StructField(s"payload_$i", StringType, nullable = false)
    }.toArray
    new StructType(core ++ payload)
  }

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

  /**
   * Spark MLlib `BucketedRandomProjectionLSH` baseline (L2). The realistic non-Lance
   * answer for users who don't want to write a Lance dataset. Builds the LSH model
   * over R, runs `approxSimilarityJoin(L, R, threshold)` to get candidate (l, r) pairs
   * by hash-bucket collisions, computes exact L2 on each pair, takes top-K per L row.
   *
   * == Knob choices ==
   *
   * - `bucketLength`: heuristic 2.0 — typical LSH guidance for L2 with normalized-ish
   *   vectors. Smaller = stricter buckets = more recall, more candidates, slower.
   * - `numHashTables`: 5 — common starting point. More tables = better recall, more shuffle.
   * - `threshold`: 1e9 (effectively no threshold). LSH's threshold is on the bucket
   *   distance, not the final top-K. We let everything through and rely on the
   *   post-filter for ranking. A small threshold would speed it up at recall cost.
   *
   * The cost profile is dominated by `approxSimilarityJoin`'s explode-by-hash-collision
   * step, which on wide R is the LSH equivalent of B-narrow's temp-Lance write — both
   * pay a per-R-row cost up-front.
   */
  private def lshKnnJoin(
      spark: SparkSession,
      left: DataFrame,
      right: DataFrame,
      dim: Int,
      k: Int): DataFrame = {
    // Both DataFrames must use the SAME input column name for approxSimilarityJoin to
    // work — `BucketedRandomProjectionLSHModel.transform` looks up the column by the
    // name configured at fit time. Use "vec" on both sides.
    val toMlVec = udf((arr: Seq[Float]) => Vectors.dense(arr.toArray.map(_.toDouble)))
    val r = right.select(col("rid"), toMlVec(col("rvec")).as("vec"))
    val l = left.select(col("lid"), toMlVec(col("lvec")).as("vec"))

    val lsh = new BucketedRandomProjectionLSH()
      .setBucketLength(2.0)
      .setNumHashTables(5)
      .setInputCol("vec")
      .setOutputCol("hashes")
    val model = lsh.fit(r)

    // approxSimilarityJoin returns (datasetA, datasetB, distCol) where distCol holds
    // the EXACT distance between vectors that collided in some hash bucket. We
    // re-derive top-K per left row.
    val similarityThreshold = 1e9
    val pairs = model.approxSimilarityJoin(l, r, similarityThreshold, "__dist")
    pairs
      .select(
        col("datasetA.lid").as("lid"),
        col("datasetB.rid").as("rid"),
        col("__dist"))
      .groupBy("lid")
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
    Result(scale, config, indexBuildMs = None, totalMs = median, runs = runs)
  }

  /**
   * Time a path where the FIRST run also builds an index. Reports first-run total +
   * subsequent-run median so build amortization is visible.
   */
  private def timeWithBuild(scale: String, config: String, repeats: Int)(
      f: () => DataFrame): Result = {
    print(s"  $config ... ")
    System.out.flush()

    val firstStart = System.nanoTime()
    runFull(f())
    val firstMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - firstStart)

    val warmRuns = (0 until repeats).map { _ =>
      val t0 = System.nanoTime()
      runFull(f())
      TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0)
    }
    val sortedWarm = warmRuns.sorted
    val medianWarm = if (sortedWarm.isEmpty) firstMs else sortedWarm(sortedWarm.length / 2)
    val approxBuildMs = math.max(0L, firstMs - medianWarm)
    println(
      f"first(build+query)=$firstMs%d ms, warm runs=${warmRuns.mkString("[", ",", "]")} ms, " +
        f"median warm=$medianWarm%d ms, approx build=$approxBuildMs%d ms")
    Result(scale, config, indexBuildMs = Some(approxBuildMs), totalMs = medianWarm, runs = warmRuns)
  }

  // -- reporting ---------------------------------------------------------------------------

  private def printSummary(results: Seq[Result]): Unit = {
    val byScale = results.groupBy(_.scale)
    println(
      f"${"scale"}%-8s  ${"config"}%-50s  ${"med ms"}%8s  ${"build ms"}%9s  ${"vs A"}%6s  ${"vs B"}%6s")
    println("-" * 100)
    val scaleOrder = Scales.keys.toSeq.filter(byScale.contains).sortBy(Scales(_).numR)
    scaleOrder.foreach { sc =>
      val rs = byScale(sc)
      val baselineA = rs.find(_.config.startsWith("A:")).map(_.totalMs).getOrElse(0L)
      // For "vs B" we compare against the apples-to-apples narrow projection. The wide
      // variant's column reflects the tradeoff but isn't itself the apples-to-apples
      // baseline, so it doesn't get a "vs B" speedup column either.
      val baselineB = rs.find(_.config.startsWith("B-narrow:")).map(_.totalMs)
        .orElse(rs.find(_.config.startsWith("B:")).map(_.totalMs))
        .getOrElse(0L)
      val _ =
        rs.find(_.config.startsWith("C-indexed-narrow:")).map(
          _.totalMs
        ) // C reference; printed in row
      rs.foreach { r =>
        val buildStr = r.indexBuildMs.map(_.toString).getOrElse("—")
        val vsA =
          if (baselineA > 0 && r.totalMs > 0) f"${baselineA.toDouble / r.totalMs}%.1fx" else "—"
        val vsB =
          if (baselineB > 0 && r.totalMs > 0) f"${baselineB.toDouble / r.totalMs}%.1fx" else "—"
        println(
          f"${r.scale}%-8s  ${r.config}%-50s  ${r.totalMs}%8d  $buildStr%9s  $vsA%6s  $vsB%6s")
      }
      println()
    }
  }
}
