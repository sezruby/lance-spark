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
import org.apache.spark.sql.types._
import org.lance.spark.knn.LanceKnnImplicits._

import java.lang.management.ManagementFactory
import java.util.Random
import java.util.concurrent.{ConcurrentLinkedQueue, Executors, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.JavaConverters._

/**
 * Concurrent sustained-load soak test for `IndexedNearestJoin` / `df.kNearestJoin`.
 *
 * Production-readiness validation #2 ("sustained concurrent load") from the must-validate
 * list: run N concurrent queries/second for M minutes and watch for:
 *
 *   - Memory growth (JVM heap + off-heap direct buffers + Arrow allocator)
 *   - File handle leaks (executor-side Lance dataset handles)
 *   - GC pressure trends (Old-gen growth over time ⇒ a leak)
 *   - Per-query latency drift (early queries vs late queries ⇒ resource contention)
 *   - Failure rate (any query throwing indicates correctness regression under load)
 *
 * The unit benchmark ([[IndexedNearestJoinBenchmark]]) measures single-query throughput;
 * this benchmark measures behavior under many simultaneous queries.
 *
 * == Cluster run ==
 *
 * Build the benchmark fat JAR first (same profile as the throughput benchmark):
 *
 * {{{
 *   ./mvnw -pl lance-spark-knn_2.12 package -Pbenchmark -DskipTests
 *   # → target/lance-spark-knn_2.12-<version>-benchmark.jar
 * }}}
 *
 * Submit via your cluster's job API, passing these environment variables:
 *
 *   - `BENCH_CLUSTER_MODE=true`   — skips `.master()` and driver bind-address config
 *   - `BENCH_DATA_PATH=<uri>`     — shared path for the synthetic right-side Lance dataset
 *                                   (s3://, abfs://, gs://, hdfs://)
 *   - `SOAK_RIGHT_ROWS=100000000` — size of the right-side Lance dataset (default: 10M)
 *   - `SOAK_LEFT_ROWS=1000`       — left rows per query (default: 100)
 *   - `SOAK_DIM=128`              — vector dimension (default: 128)
 *   - `SOAK_K=10`                 — top-K (default: 10)
 *   - `SOAK_DURATION_MIN=60`      — total soak wall-clock minutes (default: 5 for smoke)
 *   - `SOAK_CONCURRENCY=8`        — concurrent queries in flight (default: 4)
 *   - `SOAK_QPS_TARGET=2`         — target queries per second per driver thread (default: 2)
 *   - `SOAK_PROBE_PARALLELISM=1`  — `probeParallelism` per query (default: 1)
 *   - `SOAK_SEED=1337`            — RNG seed for reproducibility (default: 1337)
 *   - `SOAK_SETUP_ONLY=false`     — if "true", writes the right-side dataset and exits
 *                                   (so you can pre-warm once and run the soak many times)
 *   - `SOAK_SKIP_SETUP=false`     — if "true", assumes the dataset at `BENCH_DATA_PATH`
 *                                   already exists (for re-running the soak without rewriting)
 *
 * Report at end: p50/p95/p99/max latency, queries/sec, failure count, heap & off-heap
 * snapshots at start / midpoint / end, GC counts & times. Streams per-query timings to
 * stdout every 60 s so you can `tail -f` driver logs during a long run.
 *
 * == What this does NOT measure ==
 *
 *   - Executor-side resource trends — those need cluster-level monitoring (Grafana /
 *     Spark UI / JMX). The driver-side snapshots here are an upper-bound check; a
 *     worker-side leak won't show up here.
 *   - Correctness under load — each query uses random left vectors; we don't validate
 *     against an oracle per query (too expensive). Any thrown exception IS counted and
 *     the stack trace is logged.
 *   - Very long-running behavior (days). Default is 5 minutes for smoke; set
 *     `SOAK_DURATION_MIN` higher for real soak (4-24h recommended for production
 *     qualification).
 */
object IndexedNearestJoinSoakTest {

  private val Dim: Int = sys.env.get("SOAK_DIM").map(_.toInt).getOrElse(128)
  private val K: Int = sys.env.get("SOAK_K").map(_.toInt).getOrElse(10)
  private val RightRows: Long =
    sys.env.get("SOAK_RIGHT_ROWS").map(_.toLong).getOrElse(10000000L)
  private val LeftRows: Int = sys.env.get("SOAK_LEFT_ROWS").map(_.toInt).getOrElse(100)
  private val DurationMin: Int =
    sys.env.get("SOAK_DURATION_MIN").map(_.toInt).getOrElse(5)
  private val Concurrency: Int =
    sys.env.get("SOAK_CONCURRENCY").map(_.toInt).getOrElse(4)
  private val QpsTarget: Double =
    sys.env.get("SOAK_QPS_TARGET").map(_.toDouble).getOrElse(2.0)
  private val ProbeParallelism: Int =
    sys.env.get("SOAK_PROBE_PARALLELISM").map(_.toInt).getOrElse(1)
  private val Seed: Long = sys.env.get("SOAK_SEED").map(_.toLong).getOrElse(1337L)
  private val SetupOnly: Boolean =
    sys.env.get("SOAK_SETUP_ONLY").exists(_.equalsIgnoreCase("true"))
  private val SkipSetup: Boolean =
    sys.env.get("SOAK_SKIP_SETUP").exists(_.equalsIgnoreCase("true"))

  private val ClusterMode: Boolean =
    sys.env.get("BENCH_CLUSTER_MODE").exists(_.equalsIgnoreCase("true"))
  private val DataPath: String = sys.env.getOrElse(
    "BENCH_DATA_PATH",
    "/tmp/lance-knn-soak")

  // Stats collected across all queries.
  private val completed = new AtomicInteger(0)
  private val failed = new AtomicInteger(0)
  private val latenciesNs = new ConcurrentLinkedQueue[Long]()

  def main(args: Array[String]): Unit = {
    val spark = buildSparkSession()
    try {
      logBanner(spark)

      val rightUri = s"$DataPath/right_soak_${RightRows}_${Dim}"
      if (SkipSetup) {
        println(s"[soak] SOAK_SKIP_SETUP=true → using existing dataset at $rightUri")
      } else {
        setupRightDataset(spark, rightUri)
      }

      if (SetupOnly) {
        println("[soak] SOAK_SETUP_ONLY=true → dataset written, exiting before load phase.")
        return
      }

      runSoak(spark, rightUri)
      printFinalReport(spark)
    } finally {
      spark.stop()
    }
  }

  // -- setup ----------------------------------------------------------------------------------

  private def buildSparkSession(): SparkSession = {
    val b = SparkSession.builder().appName("IndexedNearestJoin-SoakTest")
    if (!ClusterMode) {
      b.master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
    }
    // Schedule FIFO→FAIR so concurrent queries actually run concurrently on the same
    // SparkContext. Without FAIR, queries serialise behind each other regardless of
    // SOAK_CONCURRENCY.
    b.config("spark.scheduler.mode", "FAIR")
      .getOrCreate()
  }

  private def logBanner(spark: SparkSession): Unit = {
    println("=" * 80)
    println(s"IndexedNearestJoinSoakTest")
    println("=" * 80)
    println(f"  Spark version:      ${spark.version}")
    println(f"  master:             ${spark.sparkContext.master}")
    println(f"  applicationId:      ${spark.sparkContext.applicationId}")
    println(f"  default parallelism: ${spark.sparkContext.defaultParallelism}")
    println(f"  cluster mode:       $ClusterMode")
    println(f"  data path:          $DataPath")
    println("  -- load knobs --")
    println(f"  right rows:         $RightRows%,d")
    println(f"  left rows/query:    $LeftRows%,d")
    println(f"  dim:                $Dim")
    println(f"  K:                  $K")
    println(f"  probeParallelism:   $ProbeParallelism")
    println(f"  concurrency:        $Concurrency")
    println(f"  qps target:         $QpsTarget%.2f queries/sec")
    println(f"  duration:           $DurationMin minutes")
    println(f"  seed:               $Seed")
    println("=" * 80)
    println()
  }

  private def setupRightDataset(spark: SparkSession, uri: String): Unit = {
    println(s"[soak] writing right-side Lance dataset: $RightRows rows × dim=$Dim → $uri")
    val t0 = System.nanoTime()

    // Build the right DF via `spark.range` + a UDF-ish map so the data generation is
    // distributed. For very large RightRows this is the critical scalability path.
    val schema = new StructType(Array(
      StructField("rid", LongType, nullable = false),
      StructField(
        "rvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))

    val capturedDim = Dim
    val capturedSeed = Seed
    val rdd = spark.sparkContext
      .range(0L, RightRows, 1L, math.max(spark.sparkContext.defaultParallelism * 4, 16))
      .mapPartitionsWithIndex { case (partIdx, iter) =>
        val rng = new Random(capturedSeed + partIdx.toLong)
        iter.map { i =>
          val v = new Array[Float](capturedDim)
          var j = 0
          while (j < capturedDim) { v(j) = rng.nextFloat(); j += 1 }
          // Pass the Array[Float] directly - Spark's ArrayType encoder on 2.12 expects
          // either an Array or a scala.collection.Seq, NOT a Java List (which is what
          // `.toSeq.asJava` produces and causes `Wrappers$SeqWrapper incompatible with
          // scala.collection.Seq` at encode time).
          RowFactory.create(java.lang.Long.valueOf(i), v): Row
        }
      }
    val df = spark.createDataFrame(rdd, schema)
    // Lance's catalog treats mode("overwrite") as "drop then create", which throws
    // NoSuchTableException when the path is new. Default (ErrorIfExists) is correct for
    // first-run setup; reuse across runs should set SOAK_SKIP_SETUP=true instead.
    df.write.format("lance").save(uri)

    val elapsedSec = (System.nanoTime() - t0) / 1e9
    println(f"[soak] right dataset written in $elapsedSec%.1f s")
    println()
  }

  // -- soak loop ------------------------------------------------------------------------------

  private def runSoak(spark: SparkSession, rightUri: String): Unit = {
    val right = spark.read.format("lance").load(rightUri)

    // Per-query left DataFrames are generated in the driver; cheap at LeftRows ≤ few-K.
    // Cache the schema and random generation once.
    val leftSchema = new StructType(Array(
      StructField("lid", LongType, nullable = false),
      StructField(
        "qvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))

    val deadline = System.currentTimeMillis() + DurationMin.toLong * 60L * 1000L
    val pool = Executors.newFixedThreadPool(Concurrency)
    val ticker = Executors.newSingleThreadScheduledExecutor()
    val halfway = System.currentTimeMillis() + (DurationMin.toLong * 30L * 1000L)

    val snapStart = snapshotProcess(spark)
    var snapMid: ProcessSnapshot = null

    // Per-60-s progress printer.
    ticker.scheduleAtFixedRate(
      new Runnable {
        override def run(): Unit = {
          val done = completed.get()
          val bad = failed.get()
          val snap = snapshotProcess(spark)
          println(f"[soak] t=${elapsedSec()}%6.1fs  completed=$done%,d  failed=$bad%,d  " +
            f"heap=${snap.heapUsedMb}%,d MB  directMem=${snap.directUsedMb}%,d MB  " +
            f"gcCount=${snap.gcCount}  gcTimeMs=${snap.gcTimeMs}")
        }
      },
      60L,
      60L,
      TimeUnit.SECONDS)

    val intervalMs = math.max(1L, (1000.0 / QpsTarget).toLong)
    val startTime = System.currentTimeMillis()
    var queriesSubmitted = 0L

    try {
      while (System.currentTimeMillis() < deadline) {
        val qid = queriesSubmitted
        queriesSubmitted += 1
        val task: Runnable = new Runnable {
          override def run(): Unit = runOneQuery(spark, right, leftSchema, qid)
        }
        pool.submit(task)

        if (snapMid == null && System.currentTimeMillis() >= halfway) {
          snapMid = snapshotProcess(spark)
          println(f"[soak] midpoint snapshot: heap=${snapMid.heapUsedMb}%,d MB  " +
            f"directMem=${snapMid.directUsedMb}%,d MB  gcCount=${snapMid.gcCount}")
        }

        // Throttle submission rate. Without this the pool fills with backlog and the
        // `SOAK_QPS_TARGET` knob is meaningless.
        val targetSubmitTime = startTime + queriesSubmitted * intervalMs
        val sleepMs = targetSubmitTime - System.currentTimeMillis()
        if (sleepMs > 0) Thread.sleep(sleepMs)
      }
    } finally {
      ticker.shutdown()
      pool.shutdown()
      pool.awaitTermination(2, TimeUnit.MINUTES)
      ticker.awaitTermination(5, TimeUnit.SECONDS)
    }
  }

  private def runOneQuery(
      spark: SparkSession,
      right: DataFrame,
      leftSchema: StructType,
      qid: Long): Unit = {
    val t0 = System.nanoTime()
    try {
      val rng = new Random(Seed + qid)
      val rows = new java.util.ArrayList[Row](LeftRows)
      var i = 0
      while (i < LeftRows) {
        val v = new Array[Float](Dim)
        var j = 0
        while (j < Dim) { v(j) = rng.nextFloat(); j += 1 }
        // Same Array[Float]-not-Java-List rule as setupRightDataset.
        rows.add(RowFactory.create(java.lang.Long.valueOf(i.toLong), v))
        i += 1
      }
      val left = spark.createDataFrame(rows, leftSchema)

      val joined = left.kNearestJoin(
        right = right,
        leftVecCol = "qvec",
        rightVecCol = "rvec",
        k = K,
        metric = "l2",
        probeParallelism = ProbeParallelism)

      // Consume via count() — exercises the ColumnPruning-sensitive path AND avoids
      // pulling all rows to the driver. Both are important under load.
      val n = joined.count()
      if (n != LeftRows.toLong * K.toLong) {
        throw new IllegalStateException(
          s"query $qid returned $n rows; expected ${LeftRows * K}")
      }
      latenciesNs.add(System.nanoTime() - t0)
      completed.incrementAndGet()
    } catch {
      case t: Throwable =>
        failed.incrementAndGet()
        println(s"[soak] query $qid FAILED: ${t.getClass.getSimpleName}: ${t.getMessage}")
        t.printStackTrace()
    }
  }

  // -- reporting ------------------------------------------------------------------------------

  private def printFinalReport(spark: SparkSession): Unit = {
    val snapEnd = snapshotProcess(spark)
    val done = completed.get()
    val bad = failed.get()
    val lats = latenciesNs.asScala.toVector.sorted

    println()
    println("=" * 80)
    println("SOAK TEST FINAL REPORT")
    println("=" * 80)
    println(f"  completed queries:  $done%,d")
    println(f"  failed queries:     $bad%,d")
    if (done > 0) {
      println(f"  failure rate:       ${bad.toDouble / (done + bad) * 100.0}%.3f%%")
    }
    println()

    if (lats.nonEmpty) {
      val p50 = lats(lats.length / 2) / 1e6
      val p95 = lats((lats.length * 95) / 100) / 1e6
      val p99 = lats((lats.length * 99) / 100) / 1e6
      val max = lats.last / 1e6
      val mean = lats.sum.toDouble / lats.length / 1e6
      println("  LATENCY (ms)")
      println(f"    p50:              $p50%,10.2f")
      println(f"    p95:              $p95%,10.2f")
      println(f"    p99:              $p99%,10.2f")
      println(f"    max:              $max%,10.2f")
      println(f"    mean:             $mean%,10.2f")
      println()

      // Early-vs-late drift. Splitting the sorted latencies doesn't work for this — we
      // want time-order. Instead split by query-id order. Requires the latencies in
      // submission order, which we don't have (we pushed in completion order). Skip for
      // now; a proper drift metric needs per-query (qid, nanos) pairs. Left as follow-up.
    }
    println()
    println(f"  DRIVER RESOURCE TOTALS (end of run)")
    println(f"    heap used:        ${snapEnd.heapUsedMb}%,d MB / ${snapEnd.heapMaxMb}%,d MB")
    println(f"    direct memory:    ${snapEnd.directUsedMb}%,d MB")
    println(f"    GC count:         ${snapEnd.gcCount}")
    println(f"    GC time:          ${snapEnd.gcTimeMs}%,d ms")
    println()

    // Exit code hint for CI: non-zero if any query failed.
    if (bad > 0) {
      System.err.println(s"[soak] FAILURE: $bad queries failed; see stderr above")
      System.exit(2)
    }
  }

  // -- process snapshots ----------------------------------------------------------------------

  private case class ProcessSnapshot(
      heapUsedMb: Long,
      heapMaxMb: Long,
      directUsedMb: Long,
      gcCount: Long,
      gcTimeMs: Long)

  /**
   * Snapshot of driver-side memory + GC. Off-heap `direct` bytes are tracked via the
   * JDK's `BufferPoolMXBean` for the "direct" pool — this catches `ByteBuffer.allocateDirect`
   * but NOT native allocations (Arrow allocator, JNI). For native accounting, cluster-level
   * JMX / /proc-scrape is required; this is a best-effort driver snapshot.
   */
  private def snapshotProcess(spark: SparkSession): ProcessSnapshot = {
    val memoryMx = ManagementFactory.getMemoryMXBean
    val heap = memoryMx.getHeapMemoryUsage
    val heapUsedMb = heap.getUsed / (1024L * 1024L)
    val heapMaxMb = (if (heap.getMax > 0) heap.getMax else heap.getCommitted) / (1024L * 1024L)

    val directUsedMb: Long = ManagementFactory
      .getPlatformMXBeans(classOf[java.lang.management.BufferPoolMXBean])
      .asScala
      .find(_.getName == "direct")
      .map(_.getMemoryUsed / (1024L * 1024L))
      .getOrElse(0L)

    val gcBeans = ManagementFactory.getGarbageCollectorMXBeans.asScala
    val gcCount = gcBeans.map(_.getCollectionCount).sum
    val gcTimeMs = gcBeans.map(_.getCollectionTime).sum

    ProcessSnapshot(heapUsedMb, heapMaxMb, directUsedMb, gcCount, gcTimeMs)
  }

  private def elapsedSec(): Double =
    (System.currentTimeMillis() - appStartMillis) / 1000.0

  private lazy val appStartMillis: Long = System.currentTimeMillis()
}
