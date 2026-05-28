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

import org.apache.spark.SparkEnv
import org.apache.spark.sql.SparkSession

import java.lang.management.ManagementFactory

/**
 * Pre-bench cluster health probe. Runs a fixed-cost CPU loop on every task slot and
 * reports per-executor wall-clock + JIT-warmed nanos. Outliers indicate noisy
 * neighbors (other tenants saturating cores), pinned cores, thermal throttling, or
 * uneven hardware in the executor pool — all of which make config-vs-config medians
 * unreliable.
 *
 * Output is a table sorted by median time per executor, with min/max highlighted so
 * the slowest executor is obvious. With `failRatio` set, throws if
 * `slowest_executor_median / pool_median > failRatio`.
 *
 * == Why a separate stage, not just `defaultParallelism` ==
 *
 * `defaultParallelism` only tells you the pool size, not whether the cores are
 * actually free. A 64-core pool where 32 cores are saturated by another job will
 * still report 64; this probe will show those 32 cores as ~2× slower than the
 * others. That's the data we need to decide whether to trust the bench numbers.
 *
 * == Why repeated iterations ==
 *
 * The first iteration includes JIT warmup. We discard it and report the median of
 * the remaining iterations. That gives a JIT-stable per-core compute number that
 * isolates the cluster's compute-readiness from JVM startup cost.
 */
object ExecutorCpuCheck {

  // ~80M float-mul-add ops per iteration — sized to take ~80-120 ms warm on a
  // modern x86 core. Long enough to drown out scheduler noise (~1-5 ms) and short
  // enough that the whole probe finishes in a few seconds even with many cores.
  private val OpsPerIter: Int = 80000000

  // 1 warmup + 3 measured iters per task. Per-iter ~100 ms × 4 = ~400 ms wall on
  // a healthy executor. With 64 cores the probe stage runs in parallel so total
  // wall is bounded by ~400 ms, not 64 × 400.
  private val IterCount: Int = 4

  /**
   * Run the probe. Tasks emitted: max(defaultParallelism × 2, 32) — slightly
   * over-subscribe so each core is touched. Spark's scheduler picks which task
   * goes to which executor; we record the executor host in each task's result and
   * group by host afterwards.
   */
  def run(spark: SparkSession, failRatio: Option[Double]): Unit = {
    val sc = spark.sparkContext
    val parallelism = sc.defaultParallelism
    val tasks = math.max(parallelism * 2, 32)

    println("─" * 96)
    println(f"Executor CPU probe  (defaultParallelism=$parallelism, tasks=$tasks, " +
      f"warmup+measured=${IterCount} iters/task)")
    println("─" * 96)

    val started = System.currentTimeMillis()

    // Each task runs the compute IterCount times. The first iter is warmup
    // (discarded). Returns (executorId, host, perIterNanosAfterWarmup).
    val rdd = sc.parallelize(0 until tasks, tasks).map { taskId =>
      val env = SparkEnv.get
      val executorId = if (env != null) env.executorId else "driver"
      val mxBean = ManagementFactory.getRuntimeMXBean
      val host =
        try {
          java.net.InetAddress.getLocalHost.getHostName
        } catch { case _: Throwable => "unknown" }

      val perIter = new Array[Long](IterCount)
      var iter = 0
      while (iter < IterCount) {
        val t0 = System.nanoTime()
        var acc = 1.0d
        var i = 0
        while (i < OpsPerIter) {
          // Multiply-add chain. Sequential dependency prevents the JIT from
          // hoisting the loop body, so we actually do the work.
          acc = acc * 1.0000001 + (taskId & 1).toDouble
          i += 1
        }
        // Use acc to prevent dead-code elimination.
        if (acc.isNaN) {
          throw new IllegalStateException("compute degenerate")
        }
        perIter(iter) = System.nanoTime() - t0
        iter += 1
      }
      // Drop the first iter (warmup), median the rest.
      val measured = perIter.drop(1).sorted
      val medianNanos = measured(measured.length / 2)
      // pid for finer-grained breakdown when cores are unevenly assigned across procs
      val pid = mxBean.getName
      ProbeRow(
        executorId = executorId,
        host = host,
        pid = pid,
        taskId = taskId,
        medianNanos = medianNanos,
        allNanos = perIter.toSeq)
    }

    val rows = rdd.collect().toSeq
    val elapsedMs = System.currentTimeMillis() - started

    if (rows.isEmpty) {
      println(f"  (no tasks ran — defaultParallelism=$parallelism)")
      println("─" * 96)
      println()
      return
    }

    // Group by executorId. For each, report median across that executor's tasks +
    // count of tasks landed there.
    case class ExecStats(
        executorId: String,
        host: String,
        taskCount: Int,
        medianMs: Double,
        minMs: Double,
        maxMs: Double)

    val perExec = rows
      .groupBy(_.executorId)
      .map { case (execId, execRows) =>
        val ms = execRows.map(_.medianNanos / 1e6).sorted
        val median = ms(ms.length / 2)
        ExecStats(
          executorId = execId,
          host = execRows.head.host,
          taskCount = execRows.size,
          medianMs = median,
          minMs = ms.head,
          maxMs = ms.last)
      }
      .toSeq
      .sortBy(_.medianMs)

    val poolMedian = {
      val all = perExec.map(_.medianMs).sorted
      all(all.length / 2)
    }
    val slowest = perExec.last
    val fastest = perExec.head
    val ratio = slowest.medianMs / fastest.medianMs

    // Print table.
    println(f"  ${"executor"}%-16s ${"host"}%-30s ${"tasks"}%5s   " +
      f"${"median ms"}%9s   ${"min"}%7s   ${"max"}%7s   ${"vs fastest"}%10s")
    perExec.foreach { e =>
      val rel = e.medianMs / fastest.medianMs
      val flag = if (rel >= 1.5) " ⚠"
      else if (rel >= 1.25) " ·"
      else ""
      println(f"  ${e.executorId}%-16s ${e.host}%-30s ${e.taskCount}%5d   " +
        f"${e.medianMs}%9.1f   ${e.minMs}%7.1f   ${e.maxMs}%7.1f   ${rel}%9.2fx$flag")
    }
    println()
    println(f"  pool median:    $poolMedian%9.1f ms")
    println(
      f"  slowest/fastest: $ratio%9.2fx (executor=${slowest.executorId}, host=${slowest.host})")
    println(f"  probe wall:     $elapsedMs%d ms")

    if (ratio >= 1.5) {
      println(f"  ⚠ WARNING: slowest executor is ≥1.5× the fastest — measurements " +
        f"under contention.")
    } else if (ratio >= 1.25) {
      println(f"  · note: slowest executor is ≥1.25× the fastest — minor variance, " +
        f"medians should still be meaningful.")
    } else {
      println(f"  ✓ executor pool is uniform (≤1.25× spread).")
    }

    failRatio match {
      case Some(t) if ratio > t =>
        println("─" * 96)
        throw new IllegalStateException(
          f"executor CPU spread (${ratio}%.2fx) exceeds BENCH_CPU_CHECK_FAIL_RATIO ($t%.2fx); " +
            "cluster is too noisy for trustworthy measurements. Set BENCH_CPU_CHECK_SKIP=true " +
            "to override.")
      case _ =>
    }

    println("─" * 96)
    println()
  }

  private case class ProbeRow(
      executorId: String,
      host: String,
      pid: String,
      taskId: Int,
      medianNanos: Long,
      allNanos: Seq[Long])
}
