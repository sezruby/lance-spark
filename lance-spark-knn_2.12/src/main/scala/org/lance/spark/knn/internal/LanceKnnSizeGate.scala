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
package org.lance.spark.knn.internal

import org.apache.spark.network.util.{ByteUnit, JavaUtils}
import org.apache.spark.sql.SparkSession
import org.lance.{Dataset, ReadOptions}
import org.lance.spark.LanceRuntime

import scala.collection.JavaConverters._

/**
 * Plan-time size gate for the indexed native KNN probe.
 *
 * Each probe task opens R's whole-dataset index (`fragmentIds = None`), so per-executor resident
 * memory grows with `|R|`, not with query count. We model that resident footprint as
 *
 * {{{
 *   residentFootprint(|R|)  ≈  a + b · |R|          # per executor
 *   apply native probe   iff  residentFootprint(|R|)  ≤  SAFETY · executorRAM
 * }}}
 *
 * and compare it to a safe fraction of the executor container budget
 * (`spark.executor.memory` + `spark.executor.memoryOverhead`). The constants were measured on a
 * standalone Spark cluster (synthetic uniform `dim=128` vectors, IVF-PQ, 4 concurrent probes per
 * executor): `a ≈ 0.70 GiB`, `b ≈ 64 MiB per 1M rows` at `dim=128`. The per-row slope is scaled
 * from that single `dim=128` point (see [[perRowBytesForDim]]); that extrapolation is UNVALIDATED
 * at other dimensions — the resident PQ footprint tracks the index's `num_sub_vectors`, not the raw
 * vector dimension — so treat the number as advisory. See sezruby/lance-spark#11 "Memory & size
 * gate" for the full write-up.
 *
 * This is a soft, latency-oriented guardrail rather than a hard-OOM safety check: the measured
 * `VmHWM` growth is dominated by reclaimable memory-mapped index pages, so past the budget the
 * native path degrades in latency (index pages fault in and out) rather than crashing. And the
 * native probe is the ONLY path that scales here — the default cross-product rewrite is strictly
 * worse and would OOM at the same `|R|` — so the gate never reroutes anywhere; it only surfaces the
 * working-set cliff. That is why the default mode is `warn`, not a hard failure.
 *
 * == Modes (`spark.lance.knn.sizeGate.mode`) ==
 *   - `warn` (default): log a warning with the numbers and proceed with the native probe. The
 *     estimate is advisory and the native path degrades rather than crashes, so the default never
 *     blocks a job that would otherwise complete.
 *   - `error`: throw when the estimate exceeds the budget — opt-in fail-fast for callers who would
 *     rather not start an under-provisioned job at all.
 *   - `off`: skip the check entirely (no dataset open).
 *
 * == Overrides ==
 *   - `spark.lance.knn.sizeGate.safetyFraction`  (default 0.7)
 *   - `spark.lance.knn.sizeGate.baselineBytes`   (default 0.70 GiB — the `a` term)
 *   - `spark.lance.knn.sizeGate.bytesPerRow`     (default extrapolated from dim — see caveat above)
 *   - `spark.lance.knn.sizeGate.executorRamBytes` (default `spark.executor.memory` + overhead)
 *
 * The two arithmetic helpers ([[budgetBytes]], [[estimateFor]]) are pure and deployment-agnostic so
 * they can be unit-tested without a Spark session or a live Lance dataset.
 */
object LanceKnnSizeGate {

  private val LOG = org.slf4j.LoggerFactory.getLogger("org.lance.spark.knn.sizeGate")

  // ---- configuration keys ------------------------------------------------------------------
  val ModeConfKey: String = "spark.lance.knn.sizeGate.mode"
  val SafetyFractionConfKey: String = "spark.lance.knn.sizeGate.safetyFraction"
  val BaselineBytesConfKey: String = "spark.lance.knn.sizeGate.baselineBytes"
  val BytesPerRowConfKey: String = "spark.lance.knn.sizeGate.bytesPerRow"
  val ExecutorRamBytesConfKey: String = "spark.lance.knn.sizeGate.executorRamBytes"

  // ---- measured constants (sezruby/lance-spark#11 "Memory & size gate") --------------------
  private val GiB: Long = 1L << 30
  private val MiB: Long = 1L << 20

  /** `a` — baseline per-executor footprint (JVM + native runtime + per-task buffers). */
  val DefaultBaselineBytes: Long = (0.70 * GiB).toLong

  /** Reference dimension the per-row slope `b` was measured at. */
  val ReferenceDim: Int = 128

  /** `b` at the reference dim: 64 MiB per 1M rows ≈ 67.1 bytes/row. */
  val BytesPerRowAtReferenceDim: Double = 64.0 * MiB / 1e6

  val DefaultSafetyFraction: Double = 0.7

  // Spark's own defaults for the executor container budget.
  private val DefaultExecutorMemory: String = "1g"
  private val MinOverheadBytes: Long = 384L * MiB
  private val DefaultOverheadFactor: Double = 0.1

  /**
   * The result of a size estimate. `numRows`/`dim` come from R; the rest are the resolved
   * model inputs. `fits` is the gate decision.
   */
  final case class Estimate(
      numRows: Long,
      dim: Option[Int],
      bytesPerRow: Double,
      baselineBytes: Long,
      footprintBytes: Long,
      execRamBytes: Long,
      safety: Double,
      mode: String) {
    def budgetBytes: Long = (safety * execRamBytes).toLong
    def fits: Boolean = footprintBytes <= budgetBytes
    def footprintGiB: String = f"${footprintBytes.toDouble / GiB}%.1f"
    def budgetGiB: String = f"${budgetBytes.toDouble / GiB}%.1f"
    def execRamGiB: String = f"${execRamBytes.toDouble / GiB}%.1f"
  }

  /**
   * Plan-time entry point. Reads config off `spark`, estimates the per-executor footprint for a
   * whole-dataset probe of R, and acts per [[ModeConfKey]]. `warn` (default) logs and proceeds,
   * `error` throws (opt-in fail fast), `off` skips without opening the dataset.
   */
  def check(
      spark: SparkSession,
      datasetUri: String,
      version: Option[Long],
      vectorColumn: String): Unit = {
    val mode = modeOf(spark)
    if (mode == "off") return
    val est = estimate(spark, datasetUri, version, vectorColumn, mode)
    if (!est.fits) {
      val sizing =
        s"KNN native probe estimated at ~${est.footprintGiB} GiB/exec vs budget " +
          s"${est.safety}×${est.execRamGiB} = ${est.budgetGiB} GiB at |R|=${est.numRows}" +
          est.dim.map(d => s" (dim=$d)").getOrElse("") + "."
      mode match {
        // Opt-in fail-fast: refuse to start an under-provisioned job.
        case "error" =>
          throw new IllegalArgumentException(
            sizing + s" Failing fast ($ModeConfKey=error): raise spark.executor.memory, lower " +
              s"executor cores, or set $ModeConfKey=warn to proceed anyway.")
        // Default `warn` (and any unrecognized value): the estimate is advisory and the native
        // path degrades in latency rather than crashing, so never block — just surface it.
        case _ =>
          LOG.warn(
            sizing + " Proceeding: the native probe still runs (it degrades in latency rather " +
              "than crashing, and it is the only path that scales at this size). Raise " +
              s"spark.executor.memory / lower executor cores if latency suffers, or set " +
              s"$ModeConfKey=error to fail fast instead.")
      }
    }
  }

  private def modeOf(spark: SparkSession): String =
    spark.sessionState.conf.getConfString(ModeConfKey, "warn").trim.toLowerCase

  /**
   * Resolve the model inputs from config + R's metadata and compute the [[Estimate]]. The row
   * count and vector dimension are read once via the Lance Java API on the driver.
   */
  def estimate(
      spark: SparkSession,
      datasetUri: String,
      version: Option[Long],
      vectorColumn: String,
      mode: String): Estimate = {
    val safety = optDouble(spark, SafetyFractionConfKey).getOrElse(DefaultSafetyFraction)
    val baselineBytes = optLong(spark, BaselineBytesConfKey).getOrElse(DefaultBaselineBytes)
    val execRamBytes =
      optLong(spark, ExecutorRamBytesConfKey).getOrElse(resolveExecutorRamBytes(spark))
    val bytesPerRowOverride = optDouble(spark, BytesPerRowConfKey)

    val (numRows, dim) = readRowsAndDim(datasetUri, version, vectorColumn)
    if (dim.isEmpty && bytesPerRowOverride.isEmpty) {
      LOG.warn(
        s"Could not determine the vector dimension for column '$vectorColumn' at $datasetUri; the " +
          s"size estimate falls back to the reference dim ($ReferenceDim) and may be inaccurate. " +
          s"Set $BytesPerRowConfKey to calibrate the per-row footprint.")
    }
    estimateFor(numRows, dim, safety, baselineBytes, bytesPerRowOverride, execRamBytes, mode)
  }

  /**
   * Pure footprint arithmetic — no Spark, no Lance. `bytesPerRowOverride` wins when set; otherwise
   * the per-row slope is scaled linearly from the measured `dim=128` point (falling back to the
   * reference dim when R's dimension can't be determined).
   */
  private[knn] def estimateFor(
      numRows: Long,
      dim: Option[Int],
      safety: Double,
      baselineBytes: Long,
      bytesPerRowOverride: Option[Double],
      execRamBytes: Long,
      mode: String): Estimate = {
    val bytesPerRow = bytesPerRowOverride.getOrElse(perRowBytesForDim(dim))
    val footprintBytes = baselineBytes + math.round(bytesPerRow * numRows)
    Estimate(numRows, dim, bytesPerRow, baselineBytes, footprintBytes, execRamBytes, safety, mode)
  }

  /**
   * Per-row resident-index bytes, scaled linearly from the measured `dim=128` slope. The linear
   * scaling is an UNVALIDATED extrapolation — the resident PQ footprint really tracks the index's
   * `num_sub_vectors`, which does not grow one-for-one with the raw vector dimension, so this can
   * over-predict at high dim. It is advisory only (the default `warn` mode never blocks); override
   * with [[BytesPerRowConfKey]] to calibrate against a measured workload.
   */
  private[knn] def perRowBytesForDim(dim: Option[Int]): Double =
    BytesPerRowAtReferenceDim * (dim.getOrElse(ReferenceDim).toDouble / ReferenceDim)

  /**
   * Executor container budget = `spark.executor.memory` + `spark.executor.memoryOverhead`. RSS
   * here is off-heap-dominated, so comparing against `-Xmx` (executor memory) alone would be
   * dimensionally wrong. Mirrors Spark's own overhead default:
   * `max(384 MiB, overheadFactor × executorMemory)`.
   */
  private[knn] def resolveExecutorRamBytes(spark: SparkSession): Long = {
    val sparkConf = spark.sparkContext.getConf
    val execMem = sparkConf.get("spark.executor.memory", DefaultExecutorMemory)
    val overhead = sparkConf.getOption("spark.executor.memoryOverhead")
    val overheadFactor = sparkConf.getOption("spark.executor.memoryOverheadFactor").map(_.toDouble)
    budgetBytes(execMem, overhead, overheadFactor)
  }

  /**
   * Pure budget arithmetic. Both `spark.executor.memory` and `spark.executor.memoryOverhead`
   * follow Spark's `ByteUnit.MiB` convention (a bare number means MiB); a unit suffix (`g`, `m`,
   * …) is honored. When overhead is unset, use `max(384 MiB, factor × executorMemory)`.
   */
  private[knn] def budgetBytes(
      execMem: String,
      overhead: Option[String],
      overheadFactor: Option[Double]): Long = {
    val execMemBytes = memBytes(execMem)
    val overheadBytes = overhead match {
      case Some(s) => memBytes(s)
      case None =>
        val factor = overheadFactor.getOrElse(DefaultOverheadFactor)
        math.max(MinOverheadBytes, (factor * execMemBytes).toLong)
    }
    execMemBytes + overheadBytes
  }

  /** Parse a Spark memory string (MiB convention: bare number = MiB) to bytes. */
  private def memBytes(str: String): Long = JavaUtils.byteStringAs(str, ByteUnit.MiB) * MiB

  /** Read R's total row count and the vector column's dimension via the Lance Java API. */
  private[knn] def readRowsAndDim(
      datasetUri: String,
      version: Option[Long],
      vectorColumn: String): (Long, Option[Int]) = {
    val ds = openDataset(datasetUri, version)
    try {
      (ds.countRows(), dimensionOf(ds.getSchema, vectorColumn))
    } finally ds.close()
  }

  /** Open the Lance dataset at `datasetUri`, optionally pinned to `version`. */
  private def openDataset(datasetUri: String, version: Option[Long]): Dataset = {
    val readOpts = {
      val b = new ReadOptions.Builder()
      version.foreach(v => b.setVersion(v))
      b.build()
    }
    Dataset
      .open()
      .uri(datasetUri)
      .allocator(LanceRuntime.allocator())
      .readOptions(readOpts)
      .build()
  }

  /** Vector dimension = the `FixedSizeList` width of the vector field, if the schema declares it. */
  private def dimensionOf(
      schema: org.apache.arrow.vector.types.pojo.Schema,
      vectorColumn: String): Option[Int] = {
    schema.getFields.asScala
      .find(_.getName == vectorColumn)
      .map(_.getType)
      .collect {
        case fsl: org.apache.arrow.vector.types.pojo.ArrowType.FixedSizeList => fsl.getListSize
      }
  }

  private def optDouble(spark: SparkSession, key: String): Option[Double] =
    Option(spark.sessionState.conf.getConfString(key, null)).map(_.trim.toDouble)

  private def optLong(spark: SparkSession, key: String): Option[Long] =
    Option(spark.sessionState.conf.getConfString(key, null)).map(_.trim.toLong)
}
