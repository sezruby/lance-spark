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
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{ArrayType, FloatType, IntegerType, MetadataBuilder, StructField, StructType}
import org.lance.index.external.ExternalIvfPqIndexParams
import org.lance.spark.knn.IndexedNearestJoinExternal

import java.util.concurrent.TimeUnit

import scala.collection.JavaConverters._

/**
 * Path C ("external Lance vector index over parquet") on REAL embedding data.
 *
 * The wiki caveat on the KNN-over-Parquet page said the ~30× figure was directional only:
 * synthetic vectors, warm cache, small query side, and the mega scale couldn't run. This
 * bench closes those gaps — it points the external-index path at a real Cohere-wiki parquet
 * directory (dim=1024) already on object storage, builds the index over the parquet in place
 * (no rewrite), then times:
 *   - build       (one-time index-over-parquet cost)
 *   - search cold (first timed rep — object store not yet warm for the probed pages)
 *   - search warm (subsequent reps)
 *
 * Only config E runs here (search + post-topK fetchRows). Path B / A baselines are the
 * separate IndexedNearestJoinExternalBenchmark; this is the "does Path C hold up on real
 * cold reads" question.
 *
 * Env:
 *   EXT_PARQUET_DIR   parquet dir (abfss/s3/file) whose files are R. Required.
 *   EXT_VEC_COL       embedding column name on the parquet (default "emb").
 *   EXT_DIM           embedding dim (default 1024).
 *   EXT_NUM_L         number of query vectors to sample as the left side (default 100).
 *   EXT_K             top-K (default 10).
 *   EXT_NPROBES       IVF nprobes (default 16).
 *   EXT_REFINE        refine factor (default 8).
 *   EXT_REPEATS       timed search reps AFTER the cold rep (default 2). Total timed = 1 cold + N.
 *   AZURE_STORAGE_ACCOUNT_NAME / AZURE_STORAGE_ACCOUNT_KEY (or SAS) for object-store creds.
 */
object ExternalParquetRealDataBenchmark {

  def main(args: Array[String]): Unit = {
    val parquetDir = sys.env.getOrElse(
      "EXT_PARQUET_DIR",
      sys.error("EXT_PARQUET_DIR required (parquet dir for R)"))
    val vecCol = sys.env.getOrElse("EXT_VEC_COL", "emb")
    val dim = sys.env.get("EXT_DIM").map(_.toInt).getOrElse(1024)
    val numL = sys.env.get("EXT_NUM_L").map(_.toInt).getOrElse(100)
    val k = sys.env.get("EXT_K").map(_.toInt).getOrElse(10)
    val nprobes = sys.env.get("EXT_NPROBES").map(_.toInt).getOrElse(16)
    val refine = sys.env.get("EXT_REFINE").map(_.toInt).getOrElse(8)
    val warmReps = sys.env.get("EXT_REPEATS").map(_.toInt).getOrElse(2)
    // Column(s) to materialize from the parquet for the surviving top-K rows (the fetchRows
    // step). Real parquet has no synthetic "rid" — default to the doc id. Comma-separated.
    val projCols = sys.env.getOrElse("EXT_PROJ_COL", "_id").split(",").map(_.trim).toSeq

    val spark = SparkSession.builder().appName("ExternalParquetRealDataBenchmark").getOrCreate()
    try {
      // scalastyle:off println
      println("=" * 96)
      println("Path C on REAL parquet — external Lance index over parquet (search + fetch)")
      println(s"  parquetDir=$parquetDir  vecCol=$vecCol  dim=$dim")
      println(s"  |L|=$numL  k=$k  nprobes=$nprobes  refine=$refine  warmReps=$warmReps")
      println("=" * 96)
      // scalastyle:on println

      // List the parquet files that make up R.
      val rightFilePaths = listParquetFiles(spark, parquetDir)
      require(rightFilePaths.nonEmpty, s"no .parquet files under $parquetDir")
      // scalastyle:off println
      println(s"  R = ${rightFilePaths.size} parquet files")
      // scalastyle:on println

      // Cluster health gate: probe every task slot with a fixed-cost CPU loop and print
      // per-executor timings. Outliers (noisy neighbors, pinned cores, thermal throttling)
      // make config-vs-config medians unreliable — the exact failure mode that produced a
      // 3x outlier in a prior A/B run. With BENCH_CPU_CHECK_FAIL_RATIO set, refuses to
      // proceed when slowest/fastest exceeds the ratio; BENCH_CPU_CHECK_SKIP=true skips it.
      if (!sys.env.get("BENCH_CPU_CHECK_SKIP").exists(_.equalsIgnoreCase("true"))) {
        val failRatio = sys.env.get("BENCH_CPU_CHECK_FAIL_RATIO").map(_.toDouble)
        ExecutorCpuCheck.run(spark, failRatio)
      }

      // Sample numL query vectors from the SAME parquet (the emb column). Take the first
      // numL non-null embeddings; rename to lvec with the fixed-size-list metadata the
      // probe expects on the left side. Deterministic (first numL non-null) so the query
      // set — and thus the ground truth — is identical across separate submits.
      val leftDf = sampleLeft(spark, rightFilePaths, vecCol, dim, numL)
      leftDf.cache(); leftDf.count()

      // Rerank store from env: "none" (parquet refine), "sq8" (co-located int8), or
      // "flat" (co-located full-precision f32, exact refine).
      val rerankStore = sys.env.getOrElse("EXT_RERANK", "none").toLowerCase match {
        case "sq8" => ExternalIvfPqIndexParams.RerankStore.SQ8
        case "flat" => ExternalIvfPqIndexParams.RerankStore.FLAT
        case "none" => ExternalIvfPqIndexParams.RerankStore.NONE
        case other => sys.error(s"EXT_RERANK must be none|sq8|flat, got '$other'")
      }
      val measureRecall = sys.env.getOrElse("EXT_RECALL", "true").toBoolean
      // Reuse one open index handle per executor across tasks (opt-in). Default off.
      val cacheIndex = sys.env.getOrElse("EXT_INDEX_CACHE", "false").toBoolean

      def mkParams(store: ExternalIvfPqIndexParams.RerankStore): ExternalIvfPqIndexParams =
        ExternalIvfPqIndexParams.builder()
          .numPartitions(256)
          .numSubVectors(math.min(dim / 4, 16))
          .numBitsPerSubVector(8)
          .metric(ExternalIvfPqIndexParams.Metric.L2)
          .rerankStore(store)
          .build()

      val params = mkParams(rerankStore)
      // scalastyle:off println
      println(s"  rerankStore=$rerankStore  measureRecall=$measureRecall  cacheIndex=$cacheIndex")
      // scalastyle:on println

      // One join invocation. `count()` for timing; the DataFrame is also what recall
      // collects from (grouped by lid), so `join()` is shared.
      def join(
          nprobesX: Int,
          refineX: Int,
          p: ExternalIvfPqIndexParams): DataFrame =
        IndexedNearestJoinExternal(
          left = leftDf,
          rightFilePaths = rightFilePaths,
          leftVecCol = "lvec",
          rightVecCol = vecCol,
          k = k,
          metric = "l2",
          rightProjection = Some(projCols),
          nprobes = nprobesX,
          refineFactor = refineX,
          indexParams = Some(p),
          cacheIndexPerExecutor = cacheIndex)

      def runE(): Long = {
        val t = System.nanoTime()
        val n = join(nprobes, refine, params).count()
        val ms = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t)
        // scalastyle:off println
        println(f"    (rows=$n)")
        // scalastyle:on println
        ms
      }

      // First call includes index build + cold object-store reads.
      val buildPlusCold = runE()
      // Subsequent calls: index is cached (ExternalIndexLifecycle.buildOrReuse), so these
      // isolate warm search + fetch.
      val warm = (1 to warmReps).map(_ => runE())

      // Recall@k vs an exact-ish reference: probe ALL partitions + high refine, on the
      // SAME index build as the run under test. This is the critical detail — the truth
      // reference MUST use the same `params` (same rerank store), because the lifecycle
      // caches indexes by a key that includes the rerank store: a different store means a
      // different physical index (independently-trained kmeans/PQ → different partition
      // assignments), so comparing a Flat run's candidates against a None-store index's
      // ground truth measures build nondeterminism, not rerank quality. Using `params`
      // for both isolates exactly the rerank store's effect: same candidates, exhaustive
      // reference vs the configured nprobes/refine.
      val recall: Option[Double] =
        if (measureRecall) {
          val idCol = projCols.head
          val truthNprobes = 256
          val truthRefine = math.max(refine * 4, 50)
          val truth = neighborsById(join(truthNprobes, truthRefine, params), idCol)
          val got = neighborsById(join(nprobes, refine, params), idCol)
          Some(recallAtK(truth, got, k))
        } else None

      // scalastyle:off println
      println("=" * 96)
      println(f"  E build+cold (incl. index build over parquet): ${buildPlusCold}%,d ms")
      warm.zipWithIndex.foreach { case (ms, i) =>
        println(f"  E warm rep ${i + 1}: ${ms}%,d ms  (~${ms.toDouble / numL}%.1f ms/query)")
      }
      if (warm.nonEmpty) {
        val med = warm.sorted.apply(warm.size / 2)
        println(f"  E warm median: ${med}%,d ms  (~${med.toDouble / numL}%.1f ms/query)")
      }
      recall.foreach { r =>
        println(f"  E recall@$k (vs probe-all+refine≥50 exact ref): ${r * 100}%.2f%%")
      }
      println(s"  [config] rerankStore=$rerankStore nprobes=$nprobes refine=$refine k=$k " +
        s"|L|=$numL cacheIndex=$cacheIndex")
      println("=" * 96)
      // scalastyle:on println
    } finally spark.stop()
  }

  /** Collect a join result into `lid -> set of returned right-side ids` (the top-k per query). */
  private def neighborsById(df: DataFrame, idCol: String): Map[Int, Set[String]] = {
    df.select(col("lid"), col(idCol).cast("string").as("rid"))
      .collect()
      .groupBy(_.getInt(0))
      .map { case (lid, rows) => lid -> rows.flatMap(r => Option(r.getString(1))).toSet }
  }

  /**
   * Mean recall@k over queries: for each lid, |got ∩ truth| / min(k, |truth|). Queries with no
   * ground-truth rows (shouldn't happen with an exhaustive reference) are skipped.
   */
  private def recallAtK(
      truth: Map[Int, Set[String]],
      got: Map[Int, Set[String]],
      k: Int): Double = {
    val perQuery = truth.toSeq.flatMap { case (lid, t) =>
      if (t.isEmpty) None
      else {
        val g = got.getOrElse(lid, Set.empty)
        Some(g.intersect(t).size.toDouble / math.min(k, t.size))
      }
    }
    if (perQuery.isEmpty) 0.0 else perQuery.sum / perQuery.size
  }

  private def sampleLeft(
      spark: SparkSession,
      files: Seq[String],
      vecCol: String,
      dim: Int,
      numL: Int): DataFrame = {
    val raw = spark.read.parquet(files: _*)
      .select(col(vecCol).as("v"))
      .filter(col("v").isNotNull)
      .limit(numL)
    val vecs = raw.collect().map { r =>
      r.getAs[scala.collection.Seq[Float]]("v").toArray
    }
    require(vecs.nonEmpty, "no non-null embeddings sampled for the left side")
    val fsl = new MetadataBuilder().putLong("arrow.fixed-size-list.size", dim.toLong).build()
    val schema = new StructType(Array(
      StructField("lid", IntegerType, nullable = false),
      StructField("lvec", ArrayType(FloatType, containsNull = false), nullable = false, fsl)))
    val rows = vecs.zipWithIndex.map { case (v, i) =>
      RowFactory.create(Integer.valueOf(i), v): Row
    }
    spark.createDataFrame(rows.toList.asJava, schema)
  }

  private def listParquetFiles(spark: SparkSession, dir: String): Seq[String] = {
    val hadoopPath = new org.apache.hadoop.fs.Path(dir)
    val fs = hadoopPath.getFileSystem(spark.sparkContext.hadoopConfiguration)
    val it = fs.listFiles(hadoopPath, false)
    val buf = scala.collection.mutable.ArrayBuffer.empty[String]
    while (it.hasNext) {
      val f = it.next().getPath
      if (f.getName.endsWith(".parquet")) buf += f.toString
    }
    buf.sorted.toSeq
  }
}
