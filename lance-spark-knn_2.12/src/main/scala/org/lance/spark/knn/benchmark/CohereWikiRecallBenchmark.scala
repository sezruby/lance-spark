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

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.lance.{Dataset, ReadOptions}
import org.lance.index.{IndexOptions, IndexParams, IndexType => LanceIndexType}
import org.lance.index.vector.VectorIndexParams
import org.lance.spark.LanceRuntime
import org.lance.spark.knn.IndexedNearestJoin
import org.lance.spark.knn.internal.Metric

/**
 * Cohere Wikipedia dim=768 recall benchmark -- production-shape companion to
 * [[SiftRecallBenchmark]].
 *
 * SIFT validates the mechanics at 128-dim with natural image features. Real RAG /
 * production embedding workloads are dim=768 (`bge-base`, `E5-base`,
 * `sentence-transformers/all-mpnet-base-v2`) or dim=1024-1536 (`bge-large`, `text-embedding-3`).
 * IVF-PQ behavior at high dim is materially different: PQ quantization error grows with
 * dim, centroid Voronoi cells become more uniform, and `refineFactor` matters much more.
 *
 * Input: Cohere's `wikipedia-22-12` embeddings, pre-computed with
 * `Cohere/multilingual-22-12` at dim=768. Hosted on HuggingFace:
 *   https://huggingface.co/datasets/Cohere/wikipedia-22-12
 * Ships as Parquet files with columns `(id, title, text, url, wiki_id, paragraph_id,
 * langs, emb)` where `emb` is `list<float>` at dim=768.
 *
 * Unlike SIFT, **no ground truth is shipped**. We compute it ourselves via a brute-force
 * Spark crossJoin + top-K pass on a held-out query sample. The base set is the remainder.
 *
 * == Downloading the data ==
 *
 * English Wikipedia chunks only (35M rows), partitioned across many Parquet files:
 *
 * {{{
 *   pip install huggingface_hub
 *   huggingface-cli download Cohere/wikipedia-22-12 \
 *     --repo-type dataset \
 *     --include 'en/[star].parquet' \
 *     --local-dir /tmp/cohere-wiki
 *   # -> /tmp/cohere-wiki/en/[star].parquet  (~100 GB on disk for full EN)
 *   # (Replace [star] with the literal glob `*`. Written this way because Scaladoc's
 *   #  parser interprets `[star]/` inside a comment as the end marker.)
 * }}}
 *
 * For initial validation you don't want the full 35M. Point `COHERE_SOURCE_LIMIT` at
 * 1-10M and we'll sample via Spark.
 *
 * == What this benchmark measures ==
 *
 * 1. **Index build time** for IVF-PQ / IVF-FLAT on N real Cohere embeddings. Expect
 *    materially slower than SIFT at same N because dim=768 is 6x wider.
 * 2. **Recall@K** with ground truth computed by brute-force Spark. Distance is L2 by
 *    default; cosine also supported since these are unit-normalized embeddings and
 *    cosine ≡ L2 up to a monotone transform.
 * 3. **Latency** per query at each (nprobes, refineFactor) point. Useful for picking a
 *    production config given a recall target.
 *
 * == Cluster run ==
 *
 * {{{
 *   ./mvnw -pl lance-spark-knn_2.12 package -Pbenchmark -DskipTests
 *   # upload the fat jar + (optionally) the parquet files
 *
 *   BENCH_CLUSTER_MODE=true \
 *   BENCH_DATA_PATH=s3://bucket/cohere-bench \
 *   COHERE_PARQUET=s3://my-bucket/cohere-wiki/en \
 *   COHERE_SOURCE_LIMIT=1000000 \
 *   COHERE_NUM_QUERIES=1000 \
 *   COHERE_K=10 \
 *   COHERE_NUM_PARTITIONS=1024 \
 *   COHERE_NUM_SUB_VECTORS=96 \
 *   COHERE_NPROBES_LIST=1,4,16,64 \
 *   COHERE_REFINE_LIST=1,4,16 \
 *   spark-submit --class org.lance.spark.knn.benchmark.CohereWikiRecallBenchmark <jar>
 * }}}
 *
 * == Env knobs ==
 *
 *   - `COHERE_PARQUET=<path>`         -- source parquet dir (required). Local or object store.
 *   - `COHERE_EMB_COL=emb`            -- embedding column name (default `emb`).
 *   - `COHERE_SOURCE_LIMIT=1000000`   -- sample this many rows total (base + queries)
 *                                        before splitting (default 1M). Set `0` for full
 *                                        dataset.
 *   - `COHERE_NUM_QUERIES=1000`       -- held-out query count (default 1000). Brute-force
 *                                        GT is O(Nqueries × Nbase), so keep this modest.
 *   - `COHERE_K=10`                   -- top-K to measure (default 10).
 *   - `COHERE_METRIC=l2`              -- l2 | cosine | dot (default l2). Cohere embeddings
 *                                        are unit-normalized so cosine ≡ (1 - dot) and
 *                                        L2² = 2 - 2·dot; they produce the same top-K ordering.
 *   - `COHERE_NUM_PARTITIONS=1024`    -- IVF cluster count (default 1024 for ~1M rows;
 *                                        rule of thumb: sqrt(N) to N^(2/3)).
 *   - `COHERE_NUM_SUB_VECTORS=96`     -- PQ subvectors; must divide 768 evenly
 *                                        (default 96 = 8 dims per subvector, 8-bit codes).
 *   - `COHERE_NPROBES_LIST=1,4,16,64` -- grid of nprobes to test.
 *   - `COHERE_REFINE_LIST=1,4,16`     -- grid of refineFactor (IVF-PQ only).
 *   - `COHERE_INDEX=both`             -- `ivfpq` | `ivfflat` | `both` (default `both`).
 *   - `COHERE_SEED=1337`              -- RNG seed for base/query split.
 *   - `COHERE_SKIP_PREP=false`        -- if "true", assume Lance base + query parquet + GT
 *                                        parquet already exist at `BENCH_DATA_PATH`.
 *   - `COHERE_SKIP_INDEX=false`       -- if "true", skip index build (reuse existing).
 *   - `BENCH_CLUSTER_MODE`, `BENCH_DATA_PATH` -- same as other benchmarks.
 *
 * == What this does NOT do ==
 *
 *   - Build ground truth in parallel with index grid sweeps. GT is computed once upfront,
 *     materialized to parquet under `BENCH_DATA_PATH/gt`, then all grid points compare
 *     against the same GT.
 *   - Support cross-lingual subsets (only English configured by default). Pass
 *     `COHERE_PARQUET` at a different subdir to benchmark German / French / etc.
 *   - Assert on specific recall numbers. Unlike SIFT, published IVF-PQ numbers for real
 *     dim=768 embeddings are less standardized. Use this benchmark to pick a nprobes /
 *     refine point for YOUR recall target, not to validate against a fixed expectation.
 */
object CohereWikiRecallBenchmark {

  // -- env knobs ------------------------------------------------------------------------------

  private val ClusterMode: Boolean =
    sys.env.get("BENCH_CLUSTER_MODE").exists(_.equalsIgnoreCase("true"))
  private val DataPath: String =
    sys.env.getOrElse("BENCH_DATA_PATH", "/tmp/lance-cohere-wiki")
  private val ParquetPath: String = sys.env.getOrElse(
    "COHERE_PARQUET",
    sys.error("COHERE_PARQUET is required (path to Cohere wiki-22-12 parquet files)"))
  private val EmbCol: String = sys.env.getOrElse("COHERE_EMB_COL", "emb")
  private val SourceLimit: Long =
    sys.env.get("COHERE_SOURCE_LIMIT").map(_.toLong).getOrElse(1000000L)
  private val NumQueries: Int =
    sys.env.get("COHERE_NUM_QUERIES").map(_.toInt).getOrElse(1000)
  private val K: Int = sys.env.get("COHERE_K").map(_.toInt).getOrElse(10)
  private val MetricName: String = sys.env.getOrElse("COHERE_METRIC", "l2").toLowerCase
  private val NumPartitions: Int =
    sys.env.get("COHERE_NUM_PARTITIONS").map(_.toInt).getOrElse(1024)
  private val NumSubVectors: Int =
    sys.env.get("COHERE_NUM_SUB_VECTORS").map(_.toInt).getOrElse(96)
  private val NprobesList: Seq[Int] = sys.env
    .getOrElse("COHERE_NPROBES_LIST", "1,4,16,64").split(",").map(_.trim.toInt).toSeq
  private val RefineList: Seq[Int] = sys.env
    .getOrElse("COHERE_REFINE_LIST", "1,4,16").split(",").map(_.trim.toInt).toSeq
  private val IndexMode: String =
    sys.env.getOrElse("COHERE_INDEX", "both").toLowerCase
  private val Seed: Long = sys.env.get("COHERE_SEED").map(_.toLong).getOrElse(1337L)
  private val SkipPrep: Boolean =
    sys.env.get("COHERE_SKIP_PREP").exists(_.equalsIgnoreCase("true"))
  private val SkipIndex: Boolean =
    sys.env.get("COHERE_SKIP_INDEX").exists(_.equalsIgnoreCase("true"))

  private lazy val BaseUri = s"$DataPath/base"
  private lazy val QueriesUri = s"$DataPath/queries_parquet"
  private lazy val GtUri = s"$DataPath/gt_parquet"

  // -- main -----------------------------------------------------------------------------------

  def main(args: Array[String]): Unit = {
    val spark = buildSparkSession()
    try {
      logBanner(spark)

      if (SkipPrep) {
        println(s"[cohere] COHERE_SKIP_PREP=true -> reusing existing Lance + queries + GT")
      } else {
        prepareDatasets(spark)
      }

      val dim = detectDim(spark)
      println(s"[cohere] detected dim=$dim from base dataset")
      require(
        dim % NumSubVectors == 0,
        s"COHERE_NUM_SUB_VECTORS=$NumSubVectors does not divide dim=$dim evenly")

      val indexesToRun: Seq[String] = IndexMode match {
        case "ivfpq" => Seq("ivfpq")
        case "ivfflat" => Seq("ivfflat")
        case "both" => Seq("ivfflat", "ivfpq")
        case other => sys.error(s"Unknown COHERE_INDEX=$other (expected ivfpq|ivfflat|both)")
      }

      if (!SkipIndex) indexesToRun.foreach(buildIndex)

      // Load queries + GT once; reuse across the grid sweep.
      val queriesDf = spark.read.parquet(QueriesUri).cache()
      val queriesCount = queriesDf.count()
      val gtDf = spark.read.parquet(GtUri).cache()
      val gtCount = gtDf.count()
      println(f"[cohere] queries: $queriesCount%,d ; ground-truth rows: $gtCount%,d " +
        f"(expected ${queriesCount * K.toLong}%,d)")

      // Build a driver-side GT lookup once: Map[qid -> Set[topK-rid]].
      val gtByQid: Map[Long, Set[Long]] = gtDf
        .groupBy("qid")
        .agg(collect_list(col("rid")).as("rids"))
        .collect()
        .map(r =>
          r.getLong(0) -> r.getSeq[Long](1).take(K).toSet)
        .toMap

      indexesToRun.foreach { idx =>
        println()
        println("=" * 80)
        println(s" RECALL GRID: index=$idx, dim=$dim, metric=$MetricName, K=$K, " +
          s"base sample=$SourceLimit, queries=$queriesCount")
        println("=" * 80)
        val refineGrid: Seq[Int] = if (idx == "ivfpq") RefineList else Seq(1)
        println(
          f"${"nprobes"}%8s  ${"refine"}%6s  ${"recall@K"}%10s  ${"mean_ms"}%10s  ${"queries"}%8s")
        for (nprobes <- NprobesList; refine <- refineGrid) {
          val (recall, meanMs) = runRecallGrid(
            spark,
            queriesDf,
            gtByQid,
            nprobes = nprobes,
            refineFactor = if (idx == "ivfpq") Some(refine) else None)
          println(f"$nprobes%8d  $refine%6d  $recall%10.4f  $meanMs%10.2f  $queriesCount%8d")
        }
      }
    } finally {
      spark.stop()
    }
  }

  // -- Spark --------------------------------------------------------------------------------

  private def buildSparkSession(): SparkSession = {
    val b = SparkSession.builder().appName("CohereWikiRecallBenchmark")
    if (!ClusterMode) {
      b.master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
    }
    b.getOrCreate()
  }

  private def logBanner(spark: SparkSession): Unit = {
    println("=" * 80)
    println("CohereWikiRecallBenchmark")
    println("=" * 80)
    println(f"  Spark version:       ${spark.version}")
    println(f"  master:              ${spark.sparkContext.master}")
    println(f"  cluster mode:        $ClusterMode")
    println(f"  source parquet:      $ParquetPath")
    println(f"  emb column:          $EmbCol")
    println(f"  source sample limit: $SourceLimit%,d (0 = no limit)")
    println(f"  num queries held out: $NumQueries%,d")
    println(f"  K:                   $K")
    println(f"  metric:              $MetricName")
    println(f"  num IVF partitions:  $NumPartitions")
    println(f"  num PQ subvectors:   $NumSubVectors")
    println(f"  nprobes grid:        ${NprobesList.mkString(",")}")
    println(f"  refine grid:         ${RefineList.mkString(",")}")
    println(f"  index:               $IndexMode")
    println(f"  data path:           $DataPath")
    println(f"  seed:                $Seed")
    println("=" * 80)
    println()
  }

  // -- preparation -------------------------------------------------------------------------

  /**
   * Reads the Cohere parquet, normalises schema, samples to `SourceLimit`, splits into
   * held-out queries + base, writes base as Lance, writes queries as parquet, computes
   * brute-force top-K ground truth and writes as parquet. All three artifacts land under
   * `DataPath/` and are reused on subsequent runs with `COHERE_SKIP_PREP=true`.
   */
  private def prepareDatasets(spark: SparkSession): Unit = {
    println(s"[cohere] reading source parquet: $ParquetPath")
    val raw = spark.read.parquet(ParquetPath)
    require(
      raw.schema.fieldNames.contains(EmbCol),
      s"Source parquet does not contain $EmbCol column; fields: ${raw.schema.fieldNames.mkString(",")}")

    // Keep only an `id` (if present) and `emb` column. Rename to `rid` / `vec` for the
    // downstream Lance schema. `id` in the Cohere dataset is a string; we synthesise a
    // stable `rid: long` via monotonically_increasing_id to match our `_rowid`-oriented
    // world.
    val embColExpr: org.apache.spark.sql.Column = col(EmbCol).cast(ArrayType(FloatType))
    val withId = raw.select(embColExpr.as("vec"))
      .withColumn("rid", monotonically_increasing_id())
      .select("rid", "vec")

    val limited = if (SourceLimit > 0) withId.limit(SourceLimit.toInt) else withId

    // Split: first `NumQueries` rows (after a shuffle for randomness) become queries;
    // the rest is base. orderBy(rand(Seed)) is the cheap way to get a deterministic random
    // split without a union-all join dance later.
    val shuffled = limited.orderBy(rand(Seed))
    val queries = shuffled.limit(NumQueries).cache()
    val base = shuffled.exceptAll(queries)

    println(s"[cohere] writing base to Lance: $BaseUri")
    // Lance requires the vector column to be `FixedSizeList<Float>` for indexing.
    // Passing via `.option("vec.arrow.fixed-size-list.size", ...)` on DataFrameWriter
    // does NOT propagate to the written Lance schema (it's only honoured by the
    // TBLPROPERTIES path in `CREATE TABLE`). The `cast(ArrayType(FloatType))` above
    // also strips any field metadata the parquet reader might have surfaced.
    //
    // Working shape (mirrors BaseVectorCreateTableTest:182): build the DataFrame from
    // fresh Rows using a StructType whose `vec` StructField has the
    // `arrow.fixed-size-list.size` metadata. The driver-side round-trip is unavoidable
    // at our dims — `cast(...).withColumn(...)` drops field metadata, and spark-sql has
    // no expression API to re-tag.
    //
    // Also drops mode("overwrite") — Lance catalog's overwrite path calls drop-then-
    // create and throws NoSuchTableException when the target path has never existed.
    // Default (ErrorIfExists) is correct for first-run; set COHERE_SKIP_PREP=true on
    // rerun to reuse the existing dataset.
    val dim = detectDimFromFirstRow(base)
    val embMeta = new MetadataBuilder()
      .putLong("arrow.fixed-size-list.size", dim.toLong)
      .build()
    val taggedSchema = new StructType(Array(
      StructField("rid", LongType, nullable = false),
      StructField(
        "vec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        embMeta)))
    val baseRows = base.collect()
    println(f"[cohere]   collected ${baseRows.length}%,d rows to driver for schema retagging")
    val javaRows = new java.util.ArrayList[Row](baseRows.length)
    var i = 0
    while (i < baseRows.length) {
      val r = baseRows(i)
      val s = r.getAs[scala.collection.Seq[Float]]("vec")
      val arr = new Array[Float](s.length)
      var j = 0
      while (j < s.length) { arr(j) = s(j); j += 1 }
      javaRows.add(org.apache.spark.sql.RowFactory.create(
        java.lang.Long.valueOf(r.getLong(0)),
        arr))
      i += 1
    }
    val taggedBase = spark.createDataFrame(javaRows, taggedSchema)
    taggedBase.write.format("lance").save(BaseUri)

    println(s"[cohere] writing queries to parquet: $QueriesUri")
    // Renumber query rids so they start at 0 -- the GT rids from the BASE set must not
    // collide with query rids.
    val qidCol = row_number().over(Window.orderBy("rid")).cast(LongType) - 1L
    val queriesOut = queries.withColumn("qid", qidCol).select("qid", "vec")
    queriesOut.write.mode("overwrite").parquet(QueriesUri)

    println(s"[cohere] computing brute-force ground truth (k=$K, metric=$MetricName) -> $GtUri")
    computeAndWriteGroundTruth(spark, queriesOut, base)
  }

  /**
   * Compute recall-1.0 ground truth by brute-force crossJoin + distance + top-K window.
   * Cost: `O(Nqueries × Nbase)` distance evaluations. At Nqueries=1000 × Nbase=1M ×
   * dim=768, that's ~1 TB of computations; still minutes on a modest cluster.
   *
   * Output schema: `(qid: long, rid: long, rank: int)` with `rank` in [0, K). Written as
   * parquet for reuse across grid sweeps.
   */
  private def computeAndWriteGroundTruth(
      spark: SparkSession,
      queriesDf: DataFrame,
      baseDf: DataFrame): Unit = {
    val distanceExpr = MetricName match {
      case "l2" => l2DistSq(col("q.vec"), col("b.vec")).as("dist")
      case "cosine" =>
        // Assuming unit-normalized vectors (true for Cohere embeddings), cosine ≡ 1 - dot.
        // Since 1 - dot is monotone in -dot, we use negative dot as the distance and sort
        // ASC below -- same top-K as cosine distance ASC.
        (-dotProduct(col("q.vec"), col("b.vec"))).as("dist")
      case "dot" =>
        // For raw dot, "nearest" = largest dot. Negate to use ASC sort semantics uniformly.
        (-dotProduct(col("q.vec"), col("b.vec"))).as("dist")
      case other => sys.error(s"Unsupported COHERE_METRIC=$other (expected l2|cosine|dot)")
    }

    val crossed = queriesDf.as("q")
      .crossJoin(baseDf.as("b"))
      .select(col("q.qid"), col("b.rid"), distanceExpr)

    val w = Window.partitionBy("qid").orderBy(col("dist").asc)
    val topK = crossed
      .withColumn("rank", row_number().over(w) - 1)
      .where(col("rank") < K)
      .select("qid", "rid", "rank")

    topK.write.mode("overwrite").parquet(GtUri)
  }

  /**
   * L2² distance via element-wise subtract + square + sum. Spark 3.5+ has
   *  `vector_l2_distance` but we can't count on version; keep it portable.
   */
  private def l2DistSq(
      a: org.apache.spark.sql.Column,
      b: org.apache.spark.sql.Column): org.apache.spark.sql.Column = {
    aggregate(
      zip_with(
        a,
        b,
        (x, y) => {
          val d = x - y
          d * d
        }),
      lit(0.0f),
      (acc, v) => acc + v)
  }

  /** Inner product: element-wise multiply + sum. */
  private def dotProduct(
      a: org.apache.spark.sql.Column,
      b: org.apache.spark.sql.Column): org.apache.spark.sql.Column = {
    aggregate(
      zip_with(a, b, (x, y) => x * y),
      lit(0.0f),
      (acc, v) => acc + v)
  }

  // -- schema / dim detection ---------------------------------------------------------------

  private def detectDim(spark: SparkSession): Int = {
    val base = spark.read.format("lance").load(BaseUri)
    detectDimFromFirstRow(base)
  }

  private def detectDimFromFirstRow(df: DataFrame): Int = {
    val first: Row = df.select("vec").head(1).head
    first.getSeq[Float](0).length
  }

  // -- index build --------------------------------------------------------------------------

  private def buildIndex(kind: String): Unit = {
    println(s"[cohere] building $kind index on $BaseUri " +
      s"(numPartitions=$NumPartitions" +
      (if (kind == "ivfpq") s", numSubVectors=$NumSubVectors" else "") + ")")
    val t0 = System.nanoTime()
    val ds = Dataset.open().uri(BaseUri).allocator(LanceRuntime.allocator())
      .readOptions(new ReadOptions.Builder().build()).build()
    try {
      val metric = MetricName match {
        case "l2" => Metric.L2
        case "cosine" => Metric.Cosine
        case "dot" => Metric.Dot
        case other => sys.error(s"Unsupported COHERE_METRIC=$other")
      }
      val vectorParams = kind match {
        case "ivfpq" =>
          VectorIndexParams.ivfPq(NumPartitions, NumSubVectors, 8, metric.lanceType, 50)
        case "ivfflat" =>
          VectorIndexParams.ivfFlat(NumPartitions, metric.lanceType)
      }
      val idxParams = IndexParams.builder().setVectorIndexParams(vectorParams).build()
      val opts = IndexOptions.builder(
        java.util.Collections.singletonList("vec"),
        LanceIndexType.VECTOR,
        idxParams).build()
      ds.createIndex(opts)
    } finally ds.close()
    val sec = (System.nanoTime() - t0) / 1e9
    println(f"[cohere] $kind build complete in $sec%.1f s")
  }

  // -- recall evaluation --------------------------------------------------------------------

  /**
   * Run all queries through the indexed nearest-join at one (nprobes, refine) point,
   * compute mean recall@K vs `gtByQid`. Returns (meanRecall, meanLatencyMs).
   */
  private def runRecallGrid(
      spark: SparkSession,
      queriesDf: DataFrame,
      gtByQid: Map[Long, Set[Long]],
      nprobes: Int,
      refineFactor: Option[Int]): (Double, Double) = {
    // Normalise the queries DF schema to what kNearestJoin expects (a left vector column
    // named `qvec`). The stored queries parquet has `(qid, vec)`; rename.
    val left = queriesDf.select(col("qid").as("lid"), col("vec").as("qvec"))

    val t0 = System.nanoTime()
    val joined = IndexedNearestJoin(
      left = left,
      rightLanceUri = BaseUri,
      leftVecCol = "qvec",
      rightVecCol = "vec",
      k = K,
      metric = MetricName,
      rightProjection = Some(Seq("rid")),
      nprobes = Some(nprobes),
      refineFactor = refineFactor)
    val collected = joined.collect()
    val elapsedMs = (System.nanoTime() - t0) / 1e6
    val nQueries = left.count()

    // Schema of `collected`: [lid, qvec, rid, __score]. Positions: 0,1,2,3.
    val actualByQid = collected
      .groupBy(_.getLong(0))
      .map { case (qid, rowsArr) =>
        qid -> rowsArr.toSeq.sortBy(_.getFloat(3)).take(K).map(_.getLong(2)).toSet
      }

    var recallSum = 0.0
    gtByQid.keys.foreach { qid =>
      val expected = gtByQid(qid)
      val got = actualByQid.getOrElse(qid, Set.empty[Long])
      recallSum += got.intersect(expected).size.toDouble / K
    }
    val meanRecall = recallSum / gtByQid.size
    val meanLatencyMs = elapsedMs / nQueries
    (meanRecall, meanLatencyMs)
  }
}
