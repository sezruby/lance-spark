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

import org.apache.spark.sql.{Row, RowFactory, SparkSession}
import org.apache.spark.sql.types._
import org.lance.{Dataset, ReadOptions}
import org.lance.index.IndexParams
import org.lance.index.vector.{IvfBuildParams, PQBuildParams, VectorIndexParams}
import org.lance.spark.LanceRuntime
import org.lance.spark.knn.IndexedNearestJoin
import org.lance.spark.knn.internal.Metric

import java.io.{BufferedInputStream, DataInputStream, File, FileInputStream, IOException}
import java.nio.{ByteBuffer, ByteOrder}

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * SIFT1M / SIFT10K recall benchmark.
 *
 * Validates the IVF-PQ and IVF-FLAT indexed paths against the canonical SIFT1M ground
 * truth from http://corpus-texmex.irisa.fr. For each query the dataset ships, we compute
 * the top-K nearest in the base set using Lance's indexed nearest-join and compare
 * against the shipped ground truth. Reports recall@K for each configuration.
 *
 * This is production-readiness validation #3 ("real-embeddings recall validation"). Where
 * [[IndexedNearestJoinIvfPqRecallTest]] uses synthetic random vectors and so only proves
 * the mechanics, this benchmark uses the standard ANN-benchmark corpus — so the recall
 * numbers are comparable to published IVF-PQ results from the HNSWlib / FAISS papers.
 *
 * == Downloading the data ==
 *
 * SIFT1M (small -- 168 MB compressed, 500 MB uncompressed):
 * {{{
 *   curl -L -o /tmp/sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
 *   tar -xzf /tmp/sift.tar.gz -C /tmp/
 *   # produces /tmp/sift/{sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs,
 *   #                    sift_learn.fvecs}
 * }}}
 *
 * SIFT10K (tiny -- 10 MB, useful for smoke testing):
 * {{{
 *   curl -L -o /tmp/siftsmall.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
 *   tar -xzf /tmp/siftsmall.tar.gz -C /tmp/
 * }}}
 *
 * Formats:
 *   - `.fvecs` — [int32 dim, float32 * dim, int32 dim, float32 * dim, ...] (little-endian).
 *     Dim header repeats per vector; we read the first to establish dim and assume all
 *     match.
 *   - `.ivecs` — same but int32 payloads (used for ground-truth top-K indices).
 *
 * == Cluster run ==
 *
 * {{{
 *   ./mvnw -pl lance-spark-knn_2.12 package -Pbenchmark -DskipTests
 *   # upload target/lance-spark-knn_2.12-<v>-benchmark.jar + the unpacked SIFT dir
 *
 *   BENCH_CLUSTER_MODE=true \
 *   BENCH_DATA_PATH=s3://bucket/path \
 *   SIFT_DIR=/tmp/sift \
 *   SIFT_K=10 \
 *   SIFT_NUM_PARTITIONS=256 \
 *   SIFT_NUM_SUB_VECTORS=16 \
 *   SIFT_NPROBES_LIST=1,4,16,64 \
 *   SIFT_REFINE_LIST=1,4,8 \
 *   SIFT_NUM_QUERIES=1000 \
 *   spark-submit --class org.lance.spark.knn.benchmark.SiftRecallBenchmark <jar>
 * }}}
 *
 * == Env knobs ==
 *
 *   - `SIFT_DIR=/path/to/sift`       -- directory containing the extracted .fvecs/.ivecs
 *                                       (required). Expected files:
 *                                       `sift_base.fvecs`, `sift_query.fvecs`,
 *                                       `sift_groundtruth.ivecs`.
 *                                       For siftsmall, files are prefixed `siftsmall_`;
 *                                       set `SIFT_PREFIX=siftsmall` to use them.
 *   - `SIFT_PREFIX=sift`             -- file prefix (default: `sift`; set `siftsmall` for
 *                                       the 10K subset).
 *   - `SIFT_K=10`                    -- top-K to measure recall@K against ground truth
 *                                       (default 10). SIFT ships 100 ground-truth neighbors
 *                                       per query, so K ≤ 100.
 *   - `SIFT_NUM_QUERIES=1000`        -- how many queries to run (default: all 10000). Lower
 *                                       numbers give faster feedback loops.
 *   - `SIFT_NUM_PARTITIONS=256`      -- IVF cluster count for IVF-PQ (default: sqrt(N)).
 *   - `SIFT_NUM_SUB_VECTORS=16`      -- PQ sub-vector count; must divide 128 (SIFT dim).
 *                                       Default: 16 (=> 8-byte PQ codes).
 *   - `SIFT_NPROBES_LIST=1,4,16,64`  -- comma list of `nprobes` values to test.
 *   - `SIFT_REFINE_LIST=1,4,8`       -- comma list of `refineFactor` values to test.
 *   - `SIFT_INDEX=ivfpq`             -- `ivfpq` | `ivfflat` | `both` (default: `both`).
 *   - `SIFT_SKIP_WRITE=false`        -- if "true", assumes the Lance dataset already
 *                                       exists at `BENCH_DATA_PATH`/sift.
 *   - `SIFT_SKIP_INDEX=false`        -- if "true", assumes an index already exists. Useful
 *                                       for running a grid sweep against a pre-built index.
 *   - `BENCH_CLUSTER_MODE`, `BENCH_DATA_PATH` -- same as other benchmarks.
 *
 * == Expected numbers ==
 *
 * On SIFT1M × 128-dim × 10K queries, with IVF-PQ(256 clusters, 16 subvectors, 8 bits) at
 * K=10, published ANN-benchmark numbers for FAISS IVF-PQ are in this ballpark:
 *
 *   nprobes=1:   ~0.20 recall@10, ~1 ms/query  (barely touches the right centroid)
 *   nprobes=4:   ~0.55 recall@10
 *   nprobes=16:  ~0.85 recall@10, ~5 ms/query
 *   nprobes=64:  ~0.97 recall@10, ~15 ms/query
 *   nprobes=256: 1.00 recall@10 (visits every centroid -> exact)
 *
 * With `refineFactor=8` the numbers shift up meaningfully: PQ compresses distance to
 * 8-byte codes so Voronoi selection is accurate but ranking-within-cluster is lossy; the
 * refine pass re-ranks top-K*8 by exact distance. Good recall targets with refine:
 *
 *   nprobes=4,  refine=8: ~0.85 recall@10
 *   nprobes=16, refine=8: ~0.97 recall@10
 *
 * If the numbers this benchmark prints are dramatically lower than published figures,
 * that's a signal the indexed-probe path or IVF-PQ construction has a bug.
 */
object SiftRecallBenchmark {

  // -- env knobs ------------------------------------------------------------------------------

  private val ClusterMode: Boolean =
    sys.env.get("BENCH_CLUSTER_MODE").exists(_.equalsIgnoreCase("true"))
  private val DataPath: String = sys.env.getOrElse("BENCH_DATA_PATH", "/tmp/lance-sift")
  private val SiftDir: String =
    sys.env.getOrElse("SIFT_DIR", sys.error("SIFT_DIR is required (path to unpacked sift/*.fvecs)"))
  private val SiftPrefix: String = sys.env.getOrElse("SIFT_PREFIX", "sift")
  private val K: Int = sys.env.get("SIFT_K").map(_.toInt).getOrElse(10)
  private val NumQueries: Int =
    sys.env.get("SIFT_NUM_QUERIES").map(_.toInt).getOrElse(10000)
  private val NumPartitions: Int =
    sys.env.get("SIFT_NUM_PARTITIONS").map(_.toInt).getOrElse(256)
  private val NumSubVectors: Int =
    sys.env.get("SIFT_NUM_SUB_VECTORS").map(_.toInt).getOrElse(16)
  private val NprobesList: Seq[Int] =
    sys.env.getOrElse("SIFT_NPROBES_LIST", "1,4,16,64").split(",").map(_.trim.toInt).toSeq
  private val RefineList: Seq[Int] =
    sys.env.getOrElse("SIFT_REFINE_LIST", "1,4,8").split(",").map(_.trim.toInt).toSeq
  private val IndexType: String =
    sys.env.getOrElse("SIFT_INDEX", "both").toLowerCase
  private val SkipWrite: Boolean =
    sys.env.get("SIFT_SKIP_WRITE").exists(_.equalsIgnoreCase("true"))
  private val SkipIndex: Boolean =
    sys.env.get("SIFT_SKIP_INDEX").exists(_.equalsIgnoreCase("true"))

  // -- main -----------------------------------------------------------------------------------

  def main(args: Array[String]): Unit = {
    val spark = buildSparkSession()
    try {
      logBanner(spark)

      val baseFile = s"$SiftDir/${SiftPrefix}_base.fvecs"
      val queryFile = s"$SiftDir/${SiftPrefix}_query.fvecs"
      val gtFile = s"$SiftDir/${SiftPrefix}_groundtruth.ivecs"
      Seq(baseFile, queryFile, gtFile).foreach { path =>
        if (!new File(path).exists()) {
          sys.error(s"Missing SIFT file: $path. See scaladoc for download instructions.")
        }
      }

      val lanceUri = s"$DataPath/${SiftPrefix}_base"

      if (SkipWrite) {
        println(s"[sift] SOAK_SKIP_WRITE=true -> using existing Lance dataset at $lanceUri")
      } else {
        writeBaseAsLance(spark, baseFile, lanceUri)
      }

      val indexesToRun: Seq[String] = IndexType match {
        case "ivfpq" => Seq("ivfpq")
        case "ivfflat" => Seq("ivfflat")
        case "both" => Seq("ivfflat", "ivfpq")
        case other => sys.error(s"Unknown SIFT_INDEX=$other (expected ivfpq|ivfflat|both)")
      }

      if (!SkipIndex) {
        indexesToRun.foreach(idx => buildIndex(lanceUri, idx))
      } else {
        println("[sift] SOAK_SKIP_INDEX=true -> using existing index(es); no build")
      }

      val queries = loadFvecs(queryFile, limit = NumQueries)
      val groundTruth = loadIvecs(gtFile, limit = NumQueries)
      println(f"[sift] loaded ${queries.size}%,d queries × dim ${queries.head.length}, " +
        f"ground truth with ${groundTruth.head.length}%,d neighbors per query")
      require(
        groundTruth.head.length >= K,
        s"K=$K > ground-truth neighbors-per-query (${groundTruth.head.length}). " +
          s"Lower K or rebuild ground truth.")

      indexesToRun.foreach { idx =>
        println()
        println("=" * 80)
        println(s" RECALL GRID: index=$idx, K=$K, nprobes=${NprobesList.mkString(",")}" +
          (if (idx == "ivfpq") s", refineFactor=${RefineList.mkString(",")}" else ""))
        println("=" * 80)
        // IVF-FLAT ignores refineFactor (no PQ codes to re-rank), so only iterate `nprobes`.
        val refineGrid: Seq[Int] = if (idx == "ivfpq") RefineList else Seq(1)
        println(
          f"${"nprobes"}%8s  ${"refine"}%6s  ${"recall@K"}%10s  ${"mean_ms"}%10s  ${"queries"}%8s")
        for (nprobes <- NprobesList; refine <- refineGrid) {
          val (recall, meanMs) = runRecallGrid(
            spark,
            lanceUri,
            queries,
            groundTruth,
            nprobes = nprobes,
            refineFactor = if (idx == "ivfpq") Some(refine) else None)
          println(f"$nprobes%8d  $refine%6d  $recall%10.4f  $meanMs%10.2f  ${queries.size}%8d")
        }
      }
    } finally {
      spark.stop()
    }
  }

  // -- Spark session --------------------------------------------------------------------------

  private def buildSparkSession(): SparkSession = {
    val b = SparkSession.builder().appName("SiftRecallBenchmark")
    if (!ClusterMode) {
      b.master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
    }
    b.getOrCreate()
  }

  private def logBanner(spark: SparkSession): Unit = {
    println("=" * 80)
    println(s"SiftRecallBenchmark")
    println("=" * 80)
    println(f"  Spark version:     ${spark.version}")
    println(f"  master:            ${spark.sparkContext.master}")
    println(f"  cluster mode:      $ClusterMode")
    println(f"  SIFT dir:          $SiftDir")
    println(f"  SIFT prefix:       $SiftPrefix")
    println(f"  data path:         $DataPath")
    println(f"  index type:        $IndexType")
    println(f"  num IVF parts:     $NumPartitions")
    println(f"  num PQ subvectors: $NumSubVectors")
    println(f"  K:                 $K")
    println(f"  num queries:       $NumQueries")
    println(f"  nprobes grid:      ${NprobesList.mkString(",")}")
    println(f"  refine grid:       ${RefineList.mkString(",")}")
    println("=" * 80)
    println()
  }

  // -- fvecs / ivecs I/O ----------------------------------------------------------------------

  /**
   * Read an .fvecs / .ivecs file into a Seq of arrays. The file format is a concatenation
   * of (int32 dim, payload[dim]) records, little-endian. `limit` caps the number of
   * records read; the loader stops gracefully at EOF. `readElement` reads one 4-byte
   * payload element (float via `intBitsToFloat` for fvecs, raw int for ivecs).
   */
  private def loadVecs[T: scala.reflect.ClassTag](
      path: String,
      limit: Int,
      readElement: Int => T): IndexedSeq[Array[T]] = {
    val in = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))
    try {
      val out = new mutable.ArrayBuffer[Array[T]]()
      var continue = true
      var i = 0
      while (continue && i < limit) {
        val buf = new Array[Byte](4)
        val n = in.read(buf)
        if (n < 4) { continue = false }
        else {
          val dim = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).getInt
          if (dim < 0) throw new IOException(s"Invalid dim $dim at vector $i in $path")
          val v = new Array[T](dim)
          var j = 0
          while (j < dim) {
            v(j) = readElement(readLeInt(in))
            j += 1
          }
          out += v
          i += 1
        }
      }
      out.toIndexedSeq
    } finally in.close()
  }

  private def loadFvecs(path: String, limit: Int = Int.MaxValue): IndexedSeq[Array[Float]] =
    loadVecs[Float](path, limit, java.lang.Float.intBitsToFloat)

  private def loadIvecs(path: String, limit: Int = Int.MaxValue): IndexedSeq[Array[Int]] =
    loadVecs[Int](path, limit, identity)

  /** Read one little-endian int32 from the stream. Throws `IOException` on short read. */
  private def readLeInt(in: DataInputStream): Int = {
    val buf = new Array[Byte](4)
    val n = in.read(buf)
    if (n != 4) throw new IOException(s"Short read: expected 4 bytes, got $n")
    ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).getInt
  }

  // -- Lance dataset write + index build ------------------------------------------------------

  /**
   * Write SIFT base vectors to a Lance dataset. Uses distributed Spark write -- avoids
   * loading all 1M × 128-dim × 4B = 512 MB into driver memory.
   */
  private def writeBaseAsLance(spark: SparkSession, baseFile: String, lanceUri: String): Unit = {
    println(s"[sift] writing base dataset to Lance: $baseFile -> $lanceUri")
    val t0 = System.nanoTime()

    // Load base vectors on driver (file is ~500 MB for SIFT1M; fine for driver heap of 4 GB+).
    val baseVecs = loadFvecs(baseFile, limit = Int.MaxValue)
    val dim = baseVecs.head.length
    println(f"[sift] base has ${baseVecs.size}%,d vectors × dim $dim")

    val schema = new StructType(Array(
      StructField("rid", LongType, nullable = false),
      StructField(
        "vec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", dim.toLong).build())))

    // Parallelize to cluster default parallelism * 4 for good write fan-out.
    val parallelism = math.max(spark.sparkContext.defaultParallelism * 4, 16)
    val rdd = spark.sparkContext
      .parallelize(baseVecs.indices, parallelism)
      .map { i =>
        // Pass Array[Float] directly; Scala 2.12 `.toSeq.asJava` produces a Java List via
        // Wrappers$SeqWrapper that Spark's ArrayType encoder rejects.
        RowFactory.create(java.lang.Long.valueOf(i.toLong), baseVecs(i)): Row
      }
    // mode("overwrite") on a non-existent Lance path throws NoSuchTableException via the
    // catalog's drop-before-create path. Default ErrorIfExists is correct for first-run;
    // set SIFT_SKIP_WRITE=true to reuse an existing dataset.
    spark.createDataFrame(rdd, schema)
      .write.format("lance").save(lanceUri)

    val sec = (System.nanoTime() - t0) / 1e9
    println(f"[sift] write complete in $sec%.1f s")
  }

  /**
   * Build an IVF-PQ or IVF-FLAT index on the Lance dataset. Must be called on the driver
   * (Lance's Java SDK builds the index in-process using the opened dataset handle).
   */
  private def buildIndex(lanceUri: String, kind: String): Unit = {
    println(s"[sift] building $kind index on $lanceUri (numPartitions=$NumPartitions" +
      (if (kind == "ivfpq") s", numSubVectors=$NumSubVectors" else "") + ")")
    val t0 = System.nanoTime()
    val ds = Dataset.open().uri(lanceUri).allocator(LanceRuntime.allocator())
      .readOptions(new ReadOptions.Builder().build()).build()
    try {
      val vectorParams = kind match {
        case "ivfpq" =>
          // Use the explicit builder form — the 5-arg positional `ivfPq(partitions,
          // subvectors, bits, metric, maxIters)` path had the bits/subvectors params swapped
          // somewhere between Scala call site and Rust side, yielding
          // "num_bits 16 not supported" on a call that passed bits=8. Named setters
          // eliminate that ambiguity.
          val ivf = new IvfBuildParams.Builder()
            .setNumPartitions(NumPartitions)
            .setMaxIters(50)
            .build()
          val pq = new PQBuildParams.Builder()
            .setNumSubVectors(NumSubVectors)
            .setNumBits(8)
            .setMaxIters(50)
            .build()
          VectorIndexParams.withIvfPqParams(Metric.L2.lanceType, ivf, pq)
        case "ivfflat" =>
          VectorIndexParams.ivfFlat(NumPartitions, Metric.L2.lanceType)
      }
      val idxParams = IndexParams.builder().setVectorIndexParams(vectorParams).build()
      // Give each index a kind-specific name so `SIFT_INDEX=both` can build IVF-FLAT and
      // IVF-PQ on the same `vec` column. Lance's default is `<column>_idx` which collides
      // when a second index is built on the same column.
      val opts = org.lance.index.IndexOptions.builder(
        java.util.Collections.singletonList("vec"),
        org.lance.index.IndexType.VECTOR,
        idxParams)
        .withIndexName(s"vec_${kind}_idx")
        .build()
      ds.createIndex(opts)
    } finally ds.close()
    val sec = (System.nanoTime() - t0) / 1e9
    println(f"[sift] $kind build complete in $sec%.1f s")
  }

  // -- recall evaluation ----------------------------------------------------------------------

  /**
   * Run all `queries` against the indexed Lance dataset and compute recall@K against the
   * shipped ground truth. Returns (meanRecall, meanLatencyMs).
   *
   * All queries are submitted as a single `IndexedNearestJoin.apply` call (left DF has one
   * row per query). Lance parallelises the probes internally + via Spark's per-task
   * `LanceProbe`. Total wall-clock divided by query count gives mean latency.
   */
  private def runRecallGrid(
      spark: SparkSession,
      lanceUri: String,
      queries: IndexedSeq[Array[Float]],
      groundTruth: IndexedSeq[Array[Int]],
      nprobes: Int,
      refineFactor: Option[Int]): (Double, Double) = {
    val dim = queries.head.length

    val leftSchema = new StructType(Array(
      StructField("lid", LongType, nullable = false),
      StructField(
        "qvec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", dim.toLong).build())))

    val rows = new java.util.ArrayList[Row](queries.size)
    var i = 0
    while (i < queries.size) {
      // Same Array[Float]-not-Java-List rule as writeBaseAsLance.
      rows.add(RowFactory.create(java.lang.Long.valueOf(i.toLong), queries(i)))
      i += 1
    }
    val left = spark.createDataFrame(rows, leftSchema)

    val right = spark.read.format("lance").load(lanceUri)

    val t0 = System.nanoTime()
    val joined = IndexedNearestJoin(
      left = left,
      rightLanceUri = lanceUri,
      leftVecCol = "qvec",
      rightVecCol = "vec",
      k = K,
      metric = "l2",
      rightProjection = Some(Seq("rid")),
      nprobes = Some(nprobes),
      refineFactor = refineFactor)
    // Bring to driver. Keeping it small (NumQueries × K rows × a couple of cols).
    val collected = joined.collect()
    val elapsedMs = (System.nanoTime() - t0) / 1e6

    // Output schema is `left.fields ++ right.fields :+ score` (see
    // IndexedNearestJoin.buildOutputSchema). Left is [lid:Long, qvec:Array], right
    // projection is [rid:Long], so indices are [0:lid, 1:qvec, 2:rid, 3:__score].
    val actual = collected.groupBy(_.getLong(0)).map { case (lid, rowsArr) =>
      lid -> rowsArr.toSeq.sortBy(_.getFloat(3)).take(K).map(_.getLong(2).toInt).toSet
    }

    var recallSum = 0.0
    var q = 0
    while (q < queries.size) {
      val expected = groundTruth(q).take(K).toSet
      val got = actual.getOrElse(q.toLong, Set.empty[Int])
      recallSum += got.intersect(expected).size.toDouble / K
      q += 1
    }
    val meanRecall = recallSum / queries.size
    val meanLatencyMs = elapsedMs / queries.size
    (meanRecall, meanLatencyMs)
  }
}
