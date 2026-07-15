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

import org.apache.spark.sql.SparkSession
import org.lance.index.external.{ExternalIvfPqIndex, ExternalIvfPqIndexParams}

import scala.collection.JavaConverters._

/**
 * Distributed build of the external IVF-PQ index across Spark executors.
 *
 * The single-box path ([[ExternalIndexLifecycle.buildOrReuse]] → `ExternalIvfPqIndex.build`) reads
 * the entire corpus on the driver — at 10M+ rows / dim=1024 that 40+ GB scan is the dominant build
 * cost and leaves executors idle. This builder splits the scan across the cluster:
 *
 *   1. '''driver''' trains the IVF centroids + PQ codebook on a sample of the whole corpus and
 *      serializes them to a broadcast payload (~KBs). Cheap — sample-sized regardless of |R|.
 *   2. '''executors''' each take a shard of the (globally-indexed) file list and, using the
 *      broadcast quantizers, assign + PQ-encode their files into a shard parquet on shared storage.
 *      This is the 40+ GB read, now `numExecutors`-way parallel.
 *   3. '''driver''' merges the shard parquets into the final `index.idx` (heap-merge by partition;
 *      only the PQ codes — ~`num_sub_vectors` bytes/row — flow back), then writes `manifest.json`.
 *
 * Correctness rests on partition assignment being pure over (centroids, codebook): two executors
 * with the same broadcast payload produce mergeable partition-binned output. This is proven at the
 * Rust layer (`merge_from_persisted_shards_matches`): a sharded build yields top-K distances
 * identical to a whole-corpus build sharing the same quantizers.
 *
 * The global `file_id` of each file (its position in the sorted list) is carried into the shard so
 * emitted rids stay globally consistent — this is what lets independently-built shards merge.
 *
 * NOTE: the rerank sidecar is NOT yet built distributed. When `params.getRerankStore` is not NONE
 * this falls back to the driver-side build so the sidecar is produced correctly; distributing the
 * sidecar (offset-aware shard writes) is a follow-up.
 */
private[knn] object DistributedExternalIndexBuild {

  /**
   * Build the index distributed and return the URI of the `<uuid>` index directory (suitable for
   * `ExternalIvfPqIndex.open`), mirroring [[ExternalIndexLifecycle.buildOrReuse]]'s return.
   *
   * @param filesPerTask how many files each executor task handles (shard granularity). One file
   *   per task maximizes parallelism; a few per task cuts task overhead. Default 1.
   */
  def build(
      spark: SparkSession,
      filePaths: Seq[String],
      vectorColumn: String,
      params: ExternalIvfPqIndexParams,
      filesPerTask: Int = 1): String = {
    require(filePaths.nonEmpty, "filePaths must be non-empty")

    // Sidecar isn't distributed yet — fall back to the driver build so it's correct.
    if (params.getRerankStore != ExternalIvfPqIndexParams.RerankStore.NONE) {
      // scalastyle:off println
      println(
        "[DistributedExternalIndexBuild] rerank store requested; sidecar build is not yet " +
          "distributed — falling back to driver-side build.")
      // scalastyle:on println
      return ExternalIndexLifecycle.buildOrReuse(spark, filePaths, vectorColumn, params)
    }

    // Sorted file order fixes each file's GLOBAL file_id (its position). Both the shard rids and
    // the manifest must agree on this order.
    val sortedFiles = filePaths.sorted.toArray
    val scratch = ExternalIndexLifecycle.resolveScratchDir(spark)
    val runId = java.util.UUID.randomUUID().toString
    val indexDir = s"$scratch/$runId"
    val shardDir = s"$indexDir/shards"

    // Phase 1 (driver): train on a sample of the FULL corpus, serialize to broadcast bytes.
    val payload: Array[Byte] =
      ExternalIvfPqIndex.trainBroadcast(sortedFiles.toList.asJava, vectorColumn, params)
    val bcPayload = spark.sparkContext.broadcast(payload)

    // Build the (globalFileId, path) work items, grouped into shards of `filesPerTask`.
    val indexed: Array[(String, Int)] = sortedFiles.zipWithIndex
    val shards: Array[(Int, Seq[String])] = indexed
      .grouped(filesPerTask.max(1))
      .map { grp => (grp.head._2, grp.map(_._1).toSeq) } // (offset = first global id, files)
      .toArray

    val vc = vectorColumn
    // Serialize params to primitives so the closure doesn't capture the Java object.
    val p = ParamsBlob.from(params)

    // Phase 2 (executors): each shard → its own parquet. Parallelism = number of shards.
    val numShards = shards.length
    val shardUris: Array[String] = spark.sparkContext
      .parallelize(shards.toSeq, numShards)
      .map { case (offset, files) =>
        val shardUri = s"$shardDir/shard-$offset.parquet"
        ExternalIvfPqIndex.buildShard(
          bcPayload.value,
          files.asJava,
          vc,
          offset,
          shardUri,
          p.toParams)
        shardUri
      }
      .collect()

    // Phase 3 (driver): merge shard parquets into <indexDir>/index.idx and write manifest.json
    // alongside (native mergeShards does both), leaving an openable index directory.
    ExternalIvfPqIndex.mergeShards(
      payload,
      shardUris.toList.asJava,
      sortedFiles.toList.asJava,
      vc,
      indexDir,
      params)

    // Clean up the whole run dir (shards + index) on app end / JVM shutdown, same as the
    // single-box lifecycle.
    LanceTempLifecycle.register(spark, indexDir)
    bcPayload.destroy()
    indexDir
  }

  /** Params captured as primitives so the map closure stays serializable. */
  final private case class ParamsBlob(
      numPartitions: Int,
      numSubVectors: Int,
      numBits: Int,
      metric: String,
      maxIters: Int,
      sampleRate: Int,
      seed: Long,
      rerank: String)
    extends Serializable {
    def toParams: ExternalIvfPqIndexParams =
      ExternalIvfPqIndexParams
        .builder()
        .numPartitions(numPartitions)
        .numSubVectors(numSubVectors)
        .numBitsPerSubVector(numBits)
        .metric(ExternalIvfPqIndexParams.Metric.valueOf(metric))
        .maxIters(maxIters)
        .sampleRate(sampleRate)
        .seed(seed)
        .rerankStore(ExternalIvfPqIndexParams.RerankStore.valueOf(rerank))
        .build()
  }

  private object ParamsBlob {
    def from(p: ExternalIvfPqIndexParams): ParamsBlob =
      ParamsBlob(
        p.getNumPartitions,
        p.getNumSubVectors,
        p.getNumBitsPerSubVector,
        p.getMetric.name(),
        p.getMaxIters,
        p.getSampleRate,
        p.getSeed,
        p.getRerankStore.name())
  }
}
