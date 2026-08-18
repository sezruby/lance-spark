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
package org.lance.spark.knn

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types._
import org.lance.spark.knn.internal.{LanceKnnJoinStage, Metric}

/**
 * Public entry point for the indexed nearest-by join over a Lance dataset.
 *
 * The join runs entirely inside one `mapPartitions` over the left DataFrame, with NO shuffle: each
 * partition opens R's index once and, per left row, runs a native `LanceProbe.probe(...)` (a
 * complete distributed top-K search in Lance's own threads), trims the overfetched candidates to
 * `k`, and late-materializes the surviving right rows by `_rowid`. See
 * [[org.lance.spark.knn.internal.LanceKnnJoinStage]] for why a probe → shuffle → merge →
 * materialize pipeline only adds cost over letting the single native call do the work.
 *
 * The result is assembled through public Spark APIs (`left.rdd.mapPartitions` +
 * `SparkSession.createDataFrame`) rather than a custom Catalyst node; the SQL module wires the same
 * `LanceKnnJoinStage.runPartition` into a physical operator for the `NEAREST BY` rewrite.
 */
object IndexedNearestJoin {

  /**
   * Run an approximate-nearest-neighbor join.
   *
   * @param left          left DataFrame; one query vector per row in `leftVecCol`
   * @param rightLanceUri Lance dataset URI for the right side
   * @param leftVecCol    name of the vector column in `left`. Must be `ArrayType[Float]`.
   * @param rightVecCol   name of the indexed (or to-be-searched) vector column on the right
   * @param k             top-K rows per left row
   * @param metric        distance/similarity metric: "l2" / "cosine" / "dot" (and synonyms)
   * @param rightProjection columns to materialize from the right side. Defaults to all data
   *                      columns. The score column is added separately.
   * @param outerJoin     when true, left rows with zero matches are preserved with NULL right-
   *                      side columns. Defaults to false (inner-join semantics).
   * @param scoreCol      name of the synthesized score column added to the output. Defaults to
   *                      `__score`.
   * @param overfetch     ratio of internal candidates to `k`. Lance fetches `k × overfetch`
   *                      candidates natively and this stage trims to `k`. Defaults to 1.
   * @param nprobes       optional override of Lance's `nprobes` for IVF-PQ indexes
   * @param version       optional Lance version pin; if unset, latest version is used
   * @param refineFactor  IVF-PQ recall knob. When set, Lance fetches `k * refineFactor`
   *                      approximate candidates, re-ranks them with exact distance, and trims
   *                      back to k. Higher = better recall, more compute. `None` leaves Lance's
   *                      default (= 1, no re-rank). Ignored for non-IVF-PQ indexes / unindexed.
   * @param ef            HNSW search depth. Higher = better recall, more compute. `None` leaves
   *                      Lance's default (the index's build-time `ef_construction` value).
   *                      Ignored for non-HNSW indexes / unindexed.
   */
  def apply(
      left: DataFrame,
      rightLanceUri: String,
      leftVecCol: String,
      rightVecCol: String,
      k: Int,
      metric: String = "l2",
      rightProjection: Option[Seq[String]] = None,
      outerJoin: Boolean = false,
      scoreCol: String = "__score",
      overfetch: Int = 1,
      nprobes: Option[Int] = None,
      version: Option[Long] = None,
      refineFactor: Option[Int] = None,
      ef: Option[Int] = None): DataFrame = {

    require(k > 0, "k must be positive")
    require(overfetch >= 1, "overfetch must be >= 1")

    val spark = left.sparkSession
    val parsedMetric = Metric.fromName(metric)
    val internalK = k * overfetch

    // Snapshot right-side schema on the driver before any executor work happens.
    val rightSchema: StructType = {
      val reader = spark.read.format("lance")
      version.foreach(v => reader.option("version", v.toString))
      val raw = reader.load(rightLanceUri)
      val pruned = rightProjection match {
        case Some(cols) if cols.nonEmpty => raw.select(cols.head, cols.tail: _*)
        case _ => raw
      }
      pruned.schema
    }

    val outputSchema = buildOutputSchema(left.schema, rightSchema, scoreCol)
    val rightProjectionCols: Seq[String] =
      rightProjection.getOrElse(rightSchema.fieldNames.toSeq)

    val conf = LanceKnnJoinStage.Conf(
      datasetUri = rightLanceUri,
      version = version,
      vectorColumn = rightVecCol,
      metric = parsedMetric,
      k = k,
      internalK = internalK,
      nprobes = nprobes,
      refineFactor = refineFactor,
      ef = ef,
      prefilter = None,
      leftVecIdx = left.schema.fieldIndex(leftVecCol),
      rightProjection = rightProjectionCols,
      rightFields = rightSchema.fields.toSeq,
      leftFieldCount = left.schema.fields.length,
      outerJoin = outerJoin,
      smallerIsBetter = parsedMetric.smallerIsBetter)

    val rowRdd = left.rdd.mapPartitions(iter => LanceKnnJoinStage.runPartition(iter, conf))
    spark.createDataFrame(rowRdd, outputSchema)
  }

  private def buildOutputSchema(
      left: StructType,
      right: StructType,
      scoreCol: String): StructType = {
    val rightNullable = right.fields.map(f => f.copy(nullable = true))
    val score = StructField(scoreCol, FloatType, nullable = true)
    StructType(left.fields ++ rightNullable :+ score)
  }
}
