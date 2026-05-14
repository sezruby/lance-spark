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
package org.lance.spark.knn.internal.staged

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeSet}
import org.apache.spark.sql.catalyst.plans.physical.{ClusteredDistribution, Distribution}
import org.apache.spark.sql.execution.{SparkPlan, UnaryExecNode}
import org.apache.spark.sql.types.StructType
import org.lance.spark.knn.internal.{LanceMaterializeStage, LanceMergeStage, LanceProbeStage, ProbedLeft, TopKHeap}

import scala.collection.mutable

/**
 * Three physical operators corresponding to [[LanceProbeLogicalPlan]],
 * [[LanceMergeLogicalPlan]], and [[LanceMaterializeLogicalPlan]]. Each `doExecute` decodes
 * its child's `RDD[InternalRow]` to typed `(Long, ProbedLeft)` tuples, runs the existing
 * staged-RDD pipeline op, and re-encodes back to `RDD[InternalRow]`.
 *
 * The decode → run → re-encode dance is what gives `df.explain()` an honest 3-stage shape.
 * The microbench (`InterStagePayloadOverheadBench`) measured the per-row encoding cost at
 * <1 % of total wall-clock at every realistic SQL benchmark scale, so the runtime cost is
 * below the noise floor of repeated runs.
 *
 * == AQE compatibility ==
 *
 * `LanceMergeExec` declares `requiredChildDistribution = ClusteredDistribution(leftId)`, so
 * Catalyst's `EnsureRequirements` rule inserts a `ShuffleExchangeExec` between the probe
 * and merge execs. That exchange is what AQE wraps as a `ShuffleQueryStageExec` — and once
 * AQE has the wrapper it can apply the usual rules (`CoalesceShufflePartitions`,
 * `OptimizeSkewJoin`, `OptimizeShuffleWithLocalRead`) to the merge stage's shuffle.
 *
 * The merge exec itself does NO shuffle internally — `doExecute` is just a per-partition
 * group-by-leftId aggregation, since after the exchange every leftId's contributions are
 * co-located in one partition.
 *
 * Caveat: when `probeParallelism > 1`, the probe stage uses
 * `LanceProbeStage.runWithFragmentGroups` which still does an internal RDD-level
 * `partitionBy` shuffle (to replicate left rows across fragment groups). That shuffle
 * remains AQE-invisible. Fixing it would need a different shape — replication is a
 * `flatMap` plus a partitioning, which doesn't fit Catalyst's `requiredChildDistribution`
 * model cleanly. Local-laptop benchmarks showed the fragment-grouped path doesn't help at
 * single-machine scale anyway, so we leave it as-is for now.
 */
private[knn] case class LanceProbeExec(
    override val child: SparkPlan,
    stageConf: LanceProbeStage.Conf,
    fragmentGroups: Option[Seq[Seq[Int]]],
    leftSchema: StructType,
    override val output: Seq[Attribute])
  extends UnaryExecNode {

  // The inter-stage attrs (leftId, leftRow, refs) appear in `output` but not in
  // `child.output`. We synthesise them from per-row probe results — declare so Spark's
  // `missingInput` check (and the `!` bang in tree-string output) doesn't flag this node.
  override def producedAttributes: AttributeSet = AttributeSet(output) -- child.outputSet

  override protected def doExecute(): RDD[InternalRow] = {
    val childRdd = child.execute()
    val schemaCaptured = leftSchema // capture for closure serialization
    val confCaptured = stageConf
    val groupsCaptured = fragmentGroups

    // Decode user's left-side InternalRows into Rows (matches the shape the existing
    // LanceProbeStage takes). copy() because Spark may reuse the InternalRow buffer
    // across iterations of the upstream operator.
    val leftEnc = ExpressionEncoder(schemaCaptured).resolveAndBind()
    val rowLeftRdd: RDD[Row] = childRdd.mapPartitions { iter =>
      val deser = leftEnc.createDeserializer()
      iter.map(ir => deser(ir.copy()))
    }
    val leftKeyed: RDD[(Long, Row)] =
      rowLeftRdd.zipWithUniqueId().map { case (row, id) => (id, row) }

    val probed: RDD[(Long, ProbedLeft)] = groupsCaptured match {
      case Some(groups) => LanceProbeStage.runWithFragmentGroups(leftKeyed, confCaptured, groups)
      case None => LanceProbeStage.run(leftKeyed, confCaptured)
    }

    probed.mapPartitions { iter =>
      val enc = new ProbedLeftCodec.Encoder(schemaCaptured)
      iter.map { case (lid, pl) => enc.encode(lid, pl) }
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): LanceProbeExec =
    copy(child = newChild)
}

private[knn] case class LanceMergeExec(
    override val child: SparkPlan,
    stageConf: LanceMergeStage.Conf,
    leftSchema: StructType,
    override val output: Seq[Attribute])
  extends UnaryExecNode {

  /**
   * Force a hash-partitioned shuffle on `leftId` between probe and merge. Without this,
   * the merge would have to do its own RDD-level shuffle (the original design) which is
   * invisible to AQE. With `ClusteredDistribution(leftId)` declared, Catalyst's
   * `EnsureRequirements` rule inserts a `ShuffleExchangeExec` automatically; AQE wraps
   * that exchange as a `ShuffleQueryStageExec` and can coalesce / re-balance / etc.
   *
   * `leftId` is always the first column of the inter-stage schema. Pulling it from
   * `child.output.head` matches what `ProbedLeftCodec.interStageSchema` produces.
   */
  override def requiredChildDistribution: Seq[Distribution] =
    ClusteredDistribution(Seq(child.output.head)) :: Nil

  override protected def doExecute(): RDD[InternalRow] = {
    val childRdd = child.execute()
    val schemaCaptured = leftSchema
    val finalK = stageConf.finalK
    val smallerIsBetter = stageConf.smallerIsBetter

    // After the upstream `ShuffleExchangeExec`, all rows with the same leftId are co-located
    // on the same partition. We just group within partition and apply `TopKHeap.merge`.
    // No reduceByKey needed.
    childRdd.mapPartitions { iter =>
      val dec = new ProbedLeftCodec.Decoder(schemaCaptured)
      val enc = new ProbedLeftCodec.Encoder(schemaCaptured)
      val byLid = mutable.LinkedHashMap.empty[Long, ProbedLeft]

      while (iter.hasNext) {
        val ir = iter.next().copy()
        val (lid, pl) = dec.decode(ir)
        byLid.get(lid) match {
          case Some(prev) =>
            val mergedRefs = TopKHeap.merge(prev.refs, pl.refs, finalK, smallerIsBetter)
            byLid(lid) = ProbedLeft(prev.leftRow, mergedRefs)
          case None =>
            // First contribution for this lid. Trim if it already overflows finalK
            // (probe stage may emit `internalK = k * overfetch` per ref array).
            val refs =
              if (pl.refs.length <= finalK) pl.refs
              else {
                val heap = new TopKHeap(finalK, smallerIsBetter)
                heap.offerAll(pl.refs)
                heap.drain()
              }
            byLid(lid) = ProbedLeft(pl.leftRow, refs)
        }
      }

      byLid.iterator.map { case (lid, pl) => enc.encode(lid, pl) }
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): LanceMergeExec =
    copy(child = newChild)
}

private[knn] case class LanceMaterializeExec(
    override val child: SparkPlan,
    stageConf: LanceMaterializeStage.Conf,
    leftSchema: StructType,
    finalSchema: StructType,
    override val output: Seq[Attribute])
  extends UnaryExecNode {

  // Final-schema attrs (left.* ++ right.* ++ score) are synthesised here, not in any child.
  override def producedAttributes: AttributeSet = AttributeSet(output) -- child.outputSet

  override protected def doExecute(): RDD[InternalRow] = {
    val childRdd = child.execute()
    val leftSchemaCaptured = leftSchema
    val finalSchemaCaptured = finalSchema
    val confCaptured = stageConf

    val keyed: RDD[(Long, ProbedLeft)] = childRdd.mapPartitions { iter =>
      val dec = new ProbedLeftCodec.Decoder(leftSchemaCaptured)
      iter.map(ir => dec.decode(ir.copy()))
    }

    val joinedRows: RDD[Row] = LanceMaterializeStage.run(keyed, confCaptured)

    val finalEnc = ExpressionEncoder(finalSchemaCaptured).resolveAndBind()
    joinedRows.mapPartitions { iter =>
      val ser = finalEnc.createSerializer()
      iter.map(row => ser(row).copy())
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): LanceMaterializeExec =
    copy(child = newChild)
}
