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

import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeSet}
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, UnaryNode}
import org.apache.spark.sql.types.StructType
import org.lance.spark.knn.internal.{LanceMaterializeStage, LanceMergeStage, LanceProbeStage}

/**
 * Three logical plan nodes that both the DataFrame API path (`IndexedNearestJoin.apply`)
 * and the SQL path (`IndexedNearestByJoinRule` in `lance-spark-knn-4.2_2.13`) build, so
 * each stage of the staged pipeline shows up as its own operator in `df.explain()`. They
 * are deliberately Lance-specific (the matching `Strategy` only knows how to lower these
 * three). Both user-facing paths emit this exact tree and share `LanceKnnStagedStrategy`
 * for the lowering.
 *
 * == Why they are not children of each other in the obvious "tree" way ==
 *
 * Probe and Merge nodes share the inter-stage attribute references (created once via
 * `ProbedLeftCodec.interStageAttributes(leftSchema)`) so Catalyst's attribute resolution
 * does not fight us — a parent that references attributes of a child requires those
 * attributes to be reachable from the child's `outputSet`. By keeping the same
 * `Seq[AttributeReference]` instances on both probe.output and merge.output we sidestep
 * having to copy or rewrite expr-ids across the inter-stage boundary.
 *
 * == producedAttributes ==
 *
 * For each node, `producedAttributes` is `output -- child.outputSet` — Catalyst convention
 * for "attributes this node introduces that don't come from its child." Probe introduces
 * the inter-stage triple from the user-shaped left input. Merge passes through. Materialize
 * introduces the final join output (which doesn't exist in any child) so all of its output
 * is produced.
 */
private[knn] case class LanceProbeLogicalPlan(
    override val child: LogicalPlan,
    stageConf: LanceProbeStage.Conf,
    fragmentGroups: Option[Seq[Seq[Int]]],
    leftSchema: StructType,
    interStageOutput: Seq[Attribute])
  extends UnaryNode {

  override def output: Seq[Attribute] = interStageOutput

  override def producedAttributes: AttributeSet = AttributeSet(output) -- child.outputSet

  override protected def withNewChildInternal(newChild: LogicalPlan): LanceProbeLogicalPlan =
    copy(child = newChild)
}

private[knn] case class LanceMergeLogicalPlan(
    override val child: LogicalPlan,
    stageConf: LanceMergeStage.Conf,
    leftSchema: StructType,
    interStageOutput: Seq[Attribute])
  extends UnaryNode {

  override def output: Seq[Attribute] = interStageOutput

  // Pass-through schema: same attrs as child. AttributeSet subtraction on identical
  // attribute references yields the empty set — merge produces nothing new.
  override def producedAttributes: AttributeSet = AttributeSet(output) -- child.outputSet

  // Every child attribute is load-bearing — the matching `LanceMergeExec.doExecute`
  // decodes the full inter-stage row (leftId + leftRow + refs) on every input. Without
  // this override, Catalyst's `ColumnPruning` sees downstream consumers that reference
  // nothing (`count(*)`, `Aggregate`, etc.) and wraps this node in `Project(Nil)`; that
  // project codegens to 0-field UnsafeRows which then crash `ProbedLeftCodec.Decoder`
  // at `ir.getLong(0)` (AssertionError under interpreter/C1, SIGSEGV under C2 JIT).
  // This pattern was initially misdiagnosed as a JVM-aarch64 bug; the actual cause
  // is this Catalyst rule. See IMPL_PLAN.md "3-exec staged split — root cause and fix".
  override lazy val references: AttributeSet = child.outputSet

  override protected def withNewChildInternal(newChild: LogicalPlan): LanceMergeLogicalPlan =
    copy(child = newChild)
}

private[knn] case class LanceMaterializeLogicalPlan(
    override val child: LogicalPlan,
    stageConf: LanceMaterializeStage.Conf,
    leftSchema: StructType,
    finalSchema: StructType,
    finalOutput: Seq[Attribute])
  extends UnaryNode {

  override def output: Seq[Attribute] = finalOutput

  // Final schema attrs do not appear in the inter-stage child, so all of them are produced.
  override def producedAttributes: AttributeSet = AttributeSet(output) -- child.outputSet

  // Same rationale as `LanceMergeLogicalPlan.references`: the matching exec decodes every
  // child attribute to rebuild the `ProbedLeft` tuple, so nothing in the child can be
  // pruned.
  override lazy val references: AttributeSet = child.outputSet

  override protected def withNewChildInternal(newChild: LogicalPlan): LanceMaterializeLogicalPlan =
    copy(child = newChild)
}
