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

import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.plans.logical.LocalRelation
import org.apache.spark.sql.types.{ArrayType, FloatType, LongType, StructField, StructType}
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.Test
import org.lance.spark.knn.internal.{LanceMaterializeStage, LanceMergeStage, LanceProbeStage, Metric}

/**
 * Regression test pinning the `references = child.outputSet` override on
 * `LanceMergeLogicalPlan` and `LanceMaterializeLogicalPlan`.
 *
 * If someone ever removes, narrows, or weakens these overrides, Catalyst's
 * `ColumnPruning` rule will insert `Project(Nil)` wrappers between nodes when downstream
 * consumers (`count(*)`, `Aggregate`, etc.) reference none of the node's output columns.
 * Those empty projections codegen to 0-field `UnsafeRow`s, which crash
 * `ProbedLeftCodec.Decoder.decode` with either `AssertionError: index (0) should < 0`
 * (interpreter/C1) or a SIGSEGV in `UnsafeRow.getLong` (C2 JIT). That bug was originally
 * misdiagnosed as a JVM-aarch64 codec interaction; the actual cause was missing
 * `references` overrides.
 *
 * Functional coverage of the same property lives in
 * [[org.lance.spark.knn.IndexedNearestJoinConsumerShapeTest]] (count / agg / lit-select
 * all succeed end-to-end). This test is the cheap structural pin: if the override goes
 * away, this test fails instantly instead of waiting for a slow ColumnPruning-driven
 * end-to-end crash.
 */
class StagedPlansReferencesTest {

  private def dummyLeftSchema: StructType = new StructType(Array(
    StructField("lid", LongType, nullable = false),
    StructField("qvec", ArrayType(FloatType, containsNull = false), nullable = false)))

  private def dummyChild(attrs: Seq[AttributeReference]): LocalRelation =
    LocalRelation(attrs)

  @Test def testLanceMergeLogicalPlanReferencesIsChildOutputSet(): Unit = {
    val leftSchema = dummyLeftSchema
    val interStageAttrs = ProbedLeftCodec.interStageAttributes(leftSchema)
    val child = dummyChild(interStageAttrs)
    val merge = LanceMergeLogicalPlan(
      child = child,
      stageConf = LanceMergeStage.Conf(finalK = 1, smallerIsBetter = true),
      leftSchema = leftSchema,
      interStageOutput = interStageAttrs)

    assertEquals(
      child.outputSet,
      merge.references,
      "LanceMergeLogicalPlan.references must equal child.outputSet so Catalyst " +
        "ColumnPruning cannot insert Project(Nil) above it")
  }

  @Test def testLanceMaterializeLogicalPlanReferencesIsChildOutputSet(): Unit = {
    val leftSchema = dummyLeftSchema
    val interStageAttrs = ProbedLeftCodec.interStageAttributes(leftSchema)
    val finalAttrs = Seq(
      AttributeReference("lid", LongType, nullable = false)(),
      AttributeReference("rid", LongType, nullable = true)(),
      AttributeReference("__score", FloatType, nullable = true)())

    val child = dummyChild(interStageAttrs)
    val materialize = LanceMaterializeLogicalPlan(
      child = child,
      stageConf = LanceMaterializeStage.Conf(
        datasetUri = "/tmp/unused",
        version = None,
        rightProjection = Seq("rid"),
        rightFields = Seq(StructField("rid", LongType, nullable = true)),
        leftFieldCount = 2,
        outerJoin = false),
      leftSchema = leftSchema,
      finalSchema = new StructType(Array(
        StructField("lid", LongType, nullable = false),
        StructField("rid", LongType, nullable = true),
        StructField("__score", FloatType, nullable = true))),
      finalOutput = finalAttrs)

    assertEquals(
      child.outputSet,
      materialize.references,
      "LanceMaterializeLogicalPlan.references must equal child.outputSet so Catalyst " +
        "ColumnPruning cannot insert Project(Nil) above it")
  }

  /**
   * Explicit positive check: `child.outputSet` must be a subset of `references`. This is
   * the literal predicate `ColumnPruning`'s `Aggregate(_, _, child, _) if !child.outputSet
   * .subsetOf(a.references)` uses to decide whether to insert a pruning Project. Our
   * override makes the subset relation hold (equality ⇒ subset), which short-circuits
   * the rule.
   */
  @Test def testColumnPruningPredicateShortCircuits(): Unit = {
    val leftSchema = dummyLeftSchema
    val interStageAttrs = ProbedLeftCodec.interStageAttributes(leftSchema)
    val child = dummyChild(interStageAttrs)
    val merge = LanceMergeLogicalPlan(
      child = child,
      stageConf = LanceMergeStage.Conf(finalK = 1, smallerIsBetter = true),
      leftSchema = leftSchema,
      interStageOutput = interStageAttrs)

    assertTrue(
      child.outputSet.subsetOf(merge.references),
      "ColumnPruning's guard is `!child.outputSet.subsetOf(references)`; " +
        "override must make the subset relation hold so pruning doesn't fire")
  }
}
