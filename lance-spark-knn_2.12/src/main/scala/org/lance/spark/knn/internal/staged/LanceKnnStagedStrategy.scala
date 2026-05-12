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

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}

/**
 * Maps the three staged-pipeline logical plans to their physical execs. Registered
 * lazily on the active `SparkSession`'s `experimentalMethods.extraStrategies` the first
 * time `IndexedNearestJoin.apply` runs against a session — see
 * [[LanceKnnStagedStrategy.ensureRegistered]].
 *
 * The strategy is `object`-singleton (no per-call state) so registration can use
 * reference-equality to avoid duplicate entries on repeated calls within the same session.
 */
private[knn] object LanceKnnStagedStrategy extends SparkStrategy {

  override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
    case p: LanceProbeLogicalPlan =>
      LanceProbeExec(
        child = planLater(p.child),
        stageConf = p.stageConf,
        fragmentGroups = p.fragmentGroups,
        leftSchema = p.leftSchema,
        output = p.output) :: Nil

    case p: LanceMergeLogicalPlan =>
      LanceMergeExec(
        child = planLater(p.child),
        stageConf = p.stageConf,
        leftSchema = p.leftSchema,
        output = p.output) :: Nil

    case p: LanceMaterializeLogicalPlan =>
      LanceMaterializeExec(
        child = planLater(p.child),
        stageConf = p.stageConf,
        leftSchema = p.leftSchema,
        finalSchema = p.finalSchema,
        output = p.output) :: Nil

    case _ => Nil
  }

  /**
   * Idempotently install this strategy on the session's planner. Called from
   * `IndexedNearestJoin.apply` so users don't have to wire up Spark session extensions
   * just to use the DataFrame API.
   *
   * `experimentalMethods.extraStrategies` is mutable but not thread-safe in its setter.
   * Synchronising on a private monitor (the singleton object itself) keeps concurrent
   * `IndexedNearestJoin.apply` calls from racing and double-installing. Idempotency uses
   * reference equality on the strategy singleton — straightforward since the strategy is
   * an `object`, not a `class`.
   */
  def ensureRegistered(spark: SparkSession): Unit = synchronized {
    val em = spark.sessionState.experimentalMethods
    val current = em.extraStrategies
    if (!current.exists(_ eq this)) {
      em.extraStrategies = current :+ this
    }
  }
}
