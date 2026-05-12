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
package org.lance.spark.knn.extensions

import org.apache.spark.sql.SparkSessionExtensions
import org.lance.spark.knn.catalyst.IndexedNearestByJoinRule
import org.lance.spark.knn.internal.staged.LanceKnnStagedStrategy

/**
 * Registers Phase 2 Catalyst integration for the indexed nearest-by join.
 *
 * Wire this into a SparkSession with:
 *
 * {{{
 *   SparkSession.builder()
 *     .config("spark.sql.extensions",
 *             "org.lance.spark.knn.extensions.LanceKnnSparkSessionExtensions")
 *     .config("spark.lance.knn.indexedNearestByJoin.enabled", "true")
 *     ...
 * }}}
 *
 * The `enabled` flag gates the rule itself — see [[IndexedNearestByJoinRule.EnabledConfKey]].
 * Off by default to keep the integration opt-in until the cost gate (Phase 3) is in place.
 *
 * == Injection point: postHocResolutionRule, NOT optimizerRule ==
 *
 * Spark's `RewriteNearestByJoin` runs in `FinishAnalysis`, which precedes the
 * `operatorOptimizationBatch` that `injectOptimizerRule` adds rules to. By the time an injected
 * optimizer rule fires, the `NearestByJoin` operator has already been replaced with the
 * cross-product + `MaxMinByK` rewrite. `injectPostHocResolutionRule` runs after analysis but
 * before any optimizer batch — this is the only injection point that sees the unrewritten
 * `NearestByJoin`. See `IndexedNearestByJoinRule`'s class doc for the full rationale.
 *
 * Coexistence: this extension does not replace `LanceSparkSessionExtensions` from the connector
 * modules; both can be wired together in a comma-separated `spark.sql.extensions` value.
 */
class LanceKnnSparkSessionExtensions extends (SparkSessionExtensions => Unit) {
  override def apply(extensions: SparkSessionExtensions): Unit = {
    extensions.injectPostHocResolutionRule(_ => IndexedNearestByJoinRule)
    // Shared with the DataFrame API path: `LanceKnnStagedStrategy` lowers the three
    // logical plans (`LanceProbeLogicalPlan` / `LanceMergeLogicalPlan` /
    // `LanceMaterializeLogicalPlan`) to the matching physical execs.
    // `IndexedNearestJoin.apply` also installs this strategy via
    // `experimentalMethods.extraStrategies` at first call; wiring it into the session
    // extension makes the SQL path self-sufficient without depending on a prior DataFrame
    // call having run.
    extensions.injectPlannerStrategy(_ => LanceKnnStagedStrategy)
  }
}
