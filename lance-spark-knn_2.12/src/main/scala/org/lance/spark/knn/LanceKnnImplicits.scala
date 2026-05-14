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

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.execution.datasources.v2.DataSourceV2Relation

/**
 * Idiomatic DataFrame extension for the indexed nearest-K join. The Phase 2 SQL syntax
 * (`APPROX NEAREST K BY DISTANCE ...`) requires Spark 4.2+ because that's where the
 * `NearestByJoin` operator landed. The DataFrame API path here works on every Spark version
 * the lance-spark connector supports (3.5, 4.0, 4.1, 4.2+) — it just calls Lance's Java probe
 * API directly through `IndexedNearestJoin.apply`, no Catalyst rule, no SQL.
 *
 * Usage:
 * {{{
 *   import org.lance.spark.knn.LanceKnnImplicits._
 *
 *   val docs = spark.read.format("lance").load("/path/to/lance/dataset")
 *   val joined = queries.kNearestJoin(
 *     right = docs,
 *     leftVecCol = "qvec",
 *     rightVecCol = "vec",
 *     k = 10,
 *     metric = "l2")
 * }}}
 *
 * The right DataFrame MUST be a Lance scan — `spark.read.format("lance").load(uri)`. The
 * extension extracts the underlying URI from the right-side analyzed plan; if it can't find a
 * `LanceTable` it throws `IllegalArgumentException`. This is intentional: the indexed path
 * uses Lance's Java API directly to open the dataset, so a non-Lance DataFrame cannot be
 * substituted (there's no general "any DataFrame" indexed path).
 *
 * == Why an extension method, not a builder ==
 *
 * Builder-style APIs (`new KNearestJoin(...).build()`) are heavier syntactically than what
 * users want for what should be a one-line call. The extension method makes the verb
 * (`kNearestJoin`) hang off the left DataFrame the same way `join` does, so users discover it
 * via IDE autocomplete and can reach for it without learning a new pattern.
 */
object LanceKnnImplicits {

  implicit class LanceKnnDataFrameOps(val df: DataFrame) extends AnyVal {

    /**
     * Approximate top-K nearest-neighbor join over a Lance-backed right DataFrame. The right
     * DataFrame must be a `spark.read.format("lance").load(uri)` (any plan that wraps a
     * `LanceTable` — `Filter`, `SubqueryAlias`, `Project` are unwrapped). For a non-Lance
     * right side or a derived plan that loses the URI, this method throws.
     *
     * @param right             Lance-backed right DataFrame
     * @param leftVecCol        name of the vector column on `this` (left)
     * @param rightVecCol       name of the vector column on `right`
     * @param k                 number of nearest neighbors per left row
     * @param metric            distance / similarity metric: "l2" | "cosine" | "dot"
     * @param rightProjection   columns to materialize from `right`. `None` = all columns.
     * @param outerJoin         left-outer mode: emit a left row even if zero neighbors found
     * @param scoreCol          name of the appended score column (default `__score`)
     * @param overfetch         multiplier on `k` during the probe before final trim
     * @param nprobes           IVF cluster count to visit per query (None = Lance default)
     * @param refineFactor      IVF-PQ exact-distance re-rank factor (None = no re-rank)
     * @param ef                HNSW search depth (None = Lance default; only meaningful for
     *                          HNSW indexes)
     * @param probeParallelism  fragment groups for Phase 1.5 probing. 1 = single task probes
     *                          the whole dataset (recommended on a single-machine setup); >1
     *                          splits fragments across N tasks for true distributed clusters
     * @param balanceFragments  when probeParallelism > 1, use row-count-aware LPT bin-packing
     *                          for fragment groups instead of round-robin
     */
    // scalastyle:off parameter.number
    def kNearestJoin(
        right: DataFrame,
        leftVecCol: String,
        rightVecCol: String,
        k: Int,
        metric: String = "l2",
        rightProjection: Option[Seq[String]] = None,
        outerJoin: Boolean = false,
        scoreCol: String = "__score",
        overfetch: Int = 1,
        nprobes: Option[Int] = None,
        refineFactor: Option[Int] = None,
        ef: Option[Int] = None,
        probeParallelism: Int = 1,
        balanceFragments: Boolean = false): DataFrame = {
      val (uri, version) = LanceKnnImplicits.extractLanceUri(right)
      IndexedNearestJoin(
        left = df,
        rightLanceUri = uri,
        leftVecCol = leftVecCol,
        rightVecCol = rightVecCol,
        k = k,
        metric = metric,
        rightProjection = rightProjection,
        outerJoin = outerJoin,
        scoreCol = scoreCol,
        overfetch = overfetch,
        nprobes = nprobes,
        version = version,
        probeParallelism = probeParallelism,
        refineFactor = refineFactor,
        ef = ef,
        balanceFragmentsByRowCount = balanceFragments)
    }
    // scalastyle:on parameter.number
  }

  /**
   * Walk a DataFrame's analyzed plan looking for a `LanceTable`-backed
   * `DataSourceV2Relation`. Skips through wrappers that don't change the underlying
   * relation: `SubqueryAlias`, `View`, `Project`, `Filter`. Returns `(uri, optional version)`
   * pulled from the relation's options. Throws `IllegalArgumentException` if no Lance scan
   * is found.
   *
   * Lance detection mirrors `IndexedNearestByJoinRule.isLanceTable` —
   * class-name match (`getClass.getName.contains("Lance")`) — to keep the user-facing
   * extension working without a hard dependency on the connector's internal types. The
   * extension only needs to be able to spot a Lance relation; it doesn't operate on it
   * directly.
   *
   * Public for tests.
   */
  private[knn] def extractLanceUri(df: DataFrame): (String, Option[Long]) = {
    val rel = findLanceRelation(df.queryExecution.analyzed).getOrElse {
      throw new IllegalArgumentException(
        "kNearestJoin requires the right DataFrame to be a Lance scan " +
          "(spark.read.format(\"lance\").load(uri)). Plan was:\n" +
          df.queryExecution.analyzed)
    }
    val opts = rel.options
    val uri = Option(opts.get("path"))
      .orElse(Option(opts.get("datasetUri")))
      .getOrElse(throw new IllegalArgumentException(
        "Lance relation found but no `path` / `datasetUri` option set; cannot extract URI"))
    val version = Option(opts.get("version")).map(_.toLong)
    (uri, version)
  }

  private def findLanceRelation(plan: LogicalPlan): Option[DataSourceV2Relation] = plan match {
    case rel: DataSourceV2Relation if isLanceTable(rel) => Some(rel)
    case other =>
      // Iterator.find avoids 2.13's `nextOption()` so this stays Scala 2.12-compatible.
      val it = other.children.iterator.map(findLanceRelation).filter(_.isDefined)
      if (it.hasNext) it.next() else None
  }

  private def isLanceTable(rel: DataSourceV2Relation): Boolean = {
    val cls = rel.table.getClass.getName
    cls.contains("Lance") || cls.contains("lance")
  }
}
