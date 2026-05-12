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
package org.apache.spark.sql

import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan

/**
 * Lives in the `org.apache.spark.sql` package solely to access `Dataset.ofRows`, which is
 * `private[sql]` and not reachable from `org.lance.spark.knn`. The `IndexedNearestJoin`
 * DataFrame API path needs to wrap a custom `LogicalPlan` (the staged probe / merge /
 * materialize tree) as a `DataFrame`; the only public entry to do that goes through
 * `SparkSession#sql()` (which expects a SQL string) or constructing `Dataset` directly,
 * both of which require one-liner trampolines like this.
 *
 * Same trick most JVM Spark connectors use when they need to bridge between user-facing
 * code and Catalyst internals. Kept minimal — exposing exactly one method, nothing else.
 *
 * == Spark 3.x vs 4.x: the `ofRows` location ==
 *
 * Spark 3.x keeps `Dataset` in `org.apache.spark.sql.Dataset`. Spark 4.0 moved the
 * concrete DataFrame implementation to `org.apache.spark.sql.classic.Dataset` (the
 * "classic" Dataset, vs Connect's remote DataFrame). `ofRows` is a `private[sql]` method
 * in BOTH packages with compatible signatures, so reflection picks either at runtime.
 *
 * Single-source policy: the knn module is Spark-version-agnostic at compile time
 * (binds only to `lance-spark-base` + `spark-sql` `provided`). If we compile-linked
 * against `org.apache.spark.sql.Dataset.ofRows`, Spark 4.x runtime would
 * `NoSuchMethodError`. Reflection avoids that — one lookup, cached at first call.
 */
object LanceKnnDatasetBridge {

  // Look up once on first call; cache for subsequent invocations.
  // Lazy initialization dodges a startup-time reflection cost when this object is loaded
  // but kNearestJoin is never called.
  private lazy val ofRowsInvoker: (SparkSession, LogicalPlan) => DataFrame = {
    val candidates = Seq(
      "org.apache.spark.sql.Dataset", // Spark 3.4 / 3.5
      "org.apache.spark.sql.classic.Dataset" // Spark 4.0+
    )
    var found: Option[(SparkSession, LogicalPlan) => DataFrame] = None
    val errors = scala.collection.mutable.ArrayBuffer.empty[String]
    for (cls <- candidates if found.isEmpty) {
      try {
        val companion = Class.forName(cls + "$").getField("MODULE$").get(null)
        // Find `ofRows` by shape rather than by exact parameter types: Spark 4.0+ takes
        // `classic.SparkSession` as the first parameter instead of the abstract
        // `sql.SparkSession`, so `getDeclaredMethod(..., classOf[SparkSession], ...)`
        // won't match. `getMethods` + filter on name + arity + 2nd-param type
        // (`LogicalPlan`, stable across versions) picks the canonical 2-arg overload
        // and skips the 3-arg `(session, plan, tracker)` and 4-arg variants introduced
        // in Spark 4.0+.
        val method = companion.getClass.getMethods.find { m =>
          m.getName == "ofRows" && m.getParameterCount == 2 &&
          m.getParameterTypes()(1) == classOf[LogicalPlan]
        }.getOrElse(throw new NoSuchMethodException(s"ofRows not found on $cls"))
        method.setAccessible(true)
        found = Some((s: SparkSession, p: LogicalPlan) => {
          method.invoke(companion, s, p).asInstanceOf[DataFrame]
        })
      } catch {
        case _: ClassNotFoundException | _: NoSuchMethodException =>
          errors += cls
      }
    }
    found.getOrElse(
      throw new UnsupportedOperationException(
        s"Could not locate Dataset.ofRows in any of: ${errors.mkString(", ")}. " +
          "Unsupported Spark version."))
  }

  def asDataFrame(spark: SparkSession, plan: LogicalPlan): DataFrame =
    ofRowsInvoker(spark, plan)
}
