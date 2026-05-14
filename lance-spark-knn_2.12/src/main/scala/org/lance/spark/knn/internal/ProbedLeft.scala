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

import org.apache.spark.sql.Row

/**
 * Tuple shipped from the probe stage to the merge stage. Carries the left row alongside the
 * top-K row references so the materialize stage can reconstruct join output without a separate
 * join back to the left source.
 *
 * Phase 1 trades shuffle bandwidth for simplicity: shipping the left payload through the shuffle
 * loses the "refs only ~24B per ref" bandwidth advantage that the IMPL_PLAN positions as the
 * eventual win. Phase 2/3 plan: pre-partition the left RDD by `leftId` and ship only
 * `(leftId, refs)` through the shuffle, then cogroup against the co-partitioned left payload at
 * materialize time. Doing so requires either a stable user-supplied join key or a synthetic
 * `leftId` carried alongside the payload — orthogonal to the staging refactor done here.
 */
final case class ProbedLeft(leftRow: Row, refs: Array[ScoredRowRef]) extends Serializable
