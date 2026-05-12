# Indexed `NearestByJoin` via Lance — a PoC for SPARK-56395

**Context:** [SPARK-56395](https://issues.apache.org/jira/browse/SPARK-56395) adds
`NearestByJoin` as a first-class logical operator in Spark 4.2, currently lowered by the
built-in `RewriteNearestByJoin` rule to a cross-product + `min_by_k` aggregate
(`BroadcastNestedLoopJoin` + heap aggregate). The [design
doc](https://docs.google.com/document/d/1opFVcQJgEWDWUVB7uVlFMlNomRwxqRu8iW0JmvCvxF0/)
calls out that a true indexed path is out of scope for that ticket but is the natural
follow-up — "any vector-index-backed data source should be able to intercept
`NearestByJoin` before the cross-product rewrite and substitute an index-backed plan."

This repo (`lance-spark-knn`) is **one concrete implementation** of that hook, for Lance
datasets. The purpose of this doc is to share the shape of that implementation with Spark
maintainers as a reference point, and to sketch how the same pattern could extend to
non-Lance formats (parquet, delta) via the Lance **sidecar index** pattern — without
Spark itself having to ship a vector-index backend.

**Not a proposal to change apache/spark.** The PoC lives entirely in an ecosystem
connector. It depends on only the public extension points SPARK-56395 provides (and one
pre-existing one, `injectPostHocResolutionRule`). The discussion questions at the end
are the places where a small Spark-side change *might* help; they are intentionally left
open for maintainers to weigh in on.

## What's in apache/spark, what's in this connector

| Apache/Spark side | Ecosystem connector side |
|---|---|
| `NearestByJoin` logical plan (SPARK-56395) | `IndexedNearestByJoinRule` Catalyst rule (postHocResolutionRule) |
| `RewriteNearestByJoin` optimizer rule (default: brute-force) | Three `LogicalPlan` + `SparkPlan` nodes that form a Catalyst-visible staged pipeline |
| `VectorL2Distance` / `VectorCosineSimilarity` / `VectorInnerProduct` ranking expressions | `LanceKnnStagedStrategy` lowering the three logical plans to three execs |
| Extension points: `injectPostHocResolutionRule`, `injectPlannerStrategy` | Registration via `spark.sql.extensions` — opt-in per-session |

No changes to apache/spark are required for the Lance-specific implementation. The
hookpoints are what SPARK-56395 and the older extensions API already provide.

## What the connector does — shape summary

Full details: [`DESIGN.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/DESIGN.md).
This section is the one-page summary for context.

### 1. Catalyst rule

```
Rule:   injectPostHocResolutionRule (NOT injectOptimizerRule)
Pattern: NearestByJoin(approx=true, recognized-ranking, Lance-scan-on-right)
Action:  rewrite to a 3-logical-plan tree
```

**Why `postHocResolutionRule` and not `injectOptimizerRule`** — Spark's
`RewriteNearestByJoin` runs in the optimizer's `FinishAnalysis` batch, which precedes the
`operatorOptimizationBatch` that `injectOptimizerRule` adds rules to. An injected
optimizer rule fires after `RewriteNearestByJoin` has already rewritten the operator to
a cross-product; there's nothing left to pattern-match. `injectPostHocResolutionRule`
runs after analysis but before any optimizer batch — the only injection point that sees
the unrewritten `NearestByJoin`.

This is the **single load-bearing constraint** any future engine needs to respect to
substitute an alternative physical strategy for `NearestByJoin`. It would be worth
calling out in the `NearestByJoin` scaladoc.

### 2. Three-stage plan

```
left.logicalPlan
  LanceProbeLogicalPlan       (per-task probe: left vectors → top-K (rowId, score))
  LanceMergeLogicalPlan       (co-locate by _leftId, bounded TopKHeap.merge down to final K)
  LanceMaterializeLogicalPlan (point-fetch right rows by _rowid, assemble join rows)
  [Project to drop the trailing score attr so output matches NearestByJoin.output]
```

Lowered via the shared strategy to:

```
LanceProbeExec
  ↓
ShuffleExchangeExec hashpartitioning(_leftId)   ← Catalyst inserts this via
  ↓                                                EnsureRequirements (driven by
LanceMergeExec                                     LanceMergeExec.requiredChildDistribution
  ↓                                                = ClusteredDistribution(_leftId))
LanceMaterializeExec
```

Wrapped by `AdaptiveSparkPlanExec` when AQE is on. The Exchange is AQE-visible, so
`CoalesceShufflePartitions` / `OptimizeSkewJoin` / `OptimizeShuffleWithLocalRead` all
engage on the merge shuffle.

### 3. Prefilter pushdown

`SELECT * FROM lance WHERE p APPROX NEAREST K BY ...` — the rule detects the
`Filter(p, lance)` shape, translates the predicate to a Lance SQL filter string
(conservative: bare `attr <op> literal`, `IN`, `IS [NOT] NULL`, `AND`/`OR`/`NOT`), and
threads it into the probe. Lance applies the filter **before** the index lookup, so
top-K is computed over matching rows only — avoiding the
"index-returns-K-rows-all-filtered-out-post-join" recall bug.

Untranslatable predicates (UDFs, computed expressions, left-side references) cause the
rule to **refuse** the rewrite and fall through to Spark's brute-force cross-product.
Refusal — not partial pushdown — because dropping a residual would silently change query
semantics.

### 4. Correctness and scale validation

- `recall=1.0` vs brute-force oracle on 1K × 100 × dim=16 (unindexed Lance) and on
  IVF-PQ-indexed datasets with `refineFactor=8`.
- 100–200× faster than Spark `RewriteNearestByJoin` cross-product on real Cohere Wikipedia
  embeddings (dim=1024, 1K × 50), on an 8 × 4-core / 16-GB OSS Spark 3.5 cluster.
- SIFT1M IVF-FLAT recall@10 = 0.98 at nprobes=16, 1.00 at nprobes=64 — within noise of
  published FAISS numbers.
- Local M5 Max (100K × 100 × dim=128): 17× vs Spark's brute-force. Smaller gap than the
  headline because Spark's built-in is already `min_by_k` over
  `BroadcastNestedLoopJoin` — not a full `|L|×|R|` materialization. The remaining 17×
  comes from Lance's native-SIMD distance kernels beating Catalyst's JVM expression
  evaluation per pair.

Full numbers: [`BENCHMARK_RESULTS.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/BENCHMARK_RESULTS.md).

## Extending the same shape to parquet / delta — Lance sidecar pattern

The interesting part, and the reason to share this doc.

### The observation

Only the **probe** and **materialize** stages are format-specific. They open a Lance
dataset and call Lance's native nearest-search / row-id point-fetch APIs. The rest of
the pipeline — the Catalyst rule, the three logical/physical plans, the inter-stage
schema, the bounded-merge heap, the Exchange insertion — is format-agnostic Catalyst
plumbing.

So a user who has their **primary data** in parquet or delta can still get the indexed
path by building a Lance **sidecar** keyed by a shared row identifier:

```
Primary data: a parquet or delta table with (user_id, name, ..., embedding: array<float>)

Sidecar:     a Lance dataset with just (user_id, embedding) — built once, maintained
             incrementally as the primary table grows. Vector index (IVF-PQ or HNSW)
             is built on the Lance sidecar.

Query time:  APPROX NEAREST K BY DISTANCE vector_l2_distance(q.vec, d.embedding)
             with the rule configured to probe the sidecar's Lance URI, then
             materialize by a foreign-key join to the primary parquet/delta table:

             SELECT q.qid, p.name
             FROM queries q INNER JOIN parquet_primary p
               APPROX NEAREST K BY DISTANCE vector_l2_distance(q.lvec, p.embedding)
```

The rule's right-side detection (currently "is the scan a Lance DSv2 relation?") would
need one small extension: "is there a registered sidecar index for this table?" The
registration mechanism is a small catalog-side detail — could be a table property
(`lance.sidecar.uri`), a config map, or a Spark session extension. The probe then runs
against the sidecar; materialize runs against the **primary** table, joining on the
shared key.

### What changes vs. the Lance-native implementation

The rule's detection predicate expands from "right side is Lance" to "right side is Lance
*or* has a registered Lance sidecar":

```scala
case rel: DataSourceV2Relation if isLanceTable(rel.table) =>
  LanceScanInfo(uri = rel.options.get("path"), ...)
case rel: DataSourceV2Relation if hasRegisteredSidecar(rel.table) =>
  val sidecar = lookupSidecar(rel.table)
  // probe against sidecar.uri, materialize against rel's primary URI
  SidecarScanInfo(sidecarUri = sidecar.uri, primaryRel = rel, joinKey = sidecar.joinKey, ...)
```

Materialize changes from "point-fetch from Lance by `_rowid`" to "recover the primary
table's payload for the surviving top-K row keys." The right mechanism depends on the
format:

#### Why parquet/delta can't do cheap point-fetch

Lance's `_rowid IN (...)` materialize path works because Lance has a row-id index that
translates a rowId to a `(fragment, offset)` address and the columnar reader supports
random-access within a fragment — it's comparable in cost to a secondary-index lookup in
an OLTP engine.

Parquet and delta don't have this. A `WHERE user_id IN (k1, ..., kN)` over parquet
pushes down to row-group-level min/max filtering (column stats), then still reads and
decodes every row group that *might* contain a match. Delta adds optional file skipping
via its own min/max stats and optional `DataSkippingNumIndexedCols` / Z-order /
bloom-filter-on-column indexes, but the fundamental primitive is still "read whole row
groups, filter predicates, emit matches" — not true random access. With an arbitrary
set of `|L| × K` keys drawn from the full key space (which is typically the case for
vector similarity — nearest neighbors are distributed, not clustered), the scan
degenerates to "read most of the table."

So the naive "just equi-join the merged top-K back to the primary table" is **not**
free on parquet/delta. The top-K is small (`|L| × K` ≤ 10⁴ rows for typical queries)
but the primary table is large (10⁶–10⁹ rows). A broadcast-join the small side onto a
full parquet scan costs one full table scan per query. At that point, the brute-force
cross-product isn't obviously worse — both are `O(|right|)` reads.

#### Three materialize strategies, in order of increasing sophistication

1. **Carry the needed columns in the sidecar (simplest, works today).** If the user
   knows which columns are projected in the APPROX NEAREST query at sidecar-build time,
   store those columns in the Lance sidecar alongside the embedding. Materialize runs
   entirely against Lance (the current Lance-native path). Cost: sidecar duplicates
   columns. Benefit: works without any change to the probe/materialize exec split.

   Best fit: queries always project a stable small set of columns
   (`SELECT q.qid, d.title, d.url FROM ...`). Store `(user_id, embedding, title, url)`
   in the sidecar.

2. **Equi-join broadcast the top-K list against primary (works for small-enough tables).**
   Materialize emits `(leftId, rightKey)` pairs; a subsequent equi-join materializes
   payload. Cost: one full scan of the primary table per query.

   Best fit: tables small enough that a full parquet scan is already acceptable
   (< ~100M rows at typical parquet compression), or queries already touching the
   primary table for other reasons (the scan amortizes). Delta's data-skipping stats
   help here if the primary table is partitioned and the top-K keys cluster on a
   partition column — but vector-nearest rarely clusters on anything the user
   partitioned by.

3. **Format-native point-fetch, when the format supports it (parquet with a row-index
   extension; delta with the row-id preview; iceberg with its row lineage feature).**
   These formats have been adding optional row-id / row-position metadata exactly to
   support this kind of random access. A future materialize stage could consult that
   metadata and do O(K) I/O per query instead of O(|right|). Requires the format to
   persist the sidecar-key → file-offset mapping, which none of parquet/delta does by
   default today — but the trajectory in all three communities is toward supporting it.

The PoC today implements option 1 by construction (Lance is both sidecar and primary).
Option 2 is a ~100-LoC extension — replace `LanceMaterializeExec` with a
`ForeignKeyJoinMaterializeExec` that equi-joins against the primary relation. Option 3
depends on format work outside Spark's control.

**Honest assessment:** Option 2 only beats the SPARK-56395 brute-force cross-product on
perf if the primary table is under a certain size threshold (workload-dependent,
roughly hundreds of millions of rows for typical column counts). For larger primary
tables, the user is better off either (a) using option 1 with the columns carried in
the sidecar, or (b) waiting for option 3 as parquet/delta row-indexing matures. This is
worth stating plainly — the sidecar pattern isn't a universal win, and reviewers should
know where it breaks down.

### Format-agnostic extraction, if the appetite exists

If multiple ecosystems (Lance, iceberg-spark, delta, hudi, native parquet with a vector-
index extension) converge on this shape, a cleaner long-term split is:

- **`NearestByJoinIndexProvider` trait** in `apache/spark` (or an ecosystem-shared location):
  ```scala
  trait NearestByJoinIndexProvider {
    def probe(left: DataFrame, metric: Metric, k: Int, prefilter: Option[Expression])
      : RDD[(Long, Seq[ScoredRowRef])]
    def materialize(rowIds: RDD[(Long, Seq[Long])]): RDD[Row]
  }
  ```
- **`IndexedNearestByJoinRule`** in Spark (or ecosystem-shared) — format-agnostic
  pattern match, dispatches to whichever provider is registered for the right-side
  relation.
- **Per-backend module** — Lance, FAISS-on-parquet, delta-with-vector-index — each ships
  a `NearestByJoinIndexProvider` implementation.

That's explicitly **not** something this PoC proposes to do in apache/spark today.
Refactoring to the trait shape costs a nontrivial amount of complexity for a trait
that, right now, has one implementation. But the shape of this PoC is intentionally
close to what such a generalized interface would look like — the three stages, the
inter-stage schema, the Exchange-on-`_leftId`, and the prefilter contract are all
format-agnostic.

## Open questions for Spark maintainers

The following are points where maintainer input would materially change the upstream
story. They're phrased as questions because the PoC works without resolving them; but
each is a place where a small apache/spark change would help.

1. **Scaladoc on `NearestByJoin` about the `injectPostHocResolutionRule` constraint.**
   Any future engine wanting to substitute a different physical plan must know that
   `injectOptimizerRule` is too late. A one-line note on `NearestByJoin`'s class
   scaladoc (or on `RewriteNearestByJoin`) would save the next implementer the hour of
   debugging we spent learning this.

2. **Extending the ranking-expression allowlist.** The rule pattern-matches on
   `VectorL2Distance` / `VectorCosineSimilarity` / `VectorInnerProduct` — the three
   expressions SPARK-56395 recognizes. If a future PR adds, e.g., Hamming distance for
   binary vectors, downstream indexed-path implementations would have to be updated in
   lock-step. A stable "is this a recognized vector-distance expression" predicate (or
   a trait on the expression class) would let ecosystem rules match forward-
   compatibly.

3. **`NearestByJoin` attribute stability across rewrites.** The PoC's rule preserves
   `NearestByJoin.output` attribute-for-attribute (same `ExprId`s) to avoid unresolving
   references from parent operators — the same contract `RewriteNearestByJoin`
   honors. That contract isn't explicitly documented on the class today; documenting it
   would make future rewrite implementations safer.

4. **Is there interest in an `@DeveloperApi` hook on `NearestByJoin` to register
   alternative physical strategies declaratively?** Instead of every ecosystem needing
   to write a `postHocResolutionRule` + a `SparkStrategy`, Spark could provide a
   single registration point (e.g.,
   `NearestByJoinStrategyRegistry.register(predicate, strategy)`). If the broader
   community is interested, this would be a small, focused apache/spark PR. If it's
   not, the current extension points are sufficient and the PoC lives entirely
   downstream.

5. **Row-identity metadata as a Spark-level contract for sidecar materialize.** The
   "extending to parquet/delta via sidecar" story (above) depends on the primary
   table exposing a stable row identifier that the sidecar can key against.
   Parquet/delta/iceberg each have in-flight work to surface this (parquet row-index
   extension, delta's row-id preview, iceberg row lineage) but the APIs differ across
   formats. A Spark-level abstraction — e.g., a `SupportsRowIdentifier` mixin on DSv2
   tables that returns a row-id column the optimizer can treat as unique + stable —
   would let any format plug into a sidecar materialize path without the sidecar
   needing format-specific knowledge. This is a bigger design discussion than the
   other items here; flagging it because sidecar materialize is the primary
   blocker to making the SPARK-56395 indexed path useful outside Lance-native
   storage.

## References

- **Implementation:** [`lance-spark-knn-4.2_2.13` module](https://github.com/sezruby/lance-spark/tree/knn-phase0/lance-spark-knn-4.2_2.13) (Catalyst rule + session extension), [`lance-spark-knn_2.12` module](https://github.com/sezruby/lance-spark/tree/knn-phase0/lance-spark-knn_2.12) (shared logical/physical plans + RDD stages).
- **Design overview:** [`DESIGN.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/DESIGN.md).
- **Reviewer reading order:** [`REVIEWER_GUIDE.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/REVIEWER_GUIDE.md).
- **Benchmark numbers:** [`BENCHMARK_RESULTS.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/BENCHMARK_RESULTS.md).
- **SPARK-56395 Spark JIRA:** https://issues.apache.org/jira/browse/SPARK-56395
- **SPARK-56395 design doc:** https://docs.google.com/document/d/1opFVcQJgEWDWUVB7uVlFMlNomRwxqRu8iW0JmvCvxF0/

**Status:** PoC on `sezruby/lance-spark:knn-phase0`. Recall=1.0 on unindexed + indexed
paths, 100-200× speedup vs cross-product on real embeddings, 60+17 tests passing across
Scala 2.12/2.13 and Spark 3.5/4.0/4.2-SNAPSHOT.

**What's NOT claimed:** this isn't an RFC for apache/spark; it's a reference point for
one concrete way to implement the indexed path the SPARK-56395 design doc mentions as
future work.
