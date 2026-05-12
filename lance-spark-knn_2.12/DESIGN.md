# Indexed nearest-by join — feature design

> **Audience.** Human reviewers approaching this feature for the first time. Read this top-to-
> bottom before reading diffs. Several of the design choices look arbitrary in isolation but
> are forced by Spark's existing rule ordering or by Lance's index shape — the rationale is
> here, not in the source.
>
> **PoC scope.** All phases (0 / 1 / 1.5 / 2 / 3 / 3.x) currently ship together on the
> `knn-phase0` fork branch (this is that branch). For upstream delivery to
> `lance-format/lance-spark:main`, the branch will be split into 7 smaller PRs — see
> [`UPSTREAM_DELIVERY_PLAN.md`](UPSTREAM_DELIVERY_PLAN.md) for the split. The phase
> labels here describe the design layers — the order in which a reviewer should read the
> code — not the eventual per-PR release boundaries.

## TL;DR

For SQL of the shape

```sql
SELECT *
FROM   queries q
LEFT OUTER JOIN documents d
  APPROX NEAREST 10 BY DISTANCE vector_l2_distance(q.vec, d.vec)
```

— route execution through a per-fragment Lance index probe + Spark-side merge instead of the
default `O(|L| × |R|)` cross-product. Fast where Lance has a vector index; falls back to
Spark's brute-force rewrite when conditions don't hold. **No public API change** for users
once Phase 2 ships: registering one Spark session extension is the entire opt-in.

The work is split across two modules:

| Module | What it owns | Spark version | Status |
|---|---|---|---|
| `lance-spark-knn_2.12` (+ `_2.13` cross-build) | Pipeline primitives, public DataFrame API (`IndexedNearestJoin.apply`, `df.kNearestJoin` extension), 3-exec Catalyst staged plan, Phase 1.5 fragment grouping, Phase 3 recall knobs. | 3.4 / 3.5 / 4.0 (reflection bridge in `LanceKnnDatasetBridge`); 2.12 and 2.13 cross-build. | Phase 0 / 1 / 1.5 / 3 done. |
| `lance-spark-knn-4.2_2.13` | Catalyst rule + logical + physical operators that intercept SQL `NearestByJoin`. | 4.2-SNAPSHOT (SPARK-56395 `NearestByJoin` only exists in master as of writing; re-pin once 4.2.0 releases). | Phase 2 done, but held out of upstream delivery until Spark 4.2 ships. |

## Why this works on Lance specifically

- Lance vector indexes (IVF-PQ, HNSW) are **fragment-local**. Per-fragment probes are
  independent, which makes parallelism trivial.
- Lance's Java API exposes single-vector nearest search via
  `org.lance.ipc.Query` + `LanceScanner.create(dataset, ScanOptions, allocator)`. We call into
  this primitive from Spark tasks.
- `_rowid` (Lance virtual column) makes late materialization cheap: the probe stage emits
  row IDs only, the materialize stage point-fetches by ID. We use `_rowid` rather than
  `_rowaddr` because Lance's INDEXED nearest-search path materializes `_rowid` but not
  `_rowaddr` — using `_rowid` works on both indexed and non-indexed paths uniformly.
- Lance versioning gives consistent snapshots across distributed tasks.

## Architecture

### The three-stage pipeline

```
left.logicalPlan
  -- LanceProbeLogicalPlan           --> [_leftId, leftRow fields..., _refs]
  -- LanceMergeLogicalPlan           --> same shape
  -- LanceMaterializeLogicalPlan     --> final join output schema

lowered via LanceKnnStagedStrategy to:

  LanceProbeExec
    --> (per-task) open LanceProbe, nearest-search per left row, emit inter-stage rows
  ShuffleExchangeExec hashpartitioning(_leftId)
    --> inserted by EnsureRequirements, wrapped by AdaptiveSparkPlanExec when AQE is on
  LanceMergeExec
    --> per-partition group-by-leftId, TopKHeap.merge, re-emit merged rows
  LanceMaterializeExec
    --> open LanceProbe, point-fetch right rows by _rowid, assemble join Rows
```

`IndexedNearestJoin.apply` builds the three-logical-plan tree on top of the user's left
analyzed plan, registers `LanceKnnStagedStrategy` on the session's
`experimentalMethods.extraStrategies` (idempotent), and wraps the root via
`LanceKnnDatasetBridge.asDataFrame` (a trampoline to `Dataset.ofRows` — that method is
`private[sql]`).

`df.explain()` shows four Catalyst nodes (`LanceProbe → Exchange → LanceMerge →
LanceMaterialize`) wrapped by `AdaptiveSparkPlanExec`. With AQE enabled,
`AQEShuffleRead coalesced` appears on the merge-side shuffle after the first collection.

The pipeline objects live in `org.lance.spark.knn.internal`:

- **`LanceProbeStage`** — the RDD-level primitive. Opens a `LanceProbe` per task, probes
  Lance's nearest-search per left row, emits `(leftId, ProbedLeft)`. Map-side combine via
  `TopKHeap` keeps state at exactly K entries when the task probes multiple fragments.
  When `probeParallelism > 1`, `runWithFragmentGroups` replicates left rows across G
  fragment groups via an internal RDD-level `partitionBy` so each task sees one group.
  (The fragment-grouped probe path's internal shuffle remains AQE-invisible — tracked as
  future work.)
- **`LanceMergeStage`** — per-partition group-by-leftId + `TopKHeap.merge`. No shuffle
  inside the stage itself — the shuffle is the `ShuffleExchangeExec` above it, inserted
  by Catalyst from `LanceMergeExec.requiredChildDistribution = ClusteredDistribution(leftId)`.
  `merge` is associative + commutative so per-partition aggregation is equivalent to the
  prior `reduceByKey` formulation.
- **`LanceMaterializeStage`** — opens `LanceProbe` again per task, calls
  `materialize(rowIds)` which lowers to `_rowid IN (...)` (Lance's row-id lookup path),
  assembles join rows from the carried left payload + materialized right rows.

The custom Catalyst nodes live in `org.lance.spark.knn.internal.staged`:

- **`StagedPlans.scala`** — three `LogicalPlan` nodes. Critical detail:
  `LanceMergeLogicalPlan` and `LanceMaterializeLogicalPlan` override
  `lazy val references = child.outputSet`. This blocks Catalyst's `ColumnPruning` rule
  from inserting `Project(Nil)` wrappers between the custom nodes when a downstream
  consumer references no columns (`count(*)`, etc.) — an oversight in the initial 3-exec
  implementation that caused `AssertionError` / SIGSEGV at runtime. See `IMPL_PLAN.md`
  "3-exec staged split — root cause and fix" for the full post-mortem.
- **`StagedExecs.scala`** — the three `SparkPlan` execs. Each `doExecute` decodes the
  inter-stage `InternalRow`s via `ProbedLeftCodec.Decoder`, runs the stage primitive, and
  re-encodes. `LanceMergeExec.requiredChildDistribution` is what triggers the Exchange.
- **`ProbedLeftCodec.scala`** — flat inter-stage schema (`_leftId`, leftSchema fields
  inlined, `_refs: array<struct<rowAddr, score>>`). Single `ExpressionEncoder` pass for
  encode + direct `InternalRow` accessors for decode; earlier multi-pass codec attempts
  introduced binary-layout issues.
- **`LanceKnnStagedStrategy.scala`** — registered once per session; lowers each logical
  plan to its matching exec.
- **`LanceKnnDatasetBridge.scala`** (in `org.apache.spark.sql` package) — one-method
  trampoline to the package-private `Dataset.ofRows`.

Phase 2's `IndexedNearestByJoinRule` emits the same three logical plans described above.
Both the DataFrame API path and the SQL path lower through `LanceKnnStagedStrategy` into
the identical `LanceProbeExec → ShuffleExchangeExec → LanceMergeExec → LanceMaterializeExec`
chain — the Catalyst rule is the only SQL-specific piece.

### Why `_rowid` not `_rowaddr`

Lance has two virtual columns that identify a row:

| Column | Encoding | Available on |
|---|---|---|
| `_rowaddr` | physical address `(frag_id << 32) \| row_in_frag` | non-indexed scans only |
| `_rowid`   | logical Lance-assigned ID | indexed AND non-indexed scans |

The probe stage emits one row-identifier value per ranked refsult; the materialize stage
filters by that same identifier (`<col> IN (rowIds...)`) for point-fetch. So the column
choice has to work on both code paths the probe stage exercises.

The Phase 0/1 prototype used `_rowaddr` because it's the natural physical pointer. That
worked on the no-index path. When the IVF-PQ recall test built an actual vector index, the
probe failed with:

```
LanceError(Schema): Schema error: No field named _rowaddr. Did you mean '_rowid'?
```

Lance's indexed nearest-search materializes `_rowid` but not `_rowaddr`. So the whole
pipeline uses `_rowid` (shipped as part of the Phase 3 hardening commit):

- `ScanOptions.Builder.withRowAddress(true)` → `withRowId(true)` in `LanceProbe.probe`.
- `_rowaddr IN (...)` → `_rowid IN (...)` in `LanceProbe.materialize` filter.
- `LanceProbe.RowAddressColumn` constant → renamed to `RowIdColumn`, sourced from
  `LanceConstant.ROW_ID`.

Behavior on the no-index path is identical (both columns work there); the indexed path now
works at all. The existing oracle test still passes — `_rowid` lookups via the row-id index
have the same point-fetch semantics as `_rowaddr` lookups did on the row-address index.

Variable names elsewhere (`rowAddrs: Seq[Long]`, `extractRowAddr`, `ScoredRowRef.rowAddr`)
are retained for source-compat — the field type and lookup semantics are unchanged, only
the underlying virtual column the value is read from is different.

### Bandwidth math

The substantive performance argument for the staged design is shuffle bandwidth.

```
Brute-force rewrite (Spark default):
  cross-product = |L| × |R|     rows shipped through shuffle
  payload = full right row      ~hundreds of bytes to KBs

Indexed staged pipeline:
  shuffle volume = |L| × N × K  refs   N = number of probe tasks
  payload per ref = ~24B               (8B addr + 4B score + overhead)
```

For `|L| = 10⁶, |R| = 10⁹, N = 100, K = 10`:
- brute-force = 10⁶ × 10⁹ = 10¹⁵ pair evaluations.
- staged     = 10⁶ × 100 × 10 = 10⁹ refs through shuffle.

That's six orders of magnitude. The win comes from late materialization — the probe stage
emits refs only, and the materialize stage fetches payloads after the merge has already
narrowed to top-K.

### Why Lance brute-force still beats Spark cross-product (no index needed)

A subtle finding from the benchmark: even WITHOUT a vector index, Lance's per-fragment scan
beats Spark's `RewriteNearestByJoin` (`min_by_k` + `BroadcastNestedLoopJoin`) by ~17× on
the same |L|×|R| pair-evaluation workload. Both paths do 10M pair evaluations on the
small-scale benchmark; Spark takes 3,700 ms, Lance takes 220 ms. Why:

1. **Native SIMD vs JVM expression evaluation.** Lance's distance kernel is hand-tuned Rust
   with AVX-512 / NEON intrinsics. One L2 distance over a dim-128 vector takes ~8 cycles in
   the SIMD kernel. Spark's `vector_l2_distance` is a `RuntimeReplaceable` lowered to
   `StaticInvoke(VectorFunctionImplUtils.vectorL2Distance)` — JVM bytecode through Catalyst
   expression evaluation per row. JIT can auto-vectorize the inner loop but loses to
   hand-written intrinsics by ~5-10×.
2. **Columnar contiguous arrays vs per-row deserialization.** Lance stores the vector column
   as a contiguous Arrow `float32` array per fragment; the kernel iterates contiguous memory
   (cache-friendly, prefetchable). Spark feeds each right row through Catalyst's iterator —
   each `ArrayType(FloatType)` cell goes through `UnsafeArrayData.toFloatArray()` per-row
   (per-row malloc + scattered loads).
3. **Catalyst expression-evaluation overhead.** Spark's `RewriteNearestByJoin` lowers to
   `Generate(Inline(Aggregate(min_by_k(struct(right.*), distance(L, R), K),
   BroadcastNestedLoopJoin(LeftOuter, leftWith__qid, right))))`. Each (L, R) pair passes
   through ~5 layers of expression evaluation: BNL row iterator → tag projection →
   aggregate input projection → `vector_l2_distance` evaluation → `min_by_k` heap update.
   Lance's path is just: scanner pulls a column batch, kernel iterates the float array,
   updates a top-K heap. No JOIN, no struct serialization, no broadcast.

Rough math sanity check (small scale, 10M pair evaluations across 18 cores):

```
Spark:  3,676 ms / 10M = 370 ns/pair  (≈80 cycles/core/pair @ 4 GHz)
Lance:    223 ms / 10M =  22 ns/pair  (≈ 5 cycles/core/pair @ 4 GHz)
```

The Lance number is consistent with AVX-512 doing 16 float ops/cycle on dim-128 vectors
(128/16 ≈ 8 cycles for the math, plus heap-maintenance and Arrow pointer dereferences).
The Spark number is consistent with ~5 layers of Catalyst expression evaluation overhead
on top of the underlying SIMD math.

**Implication for the design.** The 17× SQL speedup (608× DataFrame) is real on a
*no-index* dataset. An index then adds another order of magnitude on top. So the staged
pipeline's value isn't conditional on having a vector index — Lance's native scan plus
fragment-local parallelism beats Catalyst's per-pair JVM overhead even on the brute-force
path. The index is a multiplier, not a prerequisite.

### Recall

Lance's vector indexes are approximate (IVF-PQ probes a subset of the inverted file). Recall is
tunable via `nprobes` and an overfetch ratio (probe `K × overfetch`, then trim to `K` after
the merge). Without an index, Lance falls back to brute-force per-fragment scan, which is
exact (recall = 1.0) — that's how the Phase 0 oracle test is constructed.

## Phases

### Phase 0 — pure DataFrame API
**Module: `lance-spark-knn_2.12`**.

`IndexedNearestJoin.apply(left, lanceUri, leftVecCol, rightVecCol, k, ...)` Scala function. Pure
RDD primitives wrapped around the `LanceProbe` per-task primitive. **No shuffle** — probe and
materialize run in the same `mapPartitions` block. Existed first to validate per-task Lance
access works at all and to baseline recall against a brute-force oracle.

### Phase 1 — staged RDD pipeline + 3-exec Catalyst split (production path)
**Module: `lance-spark-knn_2.12`**.

Phase 0's inline `mapPartitions` was split into three stage objects (probe / merge /
materialize). Public API unchanged. The DataFrame API path then split further into three
explicit `SparkPlan` operators (`LanceProbeExec` / `LanceMergeExec` / `LanceMaterializeExec`)
with a Catalyst-inserted `ShuffleExchangeExec` between probe and merge — this is the
production path today, AQE-visible merge shuffle.

The 3-exec split had a noteworthy debugging history during development: an early
implementation produced reproducible `AssertionError` / SIGSEGV on `count()`-style
consumers. Initial diagnosis blamed a JVM-aarch64 + JIT C2 interaction; that was wrong.
The real root cause was Catalyst's `ColumnPruning` rule inserting `Project(Nil)`
wrappers between the custom nodes when downstream consumers referenced no columns; the
project codegens to 0-field `UnsafeRow`s which crash `ProbedLeftCodec.Decoder` at
`ir.getLong(0)`. The fix is a `references = child.outputSet` override on the Merge /
Materialize logical plans, which short-circuits ColumnPruning's subset guard. See
`IMPL_PLAN.md` "3-exec staged split — root cause and fix" for the full post-mortem.

The shuffle is structurally present but degenerate when `probeParallelism = 1` — each
`leftId` has one contributor, so the merge function never fires. The full bandwidth win
lands when fragment-grouping arrives in Phase 1.5.

### Phase 2 — Catalyst integration
**Module: `lance-spark-knn-4.2_2.13`**. Spark 4.2-SNAPSHOT only.

A `postHocResolutionRule` pattern-matches `NearestByJoin(approx = true, ...)` over a Lance scan
with a recognized vector-distance ranking expression and rewrites it to the same
three-logical-plan tree produced by the DataFrame API path
(`LanceProbeLogicalPlan → LanceMergeLogicalPlan → LanceMaterializeLogicalPlan`). The shared
`LanceKnnStagedStrategy` then lowers that tree to the Probe/Merge/Materialize execs — SQL
and DataFrame paths converge on the same physical shape.

After Phase 2, users get the indexed path automatically from `APPROX NEAREST` SQL queries.
EXACT queries and unsupported shapes flow through to Spark's existing brute-force rewrite —
no functional regression.

#### Why `injectPostHocResolutionRule`, NOT `injectOptimizerRule`

This is the single most important detail in Phase 2. Spark 4.2's optimizer:

```
Optimizer
  ├ Batch "Finish Analysis"   ← RewriteNearestByJoin lives in here
  │   ├ ReplaceExpressions     (RuntimeReplaceable → StaticInvoke)
  │   ├ ...
  │   ├ RewriteNearestByJoin   (NearestByJoin → cross-product + MaxMinByK)
  │   └ ...
  ├ Batch "Operator optimization batch"
  │   └ rules added by injectOptimizerRule fire HERE
  └ ...
```

By the time `injectOptimizerRule` rules fire, `RewriteNearestByJoin` has already replaced
`NearestByJoin` with a cross-product + `MaxMinByK` plan. Nothing left for us to pattern-match.

`injectPostHocResolutionRule` runs immediately after analysis — *before* the optimizer starts.
We see the unrewritten `NearestByJoin` and the unreplaced `VectorL2Distance` /
`VectorCosineSimilarity` / `VectorInnerProduct` ranking expressions. This same constraint
applies to *any* engine wanting to substitute a different physical strategy for `APPROX NEAREST`
queries.

#### Pattern-match preconditions

The rule rewrites only when ALL of these hold:

| Check | Why |
|---|---|
| `approx = true` | EXACT mode is contractually deterministic; brute-force keeps owning it. |
| `right` resolves to a Lance DSv2 relation (under at most a `SubqueryAlias`) | We need a URI to probe. Detection is class-name-based (`getClass.getName.contains("Lance")`); the URI comes from `options.get("path")` / `options.get("datasetUri")`. The probe + materialize stages are Lance-specific by construction (they call Lance's Java API directly), so there's no general "any vector backend" plug-in point — keeping detection simple is consistent with that. |
| Ranking is one of `VectorL2Distance` (with `NearestByDistance`), `VectorCosineSimilarity` (with `NearestBySimilarity`), `VectorInnerProduct` (with `NearestBySimilarity`) | Direction must match the function's natural ordering. Lance's index supports L2 / cosine / dot — anything else has no fast path. |
| Both arguments of the ranking function are bare attributes, one from each side | Mixed-side or composed expressions have no clean mapping to a Lance probe. Phase 3 may extend. |
| `spark.lance.knn.indexedNearestByJoin.enabled = true` | Opt-in until Phase 3 cost gating lands. |

When ANY condition fails the rule returns the plan unchanged and Spark's brute-force rewrite
handles the query — no regression.

#### Logical plans

The rule emits the same three logical plans described in the "Architecture" section
(`LanceProbeLogicalPlan` / `LanceMergeLogicalPlan` / `LanceMaterializeLogicalPlan`), wrapped
in a `Project` that restores the original `NearestByJoin.output` attribute-for-attribute
(including `ExprId`s) so any parent operator's references stay resolved — same contract
`RewriteNearestByJoin` honors.

The right-side Lance scan is **absorbed into the probe plan's config**, not kept as a child.
Why: if right were still a child, Catalyst would happily plan a separate scan of it,
defeating the whole optimization. The trade-off is that column-pruning / filter pushdown
that would normally happen on the right side no longer happens automatically; the rule
captures the projection set (and, for `SELECT * FROM lance WHERE ...`, the filter predicate)
at rewrite time instead.

#### Physical plans

Exactly the same physical chain as the DataFrame API path — `LanceProbeExec →
ShuffleExchangeExec → LanceMergeExec → LanceMaterializeExec` under `AdaptiveSparkPlanExec`.
`LanceKnnStagedStrategy` is shared between both paths; the Catalyst rule is the only
SQL-specific piece.

The Project-above-Materialize also strips the trailing `__score` column, since
`NearestByJoin`'s output contract doesn't include it (Phase 1 emits it internally for the
probe/merge aggregation).

### Phase 1.5 — fragment-grouped probing
**Module: `lance-spark-knn_2.12`** (same as Phase 0/1).

Adds an opt-in `probeParallelism: Int = 1` parameter on `IndexedNearestJoin.apply`. When > 1:

1. Driver enumerates Lance fragment IDs via `Dataset.getFragments()`
   (`internal/LanceFragments.scala`).
2. Round-robins into N groups (or LPT bin-packs by per-fragment row count when
   `balanceFragmentsByRowCount = true`); broadcasts the assignment to every
   `LanceProbeExec` task.
3. `LanceProbeStage.runWithFragmentGroups` replicates each left row across the N groups
   via `flatMap`, then `partitionBy(HashPartitioner(N))` so each task processes a single
   group only. Each task opens `LanceProbe` with its group's `fragmentIds`.
4. Output keyed by `leftId` with N contributions per leftId. The merge stage is a
   `LanceMergeExec` with `requiredChildDistribution = ClusteredDistribution(leftId)` —
   Catalyst's `EnsureRequirements` inserts a `ShuffleExchangeExec` above it, and the exec
   then per-partition groups by leftId and applies `TopKHeap.merge`. (The `partitionBy`
   shuffle inside step 3 remains RDD-level and is NOT AQE-visible — tracked as Phase 3.x
   future work. The merge-side Exchange inserted above step 4 IS AQE-visible.)

This is where the bandwidth win the rest of this doc promises actually lands — Phase 1
had the staged shape but a degenerate shuffle (one contributor per leftId, merge function
never fired). Phase 1.5 makes the merge stage do real work.

**Edge case** — when `probeParallelism > numFragments`, only one group has fragments and
the rule degenerates back to the Phase 1 single-task path, avoiding a replicate-shuffle
for nothing.

**Cost** — two shuffles (replicate + merge) instead of Phase 1's one. Justified by the
bandwidth win at scale; for tiny datasets stick with `probeParallelism = 1`.

### Phase 3 — hardening (partial)

Done:

- **`refineFactor` and `ef`** — IVF-PQ re-rank pass and HNSW search depth, plumbed through to
  `Query.Builder` calls in `LanceProbe.probe`.
- **Row-count-aware fragment grouping** — `balanceFragmentsByRowCount` flag uses LPT greedy
  bin-packing (4/3-optimal-makespan approximation) on `FragmentMetadata.getNumRows`.
- **Prefilter pushdown** — when the right side of `NearestByJoin` is a `Filter` over a Lance
  scan, `IndexedNearestByJoinRule` translates the predicate to a Lance SQL filter string and
  threads it through `LanceProbeStage.Conf.prefilter` → `ScanOptions.filter()`. Lance applies
  it BEFORE the index lookup (`prefilter = true` is always set), so top-K is computed over
  only matching rows. Critical for correctness, not just perf: without it, a vector probe could
  return K rows that are all later filtered out, masking truly-nearest-but-also-matching
  rows further down the index.

  Translation is conservative: bare `attr <op> literal` comparisons, `IN`, `IS [NOT] NULL`, and
  `AND`/`OR`/`NOT` over right-side attrs only. Anything else (UDFs, computed sub-expressions,
  predicates touching the LEFT input) makes the rule REFUSE the rewrite and fall through to
  Spark's brute-force cross-product. Refusal — not partial pushdown — because dropping a
  residual conjunct would silently change result semantics; a slow-but-correct query is the
  acceptable failure mode.

  Project unwrapping: `SELECT * FROM lance WHERE ...` analyzes to
  `Project(<passthrough>, Filter(cond, lance))`; the rule unwraps both. Non-passthrough
  Projects (renames, drops, computed columns) fail the unwrap check and fall through.

- **3-stage explicit physical operators (DataFrame API path)** — **Done.** The production
  path is now `LanceProbeExec → ShuffleExchangeExec → LanceMergeExec → LanceMaterializeExec`
  under `AdaptiveSparkPlanExec`. `df.explain()` shows all four nodes; AQE coalesces the
  merge shuffle (`AQEShuffleRead coalesced`). An early SIGSEGV during development was
  misattributed to JVM-aarch64; the real cause was Catalyst's `ColumnPruning` rule
  inserting `Project(Nil)` wrappers between the custom nodes when downstream consumers
  referenced no columns. Fix: `LanceMergeLogicalPlan` and `LanceMaterializeLogicalPlan`
  override `lazy val references = child.outputSet`, short-circuiting `ColumnPruning`'s
  subset guard. See `IMPL_PLAN.md` "3-exec staged split — root cause and fix" for the
  full post-mortem. 60 tests pass.

- **`df.kNearestJoin` DataFrame extension** — `LanceKnnImplicits._` provides an extension
  method that hangs off any `DataFrame`, mirrors `df.join(other, ...)`, and works on
  Spark 3.5 / 4.0 / 4.1 / 4.2+. It extracts the Lance URI from the right DataFrame's
  analyzed plan automatically; non-Lance right sides (parquet, in-memory, alias-wrapped
  non-Lance, etc.) fail fast with `IllegalArgumentException` naming the constraint.

Outstanding (Phase 3.x — see `IMPL_PLAN.md` for the full table):

- Cost gate replaces opt-in flag.
- Spark version CI matrix (compile+test verified on 3.5 and 4.0 via the
  reflection bridge; formal CI job still TODO).
- AQE-visible shuffle for the fragment-grouped probe path
  (`runWithFragmentGroups`'s internal `partitionBy` remains RDD-level; the
  merge-side shuffle IS AQE-visible).
- Per-executor `LanceProbe` cache to amortize dataset-open across small
  partitions.
- Left-side skew handling (today only the right's fragment groups are
  balanced).

## Public surface

### DataFrame API

Two equivalent forms — pick whichever fits the call site.

**Idiomatic extension method.** Lives on every `DataFrame`, hangs off the left side the
same way `join` does, and works on every Spark version the connector supports (3.5, 4.0,
4.1, 4.2+). The right side must be a Lance scan
(`spark.read.format("lance").load(uri)`); the extension extracts the Lance URI from the
right DataFrame's analyzed plan automatically.

```scala
import org.lance.spark.knn.LanceKnnImplicits._

val docs = spark.read.format("lance").load("/path/to/lance")
val joined = queries.kNearestJoin(
  right = docs,
  leftVecCol = "qvec",
  rightVecCol = "vec",
  k = 10,
  metric = "l2")          // l2 | cosine | dot
```

A `Filter` / `SubqueryAlias` / `Project(passthrough)` over the right Lance scan is
unwrapped before URI extraction; passing a non-Lance DataFrame throws
`IllegalArgumentException` with a message naming the constraint.

**URI form.** When you don't have a `DataFrame` for the right side and just have a path
string (e.g. early in a job before Spark sees the dataset), call the underlying
`IndexedNearestJoin.apply` directly.

```scala
import org.lance.spark.knn.IndexedNearestJoin

val joined = IndexedNearestJoin(
  left = leftDf,
  rightLanceUri = "/path/to/lance",
  leftVecCol = "qvec",
  rightVecCol = "vec",
  k = 10,
  metric = "l2")
```

Both forms return a `DataFrame` with schema `left.* ++ right.* ++ __score`.

### SQL (Phase 2 path)

```scala
SparkSession.builder()
  .config("spark.sql.extensions",
          "org.lance.spark.knn.extensions.LanceKnnSparkSessionExtensions")
  .config("spark.lance.knn.indexedNearestByJoin.enabled", "true")
  .getOrCreate()
```

Then any:

```sql
SELECT *
FROM left l [INNER | LEFT OUTER] JOIN right_lance r
  APPROX NEAREST k BY {DISTANCE | SIMILARITY} f(l.vec, r.vec)
```

— rewrites automatically, where `f` is `vector_l2_distance` (with `DISTANCE`),
`vector_cosine_similarity` (with `SIMILARITY`), or `vector_inner_product` (with `SIMILARITY`).
Output schema matches `NearestByJoin.output` (no score column — the user can compute that in a
project if needed).

The extension can coexist with the connector's `LanceSparkSessionExtensions` in a comma-
separated `spark.sql.extensions` value.

## Reviewer's reading order

Reviewers should start with **[`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md)** —
it's the up-to-date "start here → engine → primitives" reading path, with
a test map and trust-but-verify checklist. The below lists the specific
Catalyst-side files to read for a review of the SQL path
(`lance-spark-knn-4.2_2.13`), in order:

1. **`catalyst/IndexedNearestByJoinRule.scala`** — the load-bearing pattern
   match. The `for-yield` in `rewriteIfApplicable` short-circuits on every
   precondition, then builds the three logical plans (probe/merge/materialize)
   wrapped in a `Project` that restores the original `NearestByJoin.output`.
2. **`extensions/LanceKnnSparkSessionExtensions.scala`** — smallest file.
   Confirm the wiring uses `injectPostHocResolutionRule` (not
   `injectOptimizerRule`; see reasoning in the Phase 2 section above) and
   registers the shared `LanceKnnStagedStrategy`.

For the shared logical plans, physical execs, and strategy, see the DataFrame
API path in `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/staged/`.

## Test coverage

| Test class | What it covers |
|---|---|
| `internal/TopKHeapTest` | Metric-aware ordering, eviction, drain order, merge. Pure unit. |
| `internal/LanceFragmentsTest` | Round-robin and LPT bin-packing math. |
| `internal/LanceProbeValidationTest` | Real Lance dataset; brute-force-equivalence oracle. |
| `IndexedNearestJoinTest` | Phase 0 e2e; brute-force-equivalence oracle. Refine-factor wiring. |
| `IndexedNearestJoinPlanShapeTest` | 3-exec plan shape (LanceProbe / LanceMerge / LanceMaterialize / Exchange all present). |
| `IndexedNearestJoinAqeVisibilityTest` | Exchange hashpartitioning on `_leftId`; AQE wrap; no `!` missingInput prefix. |
| `IndexedNearestJoinConsumerShapeTest` | `count()`, `agg(count("*"))`, `select(lit(1))`, `collect()` all succeed — regression for the ColumnPruning crash. |
| `IndexedNearestJoinCorrectnessTest` | Brute-force oracle at 1K × 100 × dim 16 × K 10. |
| `IndexedNearestJoinJitStressTest` | crossJoin warmup + 20 iterations at 10K × 100 × dim 128 — durability at benchmark scale. |
| `internal/staged/StagedPlansReferencesTest` | Structural pin on `references = child.outputSet` override. |
| `IndexedNearestJoinFragmentGroupingTest` | Phase 1.5 oracle equivalence + plan-shape with `probeParallelism > 1`. |
| `catalyst/IndexedNearestByJoinRuleTest` | Phase 2 rule pattern-match (positive + negative cases); asserts the emitted `Project(LanceMaterialize(LanceMerge(LanceProbe)))` tree shape. |
| `catalyst/IndexedNearestByJoinE2ETest` | SQL `APPROX NEAREST` against real Lance, Spark 4.2-SNAPSHOT. Rule on/off plus WHERE-pushdown oracle equivalence. |

## Benchmark validation

Both benchmarks (`IndexedNearestJoinBenchmark` for DataFrame, `IndexedNearestByJoinSqlBenchmark`
for SQL) run a **pre-timing validation step** that compares EVERY config — including the
slow baseline / rule-OFF path — against an in-memory brute-force oracle on a 16-row left
subset. The benchmark `sys.error`'s out before timing if any config disagrees with ground
truth, so the quoted speedups are on equivalent results.

The 16-row subset keeps the slow baseline tractable: `O(16 × |R|)` cross-product evaluations
is sub-second even at medium scale (16 × 1M = 16M pair evaluations). The full benchmark is
unchanged in scope; the validation step is small relative to the timed runs.

### Latest validated numbers — Apple M5 Max, 18 cores, 48 GB

DataFrame benchmark (small scale, |R|=100K, |L|=100, dim=128, K=10):

```
Sanity check: all 5 configs match brute-force oracle (sample size: 16) ✅
A: Spark crossJoin baseline             109,373 ms   1.00×
B: Phase 0/1 (probeParallelism=1)           180 ms   608×
C: Phase 1.5 (probeParallelism=4)           288 ms   380×
D: Phase 1.5 (probeParallelism=8)           276 ms   396×
E: Phase 1.5 (G=8, skew-balanced)           277 ms   395×
```

SQL benchmark (small scale, same shape):

```
Sanity check: rule ON and rule OFF agree on top-K (sample size: 16) ✅
A: rule OFF (Spark RewriteNearestByJoin)  3,728 ms   1.00×
B: rule ON  (3-exec staged + shared strategy) 214 ms   17.4×
```

Why the SQL number is smaller: Spark's `RewriteNearestByJoin` is itself optimized — it
lowers to a `min_by_k` heap aggregate over a `BroadcastNestedLoopJoin`, which avoids
materializing all `|L|×|R|` pairs in JVM memory. The remaining 17.4× comes from delegating
per-fragment scans to Lance's native vector kernels (AVX-512 / NEON) instead of evaluating
`vector_l2_distance` row-by-row in the JVM.

See `BENCHMARK_RESULTS.md` for the medium-scale numbers, the honest "Phase 1.5 doesn't help
locally" finding, and where fragment-grouping would win (distributed cluster, object-store-
backed Lance, or right side too large for one machine).

### Cluster validation — OSS Spark 3.5

The local numbers above are Apple M5 Max single-machine. The indexed path has also been
validated on a real distributed OSS Spark 3.5 cluster (8 × 4 core × 16 GB executor
pods, multi-tenant infrastructure):

- **CohereLabs `wikipedia-2023-11-embed-multilingual-v3` (dim=1024, real embeddings),
  1K base × 50 queries:** indexed path is **100–200× faster than Spark crossJoin**
  (7-iter median: 64.9 s → 406 ms at E: Phase 1.5 G=8 skew-balanced, = 160×). Exact
  multiplier varies ±20% across runs due to multi-tenant CPU contention; the
  order-of-magnitude story is robust. Measured with `write.format("noop")` timing
  sink and a 16-row brute-force oracle gating correctness before each run. The
  speedup **grows** with dim (128 → 1024) because Lance's SIMD kernel advantage
  widens vs Spark's JVM UDF overhead.

- **Synthetic medium (|R|=1M, |L|=1000, dim=128) on the same cluster:** Phase 1.5 D/E
  **win** at ~55 s vs Phase 0/1 B at ~92 s. This is the OPPOSITE of the local-laptop
  finding — when `probeParallelism == numFragments`, cross-machine parallelism across 8
  executor JVMs beats Lance-internal threading on one machine. Two independent runs agree
  within 2%.

- **SIFT1M IVF-FLAT recall@10:** 0.98 at nprobes=16, 0.999 at nprobes=64 — within noise of
  published FAISS numbers.

The cluster results also surfaced two production-grade constraints documented in
`BENCHMARK_RESULTS.md` § "Cluster benchmarks": a vendor-specific executor-count knob
(some managed Spark distributions ignore `spark.executor.instances` in
standalone-per-app mode and require their own knob), and
`spark.{driver,executor}.extraClassPath` (required when the cluster's bundled Arrow-C
version is older than ours and would otherwise be first on the classpath).

**Real-backend e2e** (`IndexedNearestByJoinE2ETest`) — Phase 2 module ships a SQL-level e2e
test against an actual Lance dataset on Spark 4.2-SNAPSHOT. The trick: lance-spark-4.1's
source compiles cleanly against 4.2-SNAPSHOT and the resulting jar runs on 4.2's DSv2 API,
so we recompile it locally and use it as the runtime Lance reader. Two test cases:

- Rule on → the 3-exec chain (`LanceProbe` / `LanceMerge` / `LanceMaterialize`) is in the
  executed plan, results match brute-force oracle.
- Rule off → falls through to Spark's `RewriteNearestByJoin` (cross-product + `min_by_k`),
  results still match oracle. Confirms the opt-in fallback path doesn't break correctness.

## What this is NOT

- **Not a brute-force fallback.** That's `RewriteNearestByJoin` in Spark, kept for EXACT
  queries and unindexed cases.
- **Not a re-implementation of Lance's index.** We delegate every probe to Lance.
- **Not a vector-DB-style serving layer.** This is for batch joins inside Spark pipelines.
