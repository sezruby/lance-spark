# lance-spark-knn — phase progress & resume notes

This document is the single source of truth for picking up the indexed nearest-by join work
without the original chat context. Read it top-to-bottom before changing anything.

If you are continuing this work, **do not skip the design rationale**: several of the design
choices look arbitrary in isolation but lock with each other. The annotations are deliberate —
the *why* is recorded inline in the source comments.

## Where this lives

- Repo: `https://github.com/sezruby/lance-spark` (fork of `eto-ai/lance-spark` — yes the upstream
  org is Eto.ai).
- Branch: `knn-phase0`
- Open PR: `https://github.com/sezruby/lance-spark/pull/1` (draft; benchmarking PoC, not for
  immediate merge upstream).

## Goal

Indexed approximate-nearest-neighbor join for Spark over a Lance dataset, exposed as a Scala
function (and eventually a SQL `APPROX NEAREST 10 BY ...` clause via Catalyst). Backed by
Lance's fragment-local IVF-PQ vector indexes, executed via per-task Lance probes plus a
Spark-side merge — instead of an `O(|L| × |R|)` cross-product like the built-in
`RewriteNearestByJoin` rule.

The upstream Spark `NearestByJoin` operator (Spark 4.x) only has the cross-product rewrite. The
work here is a layer-on solution that does **not** need to ship via Spark's SPIP process —
Phase 2's Catalyst integration uses `injectPostHocResolutionRule`, not `injectOptimizerRule`,
because `RewriteNearestByJoin` runs in `FinishAnalysis` (the optimizer's first batch) and would
beat us to the punch otherwise.

## Module layout

```
lance-spark-knn_2.12/                        ← canonical sources (Phase 0/1)
  pom.xml                                    ← depends on lance-spark-base_2.12
                                              + lance-spark-3.5_2.12 as test-scope
  IMPL_PLAN.md                               ← architecture / phasing
  DESIGN.md                                  ← review-friendly overall feature design
  PHASE_PROGRESS.md                          ← this file
  src/main/scala/org/lance/spark/knn/
    IndexedNearestJoin.scala                 ← public DataFrame API
    internal/
      Metric.scala                           ← L2 / Cosine / Dot enum, smallerIsBetter flag
      ScoredRowRef.scala                     ← (rowId, score) tuple shipped through shuffle (field
                                                 named `rowAddr` for source-compat; semantically a row ID)
      ProbedLeft.scala                       ← (leftRow, refs[]) tuple, the shuffle value
      TopKHeap.scala                         ← bounded heap for map-side combine
      LanceProbe.scala                       ← per-task probe primitive (open-once dataset)
      LanceProbeStage.scala                  ← Phase 1 stage 1: probe RDD transformer
      LanceMergeStage.scala                  ← Phase 1 stage 2 config (aggregation lives in LanceMergeExec)
      LanceMaterializeStage.scala            ← Phase 1 stage 3: point-fetch right rows
  src/test/scala/org/lance/spark/knn/
    IndexedNearestJoinTest.scala             ← oracle equivalence (recall=1.0 vs. brute force)
    IndexedNearestJoinPlanShapeTest.scala    ← Phase 1 plan-shape assertions
    internal/
      TopKHeapTest.scala                     ← TopKHeap unit tests
      LanceProbeValidationTest.scala         ← LanceProbe primitive validation

lance-spark-knn_2.13/                        ← cross-build pom only
  pom.xml                                    ← sources point at ../lance-spark-knn_2.12/...

lance-spark-knn-4.2_2.13/                    ← Phase 2 Catalyst integration; Spark 4.2-only
  pom.xml                                    ← Arrow 19, spark-sql 4.2.0-SNAPSHOT (provided)
  src/main/scala/org/lance/spark/knn/
    catalyst/
      IndexedNearestByJoinRule.scala         ← postHocResolutionRule pattern match; emits
                                                the 3-plan tree shared with the DataFrame path
    extensions/
      LanceKnnSparkSessionExtensions.scala   ← user-facing entry point; registers rule + shared strategy
  src/test/scala/org/lance/spark/knn/catalyst/
    IndexedNearestByJoinRuleTest.scala       ← pattern-match tests (positive + negative)
    IndexedNearestByJoinE2ETest.scala        ← SQL e2e on real Lance (Spark 4.2-SNAPSHOT)
```

The `_2.12` directory holds the canonical sources; `_2.13` is a thin pom that re-uses them. This
matches `lance-spark-base_2.12` / `lance-spark-base_2.13` and `lance-spark-3.5_*` — same
convention as the rest of the project.

## What's done

> A standalone, review-friendly design overview of the whole feature is in
> `DESIGN.md` (next to this file). Read that for the "what / why / shape", and read
> this file for "where things live, how to resume, gotchas".

### Phase 0 — pure DataFrame API

`IndexedNearestJoin.apply(left, rightLanceUri, leftVecCol, rightVecCol, k, ...)` — opens the
right Lance dataset per task, probes Lance's nearest search per left row, materializes top-K
right rows, emits join rows. All inside one `mapPartitions` block. No shuffle, no Catalyst.

This phase exists to validate the per-task Lance access pattern works at all and to baseline
recall/correctness against a brute-force oracle.

### Phase 2 — Catalyst integration (new module `lance-spark-knn-4.2_2.13`)

Spark 4.2-SNAPSHOT only. Adds a `postHocResolutionRule` that pattern-matches
`NearestByJoin(approx = true, ...)` over a Lance scan and rewrites it to the 3-plan
staged tree (`LanceProbeLogicalPlan → LanceMergeLogicalPlan → LanceMaterializeLogicalPlan`)
wrapped in a `Project` that restores `NearestByJoin.output` exactly. The shared
`LanceKnnStagedStrategy` lowers that tree to the matching physical execs — the SQL path
and the DataFrame API path converge on the same physical shape.

**Public surface unchanged.** Wiring is one extension class registration:

```scala
SparkSession.builder()
  .config("spark.sql.extensions",
          "org.lance.spark.knn.extensions.LanceKnnSparkSessionExtensions")
  .config("spark.lance.knn.indexedNearestByJoin.enabled", "true")
```

After registration, any `APPROX NEAREST k BY {DISTANCE | SIMILARITY} f(l.vec, r.vec)` SQL
query against a Lance table rewrites automatically. EXACT queries and unrecognized shapes
flow through to Spark's existing brute-force rewrite — no regression.

The rule is gated by `spark.lance.knn.indexedNearestByJoin.enabled` (default `false`) to
keep it opt-in until the Phase 3 cost gate lands.

**The single most important detail** — `injectPostHocResolutionRule`, NOT
`injectOptimizerRule`. Spark's `RewriteNearestByJoin` runs in the optimizer's
`FinishAnalysis` batch (the *first* batch). `injectOptimizerRule` adds rules to
`operatorOptimizationBatch` which runs *after*; by then `NearestByJoin` is already gone.
`injectPostHocResolutionRule` runs after analysis but before any optimizer batch — it's
the only injection point that sees the unrewritten operator.

Tests cover rule pattern-match (positive + negative) plus real-backend SQL e2e. The trick
for the e2e: lance-spark-4.1's source compiles cleanly against 4.2-SNAPSHOT
(`-Dspark41.version=4.2.0-SNAPSHOT -Darrow183.version=19.0.0`) and the resulting jar runs
on 4.2's DSv2 API, so we recompile it locally and use it as the runtime Lance reader. Three
e2e cases: rule-on routes through the 3-exec staged chain and matches oracle; WHERE-pushdown
round-trips the prefilter and matches the filtered oracle; rule-off falls through to Spark's
`RewriteNearestByJoin` and still matches oracle.

### Phase 3 — hardening (partial)

Two substantive items shipped:

  1. **`refineFactor` / `ef`** parameters on `IndexedNearestJoin.apply`. Plumbed through
     `LanceProbeStage.Conf` to `LanceProbe.probe`, which calls `Query.Builder.setRefineFactor`
     / `setEf`. IVF-PQ recall tuning + HNSW search depth respectively. Defaults preserve
     current behavior.
  2. **`balanceFragmentsByRowCount`** flag. When true, `LanceFragments` enumerates fragments
     with their row counts and runs LPT (Longest Processing Time) greedy bin-packing into N
     groups. 4/3-approximation of optimal makespan. Default false = round-robin (Phase 1.5
     behavior, fine for evenly-sized fragments).

A `SupportsApproxNearestNeighborSearch` marker trait was prototyped during
development and then dropped. The indexed-path executor calls Lance's Java API
directly, so it's Lance-specific by construction; a general-purpose extension trait
implies portability the rule can't actually deliver. Class-name detection + standard
DSv2 options give us everything we need.

`LanceProbe.vectorColumn` was moved from a constructor field to a per-call argument on
`probe()` so the materialize stage no longer constructs the probe with a `vectorColumn = ""`
placeholder — a code smell flagged in Phase 0.

`prefilter` pushdown landed: when the right side is `Filter(cond, lance)` (or
`Project(<passthrough>, Filter(cond, lance))`, the SQL `WHERE` shape), the rule translates
the predicate to Lance SQL and threads it through `LanceProbeStage.Conf.prefilter` →
`ScanOptions.filter()`. Translation handles binary comparisons, `IN`, `IS [NOT] NULL`,
`AND`/`OR`/`NOT` over right-side attrs vs. literals. Anything else (UDFs, computed
expressions, predicates referencing the LEFT input) → rule REFUSES the rewrite (returns
the original `NearestByJoin`, falls through to brute force). Refusal — not partial
pushdown — because dropping a residual conjunct would silently change semantics. Slow
but correct.

**3-stage explicit physical operators (DataFrame API path) — DONE.** The
current shape ships the operator split + AQE-visible merge shuffle (via
`ClusteredDistribution`) + single-pass inter-stage codec as one clean
commit. An early development iteration hit reproducible `AssertionError` /
SIGSEGV in `UnsafeRow.getLong` on `count()`-style consumers, initially
misdiagnosed but narrowed to the real root cause during investigation.

The initial diagnosis blamed a JVM-aarch64 + JIT C2 interaction — that was
wrong. The crash reproduces on aarch64 but isn't a JVM bug. Step-wise
isolation (`InterStageShuffleReproTest` → `InterStageShuffleWithLanceReproTest` →
`StagedExecDirectDriveReproTest`) narrowed it to the staged execs specifically, not
Spark's UnsafeRow shuffle or the Lance→Spark boundary. Diagnostic instrumentation caught
0-field `UnsafeRow`s arriving at `LanceMaterializeExec.doExecute`. Dumping the executed
plan for `count()` showed Catalyst's `ColumnPruning` rule inserting `Project(Nil)`
wrappers between the custom nodes — empty projections that codegen to 0-field
UnsafeRows. The decoder then crashed with `AssertionError` in interpreter mode, SIGSEGV
in C2 (assertion elided → unmapped-memory read).

Fix: `LanceMergeLogicalPlan` and `LanceMaterializeLogicalPlan` override
`lazy val references = child.outputSet`. That makes `child.outputSet.subsetOf(references)`
trivially true and short-circuits `ColumnPruning`'s guard, so no `Project(Nil)` ever
gets inserted. `StagedPlansReferencesTest` pins the invariant structurally.

Production today: `LanceProbeExec → ShuffleExchangeExec → LanceMergeExec →
LanceMaterializeExec` wrapped by `AdaptiveSparkPlanExec`. `df.explain()` shows all four
nodes; with AQE enabled, `AQEShuffleRead coalesced` appears on the merge-side shuffle.
60 tests pass in `lance-spark-knn_2.12`. See `IMPL_PLAN.md` "3-exec staged split — root
cause and fix" for the full post-mortem.

**`df.kNearestJoin` extension.** Idiomatic DataFrame API mirroring
`df.join(other, ...)` — works on Spark 3.5 / 4.0 / 4.1 / 4.2+ since it goes straight to
`IndexedNearestJoin.apply` without touching the Phase 2 SQL parser. Extracts the Lance
URI from the right DataFrame's analyzed plan; non-Lance right sides (parquet, in-memory
DataFrames, alias-wrapped non-Lance) fail fast with `IllegalArgumentException`.

What's left (see `IMPL_PLAN.md` for the full table): cost gate, Spark version matrix,
AQE-visible shuffle for the fragment-grouped probe path (`runWithFragmentGroups`'s
internal `partitionBy` is still RDD-level).

### Benchmarks + SQL e2e

Two oracle-validated benchmarks on M5 Max:

  - **DataFrame** (`IndexedNearestJoinBenchmark` in `lance-spark-knn_2.12`): indexed staged
    pipeline vs. naive Spark `crossJoin + array_distance UDF + row_number window`.
    Headline: **608×** at 100K × 100 (109,373 ms → 180 ms).
  - **SQL** (`IndexedNearestByJoinSqlBenchmark` in `lance-spark-knn-4.2_2.13`): same
    `APPROX NEAREST` SQL with the Phase 2 rule ON vs. OFF (= Spark's `RewriteNearestByJoin`
    cross-product + `min_by_k`). Headline: **17.4×** at 100K × 100 (3,728 ms → 214 ms).

Both run a pre-timing oracle equivalence check on a 16-row left subset comparing every
config (including the slow baseline / rule-OFF) against an in-memory brute-force ground
truth. The benchmark `sys.error`'s if any disagrees, so quoted speedups are on validated
results.

Real-backend SQL e2e against Spark 4.2-SNAPSHOT works because lance-spark-4.1's source
compiles cleanly against 4.2 and the resulting jar runs on 4.2's DSv2 API. Surfaced two
real bugs in Phase 2 during development — View wrapper not unwrapped, `producedAttributes`
missing — both fixed.

Honest finding: Phase 1.5 fragment-grouping is *slower* than Phase 0/1 at every measured
scale on this single laptop. Lance's internal cross-fragment merge already parallelizes via
vectorized native kernels; Spark task-boundary parallelism doesn't help in shared-memory
local mode. Plumbing is correct (oracle test passes); the win lands on a true distributed
cluster, object-store-backed Lance, or a right side too large for one machine. Documented
in `BENCHMARK_RESULTS.md`.

### Phase 1.5 — fragment-grouped probing

Same module as Phase 0/1 (`lance-spark-knn_2.12`). Adds an opt-in
`probeParallelism: Int = 1` parameter on `IndexedNearestJoin.apply`. When > 1:

  1. Driver enumerates Lance fragment IDs via `Dataset.getFragments()` (helper
     `internal/LanceFragments.scala`).
  2. Round-robin into N groups; broadcast.
  3. Replicate each left row across the N groups via `flatMap`, partition by
     `groupIdx` so each task handles a single group.
  4. Each task opens `LanceProbe` with its group's `fragmentIds` and probes only those.
  5. Output keyed by `leftId` produces N contributions per leftId; downstream
     `LanceMergeExec` (with `ClusteredDistribution(leftId)`) aggregates contributions
     per-partition via `TopKHeap.merge` after the Catalyst-inserted exchange co-locates
     them.

The flatMap + partitionBy is one shuffle; the Catalyst-inserted Exchange above
`LanceMergeExec` is the second. Two shuffles total — what the IMPL_PLAN's three-stage
diagram has always shown.

This is where the bandwidth win the IMPL_PLAN promises ("refs only ~24B") actually lands.
Phase 1 had the staging but degenerate (single contributor per leftId). Phase 1.5 makes
the merge stage do real work.

**Edge case**: when `probeParallelism > numFragments`, only one group has fragments and
the rule degenerates back to the Phase 1 single-task path — avoiding a replicate shuffle
for nothing.

**3 new tests**:
  - Oracle equivalence with `probeParallelism = 4` and a 4-fragment right dataset. With
    no index, every probe is exact, so the merge result must match brute force exactly.
  - Plan-shape: `probeParallelism > 1` adds a second `ShuffledRDD` to the lineage
    (verified by counting `ShuffledRDD` occurrences in `toDebugString`).
  - `probeParallelism > numFragments` (e.g. 8 over a single-fragment dataset) still
    produces correct results.

Total knn module tests: 23 (was 16).

### Phase 1 — staged RDD pipeline + Phase 3.x — explicit physical operators

The Phase 0 inline `mapPartitions` was first split into three stage objects connected via
`reduceByKey`. Phase 3.x then promoted the DataFrame path to three explicit `SparkPlan`
operators with a Catalyst-inserted `ShuffleExchangeExec` between probe and merge. Current
shape:

```
left.analyzed
  -- LanceProbeLogicalPlan → LanceProbeExec        --> (per-task) nearest-search
  -- ShuffleExchangeExec hashpartitioning(_leftId)  --> AQE wraps this when enabled
  -- LanceMergeLogicalPlan → LanceMergeExec         --> per-partition TopKHeap merge
  -- LanceMaterializeLogicalPlan → LanceMaterializeExec --> _rowid point-fetch, assemble
```

**Public API unchanged.** `IndexedNearestJoin.apply(...)` callers see the same signature
and the same output schema. Phase 0's oracle test passes unmodified — proves the
refactor preserves correctness.

**Plan-shape assertions**:
- `IndexedNearestJoinPlanShapeTest`: executed plan contains `LanceProbe`, `LanceMerge`,
  `LanceMaterialize`, and `Exchange`.
- `IndexedNearestJoinAqeVisibilityTest`: `ShuffleExchangeExec hashpartitioning(_leftId)`
  is in the tree (AQE on and AQE off), `AdaptiveSparkPlanExec` wraps it when AQE is on,
  no `!` missingInput prefix.
- `df.rdd.toDebugString` contains `ShuffledRowRDD` (the Catalyst shuffle reader — not
  the pre-Phase-3.x `ShuffledRDD` produced by RDD-level `reduceByKey`).

### What's not yet built

#### Limitations remaining after Phase 0/1/1.5/2/3

1. **Single-task probing when `probeParallelism = 1` (default)**: each `leftId` has exactly one
   probe contributor and the merge function never fires — the shuffle is structurally present
   but degenerate. Pass `probeParallelism > 1` to engage Phase 1.5's fragment-grouped path
   where the merge stage actually aggregates contributions.
2. **Left payload in shuffle**: `ProbedLeft` carries the full `leftRow` through the shuffle.
   Cost is `~payload + 24B × K` per leftId-group instead of `~24B × K`. Fixing requires
   repartitioning the left RDD by `leftId` up front and joining back at materialize via
   `cogroup`. Deferred to Phase 3.x.
3. **Synthetic leftId from `zipWithUniqueId`**, not a user-supplied join key. Means we can't
   yet co-partition the left payload alongside `(leftId, refs)`.
4. **Filter pushdown** (`prefilter`) — DONE. The Catalyst rule detects `Filter(cond, lance)`
   on the right side, translates the predicate to a Lance SQL filter string, and threads it
   through `LanceProbeStage.Conf.prefilter` → `ScanOptions.filter()`. Refuses rewrite if any
   conjunct doesn't translate (no partial pushdown).
5. **Vector column on materialize stage** — DONE. `LanceProbe.vectorColumn` is now a per-call
   `probe()` argument, not a constructor field; the materialize stage opens `LanceProbe`
   without any vector-column placeholder.

The full Phase 3.x backlog (cost gate, real-recall test, etc.) lives in `IMPL_PLAN.md`'s
"What's left" table.

## Lance Java API surface used

These were validated during Phase 0 from the upstream Lance Java sources before being used.
If a future Lance version breaks any of them, that is the first place to look:

| Class / method | Purpose |
|---|---|
| `Dataset.open()` (builder) | Open Lance dataset; takes `.uri()` `.allocator()` `.readOptions()` |
| `ReadOptions.Builder().setVersion(v)` | Pin to a specific Lance version |
| `LanceScanner.create(dataset, options, allocator)` | Create a scanner |
| `ScanOptions.Builder.nearest(query)` | Configure vector nearest search |
| `ScanOptions.Builder.prefilter(true)` | Required when `fragmentIds` is non-empty |
| `ScanOptions.Builder.withRowId(true)` | Surface `_rowid` in result (we use this; the indexed nearest-search path doesn't materialize `_rowaddr`) |
| `ScanOptions.Builder.fragmentIds(list)` | Restrict to specific fragments |
| `ScanOptions.Builder.columns(emptyList)` | Project nothing — refs only |
| `Query.Builder` (column, key, k, distanceType, nprobes) | Build the nearest-search query |
| `org.lance.index.DistanceType` | L2 / Cosine / Dot enums |

## Build & test

The whole module builds via Maven (no SBT here — lance-spark project uses Maven):

    cd /Users/esong/repos/lance-spark
    ./mvnw -pl lance-spark-knn_2.12 test-compile
    ./mvnw -pl lance-spark-knn_2.12 test
    ./mvnw -pl lance-spark-knn_2.12 test -Dtest='IndexedNearestJoinTest'
    ./mvnw -pl lance-spark-knn_2.12 test -Dtest='*PlanShape*'

Cross-build (2.13):

    ./mvnw -pl lance-spark-knn_2.13 test    # uses _2.12 sources via pom

Phase 2 module (Spark 4.2-SNAPSHOT, Scala 2.13 only) — first install Spark master locally:

    cd /path/to/spark/master
    ./build/mvn install -DskipTests -DskipChecks -Drat.skip=true -Dscalastyle.skip=true \
      -pl sql/core -am

Then in lance-spark:

    ./mvnw install -DskipTests -pl lance-spark-knn_2.13 -am
    ./mvnw -pl lance-spark-knn-4.2_2.13 test

**Don't pass `-am` to surefire-filtered runs** — surefire's test pattern then runs against
the base module too, which has zero matching tests and fails. Just run `-pl <module>` alone.

## Gotchas observed during Phase 0/1 — keep these in mind

1. **Scala 2.13 `Seq` doesn't match `mutable.ArraySeq`.** Spark `Row.get` on `ArrayType` returns
   `mutable.ArraySeq`. `case s: Seq[_]` in 2.13 only matches `immutable.Seq`. Use the root
   trait: `case s: scala.collection.Seq[_]`. See `LanceProbeStage.extractVector`.
2. **Arrow `FieldVector.getObject` returns `JsonStringArrayList`**, not Scala `Seq`. Spark's
   `RowEncoder` won't accept it for ArrayType slots. `LanceProbe.toSparkValue` recursively
   converts `java.util.List → Seq`, `Map → Map`, `Text → String`.
3. **Spark driver bind error in tests**: every `SparkSession.builder()` in tests must set
   `spark.driver.bindAddress=127.0.0.1` and `spark.driver.host=127.0.0.1` or it fails to bind
   on restricted networks (CI sandboxes, dev containers).
4. **Lance's `format("lance")` registration** comes from `lance-spark-3.5_2.12`'s
   `META-INF/services/org.apache.spark.sql.sources.DataSourceRegister`. The knn module's
   sources don't depend on that, but its **tests** do — `lance-spark-3.5_2.12` is a test-scope
   dependency in the pom for that reason.
5. **`vectorColumn = ""` on the materialize-only LanceProbe**: not a bug, the param is just
   unused on that path. Documented inline.

## Validation checklist for new changes

Before declaring a phase done:

- [ ] `./mvnw -pl lance-spark-knn_2.12 test` — all tests pass
- [ ] Phase 0 oracle test (`testInnerJoinMatchesBruteForceOracle`) passes — recall = 1.0 vs.
      plain-Scala brute force is a load-bearing correctness check
- [ ] `df.rdd.toDebugString` for the output of `IndexedNearestJoin.apply` matches the phase's
      expected staged form (current: contains `ShuffledRowRDD` — the Catalyst shuffle
      reader, produced by the 3-exec staged plan)
- [ ] Spotless / scalastyle / checkstyle all clean (`./mvnw spotless:check`)
- [ ] No new non-ASCII chars in source (especially smart quotes / em-dashes from copy-paste)
- [ ] Update IMPL_PLAN.md status table if a phase moved
- [ ] Update this file's "What's done" section

## Quick map: where to look for X

| Question | File |
|---|---|
| How does the public API work? | `IndexedNearestJoin.scala` — start at `apply` |
| Why these three stages? | `IMPL_PLAN.md` "Three-phase distributed design" |
| How is the shuffle bandwidth bounded? | `ScoredRowRef.scala` doc comment |
| Why `injectPostHocResolutionRule` for Phase 2? | `IMPL_PLAN.md` "Phase 2 — Catalyst integration" |
| What does `_rowid IN (...)` lower to in Lance? | `LanceProbe.materialize` — pushdown into row-id lookup, point-fetch path. Switched from `_rowaddr` for indexed-path compatibility; see DESIGN.md "Why `_rowid` not `_rowaddr`". |
| How does fragment-grouping (Phase 1.5) work? | Pass `probeParallelism > 1` to `IndexedNearestJoin.apply`. `LanceFragments.enumerateGroups` round-robins fragment IDs into N groups, `LanceProbeStage.runWithFragmentGroups` replicates rows × groups, downstream merge aggregates. |
| Why does `Metric` carry `smallerIsBetter`? | `Metric.scala` — drives heap eviction direction |
