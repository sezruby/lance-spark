# Upstream delivery plan — `knn-phase0` → `lance-format/lance-spark:main`

The `knn-phase0` branch is ~13,300 LoC ahead of upstream `main` across ~60
files, organized as 9 feature-boundary commits. That's too large to land
as one PR. This document lays out a split into independent, reviewable
PRs, broadly aligned with the 9-commit structure.

## Scope limits

- **Benchmarks not shipped.** The `benchmark/` directory (both `src/main/`
  and `src/test/`) is internal tooling: it justifies design decisions and
  produced the headline speedup numbers, but isn't part of the public API
  and pulls in HTTP-fetch / local-FS logic that doesn't belong in the
  connector. Keep benchmarks on the fork / document them in an external
  `BENCHMARK_RESULTS.md` post.
- **Spark 4.2 module not shipped yet.** The `lance-spark-knn-4.2_2.13`
  module depends on Spark 4.2-SNAPSHOT's `NearestByJoin` logical plan
  (SPARK-56395), which isn't in a released Spark yet. Defer this PR until
  Spark 4.2.0 publishes to Maven Central.

## Redundancy audit

Scanned the branch for dead / duplicate / unused files. Two items found + resolved; three
items checked and kept (with rationale).

### Removed during development (not present on the branch today)

1. **An earlier duplicate `IndexedNearestJoinBenchmark.scala`** sat in
   `src/test/` after the benchmark was moved to `src/main/` to ship in
   the shaded fat JAR. The duplicate is gone in the current branch.

2. **Phase 2 single-exec SQL path + `LanceMergeStage.run` (with `reduceByKey`).**
   An earlier iteration shipped `IndexedNearestByJoinPlan` +
   `IndexedNearestByJoinExec` + `IndexedNearestByJoinStrategy` — a single
   UnaryExec that called the old `LanceProbeStage.run` /
   `LanceMergeStage.run` / `LanceMaterializeStage.run` helpers (where
   `LanceMergeStage.run` did the shuffle via RDD `reduceByKey`). Once the
   DataFrame path moved to the 3-exec Catalyst-visible design, the
   SQL-specific single-exec became redundant. The current branch emits
   the 3-plan logical tree from `IndexedNearestByJoinRule` and shares
   `LanceKnnStagedStrategy` between both paths; `LanceMergeStage.run`
   is gone (merge is now a per-partition `mapPartitions` inside
   `LanceMergeExec`, fed by a Catalyst-inserted `ShuffleExchangeExec`).

### Kept — not redundant

1. **`InterStagePayloadOverheadBench.scala`** — microbenchmark that backs the
   Catalyst-struct vs Kryo-blob choice documented in `ProbedLeftCodec`'s scaladoc.
   Test-scope only, not shipped to users. Kept as re-runnable evidence for the design
   claim in the comment (cited by name from `StagedExecs.scala` and `ProbedLeftCodec.scala`).

2. **`IndexedNearestJoinJitStressTest` + `InterStageShuffleReproTest` +
   `InterStageShuffleWithLanceReproTest`** — diagnostic tests from the
   ColumnPruning investigation that led to the `references = child.outputSet`
   fix. They rule out what *wasn't* the cause (JVM-aarch64, Spark UnsafeRow
   shuffle, Lance→Spark boundary). Kept as regression coverage — they would
   catch the same class of crash if it returned in a different place.

3. **Legacy RDD path in `IndexedNearestJoin.apply`** — the public entry
   point now builds the 3-exec staged logical plan tree directly. The
   RDD-level `LanceProbeStage` / `LanceMergeStage` / `LanceMaterializeStage`
   helpers are still called from the `Exec` nodes' `doExecute`, so they're
   not dead.

4. **`LanceVectorIndexBuilder.scala`** in test/ — helper used only by
   `IndexedNearestJoinIvfPqRecallTest`. Not a duplicate; test-scoped utility.

## PR split strategy

The split axis is "minimum reviewable unit" — each PR introduces a feature
that stands on its own, with tests that exercise only that feature. Phase
ordering is preserved so reviewers can read commits chronologically.

### PR 1: Phase 0 foundation — `LanceProbe` primitive

**Goal:** Ship the per-task Lance nearest-search primitive + oracle tests.
This is the smallest self-contained unit of the feature. Nothing here is
user-facing; the primitive is private and exercised via unit tests.

Files (~700 lines):

- `lance-spark-knn_2.12/pom.xml` (new module; minimal deps)
- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/LanceProbe.scala`
- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/Metric.scala`
- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/ScoredRowRef.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/internal/LanceProbeValidationTest.scala`
- `lance-spark-knn_2.13/pom.xml` (shared-source 2.13 build)
- `pom.xml` (reactor registration of the two modules)

**Tests:** `LanceProbeValidationTest` (4) — probe returns K row refs,
ordered by distance, with correct score column.

**Why it's reviewable standalone:** no connection to Catalyst, no
DataFrame API, no extension points. Just "here's a primitive that opens
Lance, runs `nearest` search, returns `(rowId, score)` pairs." A reviewer
can decide on the API shape without worrying about Spark integration.

**Estimated review time:** 2–3 hours.

### PR 2: Phase 0/1 — `IndexedNearestJoin.apply` + staged RDD pipeline

**Goal:** Add the three-stage RDD pipeline (probe → shuffle → merge →
materialize) and the bounded-merge heap. This is the functional core.

Files (~1,500 lines):

- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/IndexedNearestJoin.scala` (entry point — RDD-only shape for this PR; PR 4 rewires it to build the 3-plan logical tree)
- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/LanceProbeStage.scala`
- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/LanceMergeStage.scala`
- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/LanceMaterializeStage.scala`
- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/ProbedLeft.scala`
- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/TopKHeap.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/internal/TopKHeapTest.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/IndexedNearestJoinTest.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/IndexedNearestJoinCorrectnessTest.scala`

**Tests:** oracle equivalence, plan-shape (`ShuffledRDD` in lineage),
outer-join / custom scoreCol / projection.

**Why split from PR 1:** even a reviewer happy with `LanceProbe`'s API may
want to debate the staged-pipeline design separately. The pipeline shape
(`zipWithUniqueId` for leftId, Catalyst-inserted hash-partitioned exchange
above `LanceMergeExec` for the merge shuffle, point-fetch materialize) is
the "claim" this PR makes.

**Estimated review time:** 4–6 hours.

**Caveat:** this temporarily regresses behavior vs the current
`knn-phase0` branch (no AQE on merge shuffle, no Catalyst operators in
`df.explain()`). PR 4 restores that.

### PR 3: Phase 1.5 — fragment-grouped probe (`probeParallelism > 1`)

**Goal:** Add the optional fragment-grouping that enables parallel
probe tasks across a distributed cluster.

Files (~600 lines):

- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/LanceFragments.scala`
- Modifications to `LanceProbeStage.scala`: add `runWithFragmentGroups`
- Modifications to `IndexedNearestJoin.scala`: `probeParallelism` +
  `balanceFragmentsByRowCount` parameters
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/internal/LanceFragmentsTest.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/IndexedNearestJoinFragmentGroupingTest.scala`

**Tests:** LPT bin-packing math, oracle equivalence with G=4 and G=8
groups, skew-balanced variant.

**Why split from PR 2:** fragment grouping is opt-in (`probeParallelism = 1`
is the default and doesn't use it). A reviewer can accept the staged
pipeline first, then debate whether / how to expose the
`probeParallelism` knob. Cluster evidence shows fragment grouping pays
off only when `probeParallelism == numFragments` — the default of 1
is correct for single-machine / single-executor.

**Estimated review time:** 2–3 hours.

### PR 4: Phase 2 — 3-exec staged Catalyst operators + AQE visibility

**Goal:** Replace the RDD-only execution path with
`LanceProbeExec → ShuffleExchangeExec → LanceMergeExec → LanceMaterializeExec`
so `df.explain()` sees the pipeline and AQE can engage on the merge
shuffle.

Files (~1,300 lines):

- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/staged/ProbedLeftCodec.scala`
- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/staged/StagedPlans.scala`
- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/staged/StagedExecs.scala`
- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/staged/LanceKnnStagedStrategy.scala`
- `lance-spark-knn_2.12/src/main/scala/org/apache/spark/sql/LanceKnnDatasetBridge.scala`
- Modifications to `IndexedNearestJoin.apply`: route through logical plans
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/IndexedNearestJoinAqeVisibilityTest.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/IndexedNearestJoinPlanShapeTest.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/IndexedNearestJoinConsumerShapeTest.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/IndexedNearestJoinJitStressTest.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/internal/staged/StagedPlansReferencesTest.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/staged/InterStageShuffleReproTest.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/staged/InterStageShuffleWithLanceReproTest.scala`

**PR description** should include the full "3-exec staged split — root
cause and fix" post-mortem from the current `IMPL_PLAN.md`: the
ColumnPruning guard, the `Project(Nil)` insertion, the 0-field UnsafeRow
crash, the `references = child.outputSet` fix. This is the most subtle
part of the feature and reviewers need the history.

**Tests:**

- `StagedPlansReferencesTest` (3) — pins the `references` override
- `IndexedNearestJoinAqeVisibilityTest` (5) — AQE engages, `AdaptiveSparkPlanExec` wraps, all 3 execs in tree
- `IndexedNearestJoinConsumerShapeTest` (4) — `count()`, `agg(count("*"))`, `select(lit(1))`, `collect()` — the shapes that crashed the reverted code
- `IndexedNearestJoinJitStressTest` (2) — 20-iteration JIT stress
- `InterStageShuffleReproTest` (6) + `InterStageShuffleWithLanceReproTest` (4) — the isolation tests from the investigation, kept as regression coverage

**Estimated review time:** 6–10 hours. This is the heaviest PR.

### PR 5: `df.kNearestJoin` DataFrame extension

**Goal:** User-facing extension method on `DataFrame` that mirrors
`df.join(other, ...)`. Wraps `IndexedNearestJoin.apply` with URI extraction
from the right DataFrame's analyzed plan.

Files (~250 lines):

- `lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/LanceKnnImplicits.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/LanceKnnImplicitsTest.scala`

**Tests:** extension method end-to-end, Filter-on-right unwrap, Lance-only
format guard (rejects non-Lance right).

**Why a separate PR:** pure syntactic sugar over PR 2-4. A reviewer can
debate the API name + ergonomics without touching the engine.

**Estimated review time:** 1 hour.

### PR 6: Phase 3 hardening — `refineFactor`, `ef`, IVF-PQ recall test

**Goal:** Add the IVF-PQ recall knobs (`refineFactor`, `ef`) + a real
recall test built against an actual Lance vector index.

Files (~600 lines):

- Modifications to `IndexedNearestJoin.apply`: new `refineFactor` / `ef` params
- Modifications to `LanceProbeStage.Conf` + `LanceProbe.probe`: plumb the knobs
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/IndexedNearestJoinIvfPqRecallTest.scala`
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/internal/LanceVectorIndexBuilder.scala` (test util)
- `lance-spark-knn_2.12/src/test/scala/org/lance/spark/knn/testutil/ClusteredEmbeddings.scala` (synthetic clustered data for recall tests)

**Tests:** recall@10 at default IVF-PQ = 0.73, with `refineFactor=8`
reaches 1.00. Clustered-embeddings recall is printed but not asserted
(run-to-run variance too large at test scales).

**Why split:** the recall knobs compose cleanly with the existing probe
API; this PR just adds parameters.

**Estimated review time:** 2 hours.

### PR 7: Spark 4.0 compatibility (reflection bridge)

**Goal:** Make the module work on Spark 4.0+ where
`org.apache.spark.sql.Dataset` moved to
`org.apache.spark.sql.classic.Dataset`.

Files (~60 lines):

- Modifications to `LanceKnnDatasetBridge.scala` — reflection-based
  `ofRows` lookup

**Tests:** 41/41 tests pass on Spark 4.0 + Scala 2.13 with this bridge
(validated on `knn-phase0` with temporary pom overrides).

**Why separate:** trivially small, easy to review, makes the
`lance-spark-knn_2.13` module able to compile against `spark40.version`.
Doesn't need to block anything else.

**Estimated review time:** 30 minutes.

## Suggested review order for upstream

```
PR 1 (Phase 0: LanceProbe primitive)
  └─ PR 2 (Phase 0/1: IndexedNearestJoin.apply, staged RDD pipeline)
       ├─ PR 3 (Phase 1.5: fragment grouping)
       ├─ PR 4 (Phase 2: 3-exec Catalyst operators, AQE visibility)
       │    └─ PR 5 (df.kNearestJoin extension)
       ├─ PR 6 (Phase 3: refineFactor / ef / IVF-PQ recall)
       └─ PR 7 (Spark 4.0 compat) — parallel-reviewable, small
```

PR 1 → 2 → 4 is the critical path for the "real" feature to work. PR 3, 5,
6, 7 can land in parallel once PR 4 is in.

## Out-of-scope (keep on fork indefinitely)

- **All benchmarks** (`benchmark/` in both `main/` and `test/` trees):
  `IndexedNearestJoinBenchmark`, `SiftRecallBenchmark`,
  `CohereWikiRecallBenchmark`, `WikipediaKnnPerfBenchmark`,
  `IndexedNearestJoinSoakTest`, `InterStagePayloadOverheadBench`.
- **Deployment-specific build config**: the Linux-x86_64 shade filter in
  `lance-spark-knn_2.12/pom.xml` (only needed for managed-Spark distributions
  with volume-upload ingress timeouts).
- **Cluster results section** in `BENCHMARK_RESULTS.md` (specific cluster
  instance; the benchmark tooling is generic).
- **`lance-spark-knn-4.2_2.13`** module (Spark 4.2 still SNAPSHOT; revisit
  after Spark 4.2.0 releases to Maven Central).

## Mechanics — how to actually create the PRs

Each PR is a separate branch cut from `origin/main`, cherry-picking only
its commits:

```sh
git checkout -b upstream/pr1-lance-probe origin/main
# cherry-pick the phase-0 commits that touch only LanceProbe.scala / Metric.scala / ScoredRowRef.scala
# resolve conflicts against upstream main
# run tests, push, open PR against lance-format/lance-spark:main
```

Commits on `knn-phase0` don't map 1:1 to PRs — the branch's history
includes the reverted-and-restored 3-exec saga and benchmark additions
that should be squashed. Each PR should present as 1–3 clean commits with
thorough messages, not the full investigation timeline.

## Known gaps to fix before submitting

- [ ] Each PR branch should rebase onto current `origin/main` and run its
      test subset.
- [ ] Open a JIRA ticket or GitHub issue on the upstream repo describing
      the feature + PR sequence before opening the first PR, so
      reviewers have context.
