# Reviewer guide — `lance-spark-knn`

This PR is ~13 K LoC across ~60 files. It lands as a single branch but the
**intended upstream delivery is 7 smaller PRs** (see
[`UPSTREAM_DELIVERY_PLAN.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/UPSTREAM_DELIVERY_PLAN.md)). This guide helps
you navigate the current monolithic branch in **review-meaningful reading
order** so you can form an opinion on the design without slogging through
files in alphabetical order.

## Start here (10 minutes)

Read these 3 files first. After them, you know what the feature is and can
decide whether you want to dig deeper:

1. **[`DESIGN.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/DESIGN.md)** — overall architecture + the "why no-index
   Lance beats Spark cross-product" SIMD/columnar breakdown.
2. **[`IndexedNearestJoin.scala`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/IndexedNearestJoin.scala)** — the public entry point. 232 lines. One function,
   `IndexedNearestJoin.apply(...)`, builds the 3-logical-plan tree and
   hands it to Catalyst. The scaladoc on `apply` documents every parameter.
3. **[`LanceKnnImplicits.scala`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/LanceKnnImplicits.scala)** — the `df.kNearestJoin(rightDf, ...)` extension method.
   ~160 lines. Syntactic sugar over `IndexedNearestJoin.apply` with URI
   auto-extraction.

After those 3 you know the shape of the feature from a user perspective.

## Next: read the engine (30 minutes)

Four files. These are the 3-exec staged Catalyst operators on top of
the RDD primitives. This is the heart of the feature and the subtlest
part to review:

1. **[`StagedPlans.scala`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/staged/StagedPlans.scala)** — the three logical plan nodes
   (`LanceProbeLogicalPlan`, `LanceMergeLogicalPlan`,
   `LanceMaterializeLogicalPlan`).
   **Critical invariant:** Merge and Materialize override
   `lazy val references = child.outputSet`. Removing that line reintroduces
   the ColumnPruning → `Project(Nil)` → 0-field UnsafeRow → SIGSEGV crash
   the scaladoc describes. `IMPL_PLAN.md` "3-exec staged split — root
   cause and fix" has the full post-mortem.
2. **[`StagedExecs.scala`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/staged/StagedExecs.scala)** — the three physical operators
   (`LanceProbeExec`, `LanceMergeExec`, `LanceMaterializeExec`).
   **Key decision:** `LanceMergeExec.requiredChildDistribution =
   ClusteredDistribution(leftId)` — that's what makes Catalyst's
   `EnsureRequirements` insert the `ShuffleExchangeExec` between probe
   and merge, which is what AQE wraps.
3. **[`ProbedLeftCodec.scala`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/staged/ProbedLeftCodec.scala)** — encode/decode for the inter-stage
   `(leftId, leftRow, refs)` rows that cross the `ShuffleExchangeExec`
   boundary. Flat schema (not nested struct) because nested struct tripped
   a Spark serializer bug on arm64 at benchmark scale.
4. **[`LanceKnnStagedStrategy.scala`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/staged/LanceKnnStagedStrategy.scala)** — the `SparkStrategy` that lowers the
   three logical plans to the three physical execs.

## Next: read the RDD primitives (30 minutes)

These are called from the `Exec` nodes' `doExecute`. They're tested
directly by `LanceProbeValidationTest` / `TopKHeapTest` and were the Phase
0/1 foundation before the Catalyst operators were added.

- **[`LanceProbe.scala`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/LanceProbe.scala)** — per-task Lance dataset handle +
  `probe()` + `materialize()`. Closes the dataset on `close()`.
- **[`LanceProbeStage.scala`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/LanceProbeStage.scala)** — two `run` methods: one for
  `probeParallelism = 1` (default, `mapPartitions`), one for Phase 1.5
  fragment-grouped via `flatMap` + `partitionBy`.
- **[`LanceMergeStage.scala`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/LanceMergeStage.scala)** — merge-stage `Conf` (the per-partition
  `TopKHeap.merge` aggregation itself lives in `LanceMergeExec.doExecute`).
- **[`LanceMaterializeStage.scala`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/LanceMaterializeStage.scala)** — point-fetch right rows by
  `_rowid`, assemble the output join rows.
- **[`TopKHeap.scala`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/TopKHeap.scala)** — min/max heap for bounded top-K.
- **[`LanceFragments.scala`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/src/main/scala/org/lance/spark/knn/internal/LanceFragments.scala)** — driver-side fragment enumeration +
  round-robin + LPT bin-packing for skew balancing.

## Optional deeper reads

**The ColumnPruning investigation** (post-mortem): [`IMPL_PLAN.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/IMPL_PLAN.md)
§ "3-exec staged split — root cause and fix". Required reading before
touching `StagedPlans.scala`'s `references` override.

**SQL integration (Spark 4.2-only)**: the [`lance-spark-knn-4.2_2.13/`](https://github.com/sezruby/lance-spark/tree/knn-phase0/lance-spark-knn-4.2_2.13) module intercepts
Spark 4.2's `NearestByJoin` operator. Uses the same `staged/` pipeline
under the hood. Note: Spark 4.2 is SNAPSHOT at time of this PR — this
module depends on an unreleased Spark version. See
[`UPSTREAM_DELIVERY_PLAN.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/UPSTREAM_DELIVERY_PLAN.md) § "Out-of-scope"
for why it's deferred from upstream delivery.

**Phase 3 hardening** (refineFactor / ef / prefilter pushdown / index-name
handling): read [`IMPL_PLAN.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/IMPL_PLAN.md) § "What's left (Phase 3.x)"
— each row in that table links the feature to its test.

**Benchmark results + infrastructure**: [`BENCHMARK_RESULTS.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/BENCHMARK_RESULTS.md)
has both M5 Max local numbers and OSS Spark 3.5 cluster numbers.
The cluster section is where the Phase 1.5 design claim was validated:
**100–200× faster than Spark crossJoin on Cohere Wikipedia embeddings at
dim=1024** (7-iter median 160×; noop sink + 16-row brute-force oracle check).

## Test map — what to run

```sh
# Phase 0/1 core — probe primitive + staged RDD pipeline
./mvnw -pl lance-spark-knn_2.12 test -Dtest='LanceProbeValidationTest,TopKHeapTest,IndexedNearestJoinTest,IndexedNearestJoinCorrectnessTest'

# Phase 1.5 fragment grouping
./mvnw -pl lance-spark-knn_2.12 test -Dtest='LanceFragmentsTest,IndexedNearestJoinFragmentGroupingTest'

# Phase 2 — 3-exec Catalyst operators + AQE + consumer-shape regression
./mvnw -pl lance-spark-knn_2.12 test -Dtest='StagedPlansReferencesTest,IndexedNearestJoinAqeVisibilityTest,IndexedNearestJoinPlanShapeTest,IndexedNearestJoinConsumerShapeTest,IndexedNearestJoinJitStressTest'

# ColumnPruning isolation tests (regression coverage for the reverted-then-restored crash)
./mvnw -pl lance-spark-knn_2.12 test -Dtest='InterStageShuffleReproTest,InterStageShuffleWithLanceReproTest'

# Phase 3 — IVF-PQ recall
./mvnw -pl lance-spark-knn_2.12 test -Dtest='IndexedNearestJoinIvfPqRecallTest'

# DataFrame extension
./mvnw -pl lance-spark-knn_2.12 test -Dtest='LanceKnnImplicitsTest'

# All of the above at once
./mvnw -pl lance-spark-knn_2.12 test
```

Or on Scala 2.13:

```sh
./mvnw -pl lance-spark-knn_2.13 test
```

60 tests across the 2.12/2.13 module. All pass.

## Trust-but-verify checklist

If you're reviewing a specific aspect, here's where to look:

| Concern | Check |
|---|---|
| **Correctness.** Does the indexed path produce the same top-K as brute force? | `IndexedNearestJoinCorrectnessTest` — brute-force oracle at 1K×100×dim=16, recall=1.0. `IndexedNearestJoinTest` — 4 end-to-end tests. `IndexedNearestJoinBenchmark` pre-timing validates ALL configs against oracle on 16-row subset. |
| **AQE actually engages.** Can the merge shuffle be coalesced / rebalanced? | `IndexedNearestJoinAqeVisibilityTest` (5) — checks `ShuffleExchangeExec hashpartitioning(_leftId)` is in the tree, AQE wraps with `AdaptiveSparkPlanExec`, and `AQEShuffleRead coalesced` appears on the merge shuffle. |
| **No regression on consumer shapes.** `count()` / `agg(count("*"))` / `select(lit(1))` don't crash. | `IndexedNearestJoinConsumerShapeTest` (4). These are the exact shapes that crashed the reverted code. |
| **No JIT-level crash at scale.** | `IndexedNearestJoinJitStressTest` — 20-iter × 10K right × 100 left × dim=128. Passes clean. |
| **The `references = child.outputSet` fix is structurally pinned.** | `StagedPlansReferencesTest` (3) — asserts the override exists on both logical plans and that ColumnPruning's subset guard short-circuits. |
| **Lance's per-stage dataset open is efficient.** | `LanceProbeValidationTest` exercises probe across batches; `LanceProbe.close()` is called in the `try`/`finally` of every stage's `doExecute`. |
| **Fragment grouping doesn't produce duplicates.** | `IndexedNearestJoinFragmentGroupingTest` — oracle equivalence with G=4 and G=8, + skew-balanced variant. |
| **IVF-PQ integration produces real recall.** | `IndexedNearestJoinIvfPqRecallTest` (3) — builds a real IVF-PQ index on a Lance dataset, recall@10 = 0.73 at defaults, **1.00 with `refineFactor=8`**. |

## Files that LOOK large but are mechanical

- `InterStageShuffleReproTest.scala` + `InterStageShuffleWithLanceReproTest.scala` — ~700 lines
  combined. They're the isolation tests from the ColumnPruning investigation.
  Each has ~5-10 lines of actual assertion logic; the rest is test data
  setup (synthetic row generators, Lance write helpers, JIT-stress loops).
  Kept as regression coverage.
- `IndexedNearestJoinBenchmark.scala` — the performance benchmark. Long
  scaladoc comments explain design trade-offs; actual code is the timing
  harness + 5 config runs. Not a regression test. See
  [`UPSTREAM_DELIVERY_PLAN.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/UPSTREAM_DELIVERY_PLAN.md) § "Out of scope"
  for why benchmarks aren't shipped upstream.

## Out of scope for upstream review

Per [`UPSTREAM_DELIVERY_PLAN.md`](https://github.com/sezruby/lance-spark/blob/knn-phase0/lance-spark-knn_2.12/UPSTREAM_DELIVERY_PLAN.md), these files
will NOT ship to `lance-format/lance-spark` (they're kept on the fork):

- All of `benchmark/` (6 files)
- The `lance-spark-knn-4.2_2.13/` module (deferred until Spark 4.2 releases)
- `lance-spark-knn_2.12/pom.xml`'s Linux-x86_64-only shade filter
  (deployment-specific to some managed-Spark distributions' ingress timeout on
  volume uploads)
- `BENCHMARK_RESULTS.md` § "Cluster benchmarks — OSS Spark 3.5 on Kubernetes"

If you're reviewing with an eye toward upstream merge, skip those.

## Commit-by-commit alternative

The branch is organized as 9 feature-boundary commits matching the
upstream delivery plan — if you prefer commit-based review, walk the
`git log` in order and read each commit's body for the "why":

1. `feat(knn): Phase 0 foundation — LanceProbe primitive + metric types`
2. `feat(knn): staged RDD pipeline + IndexedNearestJoin.apply + bounded TopKHeap`
3. `feat(knn): Phase 1.5 — fragment-grouped probing for multi-task parallelism`
4. `feat(knn): 3-exec Catalyst-visible staged plan with AQE-visible merge shuffle`
   — **the heaviest commit; read this one closely.**
5. `feat(knn): df.kNearestJoin DataFrame extension method`
6. `feat(knn): Phase 3 hardening — refineFactor, prefilter pushdown, IVF-PQ recall`
7. `feat(knn): Spark 4.2 SQL integration — IndexedNearestByJoinRule`
8. `test(knn-bench): benchmark suite — synthetic, Wikipedia perf, SIFT/Cohere recall, SQL`
9. `docs(knn): design, impl plan, reviewer guide, ANN proposal, benchmark results`

Commit #4 is the subtle one — it introduces the 3-exec staged split
with the `references = child.outputSet` override that prevents
ColumnPruning from inserting `Project(Nil)` wrappers. The commit body
has the full root-cause narrative; also see `IMPL_PLAN.md` "3-exec
staged split — root cause and fix".

## Questions, sharp edges, and known limitations

- **`probeParallelism > 1` is a tradeoff**, not a pure win. Phase 1.5
  fragment grouping pays a 2-shuffle cost for `|R|/G` per-task work. On
  local-laptop runs it's net negative. On true distributed clusters with
  `probeParallelism == numFragments` it's a ~1.7× win. Documented in
  `BENCHMARK_RESULTS.md` § "Surprise: Phase 1.5 doesn't help at
  local-laptop scale" and § "Cluster benchmarks".
- **`probeParallelism > 1` uses an RDD-level shuffle** (`partitionBy`)
  inside `runWithFragmentGroups` that is NOT AQE-visible. Fixing would
  require a different shape that fits Catalyst's
  `requiredChildDistribution` model. Documented in `IMPL_PLAN.md` as
  follow-up.
- **Cost gate** — the Phase 2 SQL rule is opt-in via
  `spark.lance.knn.indexedNearestByJoin.enabled = true`. Production-grade
  delivery needs a cost-based heuristic (on `|R|`, selectivity) to decide
  automatically. Documented as TODO.
