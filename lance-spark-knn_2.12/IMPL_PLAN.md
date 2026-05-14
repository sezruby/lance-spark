# Indexed Nearest-By-Join — Implementation Plan

This module adds an indexed approximate-nearest-neighbor (ANN) join strategy on top of Lance's vector indexes, exposed through Spark. It complements Spark's built-in `NearestByJoin` (Spark 4.x), which today only has a brute-force cross-product rewrite.

## Goal

For SQL like:

```sql
SELECT * FROM queries q
LEFT OUTER JOIN documents d
APPROX NEAREST 10 BY DISTANCE l2_distance(q.vec, d.vec)
```

— or its Scala/Python DataFrame equivalent — execute via per-fragment Lance index probes plus a Spark-side merge, instead of an `O(|L| * |R|)` cross-product.

## Why this works on Lance specifically

- Lance's vector indexes (IVF-PQ, HNSW) are **fragment-local**. Each fragment carries its own index files; per-fragment probes are independent.
- Lance's Java API exposes single-vector nearest search via `org.lance.ipc.Query` + `LanceScanner.create(dataset, ScanOptions, allocator)`. lance-spark already uses this for single-query reads (`LanceFragmentScanner.create`).
- Lance row IDs (`_rowid` virtual column) make late materialization cheap: the probe phase emits row references, the materialize phase fetches full rows by ID. (We initially used `_rowaddr` but switched to `_rowid` because the indexed nearest-search path materializes `_rowid` only — see DESIGN.md "Why `_rowid` not `_rowaddr`".)
- Snapshot pinning (Lance versioning) gives consistent results across distributed tasks.

## Three-phase distributed design

```
ProbeExec (one task per fragment-group)
  ├── opens Lance dataset, restricted to assigned fragments
  ├── per left row: maintains local top-K' heap across owned fragments (map-side combine)
  └── emits (left_id, [row_addr, score])      ← refs only, no payload
       │
  Exchange (hash by left_id)                   ← lightweight: O(|L| × N × K)
       │
       ▼
MergeExec
  ├── K-way merge across N task contributions per left_id
  └── emits (left_id, [chosen_row_addr, score])
       │
       ▼
MaterializeExec
  ├── for each chosen row_addr: fetch right row from Lance (random access)
  └── emits full join rows
```

The merge is across at most `N = numTasks` contributions per left row (not across `F` fragments) because each task does a map-side combine — this is the difference between a manageable shuffle (~hundreds of GB at scale) and a catastrophic one (~tens of TB).

## Rollout phases

### Phase 0 — Pure Scala API (this module's first deliverable)

A `IndexedNearestJoin.apply(left, lanceTablePath, ...)` Scala function that takes a left DataFrame and produces a result DataFrame, with no Catalyst integration whatsoever. Pure RDD primitives (`mapPartitions`, `reduceByKey`) on top of a `LanceProbe` JVM helper that wraps the Lance Java vector-search API.

This phase proves:
- Per-fragment Lance probes work from JVM tasks
- The shuffle volume math holds up
- Recall vs. brute-force is acceptable
- The deferred-materialization pattern delivers the expected payload-size win

Phase 0 ships **before** any Spark version dependency is needed (works on any Spark version lance-spark supports), making benchmarks possible without committing to Spark 4.x-only code paths.

### Phase 1 — Refactor execution into Spark physical operators

Same logic, but as proper `SparkPlan` subclasses (`IndexedNearestProbeExec`, `IndexedNearestMergeExec`, `IndexedNearestMaterializeExec`). The Phase 0 API becomes a thin wrapper that constructs the exec triple. Pure refactor — no new functionality, no new tests beyond plan-shape tests.

### Phase 2 — Catalyst integration (Spark 4.x only)

A `postHocResolutionRule` that pattern-matches `NearestByJoin(approx = true, ...)` over a qualifying Lance scan with a vector index, replacing it with an `IndexedNearestByJoin` logical node. A custom strategy maps that to the physical execs from Phase 1. Wired via `SparkSessionExtensions`.

Critical detail: Spark's `RewriteNearestByJoin` runs in the optimizer's first batch (`FinishAnalysis`). `injectOptimizerRule` would fire **after** the rewrite and miss the `NearestByJoin`. Only `injectPostHocResolutionRule` (which runs before the optimizer entirely) gives us access to the unrewritten operator.

After Phase 2, users get the indexed path automatically from `APPROX NEAREST` SQL queries — no API change.

### Phase 3 — Hardening

Cost gate, recall tuning (`overfetch`, `refineFactor` re-rank pass), filter pushdown into the probe (`prefilter`), skew handling, full docs, cross-version build matrix.

## Open questions / risks

These are tracked as Phase 0 validation tasks:

1. **Per-task Lance dataset open cost.** Each probe task opens a Lance dataset. If expensive, need a per-executor singleton cache.
2. **`Query` per-call overhead.** `Query` is constructed per left row. If hidden setup cost exists, batch internally or push for batched Lance API.
3. **`_rowid` filter performance.** Materialization relies on `WHERE _rowid IN (...)` resolving to point fetches inside Lance. Confirm this isn't a scan-with-post-filter. (Original Phase 0 question — switched from `_rowaddr` to `_rowid` for indexed-path compatibility; same point-fetch semantics.)
4. **Concurrent fragment scans within a task.** Can we run multiple `LanceScanner` instances in parallel within one task, or is Lance single-threaded at the dataset level?
5. **`Query.queryParallelism` semantics.** May handle some intra-query parallelism for free.
6. **`Dataset.listIndexes()` availability and metadata.** Required for Phase 2 capability detection.
7. **Snapshot pinning across stages.** All three execs must read at the same Lance version. lance-spark's read options carry this; needs explicit test.

## What this module is NOT

- Not a brute-force fallback — that's `RewriteNearestByJoin` in Spark, kept for `EXACT` queries and unindexed cases.
- Not a re-implementation of Lance's index. We delegate every probe to Lance.
- Not a vector-DB-style serving layer. This is for batch joins inside Spark pipelines.

## Status

| Phase | Status |
|---|---|
| 0 — Scala API | Done (`IndexedNearestJoin.apply`, oracle test, LanceProbe primitive) |
| 1 — Staged RDD pipeline | Done (probe / merge / materialize stages, plan-shape test) |
| 1.5 — Fragment-grouping | Done (`probeParallelism` parameter; LanceFragments enumeration; oracle equivalence + 2-shuffle plan-shape test) |
| 2 — Catalyst integration | Done (`lance-spark-knn-4.2_2.13` module: rule, logical, physical, extension; 18 tests including SQL e2e against real Lance + Spark 4.2-SNAPSHOT, with prefilter pushdown coverage) |
| 3 — Hardening | Partially done — see "What's left" below |
| 3.x — Explicit physical operators (DataFrame API) | **Done.** Production path is now `LanceProbeExec → ShuffleExchangeExec → LanceMergeExec → LanceMaterializeExec`. `df.explain()` shows all four nodes under `AdaptiveSparkPlanExec`; with AQE on, `AQEShuffleRead coalesced` appears on the merge shuffle. An early development iteration hit reproducible SIGSEGV / AssertionError on `count()`-style consumers — misdiagnosed as a JVM-aarch64 bug; the real cause was Catalyst's `ColumnPruning` rule inserting `Project(Nil)` wrappers between the custom nodes when downstream consumers referenced no columns, which codegen'd to 0-field `UnsafeRow`s and crashed `ProbedLeftCodec.Decoder` in interpreter mode (AssertionError) / C2 mode (SIGSEGV). Fix: `LanceMergeLogicalPlan` and `LanceMaterializeLogicalPlan` override `lazy val references = child.outputSet`, which short-circuits `ColumnPruning`'s `!child.outputSet.subsetOf(references)` guard. 60 tests in lance-spark-knn_2.12. See "3-exec staged split — root cause and fix" below for details. |
| 3.x — `df.kNearestJoin` extension | Done (`LanceKnnImplicits._`; works on Spark 3.5 / 4.0 / 4.1 / 4.2+; URI auto-extracted from right DataFrame's analyzed plan; non-Lance right side fails fast with `IllegalArgumentException`) |
| Benchmarks | Done (608× DataFrame, 17.4× SQL — both oracle-validated) |

See `PHASE_PROGRESS.md` for the resume-without-context overview, file inventory, and the
substantive limitations carried forward from each phase.

## What's left (Phase 3.x)

Phase 3 done so far: `refineFactor` /
`ef` recall knobs, row-count-aware fragment grouping for skew (`balanceFragmentsByRowCount`).

Phase 3.x — outstanding work:

| Item | Module | Notes |
|---|---|---|
| Cost gate replaces opt-in flag | `lance-spark-knn-4.2_2.13` | Heuristic deciding indexed vs. brute-force based on `\|R\|` cardinality and right-side selectivity. Until then `spark.lance.knn.indexedNearestByJoin.enabled` is the gate. |
| ~~`prefilter` pushdown into Lance probe~~ | DONE | Rule detects `Filter(cond, lance)` (and `Project(<passthrough>, Filter(...))` for `SELECT *` shape), translates the predicate to a Lance SQL filter string, and threads it through `LanceProbeStage.Conf.prefilter` → `ScanOptions.filter()`. Translator handles bare `attr <op> literal` for `=`, `!=`, `<`, `<=`, `>`, `>=`, plus `IN`, `IS [NOT] NULL`, and `AND`/`OR`/`NOT`. Anything else (UDFs, computed expressions, predicates touching the LEFT input) → rule REFUSES the rewrite. No partial pushdown. |
| ~~Real recall test against IVF-PQ-indexed dataset~~ | DONE | `IndexedNearestJoinIvfPqRecallTest` builds an IVF-PQ index via Lance Java's `Dataset.createIndex` and measures recall@K. With 1024 rows × dim 32 × 4 IVF partitions: recall@10 = 0.73 at defaults, **1.00 with `refineFactor = 8`** (exact-distance re-rank recovers all true neighbors). Surfaced and fixed a real bug in the process — Lance's indexed scan materializes `_rowid` not `_rowaddr`, so the whole pipeline switched to `_rowid` (works on both paths). |
| ~~`LanceProbe.vectorColumn` cleanup~~ | DONE | `vectorColumn` moved from constructor to per-call `probe()` arg. Materialize stage no longer constructs the probe with a placeholder. |
| ~~Filter pushdown's interaction with `prefilter = true`~~ | RESOLVED | We always set `prefilter = true` and call `ScanOptions.filter(sql)` from `LanceProbe.probe`. Lance applies the predicate before the index lookup, so the top-K is computed only over matching rows — confirmed by the e2e WHERE-pushdown test against a brute-force-on-filtered oracle. |
| Spark version matrix for the connector | build infra | The Phase 2 module pins to 4.2-SNAPSHOT. Once `NearestByJoin` lands in a release, re-pin and add 4.2_2.13 module path. Phase 1.5 / Phase 0/1 work on any Spark 3.4+ via the existing connector modules. |
| ~~Real-backend e2e test for Phase 2~~ | DONE | `lance-spark-knn-4.2_2.13/src/test/.../IndexedNearestByJoinE2ETest.scala`. Recompile `lance-spark-4.1_2.13` against `4.2.0-SNAPSHOT` (its source compiles cleanly against 4.2; runtime API is compatible) and use it as the test-scope Lance reader. Three test cases: rule-on goes through the 3-exec staged chain and matches oracle; WHERE-pushdown round-trips the prefilter and matches the filtered oracle; rule-off falls through to Spark's `RewriteNearestByJoin` and still matches oracle. |
| ~~Cross-version DataFrame API parity~~ | DONE (compile+test) / TODO (CI matrix) | Module compiles and tests pass against Spark **3.5 AND 4.0**. Single-source validated: flip `spark.version=${spark40.version}` + `arrow.version=${arrow18.version}` + swap test runtime `lance-spark-4.0_2.13`, 41/41 tests pass. One source fix was required — `LanceKnnDatasetBridge` used `org.apache.spark.sql.Dataset.ofRows` which moved to `org.apache.spark.sql.classic.Dataset.ofRows` in Spark 4.0. Replaced with a reflection-based lookup that tries both packages; one cache-miss per Spark session. CI matrix against 3.4 / 3.5 / 4.0 / 4.1 still TODO. End-to-end cluster validation done on OSS Spark 3.5.4. |
| ~~Production-shape benchmark (real embeddings)~~ | DONE | `WikipediaKnnPerfBenchmark` uses CohereLabs `wikipedia-2023-11-embed-multilingual-v3` (dim=1024). On 8 × 4c/16g OSS Spark 3.5 cluster: indexed path is **100-200× faster than Spark crossJoin** at small scale (7-iter median 160×; noop sink + oracle-verified). Speedup grows with dim (128 → 1024) because Lance's native SIMD advantage widens vs Spark's JVM UDF overhead. `CohereWikiRecallBenchmark` complements with IVF-FLAT recall on the same corpus: 95% recall at nprobes=16, 99% at nprobes=64, 10-16 ms/query. Numbers in `BENCHMARK_RESULTS.md` § "Cluster benchmarks". |
| ~~SIFT1M ANN-benchmark validation~~ | DONE | `SiftRecallBenchmark` against the canonical `ftp.irisa.fr/.../sift.tar.gz` corpus. OSS Spark 3.5 cluster, 1M × dim 128, IVF-FLAT 256 partitions: recall@10 = 0.98 at nprobes=16, 1.00 at nprobes=64. Within noise of published FAISS numbers. |
| ~~Sustained concurrent load soak~~ | DONE (harness) / PARTIAL (data) | `IndexedNearestJoinSoakTest` runs N concurrent queries for M minutes while sampling driver heap + GC metrics. 10-min smoke on OSS Spark 3.5 cluster: 492 queries at 8 concurrency, 0 failures, heap stable 163–266 MB (no drift). Harness has a known bookkeeping bug (post-deadline queued queries are counted as failures after pool drain races `spark.stop()`); production qualification at 2–4 hours is deferred. |
| ~~Real-world-embedding benchmark~~ | DONE | `ClusteredEmbeddings.generate` produces clustered Gaussian-mixture vectors on the unit sphere — the geometry of typical sentence-transformer / image-feature embeddings. Used in `IndexedNearestJoinIvfPqRecallTest.testClusteredEmbeddingsRecallSurvives`, which measures recall@K on production-shaped data and asserts it clears 0.5 at default IVF-PQ settings. Both uniform and clustered recall numbers are printed for comparison; we do NOT assert `clustered >= uniform` because Lance's IVF k-means init is non-deterministic across JVM sessions and the run-to-run noise on a 1024-row dataset routinely exceeds the structural advantage. A reliable comparative would need much larger N or seed-averaging. |
| ~~AQE-aware partition sizing for the merge stage~~ | **DONE** | `LanceMergeExec` declares `requiredChildDistribution = ClusteredDistribution(leftIdAttr)`; `EnsureRequirements` auto-inserts a `ShuffleExchangeExec` between probe and merge. With AQE enabled, Catalyst wraps the stage under `AdaptiveSparkPlanExec`, applies `CoalesceShufflePartitions` / `OptimizeSkewJoin` / `OptimizeShuffleWithLocalRead`, and the final plan visibly shows `AQEShuffleRead coalesced` on the merge-side shuffle. Verified by `IndexedNearestJoinAqeVisibilityTest`. |
| Per-task `LanceProbe` reuse / connection pooling | `lance-spark-knn_2.12` | Currently each Spark task opens its own dataset. For very small partitions this dominates cost. A per-executor singleton cache could amortize. |
| Skew handling for left side too | `lance-spark-knn_2.12` | Phase 1.5 / Phase 3 balance the right side's fragment groups but the left RDD's natural partitioning can still be skewed. Repartition by `leftId` before probe is the obvious next move. |

## 3-exec staged split — root cause and fix

The three-operator staged split (`LanceProbeExec → ShuffleExchangeExec → LanceMergeExec → LanceMaterializeExec`) had a debugging detour during development — reproducible `AssertionError: index (0) should < 0` / SIGSEGV in `UnsafeRow.getLong` on JVM-aarch64 was initially blamed on the JVM. That was wrong — this section preserves the investigation for anyone who hits similar "it looks like a JVM bug but it's Catalyst" symptoms in future work.

**Initial JVM-aarch64 diagnosis was wrong.** The crash reproduces on JVM-aarch64 but it isn't a JVM bug. A multi-step isolation found it:

1. `InterStageShuffleReproTest` — synthetic rows at the staged codec's schema, through `repartition(_leftId)` + 100-iteration JIT-stress. **Passes.** Rules out Spark's UnsafeRow shuffle + our schema.
2. `InterStageShuffleWithLanceReproTest` — same but with rows sourced from a real Lance scan. **Passes.** Rules out the Lance→Spark boundary.
3. `StagedExecDirectDriveReproTest` — directly drives the staged execs at tiny scale (4 left × 8 right) via `count()`. **Crashes deterministically on first invocation.** Not a JIT issue at all.

Diagnostic instrumentation on `LanceMaterializeExec.doExecute` caught a 0-field `UnsafeRow` arriving from `child.execute()`. `LanceMergeExec` emits correctly-shaped 4-field rows; something between them truncates to 0 fields. Dumping `df.queryExecution.executedPlan` for the `count()` case showed:

```
*(2) HashAggregate(partial_count(1))
+- *(2) Project                          ← 0-column projection!
   +- LanceMaterialize
      +- *(1) Project                    ← another 0-column projection!
         +- LanceMerge
            +- Exchange hashpartitioning(_leftId)
               +- LanceProbe
```

**Root cause:** Catalyst's `ColumnPruning` optimizer rule. Its `Aggregate(child)` guard is `!child.outputSet.subsetOf(a.references)`. For `count(*)`, `Aggregate.references` is empty. The custom logical plans (`LanceMergeLogicalPlan`, `LanceMaterializeLogicalPlan`) inherited `references` from `QueryPlan`'s default — empty for pass-through nodes. That made `child.outputSet.subsetOf(references)` false, and `prunedChild` inserted `Project(Nil)` wrappers. Spark's codegen'd `ProjectExec(Nil)` emits 0-field `UnsafeRow`s. `ProbedLeftCodec.Decoder.decode` reads `ir.getLong(0)` on those rows — in interpreter mode → `AssertionError`; in C2-compiled code → the assertion is elided and the read hits unmapped memory → SIGSEGV. Hence the misleading "JVM-aarch64 bug" blame on the revert.

**Fix.** `LanceMergeLogicalPlan` and `LanceMaterializeLogicalPlan` override `lazy val references = child.outputSet`. This makes `child.outputSet.subsetOf(references)` trivially true (equality ⇒ subset), short-circuiting `ColumnPruning`'s guard. No `Project(Nil)` ever gets inserted between the custom nodes. `StagedPlansReferencesTest` pins this invariant structurally.

**Verification.**

- Unit-level: `StagedPlansReferencesTest` (3 tests) pins the override on both logical plans and explicitly checks `ColumnPruning`'s subset guard.
- Plan-shape: `IndexedNearestJoinAqeVisibilityTest` (5 tests) asserts `ShuffleExchangeExec hashpartitioning(_leftId)` is in the executed plan (AQE on and off), AQE wraps with `AdaptiveSparkPlanExec`, all three custom execs appear in the tree, no `!` missingInput prefix.
- Consumer-shape: `IndexedNearestJoinConsumerShapeTest` (4 tests) — `count()`, `agg(count("*"))`, `select(lit(1))`, `collect()` all succeed. These are the exact shapes that crashed the reverted code.
- Correctness: `IndexedNearestJoinCorrectnessTest` — recall = 1.0 against brute-force oracle at 1000 right × 100 left × dim 16 × K 10.
- Durability: `IndexedNearestJoinJitStressTest` — crossJoin JIT warmup + 20 iterations of `collect()` and `count()` at 10K right × 100 left × dim 128 × K 10. No SIGSEGV.

All 60 tests pass in `lance-spark-knn_2.12`.

**AQE now engages on the merge shuffle** — `AQEShuffleRead coalesced` is visible on the post-merge plan when AQE is enabled. This was the whole point of the staged split, and it now works.

**Caveat: plan-level integration, not exec-level.** `LanceMaterializeExec` doesn't declare `requiredChildDistribution`, so the Exchange still lives as a child of `LanceMergeExec`, not inside a larger Catalyst-optimized tree. The fragment-grouped probe path's internal `partitionBy` shuffle is still AQE-invisible.
