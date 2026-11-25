# Design proposal: indexed `APPROX NEAREST` joins over Lance tables

## Summary

Spark 4.2 adds a nearest-neighbor join to SQL ([SPARK-56395](https://issues.apache.org/jira/browse/SPARK-56395)):

```sql
SELECT q.id, d.id
FROM queries q JOIN docs d
  APPROX NEAREST 10 BY DISTANCE vector_l2_distance(q.vec, d.vec)
```

Spark runs it as a cross join plus a top-k aggregate per left row (`RewriteNearestByJoin`). Every query compares against every row, so the cost grows with |queries| × |table|.

When the right side is a Lance table with a vector index on the ranked column, lance-spark could run the same join with Lance's batched index search. That search reads each IVF partition once for all the queries that probe it.

This proposal adds that execution path. It does not add new SQL: it uses Spark's own `APPROX NEAREST`, keeps `EXACT NEAREST` and every unsupported case on Spark's plan, and is opt-in.

## Sponsorship and collaboration

Looking for a maintainer sponsor. Prior prototypes are #796 and #797 (tracking issue #798). They were opened before this proposal, and they stay on hold until the approach is agreed here.

## 1. Motivation

Batch nearest-neighbor lookups are a join: deduplication, candidate generation for recommendation, retrieval evaluation, and batch RAG all match many query vectors against a large table. Spark users have two options today:
1. **Brute force:** a cross join plus a per-query top-k. In Spark 4.2 that is what `NEAREST BY` runs; on earlier versions users write it by hand. The cost is O(|queries| × |table|) distance computations.
2. **`VECTOR_SEARCH` per query:** one search per query vector. The table functions take a single query vector today.

Measured on 10M × 1024-d vectors (§6.1): the best brute-force plan takes 255 s for 1,000 queries on 30 cores. Lance's batched index search takes 0.61 s on one 32-core process, at recall@10 0.937.

## 2. How it works today

### 2.1 Spark's `NEAREST BY` plan
`RewriteNearestByJoin` replaces `NearestByJoin` with:
- a left outer join without a condition (a cartesian product);
- an `Aggregate` keyed by a per-left-row id, computing `max_by` or `min_by(struct(right.*), expr, k)`;
- a `Generate(inline)`.

`APPROX` and `EXACT` share this plan today.

### 2.2 Lance-side building blocks (already merged in lance)
- Batched IVF search: one search call takes many query vectors and reads each probed partition once for all queries that probe it ([lance#7640](https://github.com/lance-format/lance/pull/7640)).
- Parallel scoring across queries ([lance#10480](https://github.com/lance-format/lance/pull/10480)).
- Adaptive probe bounds in the batched path ([lance#10443](https://github.com/lance-format/lance/pull/10443)).
- Java binding `Query.Builder.setKeys(float[][])` ([lance#9475](https://github.com/lance-format/lance/pull/9475)).
- Namespace `multi_vector` queries running as batch search ([lance#9618](https://github.com/lance-format/lance/pull/9618)).

These ship in Lance 15 (currently beta); lance-spark pins 14.0.3.

## 3. Proposed design

### 3.1 SQL surface
None. The rule only matches Spark 4.2's `NearestByJoin` with `approx = true`.

### 3.2 When the rule applies
Otherwise the plan is left unchanged:
- the right side resolves to a Lance relation (projection and deterministic filters allowed) whose vector column has a vector index;
- the ranking expression is one of Spark's vector functions applied to a left expression and that column, and it matches the index metric:

  | Ranking expression | Index metric |
  |---|---|
  | `BY DISTANCE vector_l2_distance(...)` | L2 |
  | `BY SIMILARITY vector_cosine_similarity(...)` | cosine |
  | `BY SIMILARITY vector_inner_product(...)` | dot |
- the join type is `INNER` or `LEFT OUTER`;
- `k` is at most a configured bound;
- the right index fits on one executor (size check at plan time, below).

### 3.3 Execution
1. Repartition the left side into one partition per executor; it is not broadcast, so the number of queries is unbounded. The operator reads the executor count at planning time (`SparkStatusTracker`), and a config can override it. Tasks keep the cluster's `spark.task.cpus`; nothing in the cluster configuration changes.
2. Each task searches its left partition with batched calls (`setKeys`, up to a configured number of queries per call to bound memory) and gets back (query index, row id, distance). Lance parallelizes inside each call across the executor's cores. Every executor opens the whole right index once and keeps it cached.
3. Fetch the projected right columns for the matched row ids (`takeRows`), then join them back to the left rows.

No shuffle of the right table, and no per-query Spark jobs.

One partition per executor works because Lance's compute pool belongs to the executor process, not to the task: a task that Spark counts as one core still searches on all of the executor's cores, and one call per executor reads each probed partition once for all of that executor's queries. Spark's scheduler hands out at most one task per executor in each pass over its resource offers, so when the executors are free the tasks land one per executor. With 10 executors, 10 ordinary tasks matched tasks pinned one per executor from 4,000 queries up (§6.2). More partitions split the batch into smaller calls (200 tasks were 1.8-2.1× slower from 4,000 queries up), and fewer leave executors idle (1 task was 7-9× slower).

Lance's compute pool must cover the task's cores. Lance reads its thread settings from environment variables once per process, so the operator cannot resize the pool itself. Since [lance#11425](https://github.com/lance-format/lance/pull/11425) (merged for Lance 15), Lance holds back `min(2, cores / 4)` cores per process for I/O by default, so a 3-core executor searches on all 3 cores. With the earlier flat reservation of 2 (`LANCE_IO_CORE_RESERVATION`), a 3-core executor searched on one thread, and the same join was 2.2-2.4× slower (§6.2).

### 3.4 Design decisions and rejected alternatives
Numbers in §6.
- **`spark.task.cpus` equal to the executor's cores:** it is an application-wide setting, so the operator cannot set it for one query, and one partition per executor gives the same result without it.
- **A stage-level task resource profile** (`cpus = spark.executor.cores` for the search stage only): with dynamic allocation off, the stage shares the existing executors; with it on, Spark acquires new executors for the profile, and each would load the index cold. Not measured, and not needed given the result above.
- **Pinning one task per executor** (preferred locations naming executor ids): needs Spark-internal APIs, and plain tasks with one partition per executor matched it.
- **A per-executor lock around the search** (`synchronized`): no gain; it removes overlap between tasks without enlarging their batches.
- **A per-executor combiner** that merges concurrent tasks' queries into one call: it needs a wait window to catch all of an executor's tasks, came closest to pinned tasks on its best runs, and had the noisiest medians. One partition per executor gives every call the executor's whole batch without it.
- **Many small tasks:** slower; each call gets a smaller batch (200 tasks: 1.8-2.1× slower than one per executor from 4,000 queries up).
- **One task per index segment** (each executor owns a slice of the index and sees every query): the slowest shape measured. Every executor processes every query, and per-(query, segment) costs add up.
- **Many index segments:** 10 segments with shared centroids cost 2.6-2.9× more per query than one segment on the same machine. This path should recommend single-segment indexes.
- **Distributed centroid training:** not needed; training on a sample takes seconds.

### 3.5 Beyond one index copy per executor (future)
Splitting queries across executors is fast as long as the index fits on one executor (§6.2). Every executor still loads the whole index, so the index must fit on one executor. Sending each query only to the executors that own its probed IVF partitions would pay each partition's work once and cache each partition on one executor. That needs a partition-restricted batched search in Lance's Java API, so it is out of scope for the MVP.

## 4. Who is affected
- Spark 4.2 users running `APPROX NEAREST` joins whose right side is a Lance table with a vector index, behind a config flag.
- No change for other plans.

## 5. Risks and mitigations
- **Approximate results:** recall depends on index type and probe count, as documented for `APPROX`. Expose `nprobes` and refine settings through table options or session config.
- **Filters on the right side:** apply them as Lance prefilters (true filtered top-k); otherwise fall back.
- **Index size (the scaling limit):** every executor holds the whole index, so adding executors does not raise the ceiling.
  - At plan time, estimate the per-executor footprint (a fixed baseline plus bytes per row for the index type and dimension) and compare it with the executor's memory.
  - Above the budget, fall back to Spark's plan (or fail or warn, by config).
  - Measured: a 5-bit IVF_RQ index on 1024-d vectors takes 0.66 GB per 1M rows, so a 16 GB executor holds roughly 20M rows.
  - Measured past the limit: with each executor's Lance index cache at 2 GiB for a 6.6 GB index, every join took ~30 s regardless of query count (1,000-16,000 queries), vs 1-7 s when the index fit (both with the earlier flat 2-core I/O reservation). Partitions are reloaded from object storage on every job.
  - Larger indexes need the partition-routed execution in §3.5.
- **Version requirements:** needs Lance 15 (batched Java API) and Spark 4.2 (`NearestByJoin`). Ship as an opt-in module.
- **Busy or changing executors:** if executors are busy with other stages, or dynamic allocation changes their number after planning, two tasks can land on one executor. Results stay correct; that executor runs two smaller searches.
- **Unindexed fragments:** Lance's search already combines indexed and flat results for fragments the index does not cover.
- **Multi-segment indexes:** they are slower (§6.3). Distributed builds should merge segments before committing. For IVF_RQ, every segment must then be built with the same rotation model (`RQBuildParams.Builder.setModel`, Lance ≥ 15.0.0-beta.9), or the merge is rejected.

## 6. Measurements

**Setup:**
- Data: 10M rows of 1024-d Cohere Wikipedia embeddings in Lance (10 fragments) on Azure Blob Storage.
- Index: IVF_RQ 5-bit, 256 partitions, L2. Search: k = 10, nprobes = 16, no refine.
- Cluster: Spark 3.5 on Kubernetes, executors with 3 cores each.
- Single-process runs: one 32-core JVM making one Lance call (`Query.Builder.setKeys`).
- Times are medians of 3-5 runs after a warmup. Recall@10 is measured against an exact flat scan on 100 queries; it is identical across execution shapes within a run.

### 6.1 Brute force vs. indexed search
Brute force ran on 30 executor cores and is exact. Its L2 is a tight loop over primitive arrays, which is more generous than a SQL expression. Reading the 10M vectors alone takes 4.9 s.

| Plan, all 10M rows | 100 queries | 1,000 queries |
|---|---|---|
| Cross product + per-partition top-k + merge per query (≈ how Spark 4.2 runs `NEAREST BY`), 30 cores | 32.4 s | 255.1 s |
| **Indexed batched search, one 32-core process** (recall@10 0.937) | **0.23 s** | **0.61 s** |
| Speedup | ~140× | ~420× |

On one fragment (1M rows), the window-based plan users write on Spark 3.5-4.1 (cross product + `row_number()`) took 17.0 s for 100 queries, vs 5.7 s for the per-partition top-k plan. It shuffles every pair, so it was not run at 10M.

### 6.2 Execution shapes for the indexed search
Single-segment index; recall@10 0.928 for every shape. Every process computes on all of its cores (`LANCE_IO_CORE_RESERVATION=0`, which is what Lance's default now gives a 3-core executor). Spark runs use 10 executors × 3 cores and keep `spark.task.cpus=1`; "pinned" tasks name one executor each as their preferred location. Each table comes from one run.

**Partition count of the left side:**

| Queries | 1 task | 10 tasks (one per executor) | 30 tasks | 90 tasks | 200 tasks | 10 pinned tasks |
|---|---|---|---|---|---|---|
| 1,000 | 31.4 s | 0.63 s | 0.59 s | 0.71 s | 0.73 s | 0.40 s |
| 4,000 | 5.90 s | 0.84 s | 1.20 s | 1.60 s | 1.72 s | 0.85 s |
| 16,000 | 22.8 s | 2.55 s | 2.70 s | 3.72 s | 4.53 s | 2.54 s |

- **One partition per executor matches pinned tasks from 4,000 queries up** (0.84 vs 0.85 s, 2.55 vs 2.54 s). At 1,000 queries its median was 0.63 s vs 0.40 s, and 10 and 30 tasks were within noise of each other.
- **More partitions cost 1.06-2.05× from 4,000 queries up:** each call carries a smaller batch.
- **One partition leaves nine executors idle** and was 7-9× slower from 4,000 queries up. Three of its five 1,000-query runs took 31-35 s instead of 1.9 s. It ran first, before the other configurations had loaded the index on every executor, so those runs most likely paid a cold index load on a new executor.

**Against one process:**

| Queries | One process, 32 cores | 30 tasks | 10 pinned tasks |
|---|---|---|---|
| 1,000 | 0.48 s | 0.80 s | 0.46 s |
| 4,000 | 1.23 s | 1.29 s | 0.89 s |
| 16,000 | 3.73 s | 2.83 s | 2.86 s |

- **Spark on 30 executor cores is 1.3-1.4× faster than one 32-core process** from 4,000 queries up, and ties it at 1,000.
- **Lance's I/O reservation matters on small executors.** With Lance's earlier default (2 cores held back for I/O per process, before lance#11425), the same 1-task-per-executor runs took 1.00 / 2.08 / 6.98 s, 2.2-2.4× slower.
- **Every executor loads the whole index on first use.** A cold start took 54 s with 20 executors reading it at once, vs 4.5-6.5 s for one process.

**Per-executor coordination of 30 tasks:**

| Queries | 30 tasks | + lock | + combiner | + combiner, 50 ms wait | 10 pinned tasks |
|---|---|---|---|---|---|
| 1,000 | 0.68 s | 1.40 s | 0.87 s | 1.22 s | 0.59 s |
| 4,000 | 1.19 s | 1.28 s | 1.16 s | 1.38 s | 1.02 s |
| 16,000 | 3.00 s | 3.11 s | 2.83 s | 3.21 s | 2.61 s |

- The lock serializes each executor's three searches without enlarging them. The combiner without a wait merged only two of three tasks per executor; with a 50 ms wait it merged all three but had the noisiest medians (best runs 0.62 / 0.90 / 2.72 s).
- Neither beats plain tasks on medians. One partition per executor reaches pinned speed without either.

### 6.3 Index layout
Same rows and queries in one 32-core process, in two separate runs; 10 segments built from shared IVF centroids, one per fragment.

| Queries | 1 segment | 10 segments |
|---|---|---|
| 1,000 | 0.40 s | 1.08 s |
| 4,000 | 1.22 s | 3.14 s |
| 16,000 | 3.42 s | 10.08 s |

- Ten segments cost 2.6-2.9× more per query, even though the scanned rows are the same.
- Giving each executor one segment (so every executor sees every query) was the slowest shape measured: 1.72 / 6.46 / 24.4 s on 10 × 3 cores, vs 0.95 / 2.48 / 9.56 s for one task per executor searching the whole 10-segment index.

## 7. Incremental plan
1. Bump lance-spark to Lance 15 once a stable release is out.
2. Batched `VECTOR_SEARCH` (multiple query vectors, #873).
3. MVP rule and operator: `INNER`, L2, left side repartitioned to one partition per executor, plan-time index-size check, no right-side filters, Spark 4.2 module.
4. `LEFT OUTER`, cosine/dot, right-side filters as prefilters.
5. Docs: index layout guidance (single segment), cache sizing.

## 8. Open questions for sponsors and reviewers
- Module placement: a separate Spark 4.2 module, or inside the 4.2 build of the base module?
- Config names and defaults (enable flag, index-size check mode and safety fraction, queries per search call, max k, partition count override).
- Partition count with dynamic allocation: the executor count read at planning time can change before the stage runs. Is the planning-time count good enough, or should the operator round up to the configured maximum?
- Should the DataFrame `nearestBy` API (Spark 4.2 follow-up) be covered in the MVP?
- Where should `nprobes` and refine settings live (table options, session conf, or SQL hints)?
