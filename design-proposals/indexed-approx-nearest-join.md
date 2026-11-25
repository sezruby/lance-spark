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
1. Keep the left side partitioned; it is not broadcast, so the number of queries is unbounded. Each task runs with `spark.task.cpus` equal to the executor's cores, so it has the whole executor.
2. Each task searches its left partition with batched calls (`setKeys`, up to a configured number of queries per call to bound memory) and gets back (query index, row id, distance). Lance parallelizes inside each call across the executor's cores. Every executor opens the whole right index once and keeps it cached.
3. Fetch the projected right columns for the matched row ids (`takeRows`), then join them back to the left rows.

No shuffle of the right table, and no per-query Spark jobs.

Lance's compute pool must cover the task's cores. Lance reads its thread settings from environment variables once per process, so the operator cannot resize the pool itself. Since [lance#11425](https://github.com/lance-format/lance/pull/11425) (merged for Lance 15), Lance holds back `min(2, cores / 4)` cores per process for I/O by default, so a 3-core executor searches on all 3 cores. With the earlier flat reservation of 2 (`LANCE_IO_CORE_RESERVATION`), a 3-core executor searched on one thread, and the same join was 2.2-2.4× slower (§6.2).

### 3.4 Design decisions and rejected alternatives
All measured; numbers in §6.
- **Many small tasks:** slower than one task per executor (task launch, per-task dataset opens, stragglers).
- **One task per index segment** (each executor owns a slice of the index and sees every query): the slowest shape measured. Every executor processes every query, and per-(query, segment) costs add up.
- **Many index segments:** 10 segments with shared centroids cost 2.6-2.9× more per query than one segment on the same machine. This path should recommend single-segment indexes.
- **Distributed centroid training:** not needed; training on a sample takes seconds.

### 3.5 Beyond one task per executor (future)
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
Single-segment index; recall@10 0.928 for every shape. Every process computes on all of its cores (`LANCE_IO_CORE_RESERVATION=0`, which is what Lance's default now gives a 3-core executor).

| Queries | One process, 32 cores | Spark: 30 small tasks (10 × 3 cores) | Spark: 1 task per executor (10 × 3 cores) |
|---|---|---|---|
| 1,000 | 0.48 s | 0.80 s | 0.46 s |
| 4,000 | 1.23 s | 1.29 s | 0.89 s |
| 16,000 | 3.73 s | 2.83 s | 2.86 s |

- **One task per executor is the fastest shape.** With 30 executor cores it is 1.3-1.4× faster than one 32-core process from 4,000 queries up, and ties it at 1,000.
- **Lance's I/O reservation matters on small executors.** With Lance's earlier default (2 cores held back for I/O per process, before lance#11425), the same 1-task-per-executor runs took 1.00 / 2.08 / 6.98 s, 2.2-2.4× slower.
- **Every executor loads the whole index on first use.** A cold start took 54 s with 20 executors reading it at once, vs 4.5-6.5 s for one process.

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
3. MVP rule and operator: `INNER`, L2, partitioned left side, plan-time index-size check, no right-side filters, Spark 4.2 module.
4. `LEFT OUTER`, cosine/dot, right-side filters as prefilters.
5. Docs: index layout guidance (single segment), cache sizing.

## 8. Open questions for sponsors and reviewers
- Module placement: a separate Spark 4.2 module, or inside the 4.2 build of the base module?
- Config names and defaults (enable flag, index-size check mode and safety fraction, queries per search call, max k).
- Should the DataFrame `nearestBy` API (Spark 4.2 follow-up) be covered in the MVP?
- Where should `nprobes` and refine settings live (table options, session conf, or SQL hints)?
