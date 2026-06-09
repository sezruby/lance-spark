# Merge strategy benchmark — local results

Comparing two paths for `MERGE INTO` on a Lance table:

- **`sql_merge_into`**: today's path — `spark.sql("MERGE INTO ...")` via lance-spark's
  `SupportsRowLevelOperations` / `LancePositionDeltaOperation` (Spark Join + position-delta write).
- **`native_merge_insert`**: proposed path — direct call to lance-core's
  `Dataset.mergeInsert(MergeInsertParams, ArrowArrayStream)`. No Spark Join, no shuffle,
  no per-`_fragid` write stage.

Both paths produce equivalent results on the same input (1% of target rows merged: half
matched-update, half not-matched-insert). Tracking issue: `sezruby/lance-spark#8`.

## Setup

- Apple M5 Max, 18 cores (12P + 6E), 48 GB RAM. Spark `local[*]`.
- `lance-core` 6.0.0-beta.4 (matches lance-spark master pin).
- `max_rows_per_file = 1000` to force fragment count proportional to row count
  (synthesizes the high-fragment regime that real workloads hit).
- Schema: `(id INT NOT NULL, value INT, tag STRING)`. Source = 1% of target.
- Median of 3 timed iters after 1 warmup. Tables fresh-seeded between iters.

## Results

| Total rows | Fragments | SQL `MERGE INTO` median | Native `mergeInsert` median | Speedup |
|---:|---:|---:|---:|---:|
| 100,000 | 100 | 190 ms | 7 ms | **27.1×** |
| 500,000 | 500 | 191 ms | 14 ms | **13.6×** |
| 2,500,000 | 2500 | 309 ms | 50 ms | **6.2×** |

## Reading the trend

- Native always wins, by 6× to 27× at these scales.
- Speedup *narrows* as the table grows. At 100K/100, SQL has high constant
  Catalyst-planning + position-delta-write overhead amortized over little real work;
  native is essentially free. At 2.5M/2500, native scales linearly with target+source
  reading and writing, so the absolute gap closes.
- SQL wall-clock isn't strictly monotonic in fragment count at these scales — Spark's
  task scheduling at `local[*]` saturates around the 100-fragment point and per-stage
  overhead dominates more than per-fragment work. On a multi-executor cluster the SQL
  side would likely scale better; the native side is single-process so won't change.

## Caveats

- **Single-process baseline.** Native `MergeInsertBuilder.execute()` runs in one process.
  For very large source DataFrames the Spark-distributed path may still win once
  source materialization on one node becomes the bottleneck.
- **Sample size**. Three iters is enough to clear noise floor for the gap shape but not
  to publish definitive numbers. Run on cluster-shape hardware before quoting upstream.
- **Validation gap**. Equivalence is sanity-checked via inspection (matched IDs got the
  expected `value+1`; new IDs appeared) but a row-by-row oracle compare against a Spark
  baseline isn't yet wired into the bench. TODO before this PR goes upstream.
- **JNI classloader warning** appears in logs (`AsyncScanner class not found`). Same
  issue PSDK has been working around with init scripts; doesn't affect correctness for
  this bench shape but flagged.

## Next

- E2E validation on a real cluster (e.g. CPD).
- Larger scales (10M, 100M rows) on cluster hardware.
- Add row-by-row equivalence check between paths in the benchmark itself.
- If numbers hold up, file as upstream feature request against `lance-format/lance-spark`.
