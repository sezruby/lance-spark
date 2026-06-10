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

### Local laptop (Apple M5 Max, Spark `local[*]`)

| Total rows | Fragments | SQL `MERGE INTO` median | Native `mergeInsert` median | Speedup |
|---:|---:|---:|---:|---:|
| 100,000 | 100 | 190 ms | 7 ms | **27.1×** |
| 500,000 | 500 | 191 ms | 14 ms | **13.6×** |
| 2,500,000 | 2,500 | 309 ms | 50 ms | **6.2×** |

### CPD stage cluster (4 executors × 4 cores × 16 GB, Spark 3.5)

| Total rows | Fragments | SQL `MERGE INTO` median | Native `mergeInsert` median | Speedup |
|---:|---:|---:|---:|---:|
| 10,000,000 | 10,000 | 7,934 ms | 468 ms | **17.0×** |
| 50,000,000 | 50,000 | 11,185 ms | 2,618 ms | **4.3×** |
| 100,000,000 | 100,000 | 18,644 ms | 5,331 ms | **3.5×** |

## Reading the trend

- Native wins at every measured scale, from 100K to 100M rows.
- The gap **narrows as scale grows on the cluster** (17× → 4.3× → 3.5×) but
  doesn't invert through 100M rows / 100K fragments. SQL benefits from
  cluster cores; native is single-process. Where SQL overhead dominates
  (small batches, high fragment count), native is dramatically faster;
  where target+source reading dominates, native still wins by ~3-4×.
- For the typical multimodal-merge workload — incremental sync of 1M–100M
  rows into a Lance table with thousands of fragments — native is the right
  path. The crossover where SQL's distributed parallelism overtakes native's
  single-process limit is past the scales tested here.

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
