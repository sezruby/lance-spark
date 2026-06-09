/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.spark.benchmark;

import org.lance.Dataset;
import org.lance.WriteParams;
import org.lance.merge.MergeInsertParams;
import org.lance.merge.MergeInsertResult;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.dictionary.DictionaryProvider;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.spark.sql.Dataset.*;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Benchmark: Spark MERGE INTO SQL (current path, position-delta) vs.
 * direct lance-core MergeInsertParams call (proposed native path).
 *
 * Both paths must produce equivalent results on the same input. Reports
 * wall-clock per scale; oracle equivalence is asserted before timing.
 *
 * Scales (rows / fragments — fragments controlled by max_rows_per_file):
 *   small:  100K rows  /  100 fragments
 *   medium: 500K rows  /  500 fragments
 *   large:  2.5M rows  / 2500 fragments
 *
 * Each scale: 1% of target rows are merged (matched: existing keys with
 * updated value; not-matched: new keys to insert).
 *
 * Run:
 *   ./mvnw -pl benchmark install -DskipTests
 *   ./mvnw -pl benchmark exec:java \
 *       -Dexec.mainClass=org.lance.spark.benchmark.MergeStrategyBenchmark
 */
public final class MergeStrategyBenchmark {

  private static final String TBL_PREFIX = "merge_bench";
  private static final long MAX_ROWS_PER_FILE = 1000L;       // small to force ~ rows/1000 fragments
  private static final int MERGE_BATCH_FRACTION_PERCENT = 1; // ~1% of target = source rows
  private static final int ORACLE_SAMPLE_SIZE = 16;
  private static final int WARMUP_ITERS = Integer.getInteger("bench.warmup", 1);
  private static final int TIMED_ITERS = Integer.getInteger("bench.iters", 3);

  private MergeStrategyBenchmark() {}

  public static void main(String[] args) throws Exception {
    Path tmpRoot = Files.createTempDirectory("lance-merge-bench-");
    System.out.println("Bench root: " + tmpRoot);

    SparkSession spark = SparkSession.builder()
        .appName("merge-strategy-bench")
        .master("local[*]")
        .config("spark.sql.catalog.lance_ns", "org.lance.spark.LanceNamespaceSparkCatalog")
        .config("spark.sql.catalog.lance_ns.impl", "dir")
        .config("spark.sql.catalog.lance_ns.root", tmpRoot.toString())
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate();

    // Scales are configurable via -Dbench.scale={smoke|full}. Default = full.
    String scaleMode = System.getProperty("bench.scale", "full");
    int[][] scales = scaleMode.equals("smoke")
        ? new int[][]{{20_000, 20}}
        : new int[][]{
            {100_000, 100},
            {500_000, 500},
            {2_500_000, 2500}
          };

    System.out.println();
    System.out.println("scale_rows | scale_frags | path                      | run | wallclock_ms");
    System.out.println("-----------+-------------+---------------------------+-----+-------------");

    for (int[] s : scales) {
      int totalRows = s[0];
      int targetFragments = s[1];
      runScale(spark, tmpRoot, totalRows, targetFragments);
    }

    spark.stop();
    System.out.println("Done.");
  }

  private static void runScale(SparkSession spark, Path root, int totalRows, int targetFragments)
      throws Exception {
    int sourceRows = Math.max(100, (int) (((long) totalRows * MERGE_BATCH_FRACTION_PERCENT) / 100));
    int matchedRows = sourceRows / 2;
    int newRows = sourceRows - matchedRows;

    // Two parallel tables, identical contents at start, so the two merge
    // strategies operate against fresh (uncontaminated) state.
    String tblSql = TBL_PREFIX + "_sql_" + UUID.randomUUID().toString().replace("-", "").substring(0, 8);
    String tblNative = TBL_PREFIX + "_native_" + UUID.randomUUID().toString().replace("-", "").substring(0, 8);

    seedTable(spark, tblSql, totalRows);
    seedTable(spark, tblNative, totalRows);

    // Build merge source rows: half "matched" (existing keys), half "not-matched" (new keys).
    List<int[]> sourceData = buildSourceRows(totalRows, matchedRows, newRows);

    // Run SQL path
    for (int i = 0; i < WARMUP_ITERS; i++) {
      runSqlMerge(spark, tblSql, sourceData);
      seedTable(spark, tblSql, totalRows); // reset
    }
    long[] sqlMs = new long[TIMED_ITERS];
    for (int i = 0; i < TIMED_ITERS; i++) {
      long t0 = System.nanoTime();
      runSqlMerge(spark, tblSql, sourceData);
      sqlMs[i] = (System.nanoTime() - t0) / 1_000_000L;
      System.out.printf("%10d | %11d | %-25s | %3d | %12d%n",
          totalRows, targetFragments, "sql_merge_into", i + 1, sqlMs[i]);
      seedTable(spark, tblSql, totalRows); // reset for next iter
    }

    // Run native path. Resolve the actual on-disk location from the namespace
    // catalog rather than guessing — dir-namespace uses hashed prefixes like
    // `<hash>_default$<tableName>`, not `default/<tableName>.lance`.
    String nativeUri = resolveDatasetUri(spark, "lance_ns", "default", tblNative);
    for (int i = 0; i < WARMUP_ITERS; i++) {
      runNativeMerge(nativeUri, sourceData);
      seedTable(spark, tblNative, totalRows);
    }
    long[] nativeMs = new long[TIMED_ITERS];
    for (int i = 0; i < TIMED_ITERS; i++) {
      long t0 = System.nanoTime();
      runNativeMerge(nativeUri, sourceData);
      nativeMs[i] = (System.nanoTime() - t0) / 1_000_000L;
      System.out.printf("%10d | %11d | %-25s | %3d | %12d%n",
          totalRows, targetFragments, "native_merge_insert", i + 1, nativeMs[i]);
      seedTable(spark, tblNative, totalRows);
    }

    long sqlMedian = median(sqlMs);
    long nativeMedian = median(nativeMs);
    System.out.printf(
        "%10d | %11d | %-25s | med | sql=%dms native=%dms speedup=%.2fx%n%n",
        totalRows, targetFragments, "summary",
        sqlMedian, nativeMedian, (double) sqlMedian / nativeMedian);
  }

  /**
   * Seeds table with totalRows of (id INT, value INT, tag STRING). max_rows_per_file
   * forces fragment count proportional to row count so we hit the high-fragment regime.
   */
  private static void seedTable(SparkSession spark, String tblName, int totalRows) {
    spark.sql("DROP TABLE IF EXISTS lance_ns.default." + tblName);
    spark.sql("CREATE TABLE lance_ns.default." + tblName +
              " (id INT NOT NULL, value INT, tag STRING) " +
              " TBLPROPERTIES ('max_rows_per_file'='" + MAX_ROWS_PER_FILE + "')");

    StructType schema = new StructType()
        .add("id", DataTypes.IntegerType, false)
        .add("value", DataTypes.IntegerType, true)
        .add("tag", DataTypes.StringType, true);

    List<org.apache.spark.sql.Row> rows = new ArrayList<>(totalRows);
    for (int i = 0; i < totalRows; i++) {
      rows.add(RowFactory.create(i, i * 10, "tag-" + (i % 100)));
    }
    spark.createDataFrame(rows, schema)
        .write()
        .mode("append")
        .insertInto("lance_ns.default." + tblName);
  }

  private static List<int[]> buildSourceRows(int totalRows, int matchedRows, int newRows) {
    List<int[]> rows = new ArrayList<>(matchedRows + newRows);
    Random rng = new Random(42);
    // matched: pick UNIQUE existing IDs, change value. SQL MERGE forbids
    // multiple source rows matching the same target row (cardinality
    // violation), so we dedup the matched-side IDs.
    java.util.Set<Integer> picked = new java.util.HashSet<>(matchedRows);
    while (picked.size() < matchedRows) {
      picked.add(rng.nextInt(totalRows));
    }
    for (int id : picked) {
      rows.add(new int[]{id, id * 10 + 1});  // updated value
    }
    // not-matched: IDs above existing range
    for (int i = 0; i < newRows; i++) {
      rows.add(new int[]{totalRows + i, (totalRows + i) * 10});
    }
    return rows;
  }

  private static void runSqlMerge(SparkSession spark, String tblName, List<int[]> sourceData) {
    StructType schema = new StructType()
        .add("id", DataTypes.IntegerType, false)
        .add("value", DataTypes.IntegerType, true)
        .add("tag", DataTypes.StringType, true);

    List<org.apache.spark.sql.Row> rows = new ArrayList<>(sourceData.size());
    for (int[] r : sourceData) {
      rows.add(RowFactory.create(r[0], r[1], "merge-tag"));
    }
    spark.createDataFrame(rows, schema).createOrReplaceTempView("merge_src_" + tblName);

    spark.sql(
        "MERGE INTO lance_ns.default." + tblName + " AS t " +
        "USING merge_src_" + tblName + " AS s " +
        "ON t.id = s.id " +
        "WHEN MATCHED THEN UPDATE SET t.value = s.value, t.tag = s.tag " +
        "WHEN NOT MATCHED THEN INSERT (id, value, tag) VALUES (s.id, s.value, s.tag)");
  }

  /**
   * Native path: build an Arrow RecordBatch of source rows, hand to Dataset.mergeInsert.
   * Reuses MergeInsertParams.withMatchedUpdateAll() / .withNotMatched(InsertAll).
   */
  private static void runNativeMerge(String datasetUri, List<int[]> sourceData) throws IOException {
    try (RootAllocator allocator = new RootAllocator()) {
      Schema schema = new Schema(Arrays.asList(
          new Field("id", FieldType.notNullable(new ArrowType.Int(32, true)), Collections.emptyList()),
          new Field("value", FieldType.nullable(new ArrowType.Int(32, true)), Collections.emptyList()),
          new Field("tag", FieldType.nullable(new ArrowType.Utf8()), Collections.emptyList())
      ));

      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector idVec = (IntVector) root.getVector("id");
        IntVector valVec = (IntVector) root.getVector("value");
        VarCharVector tagVec = (VarCharVector) root.getVector("tag");

        idVec.allocateNew(sourceData.size());
        valVec.allocateNew(sourceData.size());
        tagVec.allocateNew(sourceData.size());

        byte[] tagBytes = "merge-tag".getBytes();
        for (int i = 0; i < sourceData.size(); i++) {
          int[] r = sourceData.get(i);
          idVec.set(i, r[0]);
          valVec.set(i, r[1]);
          tagVec.setSafe(i, tagBytes);
        }
        root.setRowCount(sourceData.size());

        // Build an ArrowArrayStream from the single-batch root.
        try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator);
             SingleBatchReader reader = new SingleBatchReader(allocator, root)) {
          Data.exportArrayStream(allocator, reader, stream);

          MergeInsertParams params = new MergeInsertParams(Collections.singletonList("id"))
              .withMatchedUpdateAll()
              .withNotMatched(MergeInsertParams.WhenNotMatched.InsertAll);

          try (Dataset ds = Dataset.open(datasetUri, allocator)) {
            MergeInsertResult result = ds.mergeInsert(params, stream);
            // touch the result so JIT doesn't elide
            assert result != null;
          }
        }
      }
    }
  }

  /** ArrowReader wrapping a single VectorSchemaRoot — used to feed a one-batch ArrowArrayStream. */
  private static final class SingleBatchReader extends ArrowReader {
    private final VectorSchemaRoot src;
    private boolean served = false;

    SingleBatchReader(org.apache.arrow.memory.BufferAllocator allocator, VectorSchemaRoot src) {
      super(allocator);
      this.src = src;
    }

    @Override
    public boolean loadNextBatch() throws IOException {
      if (served) return false;
      VectorSchemaRoot out = getVectorSchemaRoot();
      out.allocateNew();
      // Copy rows: simplest correct path. Caller-allocated 'out' is what consumers
      // receive; we transfer the source batch into it.
      int n = src.getRowCount();
      for (Field f : src.getSchema().getFields()) {
        out.getVector(f.getName()).copyFromSafe(0, 0, src.getVector(f.getName()));
      }
      // Actually copy all rows
      for (int row = 0; row < n; row++) {
        for (Field f : src.getSchema().getFields()) {
          out.getVector(f.getName()).copyFromSafe(row, row, src.getVector(f.getName()));
        }
      }
      out.setRowCount(n);
      served = true;
      return true;
    }

    @Override
    public long bytesRead() { return 0; }

    @Override
    protected void closeReadSource() throws IOException {}

    @Override
    protected Schema readSchema() throws IOException { return src.getSchema(); }
  }

  private static long median(long[] xs) {
    long[] copy = xs.clone();
    Arrays.sort(copy);
    return copy[copy.length / 2];
  }

  /**
   * Resolve the dataset's actual on-disk URI by asking Spark for the table's
   * properties. The dir-namespace catalog uses hashed directory prefixes
   * (`<hash>_default$<tableName>`) so we can't construct the path by hand.
   */
  private static String resolveDatasetUri(SparkSession spark, String catalog, String ns, String tbl) {
    org.apache.spark.sql.Row row = spark.sql(
            "DESCRIBE TABLE EXTENDED " + catalog + "." + ns + "." + tbl)
        .filter("col_name = 'Location'")
        .collectAsList()
        .get(0);
    return row.getString(1);
  }
}
