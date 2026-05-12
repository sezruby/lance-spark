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
package org.lance.spark.knn.benchmark

import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{Row, RowFactory, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types._
import org.lance.spark.knn.internal.{ProbedLeft, ScoredRowRef}

import java.util.{Locale, Random}
import java.util.concurrent.TimeUnit

import scala.collection.mutable

/**
 * Microbenchmark for inter-stage payload encoding cost. The question this answers:
 *
 *   If we split the 2.12 module's RDD pipeline into 3 explicit SparkPlan operators
 *   (LanceProbeExec → LanceMergeExec → LanceMaterializeExec), each boundary needs the
 *   `ProbedLeft` payload encoded as `InternalRow`. How much wall-clock does that cost
 *   relative to the actual probe + merge + materialize work?
 *
 * Key insight before we even measure: between probe and merge, the payload is per left row,
 * NOT per `(left × K)` pair. We carry the left vector + K row refs (rowAddr + score), not
 * K full right-side rows. The right rows only get fetched in the materialize stage. So the
 * encoding cost scales with |L|, not |L| × K × |right_row_size|.
 *
 * == Two encoding schemes compared ==
 *
 *   A) Catalyst struct encoding via ExpressionEncoder. Schema:
 *      `struct<leftId: long, leftRow: struct<...>, refs: array<struct<rowAddr: long,
 *      score: float>>>`. UnsafeRow-encoded; native to Catalyst's row-shuffle path.
 *
 *   B) Binary blob via Kryo. Schema: `struct<leftId: long, blob: binary>` where `blob` is
 *      the Kryo-serialized `ProbedLeft`. Simpler implementation but pays Kryo's per-call
 *      overhead.
 *
 * The "winner" is whichever cost is small enough to ignore at the realistic scales we
 * benchmark (small = 100 left rows, medium = 1000). If both are negligible, the choice is
 * code-complexity, not performance.
 *
 * Invocation:
 * {{{
 *   MAVEN_OPTS="-Xmx4g <JDK 17 add-opens flags>" \
 *     ./mvnw -pl lance-spark-knn_2.12 -q exec:java \
 *       -Dexec.classpathScope=test \
 *       -Dexec.mainClass=org.lance.spark.knn.benchmark.InterStagePayloadOverheadBench
 * }}}
 */
object InterStagePayloadOverheadBench {

  private val Dim: Int = 128
  private val K: Int = 10
  private val Seed: Long = 42L
  private val Warmup: Int = 3
  private val Iterations: Int = 5

  // Scales matching the SQL benchmark's `numLeft` settings.
  private val Scales: Seq[(String, Int)] = Seq(
    "small (numLeft=100)" -> 100,
    "medium (numLeft=1000)" -> 1000,
    "stress (numLeft=10000)" -> 10000)

  def main(args: Array[String]): Unit = {
    println("=" * 78)
    println("Inter-stage ProbedLeft encoding overhead — A: Catalyst struct vs B: Kryo blob")
    println(s"Dim=$Dim, K=$K, ${Iterations} iters/${Warmup} warmups per cell")
    println("=" * 78)

    // SparkSession just for the encoder + Kryo registry; no jobs run here.
    val spark = SparkSession.builder()
      .appName("inter-stage-payload-overhead")
      .master("local[1]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.serializer", classOf[KryoSerializer].getName)
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try {
      val rng = new Random(Seed)

      Scales.foreach { case (name, n) =>
        println()
        println(s"--- $name ---")

        val payloads = generatePayloads(rng, n)

        // Catalyst-struct encoder.
        val schema = catalystSchema()
        val enc = ExpressionEncoder(schema).resolveAndBind()
        val ser = enc.createSerializer()
        val deser = enc.createDeserializer()

        val schemeA = bench(
          "A: Catalyst struct (encode + decode)",
          () => {
            var sink = 0L
            var i = 0
            while (i < payloads.length) {
              val ir = ser(toCatalystRow(i.toLong, payloads(i)))
              val back = deser(ir)
              sink ^= back.getLong(0)
              i += 1
            }
            sink
          })

        // Kryo encoder.
        val kryoSerializerInstance = new KryoSerializer(spark.sparkContext.getConf).newInstance()
        val schemeB = bench(
          "B: Kryo binary blob (encode + decode)",
          () => {
            var sink = 0L
            var i = 0
            while (i < payloads.length) {
              val bytes = kryoSerializerInstance.serialize(payloads(i))
              val back = kryoSerializerInstance.deserialize[ProbedLeft](bytes)
              sink ^= back.refs.length.toLong
              i += 1
            }
            sink
          })

        val medianA = median(schemeA)
        val medianB = median(schemeB)
        printf(
          "  A: Catalyst struct  median=%6.2f ms  per-row=%6.2f µs  (across both inter-stage boundaries: %6.2f ms total)%n",
          medianA / 1e6,
          medianA / 1e3 / n,
          2.0 * medianA / 1e6)
        printf(
          "  B: Kryo binary blob median=%6.2f ms  per-row=%6.2f µs  (across both inter-stage boundaries: %6.2f ms total)%n",
          medianB / 1e6,
          medianB / 1e3 / n,
          2.0 * medianB / 1e6)

        // Sanity: serialized blob size for B (rough — actual rows may vary slightly).
        val sampleBlob = new KryoSerializer(spark.sparkContext.getConf).newInstance()
          .serialize(payloads.head)
        val avgBlobBytes = sampleBlob.remaining()
        printf(
          "  Per-row serialized size (Kryo blob): ~%d bytes  (×$n rows ≈ %.1f KB total payload)%n",
          avgBlobBytes,
          avgBlobBytes * n / 1024.0)
      }

      println()
      println("=" * 78)
      println("Conclusion guide:")
      println("  - Encoding overhead < 5%% of total wall-clock at the relevant SQL benchmark")
      println("    cell  ⇒  splitting into 3 execs is essentially free; do it for explainability.")
      println("  - Encoding overhead > 20%%  ⇒  the 3-exec split costs more than it informs;")
      println("    keep the single-exec wrapper and consider RDD.setName() for Spark UI clarity.")
      println("  - Anywhere between, judgement call.")
      println("=" * 78)
    } finally {
      spark.stop()
    }
  }

  // -- payload generation ------------------------------------------------------------------

  private def generatePayloads(rng: Random, n: Int): Array[ProbedLeft] = {
    val out = new Array[ProbedLeft](n)
    var i = 0
    while (i < n) {
      val vec = new Array[Float](Dim)
      var d = 0
      while (d < Dim) { vec(d) = rng.nextFloat(); d += 1 }
      val leftRow: Row = RowFactory.create(Integer.valueOf(i), vec)

      val refs = new Array[ScoredRowRef](K)
      var r = 0
      while (r < K) {
        refs(r) = ScoredRowRef(rng.nextLong(), rng.nextFloat())
        r += 1
      }
      out(i) = ProbedLeft(leftRow, refs)
      i += 1
    }
    out
  }

  // Catalyst schema mirroring the candidate Plan-A struct encoding.
  private def catalystSchema(): StructType = StructType(Seq(
    StructField("leftId", LongType, nullable = false),
    StructField(
      "leftRow",
      StructType(Seq(
        StructField("lid", IntegerType, nullable = true),
        StructField("lvec", ArrayType(FloatType, containsNull = false), nullable = true))),
      nullable = true),
    StructField(
      "refs",
      ArrayType(
        StructType(Seq(
          StructField("rowAddr", LongType, nullable = false),
          StructField("score", FloatType, nullable = false))),
        containsNull = false),
      nullable = false)))

  private def toCatalystRow(leftId: Long, pl: ProbedLeft): Row = {
    val leftStruct = Row(pl.leftRow.getInt(0), pl.leftRow.get(1))
    val refStructs = pl.refs.map(r => Row(r.rowAddr, r.score)).toSeq
    Row(leftId, leftStruct, refStructs)
  }

  // -- timing helper -----------------------------------------------------------------------

  private def bench(label: String, body: () => Long): Seq[Long] = {
    var w = 0
    while (w < Warmup) { val _ = body(); w += 1 }
    val out = mutable.ArrayBuffer.empty[Long]
    var i = 0
    while (i < Iterations) {
      val t0 = System.nanoTime()
      val _ = body()
      out += (System.nanoTime() - t0)
      i += 1
    }
    out.toSeq
  }

  private def median(xs: Seq[Long]): Long = {
    val sorted = xs.sorted
    sorted(sorted.length / 2)
  }
}
