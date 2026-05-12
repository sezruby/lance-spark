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
package org.lance.spark.knn

import org.apache.spark.sql.{Row, RowFactory, SparkSession}
import org.apache.spark.sql.types._
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.io.TempDir

import java.nio.file.Path
import java.util.Random

import scala.collection.JavaConverters._

/**
 * End-to-end correctness regression for the `InterStageShuffle.mergeViaCatalystShuffle`
 * path. Builds a right side without any vector index — Lance then falls back to an exact
 * per-fragment scan, which makes the join a recall = 1.0 oracle: the top-K refs per
 * left row MUST equal the brute-force Scala top-K computed on the driver.
 *
 * This is the strongest correctness check available short of setting up an actual
 * vector index. If the `repartition(col(_leftId))` → per-partition merge path dropped,
 * reordered, or corrupted rows in any way, this would detect it.
 */
class IndexedNearestJoinCorrectnessTest {

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  private val Dim = 16
  private val NumRight = 1000
  private val NumLeft = 100
  private val K = 10
  private val Seed = 31L

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("indexed-nearest-join-correctness")
      .master("local[4]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.sql.shuffle.partitions", "8")
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = if (spark != null) spark.stop()

  /** Every left row's top-K must match the brute-force Scala oracle exactly. */
  @Test def testTopKMatchesBruteForceOracle(): Unit = {
    val rng = new Random(Seed)
    val (rightRows, rightVecs) = generateRows(rng, NumRight, Dim, idOffset = 1000)
    val (leftRows, leftVecs) = generateRows(rng, NumLeft, Dim, idOffset = 0)

    val rightUri = writeLance(rightRows, "rid", "rvec")
    val leftDf = buildDf(leftRows, "lid", "qvec")

    val joined = IndexedNearestJoin(
      left = leftDf,
      rightLanceUri = rightUri,
      leftVecCol = "qvec",
      rightVecCol = "rvec",
      k = K,
      metric = "l2",
      rightProjection = Some(Seq("rid"))).collect()

    // Build expected map: leftLid → sorted list of rid ASCENDING by distance.
    val expected: Map[Long, Seq[Long]] = leftVecs.zipWithIndex.map { case (qvec, leftIdx) =>
      val dists = rightVecs.zipWithIndex.map { case (rvec, rightIdx) =>
        ((rightIdx + 1000).toLong, l2DistanceSquared(qvec, rvec))
      }
      val topK = dists.sortBy(_._2).take(K).map(_._1)
      leftIdx.toLong -> topK
    }.toMap

    // Group actual rows by lid, sort by __score ASC, extract rid sequence.
    val actualByLid: Map[Long, Seq[Long]] = joined
      .groupBy(_.getLong(0))
      .map { case (lid, rows) =>
        lid -> rows.toSeq.sortBy(_.getFloat(3)).map(_.getLong(2))
      }

    assertEquals(NumLeft, actualByLid.size, s"Expected $NumLeft distinct leftIds in output")

    expected.foreach { case (lid, expectedRids) =>
      val actualRids = actualByLid.getOrElse(lid, Seq.empty)
      assertEquals(
        expectedRids,
        actualRids,
        s"Top-$K rids mismatch for lid=$lid:\n  expected=$expectedRids\n  actual=$actualRids")
    }
  }

  // -- helpers ------------------------------------------------------------------------------

  private def generateRows(
      rng: Random,
      n: Int,
      dim: Int,
      idOffset: Int): (Seq[Row], Seq[Array[Float]]) = {
    val vecs = (0 until n).map(_ => randomVector(rng, dim))
    val rows = vecs.zipWithIndex.map { case (v, i) =>
      RowFactory.create(java.lang.Long.valueOf((i + idOffset).toLong), v.toSeq.asJava)
    }
    (rows, vecs)
  }

  private def writeLance(rows: Seq[Row], idCol: String, vecCol: String): String = {
    val schema = new StructType(Array(
      StructField(idCol, LongType, nullable = false),
      StructField(
        vecCol,
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val df = spark.createDataFrame(rows.asJava, schema)
    val uri = tempDir.resolve(s"ds_${System.nanoTime()}").toString
    df.write.format("lance").save(uri)
    uri
  }

  private def buildDf(rows: Seq[Row], idCol: String, vecCol: String) = {
    val schema = new StructType(Array(
      StructField(idCol, LongType, nullable = false),
      StructField(
        vecCol,
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    spark.createDataFrame(rows.asJava, schema)
  }

  private def randomVector(rng: Random, dim: Int): Array[Float] = {
    val v = new Array[Float](dim)
    var i = 0
    while (i < dim) { v(i) = rng.nextFloat(); i += 1 }
    v
  }

  private def l2DistanceSquared(a: Array[Float], b: Array[Float]): Float = {
    var s = 0.0f
    var i = 0
    while (i < a.length) {
      val d = a(i) - b(i)
      s += d * d
      i += 1
    }
    s
  }
}
