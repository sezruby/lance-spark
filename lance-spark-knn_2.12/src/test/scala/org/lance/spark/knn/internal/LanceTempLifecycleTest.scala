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
package org.lance.spark.knn.internal

import org.apache.spark.sql.{Row, RowFactory, SparkSession}
import org.apache.spark.sql.types._
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.io.TempDir

import java.io.File
import java.nio.file.{Files, Path, Paths}
import java.util.Random

import scala.collection.JavaConverters._

/**
 * Tests for [[LanceTempLifecycle]]:
 *
 *   - register adds the URI to the per-app registry, count reflects it
 *   - stopping the SparkSession (which fires SparkListenerApplicationEnd) deletes the
 *     registered temp dirs
 *   - stopForTesting (a back-door we expose explicitly so tests don't have to actually
 *     stop the session — that breaks subsequent BeforeEach setup in the same suite)
 *     also deletes
 *   - deleteUri handles a bare local path
 *   - registering the same URI twice in one app is idempotent
 *
 * Concurrent / multi-app cases are exercised by the existence of `appId`-keyed maps in
 * the lifecycle code itself — testing genuine cross-app isolation in a JUnit suite would
 * require multi-process orchestration that's not worth the test infra cost. The unit
 * tests here cover the listener-driven path and the registry mechanics.
 */
class LanceTempLifecycleTest {

  @TempDir var tempDir: Path = _
  private var spark: SparkSession = _

  private val Dim: Int = 4

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("lance-temp-lifecycle-test")
      .master("local[2]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = {
    if (spark != null) {
      // Clean up any test residue regardless of pass/fail. stopForTesting also handles
      // any URIs leftover (e.g. when a test registered without going through .stop()).
      LanceTempLifecycle.stopForTesting(spark.sparkContext.applicationId)
      spark.stop()
    }
  }

  /** Registering and then deleting via the test back-door removes the URI from disk. */
  @Test def testRegisterAndExplicitCleanup(): Unit = {
    val tempUri = writeTempLance()
    val appId = spark.sparkContext.applicationId

    assertTrue(new File(tempUri).exists(), "precondition: temp Lance dataset exists")
    assertEquals(0, LanceTempLifecycle.registeredCount(appId), "no registrations yet")

    LanceTempLifecycle.register(spark, tempUri)
    assertEquals(1, LanceTempLifecycle.registeredCount(appId))

    LanceTempLifecycle.stopForTesting(appId)
    assertFalse(new File(tempUri).exists(), "cleanup must delete the temp dir")
    assertEquals(
      0,
      LanceTempLifecycle.registeredCount(appId),
      "registry must be empty after cleanup")
  }

  /** Multiple temp URIs from the same app are all cleaned up. */
  @Test def testMultipleRegistrationsAllCleanedUp(): Unit = {
    val a = writeTempLance()
    val b = writeTempLance()
    val c = writeTempLance()
    val appId = spark.sparkContext.applicationId
    LanceTempLifecycle.register(spark, a)
    LanceTempLifecycle.register(spark, b)
    LanceTempLifecycle.register(spark, c)
    assertEquals(3, LanceTempLifecycle.registeredCount(appId))

    LanceTempLifecycle.stopForTesting(appId)
    assertFalse(new File(a).exists())
    assertFalse(new File(b).exists())
    assertFalse(new File(c).exists())
  }

  /**
   * Re-registering the same URI is a no-op. Important because LanceTempR.materialize
   * could be called repeatedly with overlapping temp URIs in retry scenarios.
   */
  @Test def testIdempotentRegistration(): Unit = {
    val tempUri = writeTempLance()
    val appId = spark.sparkContext.applicationId
    LanceTempLifecycle.register(spark, tempUri)
    LanceTempLifecycle.register(spark, tempUri)
    LanceTempLifecycle.register(spark, tempUri)
    assertEquals(1, LanceTempLifecycle.registeredCount(appId), "duplicate registers are deduped")
    LanceTempLifecycle.stopForTesting(appId)
    assertFalse(new File(tempUri).exists())
  }

  /**
   * Stopping the SparkSession fires SparkListenerApplicationEnd and triggers cleanup
   * via the listener path — the production cleanup trigger. We tear down `spark`
   * inside this test, so override @AfterEach behavior by setting `spark = null`.
   */
  @Test def testApplicationEndTriggersCleanup(): Unit = {
    val tempUri = writeTempLance()
    val appId = spark.sparkContext.applicationId
    LanceTempLifecycle.register(spark, tempUri)
    assertEquals(1, LanceTempLifecycle.registeredCount(appId))

    spark.stop()
    spark = null // prevent @AfterEach from calling stop() again

    // Listener fires on the listener bus thread; give it a moment to drain.
    val deadline = System.currentTimeMillis() + 5000
    while (new File(tempUri).exists() && System.currentTimeMillis() < deadline) {
      Thread.sleep(50)
    }
    assertFalse(
      new File(tempUri).exists(),
      "SparkListenerApplicationEnd path must delete the registered temp dir within 5s")
    assertEquals(0, LanceTempLifecycle.registeredCount(appId))
  }

  /** deleteUri handles a non-existent path gracefully (no exception). */
  @Test def testDeleteUriNonExistentNoOps(): Unit = {
    val ghost = tempDir.resolve("never_existed_" + System.nanoTime()).toString
    LanceTempLifecycle.deleteUri(ghost, spark.sparkContext.hadoopConfiguration)
    // No assertion — pass means no exception.
  }

  /** deleteUri null/empty input is a no-op. */
  @Test def testDeleteUriNullOrEmpty(): Unit = {
    LanceTempLifecycle.deleteUri(null, spark.sparkContext.hadoopConfiguration)
    LanceTempLifecycle.deleteUri("", spark.sparkContext.hadoopConfiguration)
  }

  // -- helpers --------------------------------------------------------------------------------

  /** Write a tiny Lance dataset under tempDir and return its URI. */
  private def writeTempLance(): String = {
    val schema = new StructType(Array(
      StructField("id", IntegerType, nullable = false),
      StructField(
        "vec",
        ArrayType(FloatType, containsNull = false),
        nullable = false,
        new MetadataBuilder().putLong("arrow.fixed-size-list.size", Dim.toLong).build())))
    val rows: Seq[Row] = (0 until 4).map { i =>
      RowFactory.create(Integer.valueOf(i), randomVector(new Random(i.toLong), Dim))
    }
    val df = spark.createDataFrame(rows.asJava, schema)
    val target = tempDir.resolve("temp_" + System.nanoTime())
    df.write.format("lance").save(target.toString)
    target.toString
  }

  private def randomVector(rng: Random, dim: Int): Array[Float] = {
    val v = new Array[Float](dim)
    var i = 0
    while (i < dim) { v(i) = rng.nextFloat(); i += 1 }
    v
  }
}
