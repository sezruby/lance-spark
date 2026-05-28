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

import org.apache.spark.sql.SparkSession
import org.junit.jupiter.api.{AfterEach, BeforeEach, Test}
import org.junit.jupiter.api.Assertions._

class ExecutorCpuCheckTest {

  private var spark: SparkSession = _

  @BeforeEach def setup(): Unit = {
    spark = SparkSession.builder()
      .appName("executor-cpu-check-test")
      .master("local[2]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate()
  }

  @AfterEach def teardown(): Unit = {
    if (spark != null) spark.stop()
  }

  /**
   * Probe runs on a 2-core local Spark and prints the expected sections without throwing.
   * On `local[N]` there is exactly one driver-as-executor entity, so the table will have
   * one row — exercises the formatting + collect path without depending on cluster shape.
   */
  @Test def runsAndPrintsTable(): Unit = {
    // Just confirm it doesn't throw with a generous failRatio (passes regardless of
    // local timing variation).
    ExecutorCpuCheck.run(spark, failRatio = Some(10.0))
  }

  /**
   * `failRatio` set to 0 (impossible to satisfy) should throw — verifies the gate.
   */
  @Test def throwsWhenFailRatioImpossible(): Unit = {
    val ex = assertThrows(
      classOf[IllegalStateException],
      () => ExecutorCpuCheck.run(spark, failRatio = Some(0.0)))
    assertTrue(
      ex.getMessage.contains("BENCH_CPU_CHECK_FAIL_RATIO"),
      s"expected gate message; got: ${ex.getMessage}")
  }
}
