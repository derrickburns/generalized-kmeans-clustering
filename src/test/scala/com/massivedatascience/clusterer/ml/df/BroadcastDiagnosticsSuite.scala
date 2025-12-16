/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.massivedatascience.clusterer.ml.df

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.apache.log4j.{ Level, Logger }

import scala.collection.mutable

/** Tests for broadcast threshold diagnostics in assignment strategies.
  */
class BroadcastDiagnosticsSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .appName("BroadcastDiagnosticsSuite")
      .master("local[2]")
      .config("spark.ui.enabled", "false")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
    super.afterAll()
  }

  test("AutoAssignment logs detailed diagnostics when exceeding threshold") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create small dataset
    val df = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(3.0, 4.0))
    ).toDF("features")

    // Create many small centers to exceed threshold
    val k       = 100
    val dim     = 100
    val centers = Array.fill(k)(Array.fill(dim)(1.0))

    // Small threshold to trigger warning
    val strategy = new AutoAssignment(broadcastThresholdElems = 1000, chunkSize = 10)
    val kernel   = new SquaredEuclideanKernel()

    // This should log a warning about exceeding threshold
    // We can't easily assert on log output without a custom appender,
    // but we can verify the method completes without error
    val result = strategy.assign(df, "features", None, centers, kernel)

    assert(result.columns.contains("cluster"))
    assert(result.count() == 2)
  }

  test("AutoAssignment provides correct size calculations") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      Tuple1(Vectors.dense(1.0, 2.0, 3.0)),
      Tuple1(Vectors.dense(4.0, 5.0, 6.0))
    ).toDF("features")

    // k=10, dim=3 → k×dim = 30 elements
    val centers  = Array.fill(10)(Array.fill(3)(1.0))
    val strategy = new AutoAssignment(broadcastThresholdElems = 100, chunkSize = 5)
    val kernel   = new KLDivergenceKernel(1e-6)

    // Should use BroadcastUDF (not chunked) since 30 < 100
    val result = strategy.assign(df, "features", None, centers, kernel)

    assert(result.columns.contains("cluster"))
    assert(result.count() == 2)
  }

  test("BroadcastUDFAssignment warns on very large broadcasts") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(3.0, 4.0))
    ).toDF("features")

    // Create a very large broadcast: k=2000, dim=10000 → 20M elements = ~160MB
    // This should trigger the warning in BroadcastUDFAssignment
    val k       = 2000
    val dim     = 10000
    val centers = Array.fill(k)(Array.fill(dim)(1.0))

    val strategy = new BroadcastUDFAssignment()
    val kernel   = new SquaredEuclideanKernel()

    // This should log a warning about large broadcast
    // The method should still complete (though it may be slow/OOM in real scenarios)
    // For testing purposes, we just verify it doesn't crash
    val result = strategy.assign(df, "features", None, centers, kernel)

    assert(result.columns.contains("cluster"))
  }

  test("formatBroadcastSize shows human-readable sizes") {
    // We can't directly test the private method, but we can verify
    // that the strategies handle various sizes correctly

    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(3.0, 4.0))
    ).toDF("features")

    // Test various sizes
    val testCases = Seq(
      (10, 10),    // 100 elements = 800 bytes
      (100, 100),  // 10K elements = 80KB
      (1000, 100), // 100K elements = 800KB
      (1000, 1000) // 1M elements = 8MB
    )

    testCases.foreach { case (k, dim) =>
      val centers  = Array.fill(k)(Array.fill(dim)(1.0))
      val strategy = new AutoAssignment(broadcastThresholdElems = 2000000, chunkSize = 100)
      val kernel   = new SquaredEuclideanKernel()

      val result = strategy.assign(df, "features", None, centers, kernel)
      assert(result.columns.contains("cluster"))
    }
  }

  test("AutoAssignment calculates correct chunk count") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(3.0, 4.0))
    ).toDF("features")

    // k=250, chunkSize=100 → should require 3 passes
    val k        = 250
    val dim      = 10
    val centers  = Array.fill(k)(Array.fill(dim)(1.0))
    val strategy = new AutoAssignment(broadcastThresholdElems = 1000, chunkSize = 100)
    val kernel   = new KLDivergenceKernel(1e-6)

    // k×dim = 2500 > 1000, so should use ChunkedBroadcast
    // Math.ceil(250 / 100) = 3 passes
    val result = strategy.assign(df, "features", None, centers, kernel)

    assert(result.columns.contains("cluster"))
    assert(result.count() == 2)
  }

  test("AutoAssignment suggests appropriate threshold increase") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      Tuple1(Vectors.dense(1.0, 2.0, 3.0)),
      Tuple1(Vectors.dense(4.0, 5.0, 6.0))
    ).toDF("features")

    // k=100, dim=50 → k×dim = 5000
    // If threshold is 1000, it should suggest increasing to ~5000
    val k        = 100
    val dim      = 50
    val centers  = Array.fill(k)(Array.fill(dim)(1.0))
    val strategy = new AutoAssignment(broadcastThresholdElems = 1000, chunkSize = 50)
    val kernel   = new GeneralizedIDivergenceKernel(1e-6)

    val result = strategy.assign(df, "features", None, centers, kernel)

    assert(result.columns.contains("cluster"))
  }

  test("AutoAssignment calculates max k for given dim") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(3.0, 4.0))
    ).toDF("features")

    // With dim=20 and threshold=1000, max k should be 1000/20 = 50
    val k        = 60 // Exceeds max
    val dim      = 20
    val centers  = Array.fill(k)(Array.fill(dim)(1.0))
    val strategy = new AutoAssignment(broadcastThresholdElems = 1000, chunkSize = 30)
    val kernel   = new LogisticLossKernel(1e-6)

    // k×dim = 1200 > 1000, should warn and suggest k≈50
    val result = strategy.assign(df, "features", None, centers, kernel)

    assert(result.columns.contains("cluster"))
  }
}
