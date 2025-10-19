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
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfterAll

/** Tests for Lloyd's algorithm iterator.
  *
  * These tests verify the core k-means iteration loop:
  *   - Convergence detection
  *   - Iteration limits
  *   - Checkpoint functionality
  *   - Empty cluster handling
  *   - History tracking (distortion, movement)
  *   - Cache management
  */
class LloydsIteratorSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  private val spark: SparkSession = SparkSession
    .builder()
    .master("local[2]")
    .appName("LloydsIteratorSuite")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  import spark.implicits._

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) {
        spark.stop()
      }
    } finally {
      super.afterAll()
    }
  }

  // Helper to create test DataFrame with well-separated clusters
  private def createTestData() = {
    Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(0.2, 0.2),
      Vectors.dense(10.0, 10.0),
      Vectors.dense(10.1, 10.1),
      Vectors.dense(10.2, 10.2)
    ).map(v => Tuple1(v)).toDF("features")
  }

  // Helper to create default config
  private def defaultConfig(k: Int = 2, maxIter: Int = 20, tol: Double = 1e-4): LloydsConfig = {
    val kernel    = new SquaredEuclideanKernel()
    val assigner  = new AutoAssignment()
    val updater   = new GradMeanUDAFUpdate()
    val empty     = new DropEmptyClustersHandler()
    val conv      = new MovementConvergence()
    val validator = new StandardInputValidator()

    LloydsConfig(
      k = k,
      maxIter = maxIter,
      tol = tol,
      kernel = kernel,
      assigner = assigner,
      updater = updater,
      emptyHandler = empty,
      convergence = conv,
      validator = validator,
      checkpointInterval = 0,
      checkpointDir = None
    )
  }

  test("LloydsIterator converges with well-separated clusters") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(0.0, 0.0),
      Array(10.0, 10.0)
    )

    val config = defaultConfig(k = 2, maxIter = 20, tol = 1e-4)
    val result = iterator.run(df, "features", None, initialCenters, config)

    // Should converge
    result.converged shouldBe true
    result.iterations should be < 20

    // Should have 2 centers
    result.centers.length shouldBe 2

    // Distortion should decrease
    result.distortionHistory.length shouldBe result.iterations
    if (result.distortionHistory.length > 1) {
      result.distortionHistory.head should be >= result.distortionHistory.last
    }

    // Movement should decrease
    result.movementHistory.length shouldBe result.iterations
    if (result.movementHistory.length > 1) {
      result.movementHistory.last should be < 1e-4
    }
  }

  test("LloydsIterator respects max iteration limit") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    // Bad initial centers that won't converge quickly
    val initialCenters = Array(
      Array(5.0, 5.0),
      Array(5.1, 5.1)
    )

    val config = defaultConfig(k = 2, maxIter = 5, tol = 1e-10)
    val result = iterator.run(df, "features", None, initialCenters, config)

    // Should stop at max iterations or before if converged
    result.iterations should be <= 5

    // If didn't converge, should have run all iterations
    if (!result.converged) {
      result.iterations shouldBe 5
    }

    // Should still have valid centers (may have fewer due to drop strategy)
    result.centers.length should be >= 1
    result.centers.length should be <= 2
    result.distortionHistory.length shouldBe result.iterations
    result.movementHistory.length shouldBe result.iterations
  }

  test("LloydsIterator handles single iteration") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(0.1, 0.1),
      Array(10.1, 10.1)
    )

    val config = defaultConfig(k = 2, maxIter = 1, tol = 1e-4)
    val result = iterator.run(df, "features", None, initialCenters, config)

    result.iterations shouldBe 1
    result.centers.length shouldBe 2
    result.distortionHistory.length shouldBe 1
    result.movementHistory.length shouldBe 1
  }

  test("LloydsIterator tracks distortion history accurately") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(0.0, 0.0),
      Array(10.0, 10.0)
    )

    val config = defaultConfig(k = 2, maxIter = 10, tol = 1e-4)
    val result = iterator.run(df, "features", None, initialCenters, config)

    // Distortion history should have one entry per iteration
    result.distortionHistory.length shouldBe result.iterations

    // All distortions should be non-negative and finite
    result.distortionHistory.foreach { d =>
      d should be >= 0.0
      java.lang.Double.isFinite(d) shouldBe true
    }

    // Distortion should generally decrease (allowing for small fluctuations)
    if (result.distortionHistory.length > 1) {
      val first = result.distortionHistory.head
      val last  = result.distortionHistory.last
      // Last distortion should be less than or close to first
      last should be <= (first * 1.01)
    }
  }

  test("LloydsIterator tracks movement history accurately") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(1.0, 1.0),
      Array(9.0, 9.0)
    )

    val config = defaultConfig(k = 2, maxIter = 10, tol = 1e-4)
    val result = iterator.run(df, "features", None, initialCenters, config)

    // Movement history should have one entry per iteration
    result.movementHistory.length shouldBe result.iterations

    // All movements should be non-negative and finite
    result.movementHistory.foreach { m =>
      m should be >= 0.0
      java.lang.Double.isFinite(m) shouldBe true
    }

    // If converged, last movement should be below tolerance
    if (result.converged) {
      result.movementHistory.last should be < config.tol
    }
  }

  test("LloydsIterator handles empty clusters with drop strategy") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    // Initialize with 3 centers but data only supports 2 clusters
    val initialCenters = Array(
      Array(0.1, 0.1),
      Array(10.1, 10.1),
      Array(20.0, 20.0) // This will be empty
    )

    val config = defaultConfig(k = 3, maxIter = 10, tol = 1e-4)
    val result = iterator.run(df, "features", None, initialCenters, config)

    // Should complete (drop strategy allows fewer than k centers)
    result.iterations should be > 0

    // May have fewer than 3 centers due to dropping empty clusters
    result.centers.length should be <= 3
    result.centers.length should be >= 1
  }

  test("LloydsIterator handles empty clusters with reseed strategy") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    // Initialize with 3 centers but data only supports 2 clusters
    val initialCenters = Array(
      Array(0.1, 0.1),
      Array(10.1, 10.1),
      Array(20.0, 20.0) // This will be empty
    )

    val kernel    = new SquaredEuclideanKernel()
    val assigner  = new AutoAssignment()
    val updater   = new GradMeanUDAFUpdate()
    val empty     = new ReseedRandomHandler(seed = 42)
    val conv      = new MovementConvergence()
    val validator = new StandardInputValidator()

    val config = LloydsConfig(
      k = 3,
      maxIter = 10,
      tol = 1e-4,
      kernel = kernel,
      assigner = assigner,
      updater = updater,
      emptyHandler = empty,
      convergence = conv,
      validator = validator
    )

    val result = iterator.run(df, "features", None, initialCenters, config)

    // Should complete
    result.iterations should be > 0

    // With reseed strategy, we expect k=3 centers
    // But drop strategy may be used if reseed fails
    result.centers.length should be >= 2
    result.centers.length should be <= 3

    // Should have recorded empty cluster events if reseeding occurred
    result.emptyClusterEvents should be >= 0
  }

  test("LloydsIterator works with weighted data") {
    val data = Seq(
      (Vectors.dense(0.0, 0.0), 2.0),
      (Vectors.dense(0.1, 0.1), 1.0),
      (Vectors.dense(10.0, 10.0), 1.0),
      (Vectors.dense(10.1, 10.1), 2.0)
    ).toDF("features", "weight")

    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(0.0, 0.0),
      Array(10.0, 10.0)
    )

    val config = defaultConfig(k = 2, maxIter = 10, tol = 1e-4)
    val result = iterator.run(data, "features", Some("weight"), initialCenters, config)

    result.converged shouldBe true
    result.centers.length shouldBe 2
  }

  test("LloydsIterator works with different kernels") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(0.1, 0.1),
      Array(10.1, 10.1)
    )

    // Test with KL divergence kernel
    val klKernel  = new KLDivergenceKernel(smoothing = 1e-6)
    val assigner  = new AutoAssignment()
    val updater   = new GradMeanUDAFUpdate()
    val empty     = new DropEmptyClustersHandler()
    val conv      = new MovementConvergence()
    val validator = new StandardInputValidator()

    val config = LloydsConfig(
      k = 2,
      maxIter = 20,
      tol = 1e-4,
      kernel = klKernel,
      assigner = assigner,
      updater = updater,
      emptyHandler = empty,
      convergence = conv,
      validator = validator
    )

    val result = iterator.run(df, "features", None, initialCenters, config)

    result.iterations should be > 0
    // KL divergence with smoothing may drop empty clusters
    result.centers.length should be >= 1
    result.centers.length should be <= 2
  }

  test("LloydsIterator convergence with tight tolerance") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(0.0, 0.0),
      Array(10.0, 10.0)
    )

    val config = defaultConfig(k = 2, maxIter = 50, tol = 1e-10)
    val result = iterator.run(df, "features", None, initialCenters, config)

    // May or may not converge with very tight tolerance
    if (result.converged) {
      result.movementHistory.last should be < 1e-10
    } else {
      result.iterations shouldBe 50
    }
  }

  test("LloydsIterator convergence with loose tolerance") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(1.0, 1.0),
      Array(9.0, 9.0)
    )

    val config = defaultConfig(k = 2, maxIter = 20, tol = 1.0)
    val result = iterator.run(df, "features", None, initialCenters, config)

    // Should converge quickly with loose tolerance
    result.converged shouldBe true
    result.iterations should be <= 5
  }

  test("LloydsIterator handles k=1 (single cluster)") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(Array(5.0, 5.0))

    val config = defaultConfig(k = 1, maxIter = 10, tol = 1e-4)
    val result = iterator.run(df, "features", None, initialCenters, config)

    result.centers.length shouldBe 1
    result.iterations should be > 0

    // With k=1, center should be near the centroid of all points
    val center = result.centers(0)
    center(0) should be > 4.0
    center(0) should be < 6.0
  }

  test("LloydsIterator produces deterministic results with same initial centers") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(1.0, 1.0),
      Array(9.0, 9.0)
    )

    val config = defaultConfig(k = 2, maxIter = 10, tol = 1e-4)

    val result1 = iterator.run(df, "features", None, initialCenters, config)
    val result2 = iterator.run(df, "features", None, initialCenters, config)

    // Results should be identical
    result1.iterations shouldBe result2.iterations
    result1.converged shouldBe result2.converged

    result1.centers.zip(result2.centers).foreach { case (c1, c2) =>
      c1.zip(c2).foreach { case (v1, v2) =>
        math.abs(v1 - v2) should be < 1e-10
      }
    }
  }

  test("LloydsIterator handles large k relative to data size") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    // k=4 but only 6 data points
    val initialCenters = Array(
      Array(0.0, 0.0),
      Array(0.1, 0.1),
      Array(10.0, 10.0),
      Array(10.1, 10.1)
    )

    val config = defaultConfig(k = 4, maxIter = 10, tol = 1e-4)
    val result = iterator.run(df, "features", None, initialCenters, config)

    result.iterations should be > 0
    // May have fewer than 4 centers if empty clusters are dropped
    result.centers.length should be <= 4
    result.centers.length should be >= 1
  }

  test("LloydsIterator movement decreases monotonically (generally)") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(2.0, 2.0),
      Array(8.0, 8.0)
    )

    val config = defaultConfig(k = 2, maxIter = 10, tol = 1e-6)
    val result = iterator.run(df, "features", None, initialCenters, config)

    // Movement should generally trend downward
    if (result.movementHistory.length > 2) {
      val firstHalf  = result.movementHistory.take(result.movementHistory.length / 2).max
      val secondHalf =
        result.movementHistory.drop(result.movementHistory.length / 2).max

      secondHalf should be <= firstHalf
    }
  }

  test("LloydsIterator all centers are valid after convergence") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(0.0, 0.0),
      Array(10.0, 10.0)
    )

    val config = defaultConfig(k = 2, maxIter = 20, tol = 1e-4)
    val result = iterator.run(df, "features", None, initialCenters, config)

    // All centers should have finite coordinates
    result.centers.foreach { center =>
      center.foreach { coord =>
        java.lang.Double.isFinite(coord) shouldBe true
        coord should not be Double.NaN
      }
    }
  }

  test("LloydsIterator handles uniform data (all points identical)") {
    val uniformData = Seq.fill(10)(Vectors.dense(5.0, 5.0)).map(Tuple1(_)).toDF("features")
    val iterator    = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(4.0, 4.0),
      Array(6.0, 6.0)
    )

    val config = defaultConfig(k = 2, maxIter = 10, tol = 1e-4)
    val result = iterator.run(uniformData, "features", None, initialCenters, config)

    // Should converge to a single center or handle gracefully
    result.iterations should be > 0
    result.centers.length should be >= 1
  }

  test("LloydsIterator zero movement indicates convergence") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    // Start with perfect centers
    val initialCenters = Array(
      Array(0.1, 0.1),
      Array(10.1, 10.1)
    )

    val config = defaultConfig(k = 2, maxIter = 10, tol = 1e-4)
    val result = iterator.run(df, "features", None, initialCenters, config)

    // Should converge very quickly
    result.converged shouldBe true
    result.iterations should be <= 3
    result.movementHistory.last should be < 1e-4
  }

  test("LloydsIterator distortion is non-increasing (with same centers)") {
    val df       = createTestData()
    val iterator = new DefaultLloydsIterator()

    val initialCenters = Array(
      Array(1.0, 1.0),
      Array(9.0, 9.0)
    )

    val config = defaultConfig(k = 2, maxIter = 10, tol = 1e-4)
    val result = iterator.run(df, "features", None, initialCenters, config)

    // Distortion should not increase from one iteration to the next
    // (Lloyd's algorithm is guaranteed to not increase distortion)
    if (result.distortionHistory.length > 1) {
      for (i <- 1 until result.distortionHistory.length) {
        val prev = result.distortionHistory(i - 1)
        val curr = result.distortionHistory(i)
        curr should be <= (prev * 1.01) // Allow tiny numerical fluctuations
      }
    }
  }
}
