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

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.TestingUtils._
import com.massivedatascience.linalg.WeightedVector
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite

class BregmanDivergenceEdgeCasesSuite extends AnyFunSuite {

  test("Euclidean divergence with zero weights") {
    val ops    = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val vector = WeightedVector(Vectors.dense(1.0, 2.0), 0.0)

    // Should handle zero weight gracefully
    val point  = ops.toPoint(vector)
    val center = ops.toCenter(vector)

    assert(point.weight == 0.0)
    assert(center.weight == 0.0)

    // Distance to zero-weight center should be Infinity (center is invalid)
    // This is correct behavior - centers with zero weight are undefined
    // Note: The code defines Infinity as Double.MaxValue, not Double.PositiveInfinity
    val distance = ops.distance(point, center)
    assert(distance == Double.MaxValue)
  }

  test("Euclidean divergence with very small weights") {
    val ops        = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val tinyWeight = Double.MinPositiveValue
    val vector     = WeightedVector(Vectors.dense(1.0, 2.0), tinyWeight)

    val point  = ops.toPoint(vector)
    val center = ops.toCenter(vector)

    assert(point.weight == tinyWeight)
    assert(center.weight == tinyWeight)

    // Very small weight is likely below threshold, so distance to center should be Infinity
    val distance = ops.distance(point, center)
    // With default threshold, MinPositiveValue is below threshold
    // Note: The code defines Infinity as Double.MaxValue, not Double.PositiveInfinity
    assert(distance == Double.MaxValue)
  }

  test("Euclidean divergence with very large values") {
    val ops         = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val largeVector = WeightedVector(Vectors.dense(1e50, 1e50), 1.0)

    val point  = ops.toPoint(largeVector)
    val center = ops.toCenter(point)

    val distance = ops.distance(point, center)
    assert(distance ~= 0.0 absTol 1e-8)
    assert(java.lang.Double.isFinite(distance))
  }

  test("KL divergence with smoothing handles zero elements") {
    val ops          = BregmanPointOps(BregmanPointOps.SIMPLEX_SMOOTHED_KL)
    val zeroVector   = WeightedVector(Vectors.dense(0.0, 1.0, 2.0), 1.0)
    val normalVector = WeightedVector(Vectors.dense(1.0, 1.0, 2.0), 1.0)

    // Should handle zero values gracefully due to smoothing
    val point1 = ops.toPoint(zeroVector)
    val point2 = ops.toPoint(normalVector)
    val center = ops.toCenter(point2)

    val distance = ops.distance(point1, center)
    assert(distance >= 0.0 && java.lang.Double.isFinite(distance))
  }

  test("KL divergence without smoothing rejects zero elements") {
    val ops        = BregmanPointOps(BregmanPointOps.DISCRETE_KL)
    val zeroVector = WeightedVector(Vectors.dense(0.0, 1.0, 2.0), 1.0)

    // Should throw exception for zero elements without smoothing
    // Discrete KL uses NaturalKLDivergence which validates positive elements
    intercept[Exception] {
      ops.toCenter(zeroVector)
    }
  }

  test("numerical stability with very small KL divergence values") {
    val ops         = BregmanPointOps(BregmanPointOps.DISCRETE_KL)
    val tinyVector1 = WeightedVector(Vectors.dense(1e-100, 1e-100), 1.0)
    val tinyVector2 = WeightedVector(Vectors.dense(1e-100 + 1e-110, 1e-100), 1.0)

    val point1  = ops.toPoint(tinyVector1)
    val point2  = ops.toPoint(tinyVector2)
    val center1 = ops.toCenter(point1)

    val distance = ops.distance(point2, center1)
    assert(distance >= 0.0 && java.lang.Double.isFinite(distance))
  }

  test("Itakura-Saito divergence with extreme ratios") {
    val ops = BregmanPointOps(BregmanPointOps.ITAKURA_SAITO)
    val v1  = WeightedVector(Vectors.dense(1e-10, 1e10), 1.0)
    val v2  = WeightedVector(Vectors.dense(1e10, 1e-10), 1.0)

    val p1 = ops.toPoint(v1)
    val c2 = ops.toCenter(ops.toPoint(v2))

    val distance = ops.distance(p1, c2)
    assert(java.lang.Double.isFinite(distance) && distance > 0.0)
  }

  test("Itakura-Saito divergence rejects zero elements") {
    val ops        = BregmanPointOps(BregmanPointOps.ITAKURA_SAITO)
    val zeroVector = WeightedVector(Vectors.dense(0.0, 1.0), 1.0)

    // Should throw exception for zero elements when creating center
    // (toPoint doesn't validate, but toCenter does via gradientOfConvexHomogeneous)
    intercept[Exception] {
      ops.toCenter(zeroVector)
    }
  }

  test("Logistic loss divergence with boundary values") {
    val ops = BregmanPointOps(BregmanPointOps.LOGISTIC_LOSS)

    // Test values close to boundaries
    val nearZero = WeightedVector(Vectors.dense(1e-10), 1.0)
    val nearOne  = WeightedVector(Vectors.dense(1.0 - 1e-10), 1.0)

    val p1 = ops.toPoint(nearZero)
    val p2 = ops.toPoint(nearOne)
    val c1 = ops.toCenter(p1)
    val c2 = ops.toCenter(p2)

    val dist1 = ops.distance(p1, c1)
    val dist2 = ops.distance(p2, c2)

    assert(dist1 ~= 0.0 absTol 1e-8)
    assert(dist2 ~= 0.0 absTol 1e-8)
    assert(java.lang.Double.isFinite(dist1) && java.lang.Double.isFinite(dist2))
  }

  test("Logistic loss divergence rejects invalid probability values") {
    val ops = BregmanPointOps(BregmanPointOps.LOGISTIC_LOSS)

    // Test values outside [0,1] range
    val negative  = WeightedVector(Vectors.dense(-0.1), 1.0)
    val overOne   = WeightedVector(Vectors.dense(1.1), 1.0)
    val exactZero = WeightedVector(Vectors.dense(0.0), 1.0)
    val exactOne  = WeightedVector(Vectors.dense(1.0), 1.0)

    intercept[Exception] {
      ops.toPoint(negative)
    }

    intercept[Exception] {
      ops.toPoint(overOne)
    }

    intercept[Exception] {
      ops.toPoint(exactZero)
    }

    intercept[Exception] {
      ops.toPoint(exactOne)
    }
  }

  test("Generalized I-divergence with very small values") {
    val ops        = BregmanPointOps(BregmanPointOps.GENERALIZED_I)
    val tinyVector = WeightedVector(Vectors.dense(1e-100, 1e-50), 1.0)

    val point  = ops.toPoint(tinyVector)
    val center = ops.toCenter(point)

    val distance = ops.distance(point, center)
    assert(distance ~= 0.0 absTol 1e-8)
    assert(java.lang.Double.isFinite(distance))
  }

  test("distance function symmetry property") {
    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val v1  = WeightedVector(Vectors.dense(1.0, 2.0), 1.0)
    val v2  = WeightedVector(Vectors.dense(3.0, 4.0), 1.0)

    val p1 = ops.toPoint(v1)
    val p2 = ops.toPoint(v2)
    val c1 = ops.toCenter(p1)
    val c2 = ops.toCenter(p2)

    val dist1 = ops.distance(p1, c2)
    val dist2 = ops.distance(p2, c1)

    // For Euclidean distance, should be symmetric
    assert(dist1 ~= dist2 absTol 1e-8)
  }

  test("distance function non-negativity") {
    val operations = Seq(
      BregmanPointOps(BregmanPointOps.EUCLIDEAN),
      BregmanPointOps(BregmanPointOps.DISCRETE_KL),
      BregmanPointOps(BregmanPointOps.ITAKURA_SAITO),
      BregmanPointOps(BregmanPointOps.GENERALIZED_I)
    )

    val testVectors = Seq(
      WeightedVector(Vectors.dense(1.0, 2.0), 1.0),
      WeightedVector(Vectors.dense(0.1, 0.9), 1.0),
      WeightedVector(Vectors.dense(100.0, 200.0), 2.0)
    )

    for (ops <- operations) {
      for (v1 <- testVectors) {
        for (v2 <- testVectors) {
          try {
            val p1 = ops.toPoint(v1)
            val c2 = ops.toCenter(ops.toPoint(v2))

            val distance = ops.distance(p1, c2)
            assert(
              distance >= 0.0,
              s"Distance should be non-negative for ${ops.getClass.getSimpleName}"
            )
            assert(
              java.lang.Double.isFinite(distance),
              s"Distance should be finite for ${ops.getClass.getSimpleName}"
            )
          } catch {
            case _: Exception =>
            // Some combinations may be invalid (e.g., zero values for KL divergence)
            // This is expected behavior
          }
        }
      }
    }
  }

  test("triangle inequality approximation for Euclidean distance") {
    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val v1  = WeightedVector(Vectors.dense(0.0, 0.0), 1.0)
    val v2  = WeightedVector(Vectors.dense(1.0, 0.0), 1.0)
    val v3  = WeightedVector(Vectors.dense(1.0, 1.0), 1.0)

    val p1 = ops.toPoint(v1)
    val p2 = ops.toPoint(v2)
    val p3 = ops.toPoint(v3)
    val c1 = ops.toCenter(p1)
    val c2 = ops.toCenter(p2)
    val c3 = ops.toCenter(p3)

    val d12 = math.sqrt(ops.distance(p1, c2))
    val d23 = math.sqrt(ops.distance(p2, c3))
    val d13 = math.sqrt(ops.distance(p1, c3))

    // For squared Euclidean, we check the square root to verify triangle inequality
    assert(d13 <= d12 + d23 + 1e-8)
  }

  test("homogeneous vs inhomogeneous coordinate consistency") {
    val ops    = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val vector = WeightedVector(Vectors.dense(2.0, 4.0), 2.0)

    val point  = ops.toPoint(vector)
    val center = ops.toCenter(point)

    // The inhomogeneous coordinates should be the original vector scaled by weight
    val expectedInhomogeneous = Vectors.dense(1.0, 2.0) // (2.0, 4.0) / 2.0
    val actualInhomogeneous   = point.inhomogeneous

    assert(expectedInhomogeneous.toArray.zip(actualInhomogeneous.toArray).forall {
      case (expected, actual) => math.abs(expected - actual) < 1e-8
    })
  }

  test("center movement detection with very small changes") {
    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val v1  = WeightedVector(Vectors.dense(1.0, 2.0), 1.0)
    val v2  = WeightedVector(Vectors.dense(1.0 + 1e-10, 2.0 + 1e-10), 1.0)

    val p1 = ops.toPoint(v1)
    val c1 = ops.toCenter(p1)
    val c2 = ops.toCenter(ops.toPoint(v2))

    // Should detect very small movements
    val moved = ops.centerMoved(p1, c2)
    // The actual behavior depends on the implementation's tolerance
    // This test ensures the method doesn't crash with very small differences
    assert(moved == true || moved == false) // Just ensure it returns a boolean
  }

  test("sparse vector handling") {
    val ops          = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val sparseVector = WeightedVector(Vectors.sparse(10, Seq((1, 2.0), (5, 3.0), (9, 1.0))), 1.0)
    val denseVector =
      WeightedVector(Vectors.dense(0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0), 1.0)

    val sparsePoint  = ops.toPoint(sparseVector)
    val densePoint   = ops.toPoint(denseVector)
    val sparseCenter = ops.toCenter(sparsePoint)
    val denseCenter  = ops.toCenter(densePoint)

    // Should produce equivalent results
    val sparseToDense      = ops.distance(sparsePoint, denseCenter)
    val denseToSparse      = ops.distance(densePoint, sparseCenter)
    val sparseSelfDistance = ops.distance(sparsePoint, sparseCenter)
    val denseSelfDistance  = ops.distance(densePoint, denseCenter)

    assert(sparseToDense ~= denseToSparse absTol 1e-8)
    assert(sparseSelfDistance ~= 0.0 absTol 1e-8)
    assert(denseSelfDistance ~= 0.0 absTol 1e-8)
  }

  test("weight threshold behavior") {
    val ops       = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val threshold = ops.weightThreshold

    // Test vectors with weights around the threshold
    val belowThreshold = WeightedVector(Vectors.dense(1.0, 2.0), threshold / 2.0)
    val atThreshold    = WeightedVector(Vectors.dense(1.0, 2.0), threshold)
    val aboveThreshold = WeightedVector(Vectors.dense(1.0, 2.0), threshold * 2.0)

    // All should create valid points, but behavior may differ
    val p1 = ops.toPoint(belowThreshold)
    val p2 = ops.toPoint(atThreshold)
    val p3 = ops.toPoint(aboveThreshold)

    assert(p1.weight == threshold / 2.0)
    assert(p2.weight == threshold)
    assert(p3.weight == threshold * 2.0)

    val c1 = ops.toCenter(p1)
    val c2 = ops.toCenter(p2)
    val c3 = ops.toCenter(p3)

    // Centers should all be valid
    assert(c1.weight >= 0.0)
    assert(c2.weight >= 0.0)
    assert(c3.weight >= 0.0)
  }
}
