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
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests for mathematical accuracy of Bregman kernel computations.
  *
  * These tests verify:
  *   - Distance/divergence functions compute correctly
  *   - Gradient computations are accurate
  *   - Inverse gradient mappings are correct (grad ∘ invGrad = identity)
  *   - Mathematical properties hold (non-negativity, self-distance = 0, etc.)
  */
class BregmanKernelAccuracySuite extends AnyFunSuite with Matchers {

  // Tolerances for different divergences (some are more numerically sensitive)
  private val defaultTol = 1e-10
  private val relaxedTol = 1e-8 // For log-based divergences
  private val looseTol   = 1e-6 // For inverse gradient round-trips with exp/log

  // Helper: check if two vectors are approximately equal
  private def vectorsApproxEqual(
      v1: org.apache.spark.ml.linalg.Vector,
      v2: org.apache.spark.ml.linalg.Vector,
      tol: Double
  ): Boolean = {
    val arr1 = v1.toArray
    val arr2 = v2.toArray
    if (arr1.length != arr2.length) return false
    arr1.zip(arr2).forall { case (x1, x2) => math.abs(x1 - x2) < tol }
  }

  // Helper: assert vectors are approximately equal
  private def assertVectorsEqual(
      v1: org.apache.spark.ml.linalg.Vector,
      v2: org.apache.spark.ml.linalg.Vector,
      tol: Double,
      msg: String
  ): Unit = {
    val arr1 = v1.toArray
    val arr2 = v2.toArray
    arr1.length shouldBe arr2.length
    arr1.zip(arr2).zipWithIndex.foreach { case ((x1, x2), i) =>
      withClue(s"$msg: mismatch at index $i: $x1 != $x2") {
        math.abs(x1 - x2) should be < tol
      }
    }
  }

  //
  // ========== Squared Euclidean Kernel Tests ==========
  //

  test("SquaredEuclidean: divergence is 0.5 * ||x - mu||^2") {
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)
    val mu     = Vectors.dense(4.0, 5.0, 6.0)

    val expected = 0.5 * ((1 - 4) * (1 - 4) + (2 - 5) * (2 - 5) + (3 - 6) * (3 - 6))
    val actual   = kernel.divergence(x, mu)

    math.abs(actual - expected) should be < defaultTol
    actual shouldBe 13.5
  }

  test("SquaredEuclidean: gradient is x") {
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)

    val grad = kernel.grad(x)

    assertVectorsEqual(grad, x, defaultTol, "SE gradient should equal x")
  }

  test("SquaredEuclidean: invGrad is identity") {
    val kernel = new SquaredEuclideanKernel()
    val theta  = Vectors.dense(1.0, 2.0, 3.0)

    val x = kernel.invGrad(theta)

    assertVectorsEqual(x, theta, defaultTol, "SE invGrad should equal theta")
  }

  test("SquaredEuclidean: grad ∘ invGrad = identity") {
    val kernel = new SquaredEuclideanKernel()
    val theta  = Vectors.dense(1.0, 2.0, 3.0)

    val recovered = kernel.grad(kernel.invGrad(theta))

    assertVectorsEqual(recovered, theta, defaultTol, "SE grad ∘ invGrad should be identity")
  }

  test("SquaredEuclidean: self-distance is zero") {
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)

    val dist = kernel.divergence(x, x)

    math.abs(dist - 0.0) should be < defaultTol
  }

  test("SquaredEuclidean: divergence is symmetric") {
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)
    val y      = Vectors.dense(4.0, 5.0, 6.0)

    val d1 = kernel.divergence(x, y)
    val d2 = kernel.divergence(y, x)

    math.abs(d1 - d2) should be < defaultTol
  }

  test("SquaredEuclidean: divergence is non-negative") {
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)
    val y      = Vectors.dense(4.0, 5.0, 6.0)

    val dist = kernel.divergence(x, y)

    dist should be >= 0.0
  }

  test("SquaredEuclidean: validate accepts finite values") {
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(1.0, -2.0, 3.0, 0.0)

    kernel.validate(x) shouldBe true
  }

  test("SquaredEuclidean: validate rejects NaN") {
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(1.0, Double.NaN, 3.0)

    kernel.validate(x) shouldBe false
  }

  test("SquaredEuclidean: validate rejects Inf") {
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(1.0, Double.PositiveInfinity, 3.0)

    kernel.validate(x) shouldBe false
  }

  //
  // ========== KL Divergence Kernel Tests ==========
  //

  test("KL: divergence is sum of x_i * log(x_i / mu_i)") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.3, 0.7)
    val mu     = Vectors.dense(0.5, 0.5)

    val eps      = 1e-10
    val expected =
      (0.3 + eps) * math.log((0.3 + eps) / (0.5 + eps)) + (0.7 + eps) * math.log(
        (0.7 + eps) / (0.5 + eps)
      )
    val actual   = kernel.divergence(x, mu)

    math.abs(actual - expected) should be < relaxedTol
  }

  test("KL: gradient is [log(x_i) + 1]") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, 2.0, 3.0)

    val grad     = kernel.grad(x)
    val eps      = 1e-10
    val expected =
      Vectors.dense(math.log(1.0 + eps) + 1.0, math.log(2.0 + eps) + 1.0, math.log(3.0 + eps) + 1.0)

    assertVectorsEqual(grad, expected, relaxedTol, "KL gradient mismatch")
  }

  test("KL: invGrad is [exp(theta_i - 1)]") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)
    val theta  = Vectors.dense(0.0, 1.0, 2.0)

    val x        = kernel.invGrad(theta)
    val expected = Vectors.dense(math.exp(-1.0), math.exp(0.0), math.exp(1.0))

    assertVectorsEqual(x, expected, relaxedTol, "KL invGrad mismatch")
  }

  test("KL: grad ∘ invGrad ≈ identity") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)
    val theta  = Vectors.dense(0.0, 1.0, 2.0)

    val recovered = kernel.grad(kernel.invGrad(theta))

    assertVectorsEqual(recovered, theta, looseTol, "KL grad ∘ invGrad should be near identity")
  }

  test("KL: self-distance is zero") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.3, 0.7)

    val dist = kernel.divergence(x, x)

    math.abs(dist - 0.0) should be < relaxedTol
  }

  test("KL: divergence is non-negative") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.3, 0.7)
    val mu     = Vectors.dense(0.5, 0.5)

    val dist = kernel.divergence(x, mu)

    dist should be >= 0.0
  }

  test("KL: divergence is NOT symmetric") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.3, 0.7)
    val mu     = Vectors.dense(0.5, 0.5)

    val d1 = kernel.divergence(x, mu)
    val d2 = kernel.divergence(mu, x)

    // KL is asymmetric
    d1 should not be d2
  }

  test("KL: validate accepts non-negative values") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.3, 0.7, 0.0)

    kernel.validate(x) shouldBe true
  }

  test("KL: validate rejects negative values") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.3, -0.1, 0.7)

    kernel.validate(x) shouldBe false
  }

  //
  // ========== Itakura-Saito Kernel Tests ==========
  //

  test("ItakuraSaito: divergence is sum of (x_i/mu_i - log(x_i/mu_i) - 1)") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, 2.0)
    val mu     = Vectors.dense(0.5, 1.0)

    val eps      = 1e-10
    val expected = (1.0 + eps) / (0.5 + eps) - math.log((1.0 + eps) / (0.5 + eps)) - 1.0 +
      (2.0 + eps) / (1.0 + eps) - math.log((2.0 + eps) / (1.0 + eps)) - 1.0
    val actual   = kernel.divergence(x, mu)

    math.abs(actual - expected) should be < relaxedTol
  }

  test("ItakuraSaito: gradient is [-1/x_i]") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, 2.0, 4.0)

    val grad     = kernel.grad(x)
    val eps      = 1e-10
    val expected = Vectors.dense(-1.0 / (1.0 + eps), -1.0 / (2.0 + eps), -1.0 / (4.0 + eps))

    assertVectorsEqual(grad, expected, relaxedTol, "IS gradient mismatch")
  }

  test("ItakuraSaito: invGrad is [-1/theta_i]") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)
    val theta  = Vectors.dense(-1.0, -2.0, -0.5)

    val x        = kernel.invGrad(theta)
    val expected = Vectors.dense(1.0, 0.5, 2.0)

    assertVectorsEqual(x, expected, relaxedTol, "IS invGrad mismatch")
  }

  test("ItakuraSaito: grad ∘ invGrad ≈ identity") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)
    val theta  = Vectors.dense(-1.0, -2.0, -0.5)

    val recovered = kernel.grad(kernel.invGrad(theta))

    assertVectorsEqual(recovered, theta, looseTol, "IS grad ∘ invGrad should be near identity")
  }

  test("ItakuraSaito: self-distance is zero") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, 2.0)

    val dist = kernel.divergence(x, x)

    math.abs(dist - 0.0) should be < relaxedTol
  }

  test("ItakuraSaito: divergence is non-negative") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, 2.0)
    val mu     = Vectors.dense(0.5, 1.0)

    val dist = kernel.divergence(x, mu)

    dist should be >= 0.0
  }

  test("ItakuraSaito: validate accepts positive values") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, 2.0, 3.0)

    kernel.validate(x) shouldBe true
  }

  test("ItakuraSaito: validate rejects zero") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, 0.0, 3.0)

    kernel.validate(x) shouldBe false
  }

  test("ItakuraSaito: validate rejects negative values") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, -1.0, 3.0)

    kernel.validate(x) shouldBe false
  }

  //
  // ========== Generalized I-Divergence Kernel Tests ==========
  //

  test("GeneralizedI: divergence is sum of (x_i log(x_i/mu_i) - x_i + mu_i)") {
    val kernel = new GeneralizedIDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, 2.0)
    val mu     = Vectors.dense(0.5, 1.0)

    val eps      = 1e-10
    val expected =
      (1.0 + eps) * math.log((1.0 + eps) / (0.5 + eps)) - (1.0 + eps) + (0.5 + eps) +
        (2.0 + eps) * math.log((2.0 + eps) / (1.0 + eps)) - (2.0 + eps) + (1.0 + eps)
    val actual   = kernel.divergence(x, mu)

    math.abs(actual - expected) should be < relaxedTol
  }

  test("GeneralizedI: gradient is [log(x_i)]") {
    val kernel = new GeneralizedIDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, 2.0, 3.0)

    val grad     = kernel.grad(x)
    val eps      = 1e-10
    val expected = Vectors.dense(math.log(1.0 + eps), math.log(2.0 + eps), math.log(3.0 + eps))

    assertVectorsEqual(grad, expected, relaxedTol, "GeneralizedI gradient mismatch")
  }

  test("GeneralizedI: invGrad is [exp(theta_i)]") {
    val kernel = new GeneralizedIDivergenceKernel(smoothing = 1e-10)
    val theta  = Vectors.dense(0.0, 1.0, 2.0)

    val x        = kernel.invGrad(theta)
    val expected = Vectors.dense(1.0, math.E, math.E * math.E)

    assertVectorsEqual(x, expected, relaxedTol, "GeneralizedI invGrad mismatch")
  }

  test("GeneralizedI: grad ∘ invGrad ≈ identity") {
    val kernel = new GeneralizedIDivergenceKernel(smoothing = 1e-10)
    val theta  = Vectors.dense(0.0, 1.0, 2.0)

    val recovered = kernel.grad(kernel.invGrad(theta))

    assertVectorsEqual(
      recovered,
      theta,
      looseTol,
      "GeneralizedI grad ∘ invGrad should be near identity"
    )
  }

  test("GeneralizedI: self-distance is zero") {
    val kernel = new GeneralizedIDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, 2.0)

    val dist = kernel.divergence(x, x)

    math.abs(dist - 0.0) should be < relaxedTol
  }

  test("GeneralizedI: divergence is non-negative") {
    val kernel = new GeneralizedIDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, 2.0)
    val mu     = Vectors.dense(0.5, 1.0)

    val dist = kernel.divergence(x, mu)

    dist should be >= 0.0
  }

  test("GeneralizedI: validate accepts non-negative values") {
    val kernel = new GeneralizedIDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.0, 1.0, 2.0)

    kernel.validate(x) shouldBe true
  }

  test("GeneralizedI: validate rejects negative values") {
    val kernel = new GeneralizedIDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, -1.0, 2.0)

    kernel.validate(x) shouldBe false
  }

  //
  // ========== Logistic Loss Kernel Tests ==========
  //

  test("LogisticLoss: divergence for probability vectors") {
    val kernel = new LogisticLossKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.3)
    val mu     = Vectors.dense(0.5)

    val eps      = 1e-10
    val xi       = math.max(eps, math.min(1.0 - eps, 0.3))
    val mui      = math.max(eps, math.min(1.0 - eps, 0.5))
    val expected =
      xi * math.log(xi / mui) + (1.0 - xi) * math.log((1.0 - xi) / (1.0 - mui))
    val actual   = kernel.divergence(x, mu)

    math.abs(actual - expected) should be < relaxedTol
  }

  test("LogisticLoss: gradient is [log(x_i / (1-x_i))]") {
    val kernel = new LogisticLossKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.3, 0.7)

    val grad     = kernel.grad(x)
    val eps      = 1e-10
    val x1       = math.max(eps, math.min(1.0 - eps, 0.3))
    val x2       = math.max(eps, math.min(1.0 - eps, 0.7))
    val expected =
      Vectors.dense(math.log(x1 / (1.0 - x1)), math.log(x2 / (1.0 - x2)))

    assertVectorsEqual(grad, expected, relaxedTol, "LogisticLoss gradient mismatch")
  }

  test("LogisticLoss: invGrad is sigmoid [1 / (1 + exp(-theta_i))]") {
    val kernel = new LogisticLossKernel(smoothing = 1e-10)
    val theta  = Vectors.dense(0.0, 1.0, -1.0)

    val x        = kernel.invGrad(theta)
    val expected = Vectors.dense(
      1.0 / (1.0 + math.exp(0.0)),
      1.0 / (1.0 + math.exp(-1.0)),
      1.0 / (1.0 + math.exp(1.0))
    )

    assertVectorsEqual(x, expected, relaxedTol, "LogisticLoss invGrad mismatch")
  }

  test("LogisticLoss: grad ∘ invGrad ≈ identity") {
    val kernel = new LogisticLossKernel(smoothing = 1e-10)
    val theta  = Vectors.dense(0.0, 1.0, -1.0)

    val recovered = kernel.grad(kernel.invGrad(theta))

    // Logistic loss with smoothing may have larger roundtrip error
    assertVectorsEqual(
      recovered,
      theta,
      looseTol,
      "LogisticLoss grad ∘ invGrad should be near identity"
    )
  }

  test("LogisticLoss: self-distance is zero") {
    val kernel = new LogisticLossKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.3)

    val dist = kernel.divergence(x, x)

    math.abs(dist - 0.0) should be < relaxedTol
  }

  test("LogisticLoss: divergence is non-negative") {
    val kernel = new LogisticLossKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.3)
    val mu     = Vectors.dense(0.5)

    val dist = kernel.divergence(x, mu)

    dist should be >= 0.0
  }

  test("LogisticLoss: validate accepts values in [0, 1]") {
    val kernel = new LogisticLossKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.0, 0.5, 1.0)

    kernel.validate(x) shouldBe true
  }

  test("LogisticLoss: validate rejects values > 1") {
    val kernel = new LogisticLossKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.5, 1.1)

    kernel.validate(x) shouldBe false
  }

  test("LogisticLoss: validate rejects negative values") {
    val kernel = new LogisticLossKernel(smoothing = 1e-10)
    val x      = Vectors.dense(-0.1, 0.5)

    kernel.validate(x) shouldBe false
  }

  //
  // ========== L1 (Manhattan) Kernel Tests ==========
  //

  test("L1: divergence is sum of |x_i - mu_i|") {
    val kernel = new L1Kernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)
    val mu     = Vectors.dense(4.0, 1.0, 6.0)

    val expected = math.abs(1.0 - 4.0) + math.abs(2.0 - 1.0) + math.abs(3.0 - 6.0)
    val actual   = kernel.divergence(x, mu)

    math.abs(actual - expected) should be < defaultTol
    actual shouldBe 7.0
  }

  test("L1: gradient is sign function") {
    val kernel = new L1Kernel()
    val x      = Vectors.dense(1.0, -2.0, 0.0, 3.0)

    val grad     = kernel.grad(x)
    val expected = Vectors.dense(1.0, -1.0, 0.0, 1.0)

    assertVectorsEqual(grad, expected, defaultTol, "L1 gradient mismatch")
  }

  test("L1: invGrad is identity (placeholder for K-Medians)") {
    val kernel = new L1Kernel()
    val theta  = Vectors.dense(1.0, -1.0, 0.0)

    val x = kernel.invGrad(theta)

    // L1 invGrad is not well-defined, returns theta as identity
    assertVectorsEqual(x, theta, defaultTol, "L1 invGrad should return theta")
  }

  test("L1: self-distance is zero") {
    val kernel = new L1Kernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)

    val dist = kernel.divergence(x, x)

    math.abs(dist - 0.0) should be < defaultTol
  }

  test("L1: divergence is symmetric") {
    val kernel = new L1Kernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)
    val y      = Vectors.dense(4.0, 5.0, 6.0)

    val d1 = kernel.divergence(x, y)
    val d2 = kernel.divergence(y, x)

    math.abs(d1 - d2) should be < defaultTol
  }

  test("L1: divergence is non-negative") {
    val kernel = new L1Kernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)
    val y      = Vectors.dense(4.0, 5.0, 6.0)

    val dist = kernel.divergence(x, y)

    dist should be >= 0.0
  }

  test("L1: validate accepts all finite values") {
    val kernel = new L1Kernel()
    val x      = Vectors.dense(1.0, -2.0, 0.0, 3.0)

    kernel.validate(x) shouldBe true
  }

  //
  // ========== Spherical (Cosine) Kernel Tests ==========
  //

  test("Spherical: divergence is 1 - cos(x, mu)") {
    val kernel = new SphericalKernel()
    val x      = Vectors.dense(1.0, 0.0)
    val mu     = Vectors.dense(0.0, 1.0)

    // Perpendicular vectors have cosine = 0, so distance = 1
    val actual = kernel.divergence(x, mu)

    math.abs(actual - 1.0) should be < defaultTol
  }

  test("Spherical: same direction has distance 0") {
    val kernel = new SphericalKernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)
    val mu     = Vectors.dense(2.0, 4.0, 6.0) // Same direction, different magnitude

    val actual = kernel.divergence(x, mu)

    math.abs(actual) should be < defaultTol
  }

  test("Spherical: opposite direction has distance 2") {
    val kernel = new SphericalKernel()
    val x      = Vectors.dense(1.0, 0.0)
    val mu     = Vectors.dense(-1.0, 0.0)

    val actual = kernel.divergence(x, mu)

    math.abs(actual - 2.0) should be < defaultTol
  }

  test("Spherical: gradient normalizes vector") {
    val kernel = new SphericalKernel()
    val x      = Vectors.dense(3.0, 4.0)

    val grad = kernel.grad(x)

    // Should be (0.6, 0.8) - the unit vector
    val expected = Vectors.dense(0.6, 0.8)
    assertVectorsEqual(grad, expected, defaultTol, "Spherical gradient should normalize")

    // Check norm is 1
    val norm = math.sqrt(grad.toArray.map(v => v * v).sum)
    math.abs(norm - 1.0) should be < defaultTol
  }

  test("Spherical: invGrad normalizes vector") {
    val kernel = new SphericalKernel()
    val theta  = Vectors.dense(5.0, 12.0)

    val x = kernel.invGrad(theta)

    // Should be (5/13, 12/13) - the unit vector
    val expected = Vectors.dense(5.0 / 13.0, 12.0 / 13.0)
    assertVectorsEqual(x, expected, defaultTol, "Spherical invGrad should normalize")

    // Check norm is 1
    val norm = math.sqrt(x.toArray.map(v => v * v).sum)
    math.abs(norm - 1.0) should be < defaultTol
  }

  test("Spherical: grad ∘ invGrad produces unit vector") {
    val kernel = new SphericalKernel()
    val theta  = Vectors.dense(1.0, 2.0, 3.0)

    val recovered = kernel.grad(kernel.invGrad(theta))

    // Both grad and invGrad normalize, so result should be unit vector in same direction
    val norm = math.sqrt(recovered.toArray.map(v => v * v).sum)
    math.abs(norm - 1.0) should be < defaultTol

    // Direction should be preserved (same as normalized theta)
    val thetaNorm = math.sqrt(theta.toArray.map(v => v * v).sum)
    val expected  = Vectors.dense(theta.toArray.map(_ / thetaNorm))
    assertVectorsEqual(
      recovered,
      expected,
      defaultTol,
      "Spherical grad ∘ invGrad should preserve direction"
    )
  }

  test("Spherical: self-distance is zero") {
    val kernel = new SphericalKernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)

    val dist = kernel.divergence(x, x)

    math.abs(dist) should be < defaultTol
  }

  test("Spherical: divergence is symmetric") {
    val kernel = new SphericalKernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)
    val y      = Vectors.dense(4.0, 5.0, 6.0)

    val d1 = kernel.divergence(x, y)
    val d2 = kernel.divergence(y, x)

    math.abs(d1 - d2) should be < defaultTol
  }

  test("Spherical: divergence is non-negative") {
    val kernel = new SphericalKernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)
    val y      = Vectors.dense(4.0, 5.0, 6.0)

    val dist = kernel.divergence(x, y)

    dist should be >= 0.0
  }

  test("Spherical: divergence is bounded by [0, 2]") {
    val kernel = new SphericalKernel()

    // Test various vector pairs
    val testCases = Seq(
      (Vectors.dense(1.0, 0.0), Vectors.dense(1.0, 0.0)),  // Same direction
      (Vectors.dense(1.0, 0.0), Vectors.dense(0.0, 1.0)),  // Perpendicular
      (Vectors.dense(1.0, 0.0), Vectors.dense(-1.0, 0.0)), // Opposite
      (Vectors.dense(1.0, 2.0, 3.0), Vectors.dense(4.0, 5.0, 6.0))
    )

    testCases.foreach { case (x, y) =>
      val dist = kernel.divergence(x, y)
      dist should be >= 0.0
      dist should be <= 2.0
    }
  }

  test("Spherical: scale invariance - magnitude doesn't affect distance") {
    val kernel = new SphericalKernel()
    val x1     = Vectors.dense(1.0, 2.0, 3.0)
    val x2     = Vectors.dense(10.0, 20.0, 30.0) // Same direction, 10x magnitude
    val mu     = Vectors.dense(4.0, 5.0, 6.0)

    val d1 = kernel.divergence(x1, mu)
    val d2 = kernel.divergence(x2, mu)

    math.abs(d1 - d2) should be < defaultTol
  }

  test("Spherical: validate accepts non-zero finite values") {
    val kernel = new SphericalKernel()
    val x      = Vectors.dense(1.0, -2.0, 3.0)

    kernel.validate(x) shouldBe true
  }

  test("Spherical: validate rejects zero vector") {
    val kernel = new SphericalKernel()
    val x      = Vectors.dense(0.0, 0.0, 0.0)

    kernel.validate(x) shouldBe false
  }

  test("Spherical: validate rejects NaN") {
    val kernel = new SphericalKernel()
    val x      = Vectors.dense(1.0, Double.NaN, 3.0)

    kernel.validate(x) shouldBe false
  }

  test("Spherical: validate rejects Inf") {
    val kernel = new SphericalKernel()
    val x      = Vectors.dense(1.0, Double.PositiveInfinity, 3.0)

    kernel.validate(x) shouldBe false
  }

  test("Spherical: known cosine similarity values") {
    val kernel = new SphericalKernel()

    // 45-degree angle: cos(45°) = √2/2 ≈ 0.707, distance ≈ 0.293
    val x1        = Vectors.dense(1.0, 0.0)
    val y1        = Vectors.dense(1.0, 1.0)
    val d1        = kernel.divergence(x1, y1)
    val expected1 = 1.0 - math.sqrt(2) / 2
    math.abs(d1 - expected1) should be < relaxedTol

    // 60-degree angle: cos(60°) = 0.5, distance = 0.5
    val x2 = Vectors.dense(1.0, 0.0)
    val y2 = Vectors.dense(0.5, math.sqrt(3) / 2)
    val d2 = kernel.divergence(x2, y2)
    math.abs(d2 - 0.5) should be < relaxedTol
  }

  test("Spherical: equivalent to squared Euclidean on unit sphere (scaled by 2)") {
    val spherical = new SphericalKernel()
    val euclidean = new SquaredEuclideanKernel()

    // For unit vectors: ||x - y||² = 2(1 - x·y) = 2 * cosineDistance
    val x = Vectors.dense(0.6, 0.8) // Already unit vector
    val y = Vectors.dense(0.8, 0.6) // Already unit vector

    val cosineDist    = spherical.divergence(x, y)
    val euclideanDist = euclidean.divergence(x, y)

    // euclideanDist = 0.5 * ||x-y||² = 0.5 * 2 * (1 - x·y) = cosineDist
    math.abs(euclideanDist - cosineDist) should be < relaxedTol
  }

  //
  // ========== Cross-Kernel Property Tests ==========
  //

  test("All kernels: self-distance is zero") {
    val kernels = Seq(
      new SquaredEuclideanKernel(),
      new KLDivergenceKernel(1e-10),
      new ItakuraSaitoKernel(1e-10),
      new GeneralizedIDivergenceKernel(1e-10),
      new LogisticLossKernel(1e-10),
      new L1Kernel(),
      new SphericalKernel()
    )

    val testVectors = Map(
      "SE"           -> Vectors.dense(1.0, 2.0, 3.0),
      "KL"           -> Vectors.dense(0.3, 0.7),
      "IS"           -> Vectors.dense(1.0, 2.0),
      "GeneralizedI" -> Vectors.dense(1.0, 2.0),
      "Logistic"     -> Vectors.dense(0.3, 0.7),
      "L1"           -> Vectors.dense(1.0, 2.0, 3.0),
      "Spherical"    -> Vectors.dense(1.0, 2.0, 3.0)
    )

    kernels.zip(testVectors.values).foreach { case (kernel, x) =>
      val dist = kernel.divergence(x, x)
      withClue(s"${kernel.name}: self-distance should be 0") {
        math.abs(dist) should be < relaxedTol
      }
    }
  }

  test("All kernels: divergence is non-negative") {
    val testCases = Seq(
      (new SquaredEuclideanKernel(), Vectors.dense(1.0, 2.0), Vectors.dense(3.0, 4.0)),
      (new KLDivergenceKernel(1e-10), Vectors.dense(0.3, 0.7), Vectors.dense(0.5, 0.5)),
      (new ItakuraSaitoKernel(1e-10), Vectors.dense(1.0, 2.0), Vectors.dense(0.5, 1.0)),
      (new GeneralizedIDivergenceKernel(1e-10), Vectors.dense(1.0, 2.0), Vectors.dense(0.5, 1.0)),
      (new LogisticLossKernel(1e-10), Vectors.dense(0.3), Vectors.dense(0.5)),
      (new L1Kernel(), Vectors.dense(1.0, 2.0), Vectors.dense(3.0, 4.0)),
      (new SphericalKernel(), Vectors.dense(1.0, 2.0), Vectors.dense(3.0, 4.0))
    )

    testCases.foreach { case (kernel, x, mu) =>
      val dist = kernel.divergence(x, mu)
      withClue(s"${kernel.name}: divergence should be non-negative") {
        dist should be >= 0.0
      }
      withClue(s"${kernel.name}: divergence should be finite") {
        java.lang.Double.isFinite(dist) shouldBe true
      }
    }
  }

  test("All Bregman kernels: grad ∘ invGrad ≈ identity (roundtrip)") {
    val testCases = Seq(
      (new SquaredEuclideanKernel(), Vectors.dense(1.0, 2.0, 3.0), defaultTol),
      (new KLDivergenceKernel(1e-10), Vectors.dense(0.0, 1.0, 2.0), looseTol),
      (new ItakuraSaitoKernel(1e-10), Vectors.dense(-1.0, -2.0, -0.5), looseTol),
      (new GeneralizedIDivergenceKernel(1e-10), Vectors.dense(0.0, 1.0, 2.0), looseTol),
      (new LogisticLossKernel(1e-10), Vectors.dense(0.0, 1.0, -1.0), looseTol)
    )

    testCases.foreach { case (kernel, theta, tol) =>
      val recovered = kernel.grad(kernel.invGrad(theta))
      assertVectorsEqual(
        recovered,
        theta,
        tol,
        s"${kernel.name}: grad ∘ invGrad should be near identity"
      )
    }
  }

  test("Symmetric kernels: d(x, y) = d(y, x)") {
    val symmetricKernels = Seq(
      (new SquaredEuclideanKernel(), Vectors.dense(1.0, 2.0), Vectors.dense(3.0, 4.0)),
      (new L1Kernel(), Vectors.dense(1.0, 2.0), Vectors.dense(3.0, 4.0)),
      (new SphericalKernel(), Vectors.dense(1.0, 2.0), Vectors.dense(3.0, 4.0))
    )

    symmetricKernels.foreach { case (kernel, x, y) =>
      val d1 = kernel.divergence(x, y)
      val d2 = kernel.divergence(y, x)
      withClue(s"${kernel.name} should be symmetric") {
        math.abs(d1 - d2) should be < defaultTol
      }
    }
  }

  test("Asymmetric kernels: d(x, y) ≠ d(y, x)") {
    val asymmetricKernels = Seq(
      (new KLDivergenceKernel(1e-10), Vectors.dense(0.3, 0.7), Vectors.dense(0.5, 0.5)),
      (new ItakuraSaitoKernel(1e-10), Vectors.dense(1.0, 2.0), Vectors.dense(0.5, 1.0)),
      (new GeneralizedIDivergenceKernel(1e-10), Vectors.dense(1.0, 2.0), Vectors.dense(0.5, 1.0))
    )

    asymmetricKernels.foreach { case (kernel, x, y) =>
      val d1 = kernel.divergence(x, y)
      val d2 = kernel.divergence(y, x)
      withClue(s"${kernel.name} should be asymmetric") {
        d1 should not equal d2
      }
    }
  }

  //
  // ========== Numerical Stability Tests ==========
  //

  test("All kernels handle vectors with zeros (with smoothing)") {
    val testCases = Seq(
      (new SquaredEuclideanKernel(), Vectors.dense(0.0, 1.0, 0.0)),
      (new KLDivergenceKernel(1e-6), Vectors.dense(0.0, 1.0)),           // Needs smoothing
      (new GeneralizedIDivergenceKernel(1e-6), Vectors.dense(0.0, 1.0)), // Needs smoothing
      (new L1Kernel(), Vectors.dense(0.0, 1.0, 0.0))
    )

    testCases.foreach { case (kernel, x) =>
      withClue(s"${kernel.name} should handle zeros with smoothing") {
        noException should be thrownBy {
          kernel.divergence(x, x)
          kernel.grad(x)
        }
      }
    }
  }

  test("Log-based kernels handle large values") {
    val testCases = Seq(
      (new KLDivergenceKernel(1e-10), Vectors.dense(1e6, 1e7)),
      (new ItakuraSaitoKernel(1e-10), Vectors.dense(1e6, 1e7)),
      (new GeneralizedIDivergenceKernel(1e-10), Vectors.dense(1e6, 1e7))
    )

    testCases.foreach { case (kernel, x) =>
      val grad = kernel.grad(x)
      val dist = kernel.divergence(x, x)

      grad.toArray.foreach { v =>
        withClue(s"${kernel.name} gradient should be finite") {
          java.lang.Double.isFinite(v) shouldBe true
        }
      }
      withClue(s"${kernel.name} divergence should be finite") {
        java.lang.Double.isFinite(dist) shouldBe true
      }
    }
  }

  test("All kernels reject or handle very small values appropriately") {
    val smallVal = 1e-100

    // Squared Euclidean and L1 should handle tiny values
    val seKernel = new SquaredEuclideanKernel()
    val l1Kernel = new L1Kernel()

    noException should be thrownBy {
      seKernel.divergence(Vectors.dense(smallVal, 1.0), Vectors.dense(1.0, 1.0))
      l1Kernel.divergence(Vectors.dense(smallVal, 1.0), Vectors.dense(1.0, 1.0))
    }

    // Log-based divergences with smoothing should handle tiny values
    val klKernel = new KLDivergenceKernel(1e-6)
    noException should be thrownBy {
      klKernel.divergence(Vectors.dense(smallVal, 1.0), Vectors.dense(1.0, 1.0))
    }
  }

  //
  // ========== Integration Tests ==========
  //

  test("Gradient descent: moving along negative gradient reduces divergence") {
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(1.0, 2.0)
    val mu     = Vectors.dense(4.0, 5.0)

    val initialDiv = kernel.divergence(x, mu)

    // Gradient at x with respect to D(x, mu) points towards mu
    // Moving x towards mu should reduce divergence
    val step    = 0.1
    val xArr    = x.toArray
    val muArr   = mu.toArray
    val newXArr = xArr.zip(muArr).map { case (xi, mui) => xi + step * (mui - xi) }
    val newX    = Vectors.dense(newXArr)

    val newDiv = kernel.divergence(newX, mu)

    newDiv should be < initialDiv
  }

  test("Center computation: mean minimizes squared Euclidean divergence") {
    val kernel = new SquaredEuclideanKernel()
    val points = Seq(
      Vectors.dense(1.0, 2.0),
      Vectors.dense(2.0, 3.0),
      Vectors.dense(3.0, 4.0)
    )

    // Mean center
    val mean = Vectors.dense(2.0, 3.0)

    // Any other center should have higher total divergence
    val altCenter = Vectors.dense(2.5, 3.5)
    val divToMean = points.map(p => kernel.divergence(p, mean)).sum
    val divToAlt  = points.map(p => kernel.divergence(p, altCenter)).sum

    divToMean should be < divToAlt
  }
}
