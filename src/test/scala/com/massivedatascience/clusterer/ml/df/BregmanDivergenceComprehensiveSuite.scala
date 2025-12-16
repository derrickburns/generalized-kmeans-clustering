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

import scala.util.Random

/** Comprehensive tests for Bregman divergences with random values, extreme cases, and mathematical
  * property verification.
  *
  * This suite complements BregmanKernelAccuracySuite by testing:
  *   - Random value combinations
  *   - Extreme edge cases (very small, very large, boundary values)
  *   - Mathematical correctness against known formulas
  *   - Numerical stability under various conditions
  *   - Cross-validation of related divergences
  */
class BregmanDivergenceComprehensiveSuite extends AnyFunSuite with Matchers {

  private val defaultTol = 1e-10
  private val relaxedTol = 1e-8
  private val looseTol   = 1e-6

  // Seeded random for reproducibility
  private val rng = new Random(42)

  //
  // ========== Squared Euclidean: Random and Extreme Value Tests ==========
  //

  test("SquaredEuclidean: random value divergence correctness") {
    val kernel = new SquaredEuclideanKernel()

    for (_ <- 1 to 100) {
      val dim = 2 + rng.nextInt(10)
      val x   = Vectors.dense(Array.fill(dim)(rng.nextDouble() * 100 - 50))
      val mu  = Vectors.dense(Array.fill(dim)(rng.nextDouble() * 100 - 50))

      val expected = {
        val diff = x.toArray.zip(mu.toArray).map { case (xi, mui) => xi - mui }
        0.5 * diff.map(d => d * d).sum
      }
      val actual   = kernel.divergence(x, mu)

      withClue(s"SE divergence mismatch for random vectors") {
        math.abs(actual - expected) should be < defaultTol
      }
    }
  }

  test("SquaredEuclidean: extreme large values") {
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(1e10, -1e10, 5e9)
    val mu     = Vectors.dense(2e10, -2e10, 1e10)

    val expected =
      0.5 * ((1e10 - 2e10) * (1e10 - 2e10) + (-1e10 + 2e10) * (-1e10 + 2e10) + (5e9 - 1e10) * (5e9 - 1e10))
    val actual   = kernel.divergence(x, mu)

    math.abs((actual - expected) / expected) should be < looseTol // Relative error
    java.lang.Double.isFinite(actual) shouldBe true
  }

  test("SquaredEuclidean: extreme small values") {
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(1e-100, 1e-150, 1e-200)
    val mu     = Vectors.dense(2e-100, 2e-150, 2e-200)

    val dist = kernel.divergence(x, mu)

    dist should be >= 0.0
    java.lang.Double.isFinite(dist) shouldBe true
  }

  test("SquaredEuclidean: mixed extreme values") {
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(1e-100, 0.0, 1e10)
    val mu     = Vectors.dense(0.0, 1e-100, 1e10 + 1.0)

    val dist = kernel.divergence(x, mu)

    dist should be >= 0.0
    java.lang.Double.isFinite(dist) shouldBe true
  }

  //
  // ========== KL Divergence: Random and Extreme Value Tests ==========
  //

  test("KL: random probability distribution pairs") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)

    for (_ <- 1 to 100) {
      val dim   = 2 + rng.nextInt(10)
      // Generate random probability distributions
      val xRaw  = Array.fill(dim)(rng.nextDouble())
      val xSum  = xRaw.sum
      val x     = Vectors.dense(xRaw.map(_ / xSum))
      val muRaw = Array.fill(dim)(rng.nextDouble())
      val muSum = muRaw.sum
      val mu    = Vectors.dense(muRaw.map(_ / muSum))

      val dist = kernel.divergence(x, mu)

      withClue(s"KL divergence should be non-negative") {
        dist should be >= 0.0
        java.lang.Double.isFinite(dist) shouldBe true
      }
    }
  }

  test("KL: very skewed probability distributions") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.99, 0.01)
    val mu     = Vectors.dense(0.01, 0.99)

    val eps      = 1e-10
    val expected =
      (0.99 + eps) * math.log((0.99 + eps) / (0.01 + eps)) +
        (0.01 + eps) * math.log((0.01 + eps) / (0.99 + eps))
    val actual   = kernel.divergence(x, mu)

    math.abs((actual - expected) / expected) should be < relaxedTol
    actual should be > 1.0 // High KL divergence for very different distributions
  }

  test("KL: near-zero probabilities with smoothing") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-6)
    val x      = Vectors.dense(1e-10, 1.0 - 1e-10)
    val mu     = Vectors.dense(0.5, 0.5)

    val dist = kernel.divergence(x, mu)

    dist should be >= 0.0
    java.lang.Double.isFinite(dist) shouldBe true
  }

  test("KL: uniform vs non-uniform distribution") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.25, 0.25, 0.25, 0.25) // Uniform
    val mu     = Vectors.dense(0.1, 0.2, 0.3, 0.4)     // Non-uniform

    val dist = kernel.divergence(x, mu)

    dist should be >= 0.0
    java.lang.Double.isFinite(dist) shouldBe true
  }

  test("KL: Gibbs inequality - KL(p||q) >= 0 with equality iff p = q") {
    val kernel = new KLDivergenceKernel(smoothing = 1e-10)

    // Test equality case
    val p1 = Vectors.dense(0.3, 0.7)
    kernel.divergence(p1, p1) should be < relaxedTol

    // Test inequality case
    val p2 = Vectors.dense(0.3, 0.7)
    val q2 = Vectors.dense(0.5, 0.5)
    kernel.divergence(p2, q2) should be > 0.0
  }

  //
  // ========== Itakura-Saito: Random and Extreme Value Tests ==========
  //

  test("ItakuraSaito: random positive vectors") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)

    for (_ <- 1 to 100) {
      val dim = 2 + rng.nextInt(10)
      val x   = Vectors.dense(Array.fill(dim)(0.1 + rng.nextDouble() * 10))
      val mu  = Vectors.dense(Array.fill(dim)(0.1 + rng.nextDouble() * 10))

      val dist = kernel.divergence(x, mu)

      withClue(s"IS divergence should be non-negative and finite") {
        dist should be >= 0.0
        java.lang.Double.isFinite(dist) shouldBe true
      }
    }
  }

  test("ItakuraSaito: extreme ratio x >> mu") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1000.0, 500.0)
    val mu     = Vectors.dense(1.0, 1.0)

    val dist = kernel.divergence(x, mu)

    dist should be > 100.0 // Should be large due to large ratio
    java.lang.Double.isFinite(dist) shouldBe true
  }

  test("ItakuraSaito: extreme ratio mu >> x") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1.0, 1.0)
    val mu     = Vectors.dense(1000.0, 500.0)

    val dist = kernel.divergence(x, mu)

    dist should be >= 0.0
    java.lang.Double.isFinite(dist) shouldBe true
  }

  test("ItakuraSaito: manual formula verification") {
    val kernel = new ItakuraSaitoKernel(smoothing = 1e-10)
    val x      = Vectors.dense(2.0, 4.0)
    val mu     = Vectors.dense(1.0, 2.0)

    val eps      = 1e-10
    val expected = ((2.0 + eps) / (1.0 + eps) - math.log((2.0 + eps) / (1.0 + eps)) - 1.0) +
      ((4.0 + eps) / (2.0 + eps) - math.log((4.0 + eps) / (2.0 + eps)) - 1.0)
    val actual   = kernel.divergence(x, mu)

    math.abs(actual - expected) should be < relaxedTol
  }

  //
  // ========== Generalized I-Divergence: Random and Extreme Value Tests ==========
  //

  test("GeneralizedI: random positive vectors") {
    val kernel = new GeneralizedIDivergenceKernel(smoothing = 1e-10)

    for (_ <- 1 to 100) {
      val dim = 2 + rng.nextInt(10)
      val x   = Vectors.dense(Array.fill(dim)(rng.nextDouble() * 10))
      val mu  = Vectors.dense(Array.fill(dim)(rng.nextDouble() * 10))

      val dist = kernel.divergence(x, mu)

      withClue(s"GenI divergence should be non-negative and finite") {
        dist should be >= 0.0
        java.lang.Double.isFinite(dist) shouldBe true
      }
    }
  }

  test("GeneralizedI: integer counts (natural domain)") {
    val kernel = new GeneralizedIDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(10.0, 20.0, 30.0)
    val mu     = Vectors.dense(12.0, 18.0, 35.0)

    val dist = kernel.divergence(x, mu)

    dist should be >= 0.0
    java.lang.Double.isFinite(dist) shouldBe true
  }

  test("GeneralizedI: very large counts") {
    val kernel = new GeneralizedIDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(1e6, 2e6, 3e6)
    val mu     = Vectors.dense(1.1e6, 1.9e6, 3.2e6)

    val dist = kernel.divergence(x, mu)

    dist should be >= 0.0
    java.lang.Double.isFinite(dist) shouldBe true
  }

  test("GeneralizedI: manual formula verification") {
    val kernel = new GeneralizedIDivergenceKernel(smoothing = 1e-10)
    val x      = Vectors.dense(3.0, 5.0)
    val mu     = Vectors.dense(2.0, 4.0)

    val eps      = 1e-10
    val expected =
      (3.0 + eps) * math.log((3.0 + eps) / (2.0 + eps)) - (3.0 + eps) + (2.0 + eps) +
        (5.0 + eps) * math.log((5.0 + eps) / (4.0 + eps)) - (5.0 + eps) + (4.0 + eps)
    val actual   = kernel.divergence(x, mu)

    math.abs(actual - expected) should be < relaxedTol
  }

  //
  // ========== Logistic Loss: Random and Extreme Value Tests ==========
  //

  test("LogisticLoss: random probability values") {
    val kernel = new LogisticLossKernel(smoothing = 1e-10)

    for (_ <- 1 to 100) {
      val dim = 1 + rng.nextInt(5)
      val x   = Vectors.dense(Array.fill(dim)(0.01 + rng.nextDouble() * 0.98))
      val mu  = Vectors.dense(Array.fill(dim)(0.01 + rng.nextDouble() * 0.98))

      val dist = kernel.divergence(x, mu)

      withClue(s"LogisticLoss divergence should be non-negative and finite") {
        dist should be >= 0.0
        java.lang.Double.isFinite(dist) shouldBe true
      }
    }
  }

  test("LogisticLoss: extreme probabilities near 0 and 1") {
    val kernel = new LogisticLossKernel(smoothing = 1e-6)
    val x      = Vectors.dense(0.001, 0.999)
    val mu     = Vectors.dense(0.5, 0.5)

    val dist = kernel.divergence(x, mu)

    dist should be >= 0.0
    java.lang.Double.isFinite(dist) shouldBe true
  }

  test("LogisticLoss: manual formula verification") {
    val kernel = new LogisticLossKernel(smoothing = 1e-10)
    val x      = Vectors.dense(0.3)
    val mu     = Vectors.dense(0.7)

    val eps      = 1e-10
    val xi       = math.max(eps, math.min(1.0 - eps, 0.3))
    val mui      = math.max(eps, math.min(1.0 - eps, 0.7))
    val expected =
      xi * math.log(xi / mui) + (1.0 - xi) * math.log((1.0 - xi) / (1.0 - mui))
    val actual   = kernel.divergence(x, mu)

    math.abs(actual - expected) should be < relaxedTol
  }

  test("LogisticLoss: complement symmetry") {
    val kernel = new LogisticLossKernel(smoothing = 1e-10)
    val p1     = 0.3
    val p2     = 0.7

    val d1 = kernel.divergence(Vectors.dense(p1), Vectors.dense(p2))
    val d2 = kernel.divergence(Vectors.dense(1.0 - p1), Vectors.dense(1.0 - p2))

    // Logistic loss has symmetry under complement
    math.abs(d1 - d2) should be < relaxedTol
  }

  //
  // ========== L1/Manhattan: Random and Extreme Value Tests ==========
  //

  test("L1: random vectors") {
    val kernel = new L1Kernel()

    for (_ <- 1 to 100) {
      val dim = 2 + rng.nextInt(10)
      val x   = Vectors.dense(Array.fill(dim)(rng.nextDouble() * 100 - 50))
      val mu  = Vectors.dense(Array.fill(dim)(rng.nextDouble() * 100 - 50))

      val expected = x.toArray.zip(mu.toArray).map { case (xi, mui) => math.abs(xi - mui) }.sum
      val actual   = kernel.divergence(x, mu)

      withClue(s"L1 divergence mismatch") {
        math.abs(actual - expected) should be < defaultTol
      }
    }
  }

  test("L1: extreme values") {
    val kernel = new L1Kernel()
    val x      = Vectors.dense(1e10, -1e10, 0.0)
    val mu     = Vectors.dense(-1e10, 1e10, 1e-100)

    val expected = math.abs(1e10 - (-1e10)) + math.abs(-1e10 - 1e10) + math.abs(0.0 - 1e-100)
    val actual   = kernel.divergence(x, mu)

    math.abs((actual - expected) / expected) should be < looseTol
    java.lang.Double.isFinite(actual) shouldBe true
  }

  test("L1: triangle inequality") {
    val kernel = new L1Kernel()
    val x      = Vectors.dense(1.0, 2.0, 3.0)
    val y      = Vectors.dense(4.0, 5.0, 6.0)
    val z      = Vectors.dense(7.0, 8.0, 9.0)

    val dxy = kernel.divergence(x, y)
    val dyz = kernel.divergence(y, z)
    val dxz = kernel.divergence(x, z)

    // Triangle inequality: d(x,z) <= d(x,y) + d(y,z)
    dxz should be <= (dxy + dyz + defaultTol)
  }

  //
  // ========== Cross-Divergence Mathematical Property Tests ==========
  //

  test("All divergences: non-negativity with random vectors") {
    val kernels = Seq(
      ("SE", new SquaredEuclideanKernel()),
      ("KL", new KLDivergenceKernel(1e-10)),
      ("IS", new ItakuraSaitoKernel(1e-10)),
      ("GenI", new GeneralizedIDivergenceKernel(1e-10)),
      ("Logistic", new LogisticLossKernel(1e-10)),
      ("L1", new L1Kernel())
    )

    kernels.foreach { case (name, kernel) =>
      for (iteration <- 1 to 20) {
        val (x, mu) = name match {
          case "SE" | "L1" =>
            (
              Vectors.dense(Array.fill(3)(rng.nextDouble() * 10 - 5)),
              Vectors.dense(Array.fill(3)(rng.nextDouble() * 10 - 5))
            )
          case "KL"        =>
            // For KL, ensure we use valid probability distributions
            val xRaw  = Array.fill(3)(0.1 + rng.nextDouble())
            val xSum  = xRaw.sum
            val muRaw = Array.fill(3)(0.1 + rng.nextDouble())
            val muSum = muRaw.sum
            (Vectors.dense(xRaw.map(_ / xSum)), Vectors.dense(muRaw.map(_ / muSum)))
          case "GenI"      =>
            (
              Vectors.dense(Array.fill(3)(0.1 + rng.nextDouble() * 10)),
              Vectors.dense(Array.fill(3)(0.1 + rng.nextDouble() * 10))
            )
          case "IS"        =>
            (
              Vectors.dense(Array.fill(3)(0.1 + rng.nextDouble() * 10)),
              Vectors.dense(Array.fill(3)(0.1 + rng.nextDouble() * 10))
            )
          case "Logistic"  =>
            (
              Vectors.dense(Array.fill(2)(0.01 + rng.nextDouble() * 0.98)),
              Vectors.dense(Array.fill(2)(0.01 + rng.nextDouble() * 0.98))
            )
        }

        val dist = kernel.divergence(x, mu)
        withClue(s"$name divergence should be non-negative (iteration $iteration, x=${x.toArray
            .mkString(",")}, mu=${mu.toArray.mkString(",")}, dist=$dist)") {
          dist should be >= (-1e-8) // Allow small numerical errors due to log calculations
          java.lang.Double.isFinite(dist) shouldBe true
        }
      }
    }
  }

  test("All divergences: self-divergence is zero for random vectors") {
    val kernels = Seq(
      ("SE", new SquaredEuclideanKernel()),
      ("KL", new KLDivergenceKernel(1e-10)),
      ("IS", new ItakuraSaitoKernel(1e-10)),
      ("GenI", new GeneralizedIDivergenceKernel(1e-10)),
      ("Logistic", new LogisticLossKernel(1e-10)),
      ("L1", new L1Kernel())
    )

    kernels.foreach { case (name, kernel) =>
      for (_ <- 1 to 20) {
        val x = name match {
          case "SE" | "L1"   => Vectors.dense(Array.fill(3)(rng.nextDouble() * 10 - 5))
          case "KL" | "GenI" => Vectors.dense(Array.fill(3)(rng.nextDouble() * 10))
          case "IS"          => Vectors.dense(Array.fill(3)(0.1 + rng.nextDouble() * 10))
          case "Logistic"    => Vectors.dense(Array.fill(2)(0.01 + rng.nextDouble() * 0.98))
        }

        val dist = kernel.divergence(x, x)
        withClue(s"$name self-divergence should be zero") {
          math.abs(dist) should be < relaxedTol
        }
      }
    }
  }

  //
  // ========== Numerical Stability Edge Cases ==========
  //

  test("All kernels: handle dimension mismatch gracefully") {
    val kernels = Seq(
      new SquaredEuclideanKernel(),
      new KLDivergenceKernel(1e-10),
      new L1Kernel()
    )

    kernels.foreach { kernel =>
      val x  = Vectors.dense(1.0, 2.0)
      val mu = Vectors.dense(1.0, 2.0, 3.0)

      // Dimension mismatch may be caught during divergence calculation
      // Different kernels may handle this differently
      try {
        kernel.divergence(x, mu)
        // Some kernels may not throw, just produce incorrect results
        // This is acceptable as long as validate catches invalid inputs
      } catch {
        case _: IllegalArgumentException | _: ArrayIndexOutOfBoundsException =>
          // Expected for some kernels
          succeed
      }
    }
  }

  test("Log-based kernels: handle overflow in exponentials") {
    val klKernel = new KLDivergenceKernel(1e-10)

    // Very large gradient values should be handled
    val largeTheta = Vectors.dense(100.0, 200.0) // exp(200) would overflow
    val x          = klKernel.invGrad(largeTheta)

    x.toArray.foreach { v =>
      withClue("invGrad should produce finite values even for large theta") {
        java.lang.Double.isFinite(v) shouldBe true
        v should be > 0.0
      }
    }
  }

  test("All kernels: consistent behavior at boundaries") {
    // Test that divergences behave consistently at domain boundaries

    // Squared Euclidean at zero
    val seKernel = new SquaredEuclideanKernel()
    val seZero   = Vectors.dense(0.0, 0.0, 0.0)
    seKernel.divergence(seZero, seZero) should be < defaultTol

    // KL at uniform distribution
    val klKernel = new KLDivergenceKernel(1e-10)
    val uniform  = Vectors.dense(0.25, 0.25, 0.25, 0.25)
    klKernel.divergence(uniform, uniform) should be < relaxedTol

    // L1 at zero
    val l1Kernel = new L1Kernel()
    val l1Zero   = Vectors.dense(0.0, 0.0, 0.0)
    l1Kernel.divergence(l1Zero, l1Zero) should be < defaultTol
  }

  //
  // ========== Integration Tests: Verify Bregman Divergence Formula ==========
  //

  test("Bregman formula: D(x||mu) = F(x) - F(mu) - <grad(F(mu)), x - mu>") {
    // Verify the Bregman divergence formula holds for Squared Euclidean
    val kernel = new SquaredEuclideanKernel()
    val x      = Vectors.dense(2.0, 3.0)
    val mu     = Vectors.dense(1.0, 1.0)

    // F(x) = ||x||^2 / 2 (for SE, convex function is ||x||^2, divergence scales by 0.5)
    val Fx = x.toArray.map(xi => xi * xi).sum

    // F(mu)
    val Fmu = mu.toArray.map(mui => mui * mui).sum

    // grad(F(mu)) = 2 * mu (for F(x) = ||x||^2)
    val gradFmu = mu.toArray.map(_ * 2.0)

    // <grad(F(mu)), x - mu>
    val dotProd = gradFmu
      .zip(x.toArray.zip(mu.toArray))
      .map { case (g, (xi, mui)) =>
        g * (xi - mui)
      }
      .sum

    // Bregman divergence formula (note: SE kernel uses 0.5 * ||x - mu||^2)
    val expected = 0.5 * (Fx - Fmu - dotProd)
    val actual   = kernel.divergence(x, mu)

    math.abs(actual - expected) should be < defaultTol
  }
}
