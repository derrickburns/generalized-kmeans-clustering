/*
 * Licensed to the Massive Data Science and Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Massive Data Science and Derrick R. Burns licenses this file to You under the
 * Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
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

package com.massivedatascience.clusterer.ml.df.kernels

import org.apache.spark.ml.linalg.{ SparseVector, Vector, Vectors }
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests for sparse-optimized Bregman kernel implementations.
  *
  * Validates:
  *   - Sparse divergence matches dense divergence
  *   - Sparse optimization is efficient for sparse vectors
  *   - Edge cases (zero vectors, high sparsity)
  */
class SparseBregmanKernelSuite extends AnyFunSuite with Matchers {

  private val tol        = 1e-10
  private val relaxedTol = 1e-8

  // Helper to create sparse vector
  private def sparse(size: Int, indices: Array[Int], values: Array[Double]): SparseVector = {
    new SparseVector(size, indices, values)
  }

  // ========== SparseSEKernel Tests ==========

  test("SparseSEKernel sparse divergence matches dense") {
    val kernel = new SparseSEKernel()

    val x  = sparse(5, Array(0, 2, 4), Array(1.0, 2.0, 3.0))
    val mu = sparse(5, Array(1, 2, 3), Array(1.0, 2.0, 1.0))

    val sparseDist = kernel.divergenceSparse(x, mu)
    val denseDist  = kernel.divergence(x, mu)

    math.abs(sparseDist - denseDist) should be < tol
  }

  test("SparseSEKernel handles fully disjoint indices") {
    val kernel = new SparseSEKernel()

    val x  = sparse(6, Array(0, 2, 4), Array(1.0, 2.0, 3.0))
    val mu = sparse(6, Array(1, 3, 5), Array(1.0, 2.0, 3.0))

    val sparseDist = kernel.divergenceSparse(x, mu)
    val denseDist  = kernel.divergence(x, mu)

    math.abs(sparseDist - denseDist) should be < tol
  }

  test("SparseSEKernel handles identical vectors") {
    val kernel = new SparseSEKernel()

    val x = sparse(5, Array(0, 2, 4), Array(1.0, 2.0, 3.0))

    val dist = kernel.divergenceSparse(x, x)
    dist should be < tol
  }

  test("SparseSEKernel handles empty sparse vectors") {
    val kernel = new SparseSEKernel()

    val empty = sparse(5, Array.empty, Array.empty)
    val x     = sparse(5, Array(0, 1), Array(1.0, 2.0))

    // Distance from empty to x
    val dist1 = kernel.divergenceSparse(empty, x)
    val dist2 = kernel.divergenceSparse(x, empty)

    // ||x - 0||² = ||x||²
    val expected = 0.5 * (1.0 * 1.0 + 2.0 * 2.0)
    math.abs(dist1 - expected) should be < tol
    math.abs(dist2 - expected) should be < tol
  }

  test("SparseSEKernel name is correct") {
    val kernel = new SparseSEKernel()
    kernel.name shouldBe "sparseSE"
  }

  // ========== SparseKLKernel Tests ==========

  test("SparseKLKernel sparse divergence matches dense") {
    val kernel = new SparseKLKernel(smoothing = 1e-10)

    // Positive values for KL
    val x  = sparse(5, Array(0, 2, 4), Array(0.2, 0.3, 0.5))
    val mu = sparse(5, Array(0, 1, 2), Array(0.3, 0.2, 0.5))

    val sparseDist = kernel.divergenceSparse(x, mu)
    val denseDist  = kernel.divergence(x, mu)

    math.abs(sparseDist - denseDist) should be < relaxedTol
  }

  test("SparseKLKernel handles zero entries correctly") {
    val kernel = new SparseKLKernel(smoothing = 1e-10)

    // P has zeros where Q is non-zero (adds Q contribution)
    val p = sparse(4, Array(0, 2), Array(0.5, 0.5))
    val q = sparse(4, Array(0, 1, 2, 3), Array(0.25, 0.25, 0.25, 0.25))

    val dist = kernel.divergenceSparse(p, q)
    dist should be >= 0.0
  }

  test("SparseKLKernel validates positive domain") {
    val kernel = new SparseKLKernel()

    kernel.validate(Vectors.dense(0.1, 0.2, 0.3)) shouldBe true
    kernel.validate(Vectors.dense(0.0, 0.2, 0.3)) shouldBe true // 0 is allowed
    kernel.validate(Vectors.dense(-0.1, 0.2, 0.3)) shouldBe false
  }

  test("SparseKLKernel preferSparseComputation threshold") {
    val kernel = new SparseKLKernel()

    kernel.preferSparseComputation(0.1) shouldBe true  // 10% density
    kernel.preferSparseComputation(0.29) shouldBe true // Just under 30%
    kernel.preferSparseComputation(0.5) shouldBe false // 50% density
    kernel.preferSparseComputation(0.9) shouldBe false // 90% density
  }

  test("SparseKLKernel grad and invGrad are consistent") {
    val kernel = new SparseKLKernel()
    val x      = Vectors.dense(0.2, 0.3, 0.5)

    val grad    = kernel.grad(x)
    val invGrad = kernel.invGrad(grad)

    // invGrad(grad(x)) should approximate x
    val xArr      = x.toArray
    val recovered = invGrad.toArray

    xArr.zip(recovered).foreach { case (expected, actual) =>
      math.abs(expected - actual) should be < relaxedTol
    }
  }

  test("SparseKLKernel name is correct") {
    val kernel = new SparseKLKernel()
    kernel.name shouldBe "sparseKL"
  }

  // ========== SparseL1Kernel Tests ==========

  test("SparseL1Kernel sparse divergence matches dense") {
    val kernel = new SparseL1Kernel()

    val x  = sparse(5, Array(0, 2, 4), Array(1.0, -2.0, 3.0))
    val mu = sparse(5, Array(1, 2, 3), Array(1.0, 1.0, 1.0))

    val sparseDist = kernel.divergenceSparse(x, mu)
    val denseDist  = kernel.divergence(x, mu)

    math.abs(sparseDist - denseDist) should be < tol
  }

  test("SparseL1Kernel handles overlapping indices") {
    val kernel = new SparseL1Kernel()

    val x  = sparse(4, Array(0, 1, 2), Array(1.0, 2.0, 3.0))
    val mu = sparse(4, Array(1, 2, 3), Array(1.0, 1.0, 1.0))

    // Expected: |1-0| + |2-1| + |3-1| + |0-1| = 1 + 1 + 2 + 1 = 5
    val dist = kernel.divergenceSparse(x, mu)
    math.abs(dist - 5.0) should be < tol
  }

  test("SparseL1Kernel handles empty vectors") {
    val kernel = new SparseL1Kernel()

    val empty = sparse(3, Array.empty, Array.empty)
    val x     = sparse(3, Array(0, 1, 2), Array(1.0, 2.0, 3.0))

    val dist = kernel.divergenceSparse(empty, x)
    // Sum of absolute values
    math.abs(dist - 6.0) should be < tol
  }

  test("SparseL1Kernel name is correct") {
    val kernel = new SparseL1Kernel()
    kernel.name shouldBe "sparseL1"
  }

  // ========== SparseSphericalKernel Tests ==========

  test("SparseSphericalKernel sparse divergence matches dense") {
    val kernel = new SparseSphericalKernel()

    val x  = sparse(5, Array(0, 2, 4), Array(1.0, 2.0, 2.0))
    val mu = sparse(5, Array(1, 2, 3), Array(1.0, 2.0, 1.0))

    val sparseDist = kernel.divergenceSparse(x, mu)
    val denseDist  = kernel.divergence(x, mu)

    math.abs(sparseDist - denseDist) should be < tol
  }

  test("SparseSphericalKernel handles identical vectors") {
    val kernel = new SparseSphericalKernel()

    val x = sparse(5, Array(0, 2, 4), Array(1.0, 2.0, 3.0))

    val dist = kernel.divergenceSparse(x, x)
    dist should be < tol // cos(x, x) = 1, so distance = 0
  }

  test("SparseSphericalKernel handles orthogonal vectors") {
    val kernel = new SparseSphericalKernel()

    // Orthogonal sparse vectors (no overlapping non-zero indices)
    val x  = sparse(4, Array(0, 1), Array(1.0, 1.0))
    val mu = sparse(4, Array(2, 3), Array(1.0, 1.0))

    val dist = kernel.divergenceSparse(x, mu)
    // Cosine = 0, distance = 1
    math.abs(dist - 1.0) should be < tol
  }

  test("SparseSphericalKernel handles zero vector") {
    val kernel = new SparseSphericalKernel()

    val zero = sparse(3, Array.empty, Array.empty)
    val x    = sparse(3, Array(0), Array(1.0))

    // Zero vector should give max distance
    val dist = kernel.divergenceSparse(zero, x)
    math.abs(dist - 1.0) should be < tol
  }

  test("SparseSphericalKernel name is correct") {
    val kernel = new SparseSphericalKernel()
    kernel.name shouldBe "sparseSpherical"
  }

  // ========== SparseBregmanKernel Factory Tests ==========

  test("SparseBregmanKernel.create creates correct kernel types") {
    SparseBregmanKernel.create("squaredEuclidean") shouldBe a[SparseSEKernel]
    SparseBregmanKernel.create("kl") shouldBe a[SparseKLKernel]
    SparseBregmanKernel.create("l1") shouldBe a[SparseL1Kernel]
    SparseBregmanKernel.create("manhattan") shouldBe a[SparseL1Kernel]
    SparseBregmanKernel.create("spherical") shouldBe a[SparseSphericalKernel]
    SparseBregmanKernel.create("cosine") shouldBe a[SparseSphericalKernel]
  }

  test("SparseBregmanKernel.create falls back for unsupported divergences") {
    // These should fall back to standard kernels (not sparse-optimized)
    val isKernel = SparseBregmanKernel.create("itakuraSaito")
    isKernel should not be a[SparseBregmanKernel]

    val giKernel = SparseBregmanKernel.create("generalizedI")
    giKernel should not be a[SparseBregmanKernel]
  }

  test("SparseBregmanKernel.create rejects unknown divergences") {
    an[IllegalArgumentException] should be thrownBy {
      SparseBregmanKernel.create("unknown")
    }
  }

  test("SparseBregmanKernel.hasSparseOptimization is accurate") {
    SparseBregmanKernel.hasSparseOptimization("squaredEuclidean") shouldBe true
    SparseBregmanKernel.hasSparseOptimization("kl") shouldBe true
    SparseBregmanKernel.hasSparseOptimization("l1") shouldBe true
    SparseBregmanKernel.hasSparseOptimization("manhattan") shouldBe true
    SparseBregmanKernel.hasSparseOptimization("spherical") shouldBe true
    SparseBregmanKernel.hasSparseOptimization("cosine") shouldBe true

    SparseBregmanKernel.hasSparseOptimization("itakuraSaito") shouldBe false
    SparseBregmanKernel.hasSparseOptimization("generalizedI") shouldBe false
    SparseBregmanKernel.hasSparseOptimization("logistic") shouldBe false
  }

  // ========== Performance Characteristics Tests ==========

  test("Sparse computation is preferred for high sparsity") {
    val seKernel        = new SparseSEKernel()
    val klKernel        = new SparseKLKernel()
    val l1Kernel        = new SparseL1Kernel()
    val sphericalKernel = new SparseSphericalKernel()

    // All should prefer sparse for 10% density
    seKernel.preferSparseComputation(0.1) shouldBe true
    klKernel.preferSparseComputation(0.1) shouldBe true
    l1Kernel.preferSparseComputation(0.1) shouldBe true
    sphericalKernel.preferSparseComputation(0.1) shouldBe true
  }

  test("Dense computation is preferred for low sparsity") {
    val seKernel        = new SparseSEKernel()
    val klKernel        = new SparseKLKernel()
    val l1Kernel        = new SparseL1Kernel()
    val sphericalKernel = new SparseSphericalKernel()

    // All should prefer dense for 80% density
    seKernel.preferSparseComputation(0.8) shouldBe false
    klKernel.preferSparseComputation(0.8) shouldBe false
    l1Kernel.preferSparseComputation(0.8) shouldBe false
    sphericalKernel.preferSparseComputation(0.8) shouldBe false
  }

  // ========== Numerical Stability Tests ==========

  test("SparseKLKernel handles near-zero values via smoothing") {
    val kernel = new SparseKLKernel(smoothing = 1e-10)

    // Very small values
    val x  = sparse(3, Array(0, 1, 2), Array(1e-12, 0.5, 0.5))
    val mu = sparse(3, Array(0, 1, 2), Array(0.3, 0.3, 0.4))

    val dist = kernel.divergenceSparse(x, mu)
    dist.isNaN shouldBe false
    dist.isInfinite shouldBe false
    dist should be >= 0.0
  }

  test("SparseSEKernel handles large values correctly") {
    val kernel = new SparseSEKernel()

    val x  = sparse(3, Array(0, 1), Array(1e6, 1e6))
    val mu = sparse(3, Array(0, 1), Array(1e6 + 1, 1e6 + 1))

    val dist = kernel.divergenceSparse(x, mu)
    // Should be close to 0.5 * (1^2 + 1^2) = 1.0
    math.abs(dist - 1.0) should be < 1e-6
  }
}
