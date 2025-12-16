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

/** Extended kernel interface for sparse-optimized divergence computation.
  *
  * This trait extends BregmanKernel with methods optimized for sparse vectors,
  * avoiding materialization of dense representations when possible.
  *
  * Key optimizations:
  *   - Skip zero entries where mathematically valid (0·log(0) = 0 for KL)
  *   - Use sparse iteration over non-zero indices
  *   - Maintain sparse center representations when beneficial
  */
trait SparseBregmanKernel extends BregmanKernel {

  /** Compute divergence optimized for sparse vectors.
    *
    * Default implementation converts to dense. Subclasses should override
    * for efficient sparse computation.
    *
    * @param x
    *   sparse data point
    * @param mu
    *   sparse cluster center
    * @return
    *   D_F(x, mu)
    */
  def divergenceSparse(x: SparseVector, mu: SparseVector): Double = {
    divergence(x, mu)
  }

  /** Check if sparse optimization is beneficial for given sparsity level.
    *
    * @param sparsity
    *   fraction of non-zero elements (0.0 = all zeros, 1.0 = dense)
    * @return
    *   true if sparse computation is recommended
    */
  def preferSparseComputation(sparsity: Double): Boolean = sparsity < 0.5
}

/** Sparse-optimized Squared Euclidean kernel.
  *
  * Exploits sparsity by only iterating over non-zero indices in both vectors.
  * For ||x - μ||², we compute:
  *   ||x||² + ||μ||² - 2⟨x, μ⟩
  *
  * Where each term can be computed sparsely.
  */
class SparseSEKernel extends SquaredEuclideanKernel with SparseBregmanKernel {

  override def divergenceSparse(x: SparseVector, mu: SparseVector): Double = {
    val xIndices  = x.indices
    val xValues   = x.values
    val muIndices = mu.indices
    val muValues  = mu.values

    // ||x||²
    var xNormSq = 0.0
    var i       = 0
    while (i < xValues.length) {
      xNormSq += xValues(i) * xValues(i)
      i += 1
    }

    // ||μ||²
    var muNormSq = 0.0
    var j        = 0
    while (j < muValues.length) {
      muNormSq += muValues(j) * muValues(j)
      j += 1
    }

    // ⟨x, μ⟩ using merge of sorted indices
    var dotProduct = 0.0
    i = 0
    j = 0
    while (i < xIndices.length && j < muIndices.length) {
      if (xIndices(i) == muIndices(j)) {
        dotProduct += xValues(i) * muValues(j)
        i += 1
        j += 1
      } else if (xIndices(i) < muIndices(j)) {
        i += 1
      } else {
        j += 1
      }
    }

    0.5 * (xNormSq + muNormSq - 2.0 * dotProduct)
  }

  override def name: String = "sparseSE"
}

/** Sparse-optimized KL Divergence kernel.
  *
  * KL divergence: D_KL(P||Q) = Σ_i P_i log(P_i / Q_i)
  *
  * Sparse optimization:
  *   - If P_i = 0: contribution is 0 (0·log(0) = 0 by convention)
  *   - If Q_i = 0 but P_i > 0: divergence is +∞ (but we use smoothing)
  *   - Only iterate over non-zero entries in P
  *
  * For text/document clustering, P is often very sparse (bag of words).
  *
  * @param smoothing
  *   epsilon to add to Q to avoid log(0)
  */
class SparseKLKernel(val smoothing: Double = 1e-10) extends SparseBregmanKernel {

  override def grad(x: Vector): Vector = {
    // grad(x) = log(x) + 1 for KL (element-wise)
    val arr = x.toArray.map { v =>
      val smoothed = math.max(v, smoothing)
      math.log(smoothed) + 1.0
    }
    Vectors.dense(arr)
  }

  override def invGrad(theta: Vector): Vector = {
    // invGrad(θ) = exp(θ - 1) for KL
    val arr = theta.toArray.map(t => math.exp(t - 1.0))
    Vectors.dense(arr)
  }

  override def divergence(x: Vector, mu: Vector): Double = {
    var sum = 0.0
    val n   = x.size
    var i   = 0
    while (i < n) {
      val p = math.max(x(i), smoothing)
      val q = math.max(mu(i), smoothing)
      sum += p * math.log(p / q) - p + q
      i += 1
    }
    sum
  }

  override def divergenceSparse(x: SparseVector, mu: SparseVector): Double = {
    // For KL: D(P||Q) = Σ P_i log(P_i/Q_i) - P_i + Q_i
    // When P_i = 0: contribution is just Q_i
    // When Q_i = 0 but P_i > 0: use smoothing

    val xIndices  = x.indices
    val xValues   = x.values
    val muIndices = mu.indices
    val muValues  = mu.values

    var sum = 0.0

    // Create lookup for mu values
    val muMap = new java.util.HashMap[Int, Double]()
    var j     = 0
    while (j < muIndices.length) {
      muMap.put(muIndices(j), muValues(j))
      j += 1
    }

    // Sum over non-zero P entries
    var i = 0
    while (i < xIndices.length) {
      val idx = xIndices(i)
      val p   = math.max(xValues(i), smoothing)
      val q   = if (muMap.containsKey(idx)) math.max(muMap.get(idx), smoothing) else smoothing
      sum += p * math.log(p / q) - p + q
      i += 1
    }

    // Add Q_i for entries where P_i = 0 but Q_i > 0
    j = 0
    while (j < muIndices.length) {
      val idx = muIndices(j)
      if (!xIndices.contains(idx)) {
        val q = math.max(muValues(j), smoothing)
        sum += q // -0 + q when P_i = 0
      }
      j += 1
    }

    sum
  }

  override def validate(x: Vector): Boolean = {
    x.toArray.forall(_ >= 0.0)
  }

  override def name: String = "sparseKL"

  override def preferSparseComputation(sparsity: Double): Boolean = sparsity < 0.3
}

/** Sparse-optimized L1 kernel (Manhattan distance).
  *
  * L1 distance: Σ_i |x_i - μ_i|
  *
  * Sparse optimization:
  *   - Only iterate over union of non-zero indices
  *   - |0 - 0| = 0 contributes nothing
  */
class SparseL1Kernel extends L1Kernel with SparseBregmanKernel {

  override def divergenceSparse(x: SparseVector, mu: SparseVector): Double = {
    val xIndices  = x.indices
    val xValues   = x.values
    val muIndices = mu.indices
    val muValues  = mu.values

    var sum = 0.0

    // Use merge of sorted indices
    var i = 0
    var j = 0
    while (i < xIndices.length && j < muIndices.length) {
      if (xIndices(i) == muIndices(j)) {
        sum += math.abs(xValues(i) - muValues(j))
        i += 1
        j += 1
      } else if (xIndices(i) < muIndices(j)) {
        sum += math.abs(xValues(i)) // mu is 0
        i += 1
      } else {
        sum += math.abs(muValues(j)) // x is 0
        j += 1
      }
    }

    // Handle remaining elements
    while (i < xIndices.length) {
      sum += math.abs(xValues(i))
      i += 1
    }
    while (j < muIndices.length) {
      sum += math.abs(muValues(j))
      j += 1
    }

    sum
  }

  override def name: String = "sparseL1"
}

/** Factory for creating sparse-optimized kernels.
  */
object SparseBregmanKernel {

  /** Create a sparse-optimized kernel for the given divergence type.
    *
    * @param divergence
    *   divergence name
    * @param smoothing
    *   smoothing parameter (for KL, IS)
    * @return
    *   sparse-optimized kernel if available, or standard kernel
    */
  def create(divergence: String, smoothing: Double = 1e-10): BregmanKernel = divergence match {
    case "squaredEuclidean"     => new SparseSEKernel()
    case "kl"                   => new SparseKLKernel(smoothing)
    case "l1" | "manhattan"     => new SparseL1Kernel()
    case "spherical" | "cosine" => new SparseSphericalKernel()
    // Fall back to standard kernels for others
    case "itakuraSaito"         => new ItakuraSaitoKernel(smoothing)
    case "generalizedI"         => new GeneralizedIDivergenceKernel(smoothing)
    case "logistic"             => new LogisticLossKernel(smoothing)
    case other => throw new IllegalArgumentException(s"Unknown divergence: $other")
  }

  /** Check if sparse optimization is available for given divergence.
    */
  def hasSparseOptimization(divergence: String): Boolean =
    Set("squaredEuclidean", "kl", "l1", "manhattan", "spherical", "cosine").contains(divergence)
}

/** Sparse-optimized Spherical (cosine) kernel.
  *
  * Cosine distance: 1 - cos(x, μ) = 1 - ⟨x, μ⟩ / (||x|| · ||μ||)
  *
  * For unit vectors: 1 - ⟨x, μ⟩
  */
class SparseSphericalKernel extends SphericalKernel with SparseBregmanKernel {

  override def divergenceSparse(x: SparseVector, mu: SparseVector): Double = {
    val xIndices  = x.indices
    val xValues   = x.values
    val muIndices = mu.indices
    val muValues  = mu.values

    // Compute norms and dot product sparsely
    var xNormSq    = 0.0
    var muNormSq   = 0.0
    var dotProduct = 0.0

    var i = 0
    while (i < xValues.length) {
      xNormSq += xValues(i) * xValues(i)
      i += 1
    }

    var j = 0
    while (j < muValues.length) {
      muNormSq += muValues(j) * muValues(j)
      j += 1
    }

    // Dot product via merge
    i = 0
    j = 0
    while (i < xIndices.length && j < muIndices.length) {
      if (xIndices(i) == muIndices(j)) {
        dotProduct += xValues(i) * muValues(j)
        i += 1
        j += 1
      } else if (xIndices(i) < muIndices(j)) {
        i += 1
      } else {
        j += 1
      }
    }

    val xNorm  = math.sqrt(xNormSq)
    val muNorm = math.sqrt(muNormSq)

    if (xNorm < 1e-10 || muNorm < 1e-10) {
      1.0 // Maximum distance if either is zero vector
    } else {
      val cosine = dotProduct / (xNorm * muNorm)
      1.0 - math.max(-1.0, math.min(1.0, cosine))
    }
  }

  override def name: String = "sparseSpherical"
}
