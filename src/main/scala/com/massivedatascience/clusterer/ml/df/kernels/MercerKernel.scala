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

import org.apache.spark.ml.linalg.{ DenseMatrix, Vector }

/** Mercer kernel (positive-definite kernel function) for kernel methods.
  *
  * A Mercer kernel k: X × X → ℝ satisfies:
  *   1. Symmetry: k(x, y) = k(y, x)
  *   2. Positive semi-definiteness: ∀{x_i}, the Gram matrix K_ij = k(x_i, x_j) is PSD
  *
  * This enables the "kernel trick" - computing inner products in high-dimensional
  * feature spaces without explicit mapping:
  *   k(x, y) = ⟨φ(x), φ(y)⟩
  *
  * IMPORTANT: This is distinct from BregmanKernel, which represents divergence functions.
  * We use "MercerKernel" to avoid confusion.
  *
  * @see
  *   [[BregmanKernel]] for divergence-based clustering
  */
trait MercerKernel extends Serializable {

  /** Compute the kernel value k(x, y).
    *
    * @param x
    *   first vector
    * @param y
    *   second vector
    * @return
    *   kernel evaluation k(x, y)
    */
  def apply(x: Vector, y: Vector): Double

  /** Name of this kernel for logging. */
  def name: String

  /** Compute the induced distance in feature space.
    *
    * d²(x, y) = ||φ(x) - φ(y)||² = k(x,x) - 2k(x,y) + k(y,y)
    *
    * @param x
    *   first vector
    * @param y
    *   second vector
    * @return
    *   squared distance in feature space
    */
  def squaredDistance(x: Vector, y: Vector): Double = {
    apply(x, x) - 2.0 * apply(x, y) + apply(y, y)
  }

  /** Compute the Gram matrix for a set of points.
    *
    * K_ij = k(x_i, x_j)
    *
    * @param points
    *   array of vectors
    * @return
    *   n × n Gram matrix
    */
  def gramMatrix(points: Array[Vector]): DenseMatrix = {
    val n      = points.length
    val values = new Array[Double](n * n)

    for (i <- 0 until n) {
      for (j <- i until n) {
        val k = apply(points(i), points(j))
        values(i * n + j) = k
        values(j * n + i) = k // Symmetric
      }
    }

    new DenseMatrix(n, n, values)
  }
}

/** RBF (Radial Basis Function) kernel, also known as Gaussian kernel.
  *
  * k(x, y) = exp(-γ ||x - y||²)
  *
  * Common choices for γ:
  *   - γ = 1 / (2σ²) where σ is the characteristic length scale
  *   - γ = 1 / dim (scales with dimensionality)
  *   - γ = 1 / median(||x_i - x_j||²) (median heuristic)
  *
  * Properties:
  *   - Maps to infinite-dimensional feature space
  *   - Local kernel (only nearby points have high similarity)
  *   - Universal approximator
  *
  * @param gamma
  *   bandwidth parameter (γ > 0)
  */
class RBFKernel(val gamma: Double) extends MercerKernel {
  require(gamma > 0, s"gamma must be positive, got $gamma")

  override def apply(x: Vector, y: Vector): Double = {
    val xArr = x.toArray
    val yArr = y.toArray
    var sqDist = 0.0
    var i     = 0
    while (i < xArr.length) {
      val diff = xArr(i) - yArr(i)
      sqDist += diff * diff
      i += 1
    }
    math.exp(-gamma * sqDist)
  }

  override def name: String = s"RBF(γ=$gamma)"

  override def squaredDistance(x: Vector, y: Vector): Double = {
    // For RBF: k(x,x) = 1 always, so d² = 2 - 2k(x,y) = 2(1 - k(x,y))
    2.0 * (1.0 - apply(x, y))
  }
}

object RBFKernel {

  /** Create RBF kernel with automatic gamma selection using median heuristic.
    *
    * @param data
    *   sample of data points
    * @param sampleSize
    *   number of pairs to sample for median estimation
    * @return
    *   RBF kernel with median-heuristic gamma
    */
  def withMedianHeuristic(data: Array[Vector], sampleSize: Int = 1000): RBFKernel = {
    val n         = math.min(data.length, math.sqrt(sampleSize).toInt)
    val distances = for {
      i <- 0 until n
      j <- i + 1 until n
    } yield {
      val x      = data(i).toArray
      val y      = data(j).toArray
      var sqDist = 0.0
      var k      = 0
      while (k < x.length) {
        val diff = x(k) - y(k)
        sqDist += diff * diff
        k += 1
      }
      sqDist
    }

    val sorted = distances.sorted
    val median = if (sorted.isEmpty) 1.0
    else sorted(sorted.length / 2)

    val gamma = if (median > 1e-10) 1.0 / median else 1.0
    new RBFKernel(gamma)
  }
}

/** Polynomial kernel.
  *
  * k(x, y) = (γ⟨x, y⟩ + c)^d
  *
  * Properties:
  *   - Maps to finite-dimensional feature space (dim = C(n+d, d))
  *   - Captures polynomial interactions between features
  *   - d=1, c=0 gives linear kernel
  *
  * @param degree
  *   polynomial degree (d ≥ 1)
  * @param gamma
  *   scaling factor (default: 1.0)
  * @param coef0
  *   constant term (c, default: 0.0)
  */
class PolynomialKernel(val degree: Int, val gamma: Double = 1.0, val coef0: Double = 0.0)
    extends MercerKernel {
  require(degree >= 1, s"degree must be >= 1, got $degree")
  require(gamma > 0, s"gamma must be positive, got $gamma")

  override def apply(x: Vector, y: Vector): Double = {
    val xArr = x.toArray
    val yArr = y.toArray
    var dot  = 0.0
    var i    = 0
    while (i < xArr.length) {
      dot += xArr(i) * yArr(i)
      i += 1
    }
    math.pow(gamma * dot + coef0, degree)
  }

  override def name: String = s"Polynomial(d=$degree, γ=$gamma, c=$coef0)"
}

/** Linear kernel (inner product).
  *
  * k(x, y) = ⟨x, y⟩
  *
  * Special case of polynomial kernel with d=1, γ=1, c=0.
  * Equivalent to standard K-Means in input space.
  */
class LinearKernel extends MercerKernel {

  override def apply(x: Vector, y: Vector): Double = {
    val xArr = x.toArray
    val yArr = y.toArray
    var dot  = 0.0
    var i    = 0
    while (i < xArr.length) {
      dot += xArr(i) * yArr(i)
      i += 1
    }
    dot
  }

  override def name: String = "Linear"

  override def squaredDistance(x: Vector, y: Vector): Double = {
    // For linear kernel: ||x - y||²
    val xArr   = x.toArray
    val yArr   = y.toArray
    var sqDist = 0.0
    var i      = 0
    while (i < xArr.length) {
      val diff = xArr(i) - yArr(i)
      sqDist += diff * diff
      i += 1
    }
    sqDist
  }
}

/** Sigmoid (hyperbolic tangent) kernel.
  *
  * k(x, y) = tanh(γ⟨x, y⟩ + c)
  *
  * Note: Not always positive semi-definite (not a true Mercer kernel
  * for all parameter values). Use with caution.
  *
  * @param gamma
  *   scaling factor
  * @param coef0
  *   constant term
  */
class SigmoidKernel(val gamma: Double = 1.0, val coef0: Double = 0.0) extends MercerKernel {
  require(gamma > 0, s"gamma must be positive, got $gamma")

  override def apply(x: Vector, y: Vector): Double = {
    val xArr = x.toArray
    val yArr = y.toArray
    var dot  = 0.0
    var i    = 0
    while (i < xArr.length) {
      dot += xArr(i) * yArr(i)
      i += 1
    }
    math.tanh(gamma * dot + coef0)
  }

  override def name: String = s"Sigmoid(γ=$gamma, c=$coef0)"
}

/** Laplacian kernel (exponential of negative L1 distance).
  *
  * k(x, y) = exp(-γ ||x - y||_1)
  *
  * More robust to outliers than RBF due to L1 distance.
  *
  * @param gamma
  *   bandwidth parameter
  */
class LaplacianKernel(val gamma: Double) extends MercerKernel {
  require(gamma > 0, s"gamma must be positive, got $gamma")

  override def apply(x: Vector, y: Vector): Double = {
    val xArr = x.toArray
    val yArr = y.toArray
    var l1   = 0.0
    var i    = 0
    while (i < xArr.length) {
      l1 += math.abs(xArr(i) - yArr(i))
      i += 1
    }
    math.exp(-gamma * l1)
  }

  override def name: String = s"Laplacian(γ=$gamma)"
}

/** Factory for creating Mercer kernels.
  */
object MercerKernel {

  /** Create a Mercer kernel by name.
    *
    * @param kernelType
    *   "rbf" | "polynomial" | "linear" | "sigmoid" | "laplacian"
    * @param params
    *   kernel parameters (gamma, degree, coef0)
    */
  def create(
      kernelType: String,
      gamma: Double = 1.0,
      degree: Int = 3,
      coef0: Double = 0.0
  ): MercerKernel = kernelType.toLowerCase match {
    case "rbf" | "gaussian"  => new RBFKernel(gamma)
    case "polynomial" | "poly" => new PolynomialKernel(degree, gamma, coef0)
    case "linear"              => new LinearKernel()
    case "sigmoid" | "tanh"    => new SigmoidKernel(gamma, coef0)
    case "laplacian"           => new LaplacianKernel(gamma)
    case other => throw new IllegalArgumentException(s"Unknown Mercer kernel: $other")
  }
}
