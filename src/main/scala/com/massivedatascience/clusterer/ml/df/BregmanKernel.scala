package com.massivedatascience.clusterer.ml.df

import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
 * Kernel interface for Bregman divergences in DataFrame-based clustering.
 *
 * A Bregman divergence is defined by a strictly convex function F: S → ℝ where S ⊆ ℝ^d.
 * The divergence between points x and y is:
 *   D_F(x, y) = F(x) - F(y) - ⟨∇F(y), x - y⟩
 *
 * For clustering, we need:
 * - grad(x): Gradient ∇F(x) (natural parameters)
 * - invGrad(θ): Inverse gradient mapping θ → x where θ = ∇F(x) (expectation parameters)
 * - divergence(x, μ): Compute D_F(x, μ)
 * - validate(x): Check if x is in valid domain
 *
 * The cluster centers are computed as:
 *   μ = invGrad(∑_i w_i · grad(x_i) / ∑_i w_i)
 */
trait BregmanKernel extends Serializable {

  /**
   * Compute the gradient ∇F(x) of the Bregman function at point x.
   *
   * @param x input point
   * @return gradient vector (natural parameters)
   */
  def grad(x: Vector): Vector

  /**
   * Compute the inverse gradient mapping: given θ = ∇F(x), recover x.
   *
   * This maps from natural parameters to expectation parameters.
   *
   * @param theta natural parameters (gradient of some point)
   * @return point x such that ∇F(x) = theta
   */
  def invGrad(theta: Vector): Vector

  /**
   * Compute the Bregman divergence D_F(x, mu) between point x and center mu.
   *
   * @param x    data point
   * @param mu   cluster center
   * @return D_F(x, mu) ≥ 0
   */
  def divergence(x: Vector, mu: Vector): Double

  /**
   * Validate that point x is in the valid domain of the Bregman function.
   *
   * @param x point to validate
   * @return true if x is in valid domain
   */
  def validate(x: Vector): Boolean

  /**
   * Name of the kernel for logging and identification.
   */
  def name: String

  /**
   * Whether this kernel supports expression-based cross-join optimization.
   * Only Squared Euclidean can use this fast path.
   */
  def supportsExpressionOptimization: Boolean = false
}

/**
 * Squared Euclidean distance kernel: F(x) = ½||x||²
 *
 * This is the most common kernel for k-means clustering.
 * - grad(x) = x
 * - invGrad(θ) = θ
 * - divergence(x, μ) = ½||x - μ||²
 */
class SquaredEuclideanKernel extends BregmanKernel {
  override def grad(x: Vector): Vector = x

  override def invGrad(theta: Vector): Vector = theta

  override def divergence(x: Vector, mu: Vector): Double = {
    val xArr = x.toArray
    val muArr = mu.toArray
    var sum = 0.0
    var i = 0
    while (i < xArr.length) {
      val diff = xArr(i) - muArr(i)
      sum += diff * diff
      i += 1
    }
    sum * 0.5
  }

  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    arr.forall(v => !v.isNaN && !v.isInfinity)
  }

  override def name: String = "SquaredEuclidean"

  override def supportsExpressionOptimization: Boolean = true
}

/**
 * Kullback-Leibler divergence kernel: F(x) = ∑_i x_i log(x_i)
 *
 * Used for probability distributions and count data.
 * Requires all components to be positive (probability simplex).
 * - grad(x) = [log(x_i) + 1]
 * - invGrad(θ) = [exp(θ_i - 1)]
 * - divergence(x, μ) = ∑_i x_i log(x_i / μ_i)
 *
 * @param smoothing small constant added to avoid log(0)
 */
class KLDivergenceKernel(smoothing: Double = 1e-10) extends BregmanKernel {
  require(smoothing > 0, "Smoothing must be positive")

  override def grad(x: Vector): Vector = {
    val arr = x.toArray
    val result = new Array[Double](arr.length)
    var i = 0
    while (i < arr.length) {
      result(i) = math.log(arr(i) + smoothing) + 1.0
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGrad(theta: Vector): Vector = {
    val arr = theta.toArray
    val result = new Array[Double](arr.length)
    var i = 0
    while (i < arr.length) {
      result(i) = math.exp(arr(i) - 1.0)
      i += 1
    }
    Vectors.dense(result)
  }

  override def divergence(x: Vector, mu: Vector): Double = {
    val xArr = x.toArray
    val muArr = mu.toArray
    var sum = 0.0
    var i = 0
    while (i < xArr.length) {
      val xi = xArr(i) + smoothing
      val mui = muArr(i) + smoothing
      sum += xi * math.log(xi / mui)
      i += 1
    }
    sum
  }

  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    arr.forall(v => !v.isNaN && !v.isInfinity && v >= 0.0)
  }

  override def name: String = s"KL(smoothing=$smoothing)"
}

/**
 * Itakura-Saito divergence kernel: F(x) = -∑_i log(x_i)
 *
 * Used for spectral analysis and audio processing.
 * Requires all components to be strictly positive.
 * - grad(x) = [-1/x_i]
 * - invGrad(θ) = [-1/θ_i]
 * - divergence(x, μ) = ∑_i (x_i/μ_i - log(x_i/μ_i) - 1)
 *
 * @param smoothing small constant added to avoid division by zero
 */
class ItakuraSaitoKernel(smoothing: Double = 1e-10) extends BregmanKernel {
  require(smoothing > 0, "Smoothing must be positive")

  override def grad(x: Vector): Vector = {
    val arr = x.toArray
    val result = new Array[Double](arr.length)
    var i = 0
    while (i < arr.length) {
      result(i) = -1.0 / (arr(i) + smoothing)
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGrad(theta: Vector): Vector = {
    val arr = theta.toArray
    val result = new Array[Double](arr.length)
    var i = 0
    while (i < arr.length) {
      result(i) = -1.0 / arr(i)
      i += 1
    }
    Vectors.dense(result)
  }

  override def divergence(x: Vector, mu: Vector): Double = {
    val xArr = x.toArray
    val muArr = mu.toArray
    var sum = 0.0
    var i = 0
    while (i < xArr.length) {
      val xi = xArr(i) + smoothing
      val mui = muArr(i) + smoothing
      sum += xi / mui - math.log(xi / mui) - 1.0
      i += 1
    }
    sum
  }

  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    arr.forall(v => !v.isNaN && !v.isInfinity && v > 0.0)
  }

  override def name: String = s"ItakuraSaito(smoothing=$smoothing)"
}

/**
 * Generalized I-divergence kernel: F(x) = ∑_i x_i log(x_i) - x_i
 *
 * Similar to KL but without normalization constraint.
 * Used for non-negative data like counts.
 * - grad(x) = [log(x_i)]
 * - invGrad(θ) = [exp(θ_i)]
 * - divergence(x, μ) = ∑_i (x_i log(x_i/μ_i) - x_i + μ_i)
 *
 * @param smoothing small constant added to avoid log(0)
 */
class GeneralizedIDivergenceKernel(smoothing: Double = 1e-10) extends BregmanKernel {
  require(smoothing > 0, "Smoothing must be positive")

  override def grad(x: Vector): Vector = {
    val arr = x.toArray
    val result = new Array[Double](arr.length)
    var i = 0
    while (i < arr.length) {
      result(i) = math.log(arr(i) + smoothing)
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGrad(theta: Vector): Vector = {
    val arr = theta.toArray
    val result = new Array[Double](arr.length)
    var i = 0
    while (i < arr.length) {
      result(i) = math.exp(arr(i))
      i += 1
    }
    Vectors.dense(result)
  }

  override def divergence(x: Vector, mu: Vector): Double = {
    val xArr = x.toArray
    val muArr = mu.toArray
    var sum = 0.0
    var i = 0
    while (i < xArr.length) {
      val xi = xArr(i) + smoothing
      val mui = muArr(i) + smoothing
      sum += xi * math.log(xi / mui) - xi + mui
      i += 1
    }
    sum
  }

  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    arr.forall(v => !v.isNaN && !v.isInfinity && v >= 0.0)
  }

  override def name: String = s"GeneralizedI(smoothing=$smoothing)"
}

/**
 * Logistic loss kernel: F(x) = ∑_i (x_i log(x_i) + (1-x_i)log(1-x_i))
 *
 * Used for binary probability vectors.
 * Requires all components in (0, 1).
 * - grad(x) = [log(x_i / (1-x_i))]
 * - invGrad(θ) = [1 / (1 + exp(-θ_i))]  (sigmoid)
 * - divergence(x, μ) = ∑_i (x_i log(x_i/μ_i) + (1-x_i)log((1-x_i)/(1-μ_i)))
 *
 * @param smoothing small constant to keep values away from 0 and 1
 */
class LogisticLossKernel(smoothing: Double = 1e-10) extends BregmanKernel {
  require(smoothing > 0 && smoothing < 0.5, "Smoothing must be in (0, 0.5)")

  override def grad(x: Vector): Vector = {
    val arr = x.toArray
    val result = new Array[Double](arr.length)
    var i = 0
    while (i < arr.length) {
      val xi = math.max(smoothing, math.min(1.0 - smoothing, arr(i)))
      result(i) = math.log(xi / (1.0 - xi))
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGrad(theta: Vector): Vector = {
    val arr = theta.toArray
    val result = new Array[Double](arr.length)
    var i = 0
    while (i < arr.length) {
      result(i) = 1.0 / (1.0 + math.exp(-arr(i)))
      i += 1
    }
    Vectors.dense(result)
  }

  override def divergence(x: Vector, mu: Vector): Double = {
    val xArr = x.toArray
    val muArr = mu.toArray
    var sum = 0.0
    var i = 0
    while (i < xArr.length) {
      val xi = math.max(smoothing, math.min(1.0 - smoothing, xArr(i)))
      val mui = math.max(smoothing, math.min(1.0 - smoothing, muArr(i)))
      sum += xi * math.log(xi / mui) + (1.0 - xi) * math.log((1.0 - xi) / (1.0 - mui))
      i += 1
    }
    sum
  }

  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    arr.forall(v => !v.isNaN && !v.isInfinity && v >= 0.0 && v <= 1.0)
  }

  override def name: String = s"LogisticLoss(smoothing=$smoothing)"
}

/**
 * L1 (Manhattan distance) kernel for K-Medians clustering.
 *
 * This is NOT a Bregman divergence, but we provide it for K-Medians support.
 * K-Medians uses component-wise medians instead of gradient-based means.
 *
 * - Distance: ||x - μ||_1 = ∑_i |x_i - μ_i|
 * - Centers: component-wise median (not via gradient)
 * - More robust to outliers than squared Euclidean
 *
 * Note: This kernel should be paired with MedianUpdateStrategy, not GradMeanUDAFUpdate.
 */
class L1Kernel extends BregmanKernel {

  override def grad(x: Vector): Vector = {
    // L1 is not differentiable everywhere (non-differentiable at 0)
    // Use sign function as subgradient
    val arr = x.toArray
    val result = new Array[Double](arr.length)
    var i = 0
    while (i < arr.length) {
      result(i) = if (arr(i) > 0) 1.0 else if (arr(i) < 0) -1.0 else 0.0
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGrad(theta: Vector): Vector = {
    // Inverse of sign function is not well-defined
    // For K-Medians, we compute medians directly, not via gradient
    // Return theta as identity (will be overridden by MedianUpdateStrategy)
    theta
  }

  override def divergence(x: Vector, mu: Vector): Double = {
    // Manhattan distance (L1 norm)
    val xArr = x.toArray
    val muArr = mu.toArray
    var sum = 0.0
    var i = 0
    while (i < xArr.length) {
      sum += math.abs(xArr(i) - muArr(i))
      i += 1
    }
    sum
  }

  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    arr.forall(v => !v.isNaN && !v.isInfinity)
  }

  override def name: String = "L1"
}
