package com.massivedatascience.clusterer.ml.df

import org.apache.spark.ml.linalg.{ Vector, Vectors }

/** Kernel interface for Bregman divergences in DataFrame-based clustering.
  *
  * A Bregman divergence is defined by a strictly convex function F: S → ℝ where S ⊆ ℝ^d. The
  * divergence between points x and y is: D_F(x, y) = F(x) - F(y) - ⟨∇F(y), x - y⟩
  *
  * For clustering, we need:
  *   - grad(x): Gradient ∇F(x) (natural parameters)
  *   - invGrad(θ): Inverse gradient mapping θ → x where θ = ∇F(x) (expectation parameters)
  *   - divergence(x, μ): Compute D_F(x, μ)
  *   - validate(x): Check if x is in valid domain
  *
  * The cluster centers are computed as: μ = invGrad(∑_i w_i · grad(x_i) / ∑_i w_i)
  */
trait BregmanKernel extends Serializable {

  /** Compute the gradient ∇F(x) of the Bregman function at point x.
    *
    * @param x
    *   input point
    * @return
    *   gradient vector (natural parameters)
    */
  def grad(x: Vector): Vector

  /** Compute the inverse gradient mapping: given θ = ∇F(x), recover x.
    *
    * This maps from natural parameters to expectation parameters.
    *
    * @param theta
    *   natural parameters (gradient of some point)
    * @return
    *   point x such that ∇F(x) = theta
    */
  def invGrad(theta: Vector): Vector

  /** Compute the Bregman divergence D_F(x, mu) between point x and center mu.
    *
    * @param x
    *   data point
    * @param mu
    *   cluster center
    * @return
    *   D_F(x, mu) ≥ 0
    */
  def divergence(x: Vector, mu: Vector): Double

  /** Validate that point x is in the valid domain of the Bregman function.
    *
    * @param x
    *   point to validate
    * @return
    *   true if x is in valid domain
    */
  def validate(x: Vector): Boolean

  /** Name of the kernel for logging and identification.
    */
  def name: String

  /** Whether this kernel supports expression-based cross-join optimization. Only Squared Euclidean
    * can use this fast path.
    */
  def supportsExpressionOptimization: Boolean = false
}

/** Squared Euclidean distance kernel: F(x) = ½||x||²
  *
  * This is the most common kernel for k-means clustering.
  *   - grad(x) = x
  *   - invGrad(θ) = θ
  *   - divergence(x, μ) = ½||x - μ||²
  */
class SquaredEuclideanKernel extends BregmanKernel {
  override def grad(x: Vector): Vector = x

  override def invGrad(theta: Vector): Vector = theta

  override def divergence(x: Vector, mu: Vector): Double = {
    val xArr  = x.toArray
    val muArr = mu.toArray
    var sum   = 0.0
    var i     = 0
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

/** Kullback-Leibler divergence kernel: F(x) = ∑_i x_i log(x_i)
  *
  * Used for probability distributions and count data. Requires all components to be positive
  * (probability simplex).
  *   - grad(x) = [log(x_i) + 1]
  *   - invGrad(θ) = [exp(θ_i - 1)]
  *   - divergence(x, μ) = ∑_i x_i log(x_i / μ_i)
  *
  * @param smoothing
  *   small constant added to avoid log(0)
  */
class KLDivergenceKernel(smoothing: Double = 1e-10) extends BregmanKernel {
  require(smoothing > 0, "Smoothing must be positive")

  override def grad(x: Vector): Vector = {
    val arr    = x.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = math.log(arr(i) + smoothing) + 1.0
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGrad(theta: Vector): Vector = {
    val arr    = theta.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = math.exp(arr(i) - 1.0)
      i += 1
    }
    Vectors.dense(result)
  }

  override def divergence(x: Vector, mu: Vector): Double = {
    val xArr  = x.toArray
    val muArr = mu.toArray
    var sum   = 0.0
    var i     = 0
    while (i < xArr.length) {
      val xi  = xArr(i) + smoothing
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

/** Itakura-Saito divergence kernel: F(x) = -∑_i log(x_i)
  *
  * Used for spectral analysis and audio processing. Requires all components to be strictly
  * positive.
  *   - grad(x) = [-1/x_i]
  *   - invGrad(θ) = [-1/θ_i]
  *   - divergence(x, μ) = ∑_i (x_i/μ_i - log(x_i/μ_i) - 1)
  *
  * @param smoothing
  *   small constant added to avoid division by zero
  */
class ItakuraSaitoKernel(smoothing: Double = 1e-10) extends BregmanKernel {
  require(smoothing > 0, "Smoothing must be positive")

  override def grad(x: Vector): Vector = {
    val arr    = x.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = -1.0 / (arr(i) + smoothing)
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGrad(theta: Vector): Vector = {
    val arr    = theta.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = -1.0 / arr(i)
      i += 1
    }
    Vectors.dense(result)
  }

  override def divergence(x: Vector, mu: Vector): Double = {
    val xArr  = x.toArray
    val muArr = mu.toArray
    var sum   = 0.0
    var i     = 0
    while (i < xArr.length) {
      val xi  = xArr(i) + smoothing
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

/** Generalized I-divergence kernel: F(x) = ∑_i x_i log(x_i) - x_i
  *
  * Similar to KL but without normalization constraint. Used for non-negative data like counts.
  *   - grad(x) = [log(x_i)]
  *   - invGrad(θ) = [exp(θ_i)]
  *   - divergence(x, μ) = ∑_i (x_i log(x_i/μ_i) - x_i + μ_i)
  *
  * @param smoothing
  *   small constant added to avoid log(0)
  */
class GeneralizedIDivergenceKernel(smoothing: Double = 1e-10) extends BregmanKernel {
  require(smoothing > 0, "Smoothing must be positive")

  override def grad(x: Vector): Vector = {
    val arr    = x.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = math.log(arr(i) + smoothing)
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGrad(theta: Vector): Vector = {
    val arr    = theta.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = math.exp(arr(i))
      i += 1
    }
    Vectors.dense(result)
  }

  override def divergence(x: Vector, mu: Vector): Double = {
    val xArr  = x.toArray
    val muArr = mu.toArray
    var sum   = 0.0
    var i     = 0
    while (i < xArr.length) {
      val xi  = xArr(i) + smoothing
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

/** Logistic loss kernel: F(x) = ∑_i (x_i log(x_i) + (1-x_i)log(1-x_i))
  *
  * Used for binary probability vectors. Requires all components in (0, 1).
  *   - grad(x) = [log(x_i / (1-x_i))]
  *   - invGrad(θ) = [1 / (1 + exp(-θ_i))] (sigmoid)
  *   - divergence(x, μ) = ∑_i (x_i log(x_i/μ_i) + (1-x_i)log((1-x_i)/(1-μ_i)))
  *
  * @param smoothing
  *   small constant to keep values away from 0 and 1
  */
class LogisticLossKernel(smoothing: Double = 1e-10) extends BregmanKernel {
  require(smoothing > 0 && smoothing < 0.5, "Smoothing must be in (0, 0.5)")

  override def grad(x: Vector): Vector = {
    val arr    = x.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      val xi = math.max(smoothing, math.min(1.0 - smoothing, arr(i)))
      result(i) = math.log(xi / (1.0 - xi))
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGrad(theta: Vector): Vector = {
    val arr    = theta.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = 1.0 / (1.0 + math.exp(-arr(i)))
      i += 1
    }
    Vectors.dense(result)
  }

  override def divergence(x: Vector, mu: Vector): Double = {
    val xArr  = x.toArray
    val muArr = mu.toArray
    var sum   = 0.0
    var i     = 0
    while (i < xArr.length) {
      val xi  = math.max(smoothing, math.min(1.0 - smoothing, xArr(i)))
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

/** L1 (Manhattan distance) kernel for K-Medians clustering.
  *
  * This is NOT a Bregman divergence, but we provide it for K-Medians support. K-Medians uses
  * component-wise medians instead of gradient-based means.
  *
  *   - Distance: ||x - μ||_1 = ∑_i |x_i - μ_i|
  *   - Centers: component-wise median (not via gradient)
  *   - More robust to outliers than squared Euclidean
  *
  * Note: This kernel should be paired with MedianUpdateStrategy, not GradMeanUDAFUpdate.
  */
class L1Kernel extends BregmanKernel {

  override def grad(x: Vector): Vector = {
    // L1 is not differentiable everywhere (non-differentiable at 0)
    // Use sign function as subgradient
    val arr    = x.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
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
    val xArr  = x.toArray
    val muArr = mu.toArray
    var sum   = 0.0
    var i     = 0
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

/** Spherical K-Means kernel for cosine similarity clustering.
  *
  * Spherical K-Means clusters data on the unit hypersphere, making it ideal for text embeddings,
  * document vectors, and any data where direction matters more than magnitude.
  *
  * This is NOT a Bregman divergence in the strict sense, but fits the kernel interface for
  * practical clustering. Like L1Kernel, we provide it for its widespread utility.
  *
  * The algorithm:
  *   1. Normalizes all points to unit length (L2 norm = 1)
  *   2. Uses cosine similarity as the proximity measure
  *   3. Centers are computed as normalized mean of assigned points
  *
  * Distance: 1 - cos(x, μ) = 1 - (x · μ) / (||x|| ||μ||)
  *
  * For normalized vectors (||x|| = ||μ|| = 1): D(x, μ) = 1 - x · μ
  *
  * Properties:
  *   - D(x, μ) ∈ [0, 2] for any x, μ
  *   - D(x, x) = 0 when x is normalized
  *   - Equivalent to squared Euclidean on normalized data: ||x - μ||² = 2(1 - x·μ)
  *
  * Use cases:
  *   - Text/document clustering (TF-IDF, word embeddings)
  *   - Image feature clustering
  *   - Recommendation systems
  *   - Any high-dimensional sparse data where scale is irrelevant
  *
  * Note: Input vectors are automatically normalized. Pre-normalized data will work correctly
  * but normalization is applied regardless.
  */
class SphericalKernel extends BregmanKernel {

  /** Normalize a vector to unit length.
    *
    * @param x
    *   input vector
    * @return
    *   normalized vector with L2 norm = 1, or zero vector if input is zero
    */
  private def normalize(x: Vector): Vector = {
    val arr  = x.toArray
    var norm = 0.0
    var i    = 0
    while (i < arr.length) {
      norm += arr(i) * arr(i)
      i += 1
    }
    norm = math.sqrt(norm)

    if (norm > 1e-10) {
      val result = new Array[Double](arr.length)
      i = 0
      while (i < arr.length) {
        result(i) = arr(i) / norm
        i += 1
      }
      Vectors.dense(result)
    } else {
      // Zero vector - return as-is (will be handled gracefully)
      x
    }
  }

  /** Compute the "gradient" for spherical k-means.
    *
    * For spherical k-means, the gradient is simply the normalized vector. This allows the standard
    * gradient-mean update: μ_new = normalize(mean(x_i)) = normalize(sum(x_i) / n)
    *
    * Since we normalize in invGrad, we can just return the normalized input here.
    *
    * @param x
    *   input point
    * @return
    *   normalized vector
    */
  override def grad(x: Vector): Vector = normalize(x)

  /** Compute the inverse gradient (center computation).
    *
    * For spherical k-means: μ = normalize(∑_i x_i / n) = normalize(mean gradient)
    *
    * The weighted sum of gradients (normalized vectors) is then re-normalized to produce the new
    * center on the unit sphere.
    *
    * @param theta
    *   sum of normalized vectors (will be renormalized)
    * @return
    *   normalized center vector
    */
  override def invGrad(theta: Vector): Vector = normalize(theta)

  /** Compute cosine distance: 1 - cos(x, μ).
    *
    * @param x
    *   data point (will be normalized)
    * @param mu
    *   cluster center (assumed normalized)
    * @return
    *   cosine distance in [0, 2]
    */
  override def divergence(x: Vector, mu: Vector): Double = {
    val xNorm  = normalize(x)
    val muNorm = normalize(mu)

    val xArr  = xNorm.toArray
    val muArr = muNorm.toArray

    // Compute dot product
    var dot = 0.0
    var i   = 0
    while (i < xArr.length) {
      dot += xArr(i) * muArr(i)
      i += 1
    }

    // Cosine distance: 1 - cos(x, mu)
    // Clamp dot product to [-1, 1] to handle numerical errors
    1.0 - math.max(-1.0, math.min(1.0, dot))
  }

  /** Validate that point has finite values and non-zero norm.
    *
    * @param x
    *   point to validate
    * @return
    *   true if valid for spherical k-means
    */
  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    var norm = 0.0
    var i = 0
    var valid = true
    while (i < arr.length && valid) {
      val v = arr(i)
      if (v.isNaN || v.isInfinity) {
        valid = false
      } else {
        norm += v * v
      }
      i += 1
    }
    // Must have non-zero norm (can't normalize zero vector)
    valid && norm > 1e-20
  }

  override def name: String = "Spherical"

  /** Spherical kernel does not support expression optimization (requires normalization). */
  override def supportsExpressionOptimization: Boolean = false
}
