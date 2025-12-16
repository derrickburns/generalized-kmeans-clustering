package com.massivedatascience.clusterer.ml.df.kernels

import org.apache.spark.ml.linalg.{ Vector, Vectors }

/** Spherical K-Means kernel for cosine similarity clustering.
  *
  * Spherical K-Means clusters data on the unit hypersphere, making it ideal for text embeddings,
  * document vectors, and any data where direction matters more than magnitude.
  *
  * This is NOT a Bregman divergence in the strict sense, but fits the kernel interface for
  * practical clustering. Like L1Kernel, we provide it for its widespread utility.
  *
  * The algorithm:
  *   1. Normalizes all points to unit length (L2 norm = 1) 2. Uses cosine similarity as the
  *      proximity measure 3. Centers are computed as normalized mean of assigned points
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
  * Note: Input vectors are automatically normalized. Pre-normalized data will work correctly but
  * normalization is applied regardless.
  */
private[df] class SphericalKernel extends BregmanKernel {

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
    val arr   = x.toArray
    var norm  = 0.0
    var i     = 0
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
