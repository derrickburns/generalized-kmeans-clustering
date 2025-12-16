package com.massivedatascience.clusterer.ml.df.kernels

import org.apache.spark.ml.linalg.{ Vector, Vectors }

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
private[df] class GeneralizedIDivergenceKernel(smoothing: Double = 1e-10) extends BregmanKernel {
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
