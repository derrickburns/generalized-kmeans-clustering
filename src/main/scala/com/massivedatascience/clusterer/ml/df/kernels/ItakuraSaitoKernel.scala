package com.massivedatascience.clusterer.ml.df.kernels

import org.apache.spark.ml.linalg.{ Vector, Vectors }

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
private[df] class ItakuraSaitoKernel(smoothing: Double = 1e-10) extends BregmanKernel {
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
