package com.massivedatascience.clusterer.ml.df.kernels

import org.apache.spark.ml.linalg.{ Vector, Vectors }

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
private[df] class KLDivergenceKernel(smoothing: Double = 1e-10) extends BregmanKernel {
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
