package com.massivedatascience.clusterer.ml.df.kernels

import org.apache.spark.ml.linalg.{ Vector, Vectors }

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
private[df] class LogisticLossKernel(smoothing: Double = 1e-10) extends BregmanKernel {
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
