package com.massivedatascience.clusterer.ml.df.kernels

import org.apache.spark.ml.linalg.Vector

/** Squared Euclidean distance kernel: F(x) = ½||x||²
  *
  * This is the most common kernel for k-means clustering.
  *   - grad(x) = x
  *   - invGrad(θ) = θ
  *   - divergence(x, μ) = ½||x - μ||²
  */
private[df] class SquaredEuclideanKernel extends BregmanKernel {
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
