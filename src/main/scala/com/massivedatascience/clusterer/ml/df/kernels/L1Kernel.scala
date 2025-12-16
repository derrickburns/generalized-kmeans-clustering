package com.massivedatascience.clusterer.ml.df.kernels

import org.apache.spark.ml.linalg.{ Vector, Vectors }

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
private[df] class L1Kernel extends BregmanKernel {

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
