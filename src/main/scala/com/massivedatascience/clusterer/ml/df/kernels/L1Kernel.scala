package com.massivedatascience.clusterer.ml.df.kernels

import org.apache.spark.ml.linalg.Vector

/** L1 (Manhattan distance) kernel for K-Medians clustering.
  *
  * This is NOT a Bregman divergence. It extends [[ClusteringKernel]] directly because L1 has no
  * well-defined gradient or inverse gradient. Centers must be computed via component-wise median
  * (MedianUpdateStrategy), not gradient-based averaging (GradMeanUDAFUpdate).
  *
  *   - Distance: ||x - μ||_1 = ∑_i |x_i - μ_i|
  *   - Centers: component-wise median (not via gradient)
  *   - More robust to outliers than squared Euclidean
  *
  * Note: This kernel must be paired with MedianUpdateStrategy, not GradMeanUDAFUpdate.
  */
private[df] class L1Kernel extends ClusteringKernel {

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
