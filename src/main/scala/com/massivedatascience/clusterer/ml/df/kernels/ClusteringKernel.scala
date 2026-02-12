package com.massivedatascience.clusterer.ml.df.kernels

import org.apache.spark.ml.linalg.Vector

/** Base interface for clustering distance kernels.
  *
  * This is the root trait for all kernels used in clustering. It provides the minimal interface
  * needed for assignment (computing divergence) and validation.
  *
  * Subtypes:
  *   - [[BregmanKernel]]: Kernels based on Bregman divergences (have grad/invGrad for
  *     gradient-based center computation)
  *   - Direct implementations: Non-Bregman kernels like L1 that use alternative center computation
  *     (e.g., component-wise median)
  */
trait ClusteringKernel extends Serializable {

  /** Compute the divergence/distance between point x and center mu.
    *
    * @param x
    *   data point
    * @param mu
    *   cluster center
    * @return
    *   D(x, mu) >= 0
    */
  def divergence(x: Vector, mu: Vector): Double

  /** Validate that point x is in the valid domain of this kernel.
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
