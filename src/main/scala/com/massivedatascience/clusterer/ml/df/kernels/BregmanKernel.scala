package com.massivedatascience.clusterer.ml.df.kernels

import org.apache.spark.ml.linalg.Vector

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
