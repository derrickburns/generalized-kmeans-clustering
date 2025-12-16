package com.massivedatascience.clusterer.ml

/** DataFrame-based K-means clustering algorithms.
  *
  * This package provides the core implementation of Bregman K-means clustering using Spark's
  * DataFrame API.
  *
  * ==Key Components==
  *
  *   - [[df.BregmanKernel]]: Bregman divergence kernels for different distance measures
  *   - [[df.BregmanFunction]]: Mathematical functions underlying divergences
  *   - [[df.LloydsIterator]]: Core Lloyd's algorithm implementation
  *
  * ==Strategy Types (from strategies subpackage)==
  *
  *   - [[df.AssignmentStrategy]]: Assign points to nearest cluster centers
  *   - [[df.UpdateStrategy]]: Update cluster centers from assignments
  *   - [[df.EmptyClusterHandler]]: Handle empty clusters during iterations
  *   - [[df.ConvergenceCheck]]: Check for algorithm convergence
  *   - [[df.InputValidator]]: Validate input data
  */
package object df {
  // Re-export strategy types for backward compatibility
  // Users importing com.massivedatascience.clusterer.ml.df._ get these types

  /** Factory for creating BregmanKernel instances by name.
    *
    * @param divergence
    *   divergence name: "squaredEuclidean", "kl", "itakuraSaito", etc.
    * @param smoothing
    *   smoothing parameter for divergences with domain constraints
    * @return
    *   BregmanKernel instance
    */
  object BregmanKernel {
    def create(divergence: String, smoothing: Double = 1e-10): kernels.BregmanKernel =
      divergence.toLowerCase match {
        case "squaredeuclidean" | "se" | "euclidean" => new kernels.SquaredEuclideanKernel()
        case "kl" | "kullbackleibler"                => new kernels.KLDivergenceKernel(smoothing)
        case "itakurasaito" | "is"                   => new kernels.ItakuraSaitoKernel(smoothing)
        case "generalizedi" | "geni"                 => new kernels.GeneralizedIDivergenceKernel(smoothing)
        case "logistic"                              => new kernels.LogisticLossKernel(smoothing)
        case "l1" | "manhattan"                      => new kernels.L1Kernel()
        case "spherical" | "cosine"                  => new kernels.SphericalKernel()
        case other                                   => throw new IllegalArgumentException(s"Unknown divergence: $other")
      }
  }

  // Assignment strategies
  type AssignmentStrategy          = strategies.AssignmentStrategy
  type BroadcastUDFAssignment      = strategies.BroadcastUDFAssignment
  type SECrossJoinAssignment       = strategies.SECrossJoinAssignment
  type ChunkedBroadcastAssignment  = strategies.ChunkedBroadcastAssignment
  type AdaptiveBroadcastAssignment = strategies.AdaptiveBroadcastAssignment
  type AutoAssignment              = strategies.AutoAssignment

  // Update strategies
  type UpdateStrategy       = strategies.UpdateStrategy
  type GradMeanUDAFUpdate   = strategies.GradMeanUDAFUpdate
  type MedianUpdateStrategy = strategies.MedianUpdateStrategy

  // Empty cluster handlers
  type EmptyClusterHandler      = strategies.EmptyClusterHandler
  type ReseedRandomHandler      = strategies.ReseedRandomHandler
  type DropEmptyClustersHandler = strategies.DropEmptyClustersHandler

  // Convergence checks
  type ConvergenceCheck    = strategies.ConvergenceCheck
  type MovementConvergence = strategies.MovementConvergence

  // Input validators
  type InputValidator         = strategies.InputValidator
  type StandardInputValidator = strategies.StandardInputValidator

  // Kernel types (from kernels subpackage)
  type BregmanKernel                = kernels.BregmanKernel
  type SquaredEuclideanKernel       = kernels.SquaredEuclideanKernel
  type KLDivergenceKernel           = kernels.KLDivergenceKernel
  type ItakuraSaitoKernel           = kernels.ItakuraSaitoKernel
  type GeneralizedIDivergenceKernel = kernels.GeneralizedIDivergenceKernel
  type LogisticLossKernel           = kernels.LogisticLossKernel
  type L1Kernel                     = kernels.L1Kernel
  type SphericalKernel              = kernels.SphericalKernel
}
