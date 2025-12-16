package com.massivedatascience.clusterer.ml.df

/** Re-exports strategy types from the strategies subpackage for backward compatibility.
  *
  * New code should import directly from the strategies package:
  * {{{
  * import com.massivedatascience.clusterer.ml.df.strategies._
  * }}}
  */
object Strategies {
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
}

// Also export traits at the package level for direct imports
// Note: These are re-exported from the strategies subpackage
