package com.massivedatascience.clusterer.ml.df

/** Strategy implementations for clustering algorithms.
  *
  * This package contains modular strategies for different aspects of K-means clustering:
  *
  *   - [[strategies.AssignmentStrategy]]: Assign points to clusters
  *   - [[strategies.UpdateStrategy]]: Update cluster centers
  *   - [[strategies.EmptyClusterHandler]]: Handle empty clusters
  *   - [[strategies.ConvergenceCheck]]: Check for convergence
  *   - [[strategies.InputValidator]]: Validate input data
  *
  * Assignment strategy implementations are in [[strategies.impl]].
  */
package object strategies {
  // Re-export implementation types for backward compatibility
  type BroadcastUDFAssignment = impl.BroadcastUDFAssignment
  type SECrossJoinAssignment = impl.SECrossJoinAssignment
  type ChunkedBroadcastAssignment = impl.ChunkedBroadcastAssignment
  type AdaptiveBroadcastAssignment = impl.AdaptiveBroadcastAssignment
  type AutoAssignment = impl.AutoAssignment
}
