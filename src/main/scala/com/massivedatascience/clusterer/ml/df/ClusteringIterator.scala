/*
 * Licensed to the Massive Data Science and Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Massive Data Science and Derrick R. Burns licenses this file to You under the
 * Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.massivedatascience.clusterer.ml.df

import org.apache.spark.internal.Logging
import org.apache.spark.sql.DataFrame

/** Base configuration for all clustering iterators.
  *
  * @param maxIter
  *   maximum number of iterations
  * @param tol
  *   convergence tolerance
  * @param checkpointInterval
  *   checkpoint every N iterations (0 = disabled)
  * @param checkpointDir
  *   checkpoint directory (None = use Spark default)
  */
case class IteratorConfig(
    maxIter: Int,
    tol: Double,
    checkpointInterval: Int = 10,
    checkpointDir: Option[String] = None
)

/** Generic result from any clustering iteration algorithm.
  *
  * @param centers
  *   final cluster centers (or parameters for mixture models)
  * @param iterations
  *   number of iterations performed
  * @param objectiveHistory
  *   objective function value at each iteration (distortion, likelihood, etc.)
  * @param converged
  *   whether algorithm converged
  * @param metadata
  *   algorithm-specific metadata (e.g., movement history, empty cluster events)
  */
case class ClusteringResult(
    centers: Array[Array[Double]],
    iterations: Int,
    objectiveHistory: Array[Double],
    converged: Boolean,
    metadata: Map[String, Any] = Map.empty
) {

  /** Get movement history if available (from Lloyd's-style algorithms). */
  def movementHistory: Option[Array[Double]] =
    metadata.get("movementHistory").map(_.asInstanceOf[Array[Double]])

  /** Get empty cluster event count if available. */
  def emptyClusterEvents: Option[Int] =
    metadata.get("emptyClusterEvents").map(_.asInstanceOf[Int])

  /** Get component weights if available (from mixture models). */
  def componentWeights: Option[Array[Double]] =
    metadata.get("componentWeights").map(_.asInstanceOf[Array[Double]])

  /** Get log-likelihood history if available (from EM algorithms). */
  def logLikelihoodHistory: Option[Array[Double]] =
    metadata.get("logLikelihoodHistory").map(_.asInstanceOf[Array[Double]])

  /** Convert to LloydResult for backward compatibility. */
  def toLloydResult: LloydResult = LloydResult(
    centers = centers,
    iterations = iterations,
    distortionHistory = objectiveHistory,
    movementHistory = movementHistory.getOrElse(Array.empty),
    converged = converged,
    emptyClusterEvents = emptyClusterEvents.getOrElse(0)
  )
}

/** State maintained across iterations.
  *
  * @param df
  *   current DataFrame (may include computed columns)
  * @param parameters
  *   current model parameters (centers, weights, etc.)
  * @param iteration
  *   current iteration number
  * @param objective
  *   current objective value
  */
case class IterationState(
    df: DataFrame,
    parameters: Map[String, Any],
    iteration: Int,
    objective: Double
)

/** Base trait for all clustering iteration algorithms.
  *
  * This provides a common interface and shared functionality for different clustering iteration
  * patterns:
  *   - Lloyd's algorithm (assign → update → check)
  *   - EM algorithm (E-step → M-step → likelihood)
  *   - Agglomerative (merge → update linkage)
  *   - Information Bottleneck (optimize T|Y → optimize T|X)
  *
  * Implementations override the core iteration logic while inheriting checkpointing, logging, and
  * convergence tracking.
  */
trait ClusteringIterator extends Logging with Serializable {

  /** Run the clustering algorithm until convergence or max iterations.
    *
    * @param df
    *   input DataFrame with features
    * @param featuresCol
    *   name of features column
    * @param weightCol
    *   optional weight column
    * @param initialState
    *   initial algorithm state (centers, parameters)
    * @param config
    *   iteration configuration
    * @return
    *   clustering result with final parameters and statistics
    */
  def run(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      initialState: IterationState,
      config: IteratorConfig
  ): ClusteringResult

  /** Perform a single iteration step.
    *
    * @param state
    *   current iteration state
    * @param featuresCol
    *   name of features column
    * @param weightCol
    *   optional weight column
    * @return
    *   updated state after one iteration
    */
  protected def iterate(
      state: IterationState,
      featuresCol: String,
      weightCol: Option[String]
  ): IterationState

  /** Check if algorithm has converged.
    *
    * @param previousObjective
    *   objective from previous iteration
    * @param currentObjective
    *   objective from current iteration
    * @param tol
    *   convergence tolerance
    * @return
    *   true if converged
    */
  protected def hasConverged(
      previousObjective: Double,
      currentObjective: Double,
      tol: Double
  ): Boolean = {
    val improvement = math.abs(previousObjective - currentObjective) /
      math.max(math.abs(previousObjective), 1e-10)
    improvement < tol
  }

  /** Setup checkpointing if configured.
    *
    * @param df
    *   DataFrame to configure
    * @param config
    *   iterator configuration
    */
  protected def setupCheckpointing(df: DataFrame, config: IteratorConfig): Unit = {
    config.checkpointDir.foreach { dir =>
      df.sparkSession.sparkContext.setCheckpointDir(dir)
    }
  }

  /** Checkpoint DataFrame if needed based on iteration count.
    *
    * @param df
    *   DataFrame to potentially checkpoint
    * @param iteration
    *   current iteration number
    * @param config
    *   iterator configuration
    * @return
    *   checkpointed DataFrame if checkpointing occurred, otherwise original
    */
  protected def maybeCheckpoint(
      df: DataFrame,
      iteration: Int,
      config: IteratorConfig
  ): (DataFrame, Boolean) = {
    if (config.checkpointInterval > 0 && iteration % config.checkpointInterval == 0) {
      logInfo(s"Checkpointing at iteration $iteration")
      (df.checkpoint(), true)
    } else {
      (df, false)
    }
  }
}

/** Abstract base class providing common iteration loop logic.
  *
  * Subclasses only need to implement the `iterate` method for their specific algorithm (Lloyd's,
  * EM, etc.).
  */
abstract class AbstractClusteringIterator extends ClusteringIterator {

  import scala.collection.mutable.ArrayBuffer

  override def run(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      initialState: IterationState,
      config: IteratorConfig
  ): ClusteringResult = {

    logInfo(s"Starting clustering iteration with maxIter=${config.maxIter}, tol=${config.tol}")

    setupCheckpointing(df, config)

    var state             = initialState.copy(df = df.cache())
    var previousObjective = Double.MaxValue
    var converged         = false
    val objectiveHistory  = ArrayBuffer[Double]()
    val metadataBuilder   = scala.collection.mutable.Map[String, Any]()

    try {
      while (state.iteration < config.maxIter && !converged) {
        val newState = iterate(state, featuresCol, weightCol)

        objectiveHistory += newState.objective
        logInfo(f"Iteration ${newState.iteration}: objective=${newState.objective}%.6f")

        // Check convergence
        converged = hasConverged(previousObjective, newState.objective, config.tol)

        if (converged) {
          logInfo(s"Converged after ${newState.iteration} iterations")
        }

        // Checkpoint if needed
        val (checkpointedDF, didCheckpoint) =
          maybeCheckpoint(newState.df, newState.iteration, config)
        if (didCheckpoint) {
          state.df.unpersist()
        }

        previousObjective = newState.objective
        state = newState.copy(df = checkpointedDF)
      }

      if (!converged) {
        logWarning(s"Did not converge after ${config.maxIter} iterations")
      }

      // Extract final centers from parameters
      val centers = state.parameters.get("centers") match {
        case Some(c: Array[Array[Double] @unchecked]) => c
        case _                                        => Array.empty[Array[Double]]
      }

      // Collect additional metadata from final state
      state.parameters.foreach {
        case ("centers", _) => // Skip, already extracted
        case (k, v)         => metadataBuilder += (k -> v)
      }

      ClusteringResult(
        centers = centers,
        iterations = state.iteration,
        objectiveHistory = objectiveHistory.toArray,
        converged = converged,
        metadata = metadataBuilder.toMap
      )

    } finally {
      if (state.df != df) {
        state.df.unpersist()
      }
      df.unpersist()
    }
  }
}
