package com.massivedatascience.clusterer.ml.df

import org.apache.spark.internal.Logging
import org.apache.spark.sql.DataFrame
import scala.collection.mutable.ArrayBuffer

/**
 * Configuration for Lloyd's algorithm.
 *
 * @param k                  number of clusters
 * @param maxIter           maximum number of iterations
 * @param tol               convergence tolerance (max center movement)
 * @param kernel            Bregman divergence kernel
 * @param assigner          assignment strategy
 * @param updater           center update strategy
 * @param emptyHandler      empty cluster handler
 * @param convergence       convergence check
 * @param validator         input validation
 * @param checkpointInterval checkpoint every N iterations (0 = disabled)
 * @param checkpointDir     checkpoint directory (None = use Spark default)
 */
case class LloydsConfig(
    k: Int,
    maxIter: Int,
    tol: Double,
    kernel: BregmanKernel,
    assigner: AssignmentStrategy,
    updater: UpdateStrategy,
    emptyHandler: EmptyClusterHandler,
    convergence: ConvergenceCheck,
    validator: InputValidator,
    checkpointInterval: Int = 10,
    checkpointDir: Option[String] = None)

/**
 * Result of Lloyd's algorithm.
 *
 * @param centers             final cluster centers
 * @param iterations         number of iterations performed
 * @param distortionHistory  distortion at each iteration
 * @param movementHistory    max center movement at each iteration
 * @param converged          whether algorithm converged
 * @param emptyClusterEvents number of times empty clusters were handled
 */
case class LloydResult(
    centers: Array[Array[Double]],
    iterations: Int,
    distortionHistory: Array[Double],
    movementHistory: Array[Double],
    converged: Boolean,
    emptyClusterEvents: Int)

/**
 * Core abstraction for Lloyd's algorithm (k-means iteration).
 *
 * This trait defines the single source of truth for the Lloyd's algorithm loop:
 * 1. Assign each point to nearest center
 * 2. Update centers based on assignments
 * 3. Handle empty clusters
 * 4. Check convergence
 * 5. Repeat until convergence or max iterations
 *
 * All clustering algorithms (standard k-means, soft k-means, constrained k-means, etc.)
 * can be implemented by plugging in different strategies.
 */
trait LloydsIterator extends Logging {

  /**
   * Run Lloyd's algorithm until convergence or max iterations.
   *
   * @param df             input DataFrame with features column
   * @param featuresCol    name of features column
   * @param weightCol      optional name of weight column
   * @param initialCenters initial cluster centers
   * @param config         algorithm configuration
   * @return result with final centers and statistics
   */
  def run(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      initialCenters: Array[Array[Double]],
      config: LloydsConfig): LloydResult
}

/**
 * Default implementation of Lloyd's iterator.
 *
 * This is the single implementation of the Lloyd's algorithm loop.
 * Different clustering behaviors are achieved through pluggable strategies.
 */
class DefaultLloydsIterator extends LloydsIterator {

  override def run(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      initialCenters: Array[Array[Double]],
      config: LloydsConfig): LloydResult = {

    logInfo(s"Starting Lloyd's algorithm with k=${config.k}, maxIter=${config.maxIter}, " +
      s"tol=${config.tol}, kernel=${config.kernel.name}")

    // Validate input
    config.validator.validate(df, featuresCol, weightCol, config.kernel)

    // Setup checkpoint directory if specified
    config.checkpointDir.foreach { dir =>
      df.sparkSession.sparkContext.setCheckpointDir(dir)
    }

    var centers = initialCenters
    var iter = 0
    var converged = false
    val distortionHistory = ArrayBuffer[Double]()
    val movementHistory = ArrayBuffer[Double]()
    var emptyClusterEvents = 0
    var currentDF = df.cache()

    try {
      while (iter < config.maxIter && !converged) {
        iter += 1
        logInfo(s"Starting iteration $iter")

        // Assignment step: assign each point to nearest center
        val assigned = config.assigner.assign(
          currentDF,
          featuresCol,
          weightCol,
          centers,
          config.kernel)

        // Materialize assignments
        val assignedCached = assigned.cache()
        val assignmentCount = assignedCached.count()
        logInfo(s"Iteration $iter: assigned $assignmentCount points")

        // Update step: compute new centers
        val newCenters = config.updater.update(
          assignedCached,
          featuresCol,
          weightCol,
          config.k,
          config.kernel)

        logDebug(s"Iteration $iter: computed ${newCenters.length} new centers")

        // Handle empty clusters
        val (finalCenters, emptyCount) = config.emptyHandler.handle(
          assignedCached,
          featuresCol,
          weightCol,
          newCenters,
          currentDF,
          config.kernel)

        if (emptyCount > 0) {
          emptyClusterEvents += 1
          logWarning(s"Iteration $iter: handled $emptyCount empty clusters " +
            s"(total events: $emptyClusterEvents)")
        }

        // Check convergence
        val (movement, distortion) = config.convergence.check(
          centers,
          finalCenters,
          assignedCached,
          featuresCol,
          weightCol,
          config.kernel)

        distortionHistory += distortion
        movementHistory += movement

        logInfo(f"Iteration $iter: distortion=$distortion%.6f, movement=$movement%.6f")

        converged = movement < config.tol

        if (converged) {
          logInfo(s"Converged after $iter iterations (movement $movement < tol ${config.tol})")
        }

        // Checkpoint if needed
        if (config.checkpointInterval > 0 && iter % config.checkpointInterval == 0) {
          logInfo(s"Checkpointing at iteration $iter")
          val checkpointed = assignedCached.checkpoint()
          assignedCached.unpersist()
          if (currentDF != df) {
            currentDF.unpersist()
          }
          currentDF = checkpointed
        } else {
          assignedCached.unpersist()
        }

        centers = finalCenters
      }

      if (!converged) {
        logWarning(s"Did not converge after ${config.maxIter} iterations " +
          s"(final movement: ${movementHistory.lastOption.getOrElse(Double.NaN)})")
      }

      LloydResult(
        centers = centers,
        iterations = iter,
        distortionHistory = distortionHistory.toArray,
        movementHistory = movementHistory.toArray,
        converged = converged,
        emptyClusterEvents = emptyClusterEvents)

    } finally {
      // Cleanup
      if (currentDF != df) {
        currentDF.unpersist()
      }
      df.unpersist()
    }
  }
}
