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

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import scala.collection.mutable.ArrayBuffer

/** Helper object with static methods to avoid serialization issues in UDFs. */
object ElkanHelpers {

  /** Compute squared Euclidean distance (Bregman convention: half squared). */
  def squaredEuclidean(a: Array[Double], b: Array[Double]): Double = {
    var sum = 0.0
    var i   = 0
    while (i < a.length) {
      val diff = a(i) - b(i)
      sum += diff * diff
      i += 1
    }
    sum * 0.5
  }

  /** Compute Euclidean distance (not squared) between two arrays. */
  def euclideanDistance(a: Array[Double], b: Array[Double]): Double = {
    var sum = 0.0
    var i   = 0
    while (i < a.length) {
      val diff = a(i) - b(i)
      sum += diff * diff
      i += 1
    }
    math.sqrt(sum)
  }
}

/** Elkan-accelerated Lloyd's iterator for Squared Euclidean distance.
  *
  * Uses the triangle inequality to dramatically reduce distance computations by tracking per-point
  * bounds across iterations.
  *
  * ==Algorithm Overview==
  *
  * The key insight is that cluster assignments change slowly as the algorithm converges. By
  * tracking bounds on distances, we can often prove a point's assignment hasn't changed without
  * computing any distances.
  *
  * '''Per-point state (stored as DataFrame columns):'''
  *   - `_upper_bound`: Upper bound on distance to assigned center
  *   - `_lower_bound`: Lower bound on distance to second-closest center
  *   - `_cluster`: Current cluster assignment
  *
  * '''Per-iteration:'''
  *   1. Compute center movements from previous iteration 2. Update bounds: upper +=
  *      movement[assigned], lower -= max(movements) 3. For points where upper < lower: skip
  *      (assignment unchanged) 4. For remaining points: recompute distances with pruning 5. Update
  *      centers based on assignments
  *
  * ==Speedup Characteristics==
  *
  *   - Early iterations: ~2x speedup (clusters moving significantly)
  *   - Later iterations: 10-50x speedup (most points unchanged)
  *   - Overall: 3-10x speedup typical for convergent clustering
  *
  * ==Requirements==
  *
  *   - Only works with Squared Euclidean kernel (requires metric properties)
  *   - Requires additional memory for bound columns
  *   - Best for k >= 5 (overhead not worthwhile for small k)
  *
  * @see
  *   Elkan (2003): "Using the Triangle Inequality to Accelerate k-Means"
  * @see
  *   Hamerly (2010): "Making k-means Even Faster"
  */
class ElkanLloydsIterator extends LloydsIterator {

  import ElkanHelpers._

  /** Column names for tracking state */
  private val UpperBoundCol = "_elkan_upper"
  private val LowerBoundCol = "_elkan_lower"
  private val ClusterCol    = "_elkan_cluster"

  /** Compute center movements between iterations.
    *
    * @return
    *   Array of Euclidean distances each center moved
    */
  private def computeCenterMovements(
      oldCenters: Array[Array[Double]],
      newCenters: Array[Array[Double]]
  ): Array[Double] = {
    oldCenters.zip(newCenters).map { case (old, neu) =>
      euclideanDistance(old, neu)
    }
  }

  /** Compute pairwise center-to-center distances.
    *
    * @return
    *   k×k matrix of Euclidean distances
    */
  private def computeCenterDistances(centers: Array[Array[Double]]): Array[Array[Double]] = {
    val k         = centers.length
    val distances = Array.ofDim[Double](k, k)

    var i = 0
    while (i < k) {
      var j = i + 1
      while (j < k) {
        val d = euclideanDistance(centers(i), centers(j))
        distances(i)(j) = d
        distances(j)(i) = d
        j += 1
      }
      i += 1
    }

    distances
  }

  /** Initialize bounds by computing all distances.
    *
    * @return
    *   DataFrame with cluster assignment and bounds columns
    */
  private def initializeBounds(
      df: DataFrame,
      featuresCol: String,
      centers: Array[Array[Double]]
  ): DataFrame = {
    val spark     = df.sparkSession
    val bcCenters = spark.sparkContext.broadcast(centers)
    val k         = centers.length

    // UDF to compute initial assignment and bounds
    val initBoundsUDF = udf { (features: Vector) =>
      val point = features.toArray
      val ctrs  = bcCenters.value

      // Compute all distances
      val distances = ctrs.map(c => squaredEuclidean(point, c))

      // Find closest and second-closest
      var minIdx     = 0
      var minDist    = distances(0)
      var secondDist = Double.PositiveInfinity

      var i = 1
      while (i < k) {
        val d = distances(i)
        if (d < minDist) {
          secondDist = minDist
          minDist = d
          minIdx = i
        } else if (d < secondDist) {
          secondDist = d
        }
        i += 1
      }

      // Convert to Euclidean for bounds (triangle inequality uses Euclidean)
      val upperBound = math.sqrt(2.0 * minDist)
      val lowerBound = if (k > 1) math.sqrt(2.0 * secondDist) else Double.PositiveInfinity

      (minIdx, upperBound, lowerBound)
    }

    val result = df
      .withColumn("_init", initBoundsUDF(col(featuresCol)))
      .withColumn(ClusterCol, col("_init._1"))
      .withColumn(UpperBoundCol, col("_init._2"))
      .withColumn(LowerBoundCol, col("_init._3"))
      .drop("_init")

    bcCenters.unpersist()
    result
  }

  /** Update assignments using bounds to prune computations.
    *
    * @param df
    *   DataFrame with current bounds
    * @param prevCenters
    *   centers from previous iteration
    * @param newCenters
    *   centers for current iteration (after update)
    * @param movements
    *   how much each center moved
    * @param maxMovement
    *   maximum movement across all centers
    * @return
    *   DataFrame with updated assignments and bounds
    */
  private def updateWithBounds(
      df: DataFrame,
      featuresCol: String,
      centers: Array[Array[Double]],
      movements: Array[Double],
      maxMovement: Double,
      centerDists: Array[Array[Double]]
  ): (DataFrame, Long, Long) = {

    val spark         = df.sparkSession
    val bcCenters     = spark.sparkContext.broadcast(centers)
    val bcMovements   = spark.sparkContext.broadcast(movements)
    val bcCenterDists = spark.sparkContext.broadcast(centerDists)
    val k             = centers.length

    // Accumulators for statistics
    val skippedPoints = spark.sparkContext.longAccumulator("skippedPoints")
    val totalPoints   = spark.sparkContext.longAccumulator("totalPoints")

    // UDF to update bounds and possibly reassign
    val updateBoundsUDF = udf { (features: Vector, cluster: Int, upper: Double, lower: Double) =>
      val point  = features.toArray
      val ctrs   = bcCenters.value
      val mvmts  = bcMovements.value
      val cDists = bcCenterDists.value

      totalPoints.add(1)

      // Update bounds based on center movements
      val newUpper = upper + mvmts(cluster)
      val newLower = math.max(0.0, lower - maxMovement)

      // Check if we can skip this point entirely
      if (newUpper <= newLower) {
        // Assignment definitely unchanged
        skippedPoints.add(1)
        (cluster, newUpper, newLower)
      } else {
        // Need to verify/update assignment
        // First, tighten upper bound by computing actual distance to assigned center
        val actualDist = math.sqrt(2.0 * squaredEuclidean(point, ctrs(cluster)))

        if (actualDist <= newLower) {
          // Still assigned to same cluster
          skippedPoints.add(1)
          (cluster, actualDist, newLower)
        } else {
          // May need to change assignment - compute distances with pruning
          var minIdx     = cluster
          var minDist    = actualDist
          var secondDist = Double.PositiveInfinity

          var i = 0
          while (i < k) {
            if (i != cluster) {
              // Triangle inequality pruning using center-center distances
              val lowerBoundFromCenter = cDists(cluster)(i) - minDist
              if (lowerBoundFromCenter < minDist) {
                // Cannot prune - compute actual distance
                val d = math.sqrt(2.0 * squaredEuclidean(point, ctrs(i)))
                if (d < minDist) {
                  secondDist = minDist
                  minDist = d
                  minIdx = i
                } else if (d < secondDist) {
                  secondDist = d
                }
              }
            }
            i += 1
          }

          (minIdx, minDist, secondDist)
        }
      }
    }

    val result = df
      .withColumn(
        "_update",
        updateBoundsUDF(col(featuresCol), col(ClusterCol), col(UpperBoundCol), col(LowerBoundCol))
      )
      .withColumn(ClusterCol, col("_update._1"))
      .withColumn(UpperBoundCol, col("_update._2"))
      .withColumn(LowerBoundCol, col("_update._3"))
      .drop("_update")

    bcCenters.unpersist()
    bcMovements.unpersist()
    bcCenterDists.unpersist()

    (result, skippedPoints.value, totalPoints.value)
  }

  override def run(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      initialCenters: Array[Array[Double]],
      config: LloydsConfig
  ): LloydResult = {

    // Verify we have Squared Euclidean kernel
    require(
      config.kernel.supportsExpressionOptimization,
      s"ElkanLloydsIterator only works with Squared Euclidean kernel, got ${config.kernel.name}"
    )

    val k = config.k
    logInfo(
      s"Starting Elkan-accelerated Lloyd's algorithm with k=$k, maxIter=${config.maxIter}, " +
        s"tol=${config.tol}"
    )

    // For small k, fall back to standard iterator
    if (k < 5) {
      logInfo(s"ElkanLloydsIterator: k=$k < 5, falling back to DefaultLloydsIterator")
      return new DefaultLloydsIterator().run(df, featuresCol, weightCol, initialCenters, config)
    }

    // Validate input
    config.validator.validate(df, featuresCol, weightCol, config.kernel)

    // Setup checkpoint directory if specified
    config.checkpointDir.foreach { dir =>
      df.sparkSession.sparkContext.setCheckpointDir(dir)
    }

    var centers            = initialCenters
    var iter               = 0
    var converged          = false
    val distortionHistory  = ArrayBuffer[Double]()
    val movementHistory    = ArrayBuffer[Double]()
    var emptyClusterEvents = 0

    // Initialize bounds on first iteration
    logInfo("Initializing Elkan bounds...")
    var currentDF = initializeBounds(df, featuresCol, centers).cache()
    val initCount = currentDF.count()
    logInfo(s"Initialized bounds for $initCount points")

    try {
      while (iter < config.maxIter && !converged) {
        iter += 1
        logInfo(s"Starting Elkan iteration $iter")

        // Get current assignments for center update
        val assignedDF = currentDF.withColumn("cluster", col(ClusterCol))

        // Update centers based on current assignments
        val newCenters = config.updater.update(
          assignedDF,
          featuresCol,
          weightCol,
          k,
          config.kernel
        )

        // Handle empty clusters
        val (finalCenters, emptyCount) = config.emptyHandler.handle(
          assignedDF,
          featuresCol,
          weightCol,
          newCenters,
          df,
          config.kernel
        )

        if (emptyCount > 0) {
          emptyClusterEvents += 1
          logWarning(s"Iteration $iter: handled $emptyCount empty clusters")
        }

        // Compute center movements
        val movements   = computeCenterMovements(centers, finalCenters)
        val maxMovement = movements.max

        // Check convergence
        val movement = maxMovement
        logInfo(f"Iteration $iter: max center movement = $movement%.6f")

        converged = movement < config.tol

        if (converged) {
          logInfo(s"Converged after $iter iterations (movement $movement < tol ${config.tol})")
          // Compute final distortion
          val distortion = computeDistortion(currentDF, featuresCol, ClusterCol, finalCenters)
          distortionHistory += distortion
          movementHistory += movement
        } else {
          // Update bounds and reassign
          val centerDists                 = computeCenterDistances(finalCenters)
          val (updatedDF, skipped, total) = updateWithBounds(
            currentDF,
            featuresCol,
            finalCenters,
            movements,
            maxMovement,
            centerDists
          )

          val newDF = updatedDF.cache()
          newDF.count() // Materialize

          val skipRate = if (total > 0) skipped.toDouble / total * 100 else 0.0
          logInfo(
            f"Iteration $iter: $skipped%d / $total%d points skipped ($skipRate%.1f%% pruning)"
          )

          // Compute distortion for history
          val distortion = computeDistortion(newDF, featuresCol, ClusterCol, finalCenters)
          distortionHistory += distortion
          movementHistory += movement

          logInfo(f"Iteration $iter: distortion = $distortion%.6f")

          // Checkpoint if needed
          if (config.checkpointInterval > 0 && iter % config.checkpointInterval == 0) {
            logInfo(s"Checkpointing at iteration $iter")
            val checkpointed = newDF.checkpoint()
            newDF.unpersist()
            currentDF.unpersist()
            currentDF = checkpointed
          } else {
            currentDF.unpersist()
            currentDF = newDF
          }
        }

        centers = finalCenters
      }

      if (!converged) {
        logWarning(
          s"Did not converge after ${config.maxIter} iterations " +
            s"(final movement: ${movementHistory.lastOption.getOrElse(Double.NaN)})"
        )
      }

      LloydResult(
        centers = centers,
        iterations = iter,
        distortionHistory = distortionHistory.toArray,
        movementHistory = movementHistory.toArray,
        converged = converged,
        emptyClusterEvents = emptyClusterEvents
      )

    } finally {
      currentDF.unpersist()
    }
  }

  /** Compute total distortion (sum of squared distances to assigned centers). */
  private def computeDistortion(
      df: DataFrame,
      featuresCol: String,
      clusterCol: String,
      centers: Array[Array[Double]]
  ): Double = {
    val spark     = df.sparkSession
    val bcCenters = spark.sparkContext.broadcast(centers)

    val distortionUDF = udf { (features: Vector, cluster: Int) =>
      val point  = features.toArray
      val center = bcCenters.value(cluster)
      squaredEuclidean(point, center)
    }

    val total = df
      .withColumn("_dist", distortionUDF(col(featuresCol), col(clusterCol)))
      .agg(sum("_dist"))
      .head()
      .getDouble(0)

    bcCenters.unpersist()
    total
  }
}

/** Factory for creating the appropriate Lloyd's iterator. */
object LloydsIteratorFactory {

  /** Create a Lloyd's iterator appropriate for the given kernel.
    *
    * @param kernel
    *   the Bregman kernel being used
    * @param k
    *   number of clusters
    * @param useAcceleration
    *   whether to use Elkan acceleration when possible
    * @return
    *   appropriate LloydsIterator implementation
    */
  def create(kernel: BregmanKernel, k: Int, useAcceleration: Boolean = true): LloydsIterator = {
    if (useAcceleration && kernel.supportsExpressionOptimization && k >= 5) {
      new ElkanLloydsIterator()
    } else {
      new DefaultLloydsIterator()
    }
  }
}
