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

package com.massivedatascience.clusterer.ml.df.strategies.impl

import com.massivedatascience.clusterer.ml.df.BregmanKernel
import com.massivedatascience.clusterer.ml.df.strategies.AssignmentStrategy
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/** Accelerated assignment strategy using center-distance pruning.
  *
  * Uses the triangle inequality to skip unnecessary distance computations:
  *
  * '''Key Insight (Elkan's Lemma 1):''' If d(x, c) ≤ d(c, c')/2, then d(x, c) ≤ d(x, c')
  *
  * This means: once we find a center c with distance d, we can skip any center c' where d(c, c') >=
  * 2*d (because the triangle inequality guarantees c' is farther).
  *
  * ==Algorithm==
  *
  *   1. Precompute pairwise distances between all centers: O(k²) 2. For each point:
  *      a. Compute distance to first center b. For remaining centers, check if triangle inequality
  *         allows skipping c. Only compute distance if the center might be closer 3. Track
  *         statistics on skipped computations
  *
  * ==Speedup Characteristics==
  *
  *   - Best case: O(1) per point when clusters are well-separated
  *   - Worst case: O(k) per point when clusters overlap significantly
  *   - Typical: 2-5x speedup for well-clustered data
  *
  * ==Limitations==
  *
  *   - Only works with Squared Euclidean distance (uses metric properties)
  *   - Requires k² memory for center-center distances
  *   - Overhead may not pay off for small k (< 10)
  *
  * @note
  *   This is a single-iteration optimization. For full Elkan/Hamerly acceleration with
  *   cross-iteration bounds, a stateful iterator would be needed.
  *
  * @see
  *   Elkan (2003): "Using the Triangle Inequality to Accelerate k-Means"
  * @see
  *   Hamerly (2010): "Making k-means Even Faster"
  */
class AcceleratedSEAssignment extends AssignmentStrategy with Logging {

  /** Compute squared Euclidean distance between two arrays. */
  private def squaredEuclidean(a: Array[Double], b: Array[Double]): Double = {
    var sum = 0.0
    var i   = 0
    while (i < a.length) {
      val diff = a(i) - b(i)
      sum += diff * diff
      i += 1
    }
    sum * 0.5 // Bregman convention: half squared distance
  }

  /** Compute Euclidean distance (not squared) for triangle inequality. */
  private def euclideanDistance(a: Array[Double], b: Array[Double]): Double = {
    var sum = 0.0
    var i   = 0
    while (i < a.length) {
      val diff = a(i) - b(i)
      sum += diff * diff
      i += 1
    }
    math.sqrt(sum)
  }

  /** Precompute pairwise Euclidean distances between centers.
    *
    * Returns a k×k matrix where entry (i,j) is d(center_i, center_j). Uses Euclidean (not squared)
    * for triangle inequality.
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

  /** Find the closest center with pruning based on triangle inequality.
    *
    * @param point
    *   feature vector
    * @param centers
    *   cluster centers
    * @param centerDists
    *   precomputed center-to-center distances
    * @return
    *   (closest_center_index, distance_to_closest, num_distances_computed)
    */
  private def findClosestWithPruning(
      point: Array[Double],
      centers: Array[Array[Double]],
      centerDists: Array[Array[Double]]
  ): (Int, Double, Int) = {
    val k = centers.length
    if (k == 0) return (0, Double.PositiveInfinity, 0)
    if (k == 1) return (0, squaredEuclidean(point, centers(0)), 1)

    // Start with first center
    var minIdx           = 0
    var minDist          = squaredEuclidean(point, centers(0))
    var minDistEuclidean = math.sqrt(2.0 * minDist) // Convert to Euclidean for triangle inequality
    var computations     = 1

    // Check remaining centers with pruning
    var i = 1
    while (i < k) {
      // Triangle inequality: if d(current_best, candidate) >= 2 * d(point, current_best),
      // then candidate cannot be closer (Elkan's Lemma 1)
      val centerCenterDist = centerDists(minIdx)(i)

      if (centerCenterDist < 2.0 * minDistEuclidean) {
        // Cannot prune - need to compute actual distance
        val dist = squaredEuclidean(point, centers(i))
        computations += 1

        if (dist < minDist) {
          minDist = dist
          minIdx = i
          minDistEuclidean = math.sqrt(2.0 * minDist)
        }
      }
      // else: pruned! Triangle inequality guarantees this center is farther

      i += 1
    }

    (minIdx, minDist, computations)
  }

  override def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: BregmanKernel
  ): DataFrame = {

    require(
      kernel.supportsExpressionOptimization,
      s"AcceleratedSEAssignment only works with Squared Euclidean kernel, got ${kernel.name}"
    )

    val k = centers.length

    logInfo(s"AcceleratedSEAssignment: assigning $k clusters with triangle inequality pruning")

    // For small k, overhead may not pay off - fall back to simple broadcast
    if (k < 5) {
      logInfo(
        s"AcceleratedSEAssignment: k=$k < 5, using simple broadcast (pruning overhead not worthwhile)"
      )
      return new BroadcastUDFAssignment().assign(df, featuresCol, weightCol, centers, kernel)
    }

    // Precompute center-to-center distances
    val centerDists = computeCenterDistances(centers)
    logDebug(s"AcceleratedSEAssignment: precomputed ${k * (k - 1) / 2} center-to-center distances")

    val spark         = df.sparkSession
    val bcCenters     = spark.sparkContext.broadcast(centers)
    val bcCenterDists = spark.sparkContext.broadcast(centerDists)

    // Accumulator to track pruning statistics
    val totalComputations = spark.sparkContext.longAccumulator("totalDistanceComputations")
    val totalPoints       = spark.sparkContext.longAccumulator("totalPoints")

    val assignWithStatsUDF = udf { (features: Vector) =>
      val ctrs     = bcCenters.value
      val ctrDists = bcCenterDists.value
      val pointArr = features.toArray

      val (minIdx, _, computations) = findClosestWithPruning(pointArr, ctrs, ctrDists)

      totalComputations.add(computations)
      totalPoints.add(1)

      minIdx
    }

    val result = df.withColumn("cluster", assignWithStatsUDF(col(featuresCol)))

    // Force computation to get stats
    val count = result.cache().count()

    // Log pruning statistics
    val totalComps  = totalComputations.value
    val numPoints   = totalPoints.value
    val avgComps    = if (numPoints > 0) totalComps.toDouble / numPoints else 0.0
    val maxComps    = k.toDouble
    val pruningRate = if (k > 1) (1.0 - avgComps / maxComps) * 100 else 0.0

    logInfo(
      f"AcceleratedSEAssignment: completed $count%d points, " +
        f"avg ${avgComps}%.2f / $maxComps%.0f distance computations per point " +
        f"(${pruningRate}%.1f%% pruning rate)"
    )

    bcCenters.unpersist()
    bcCenterDists.unpersist()

    result
  }
}

/** Factory for accelerated assignment strategies. */
object AcceleratedAssignment {

  /** Create an accelerated assignment strategy if applicable.
    *
    * Returns AcceleratedSEAssignment for Squared Euclidean with k >= 5, otherwise returns the
    * standard BroadcastUDFAssignment.
    *
    * @param kernel
    *   the Bregman kernel being used
    * @param k
    *   number of clusters
    * @return
    *   appropriate assignment strategy
    */
  def forKernel(kernel: BregmanKernel, k: Int): AssignmentStrategy = {
    if (kernel.supportsExpressionOptimization && k >= 5) {
      new AcceleratedSEAssignment()
    } else {
      new BroadcastUDFAssignment()
    }
  }
}
