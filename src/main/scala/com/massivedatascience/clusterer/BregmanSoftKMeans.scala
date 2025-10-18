/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
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

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.MultiKMeansClusterer.ClusteringWithDistortion
import org.apache.spark.rdd.RDD

import scala.math.{ exp, log }

/** Configuration for Bregman soft clustering (fuzzy c-means).
  *
  * @param beta
  *   Inverse temperature parameter controlling soft assignment sharpness. Higher values = sharper
  *   assignments (approaches hard clustering). Lower values = softer assignments (more fuzzy
  *   membership).
  * @param minMembership
  *   Minimum membership probability to avoid numerical issues
  * @param maxIterations
  *   Maximum number of iterations
  * @param convergenceThreshold
  *   Convergence threshold for membership changes
  * @param computeObjective
  *   Whether to compute and track the soft clustering objective
  */
case class BregmanSoftKMeansConfig(
    beta: Double = 1.0,
    minMembership: Double = 1e-10,
    maxIterations: Int = 100,
    convergenceThreshold: Double = 1e-6,
    computeObjective: Boolean = true
) extends ConfigValidator {

  requirePositive(beta, "Beta (inverse temperature)")
  requireInRange(minMembership, 0.0, 1.0, "Minimum membership")
  requirePositive(maxIterations, "Max iterations")
  requirePositive(convergenceThreshold, "Convergence threshold")
}

/** Soft assignment result containing membership probabilities and statistics.
  *
  * @param memberships
  *   RDD of (point, membership_probabilities) pairs
  * @param centers
  *   Final cluster centers
  * @param objective
  *   Final soft clustering objective value
  * @param iterations
  *   Number of iterations used
  * @param converged
  *   Whether the algorithm converged
  * @param config
  *   Configuration used
  */
case class SoftClusteringResult(
    memberships: RDD[(BregmanPoint, Array[Double])],
    centers: IndexedSeq[BregmanCenter],
    objective: Double,
    iterations: Int,
    converged: Boolean,
    config: BregmanSoftKMeansConfig
) {

  /** Convert soft assignments to hard assignments (maximum membership).
    */
  def toHardAssignments: RDD[(BregmanPoint, Int)] = {
    memberships.map { case (point, probs) =>
      val hardAssignment = probs.zipWithIndex.maxBy(_._1)._2
      (point, hardAssignment)
    }
  }

  /** Get the effective number of clusters (entropy-based measure).
    *
    * For each point, the effective number of clusters is exp(entropy). We return the average
    * effective number across all points.
    *
    * This correctly handles:
    *   - Hard assignment (entropy=0) → exp(0) = 1 effective cluster
    *   - Uniform over k clusters (entropy=ln(k)) → exp(ln(k)) = k effective clusters
    */
  def effectiveNumberOfClusters: Double = {
    val effectiveClustersPerPoint = memberships.map { case (_, probs) =>
      val entropy = -probs.map(p => if (p > config.minMembership) p * log(p) else 0.0).sum
      exp(entropy) // Convert entropy to effective number of clusters for this point
    }
    effectiveClustersPerPoint.mean() // Average across all points
  }

  /** Get comprehensive statistics about the soft clustering.
    */
  def getStats: Map[String, Double] = {
    Map(
      "objective"            -> objective,
      "iterations"           -> iterations.toDouble,
      "converged"            -> (if (converged) 1.0 else 0.0),
      "beta"                 -> config.beta,
      "numCenters"           -> centers.length.toDouble,
      "effectiveNumClusters" -> effectiveNumberOfClusters
    )
  }
}

/** Bregman soft clustering (fuzzy c-means) implementation.
  *
  * This implements probabilistic clustering where each point has a membership probability for each
  * cluster, computed using:
  *
  * p(cluster c | point x) ∝ exp(-β * D_φ(x, μ_c))
  *
  * where β is the inverse temperature parameter and D_φ is the Bregman divergence.
  *
  * Higher β values lead to sharper (more decisive) assignments, while lower β values lead to softer
  * (more fuzzy) assignments.
  */
case class BregmanSoftKMeans(config: BregmanSoftKMeansConfig = BregmanSoftKMeans.defaultConfig)
    extends MultiKMeansClusterer
    with Logging {

  /** Perform soft clustering on the given data.
    *
    * @param maxIterations
    *   Maximum iterations (overrides config if different)
    * @param pointOps
    *   Bregman point operations
    * @param data
    *   Input data points
    * @param initialCenters
    *   Initial cluster centers
    * @return
    *   Sequence of soft clustering results (one per initial center set)
    */
  def cluster(
      maxIterations: Int,
      pointOps: BregmanPointOps,
      data: RDD[BregmanPoint],
      initialCenters: Seq[IndexedSeq[BregmanCenter]]
  ): Seq[ClusteringWithDistortion] = {

    require(initialCenters.nonEmpty, "At least one set of initial centers must be provided")

    logger.info(
      s"Starting Bregman soft clustering with β=${config.beta}, ${initialCenters.length} runs"
    )

    val actualMaxIterations =
      if (maxIterations != config.maxIterations) maxIterations else config.maxIterations

    val results = initialCenters.map { centers =>
      val softResult = clusterSoft(actualMaxIterations, pointOps, data, centers)

      // Convert to hard clustering result for compatibility
      val hardObjective = computeHardObjective(softResult.memberships, softResult.centers, pointOps)
      ClusteringWithDistortion(hardObjective, softResult.centers)
    }

    logger.info(s"Soft clustering completed, best objective: ${results.map(_.distortion).min}")
    results
  }

  /** Perform soft clustering and return detailed soft clustering results.
    *
    * @param maxIterations
    *   Maximum number of iterations
    * @param pointOps
    *   Bregman point operations
    * @param data
    *   Input data points
    * @param initialCenters
    *   Initial cluster centers
    * @return
    *   Soft clustering result with membership probabilities
    */
  def clusterSoft(
      maxIterations: Int,
      pointOps: BregmanPointOps,
      data: RDD[BregmanPoint],
      initialCenters: IndexedSeq[BregmanCenter]
  ): SoftClusteringResult = {

    val numClusters = initialCenters.length
    logger.info(
      s"Starting soft clustering: k=$numClusters, β=${config.beta}, maxIter=$maxIterations"
    )

    var centers           = initialCenters
    var previousObjective = Double.MaxValue
    var iteration         = 0
    var converged         = false

    // Cache data for performance
    data.cache()

    while (iteration < maxIterations && !converged) {
      iteration += 1
      logger.debug(s"Soft clustering iteration $iteration")

      // E-step: Compute soft assignments
      val memberships = computeSoftAssignments(data, centers, pointOps)

      // M-step: Update centers based on soft assignments
      val newCenters = computeSoftCenters(memberships, numClusters, pointOps)

      // Check convergence
      if (config.computeObjective) {
        val objective   = computeSoftObjective(memberships, newCenters, pointOps)
        val improvement =
          (previousObjective - objective) / math.max(math.abs(previousObjective), 1e-10)

        logger.debug(
          f"Iteration $iteration: objective = $objective%.6f, improvement = $improvement%.8f"
        )

        if (improvement < config.convergenceThreshold) {
          converged = true
          logger.info(s"Soft clustering converged after $iteration iterations")
        }

        previousObjective = objective
      } else {
        // Use center movement for convergence detection
        val centerMovement = computeCenterMovement(centers, newCenters, pointOps)
        if (centerMovement < config.convergenceThreshold) {
          converged = true
          logger.info(s"Soft clustering converged after $iteration iterations (center movement)")
        }
      }

      centers = newCenters
    }

    if (!converged) {
      logger.warn(s"Soft clustering did not converge after $maxIterations iterations")
    }

    // Final soft assignments
    val finalMemberships = computeSoftAssignments(data, centers, pointOps)
    val finalObjective   = if (config.computeObjective) {
      computeSoftObjective(finalMemberships, centers, pointOps)
    } else {
      computeHardObjective(finalMemberships, centers, pointOps)
    }

    SoftClusteringResult(
      memberships = finalMemberships,
      centers = centers,
      objective = finalObjective,
      iterations = iteration,
      converged = converged,
      config = config
    )
  }

  /** Compute soft assignment probabilities for all points.
    *
    * Uses the Boltzmann distribution: p(c|x) ∝ exp(-β * D_φ(x, μ_c))
    *
    * Uses the log-sum-exp trick for numerical stability by subtracting the minimum distance before
    * exponentiating. This ensures the largest probability is exp(0) = 1.0.
    */
  private def computeSoftAssignments(
      data: RDD[BregmanPoint],
      centers: IndexedSeq[BregmanCenter],
      pointOps: BregmanPointOps
  ): RDD[(BregmanPoint, Array[Double])] = {

    val beta             = config.beta
    val minMembership    = config.minMembership
    val broadcastCenters = data.sparkContext.broadcast(centers)

    data.map { point =>
      val distances = broadcastCenters.value.map(center => pointOps.distance(point, center))

      // Compute unnormalized probabilities: exp(-β * distance)
      // Use min distance for numerical stability (log-sum-exp trick)
      // This ensures the largest probability is exp(0) = 1.0
      val minDistance       = distances.min
      val unnormalizedProbs = distances.map { dist =>
        math.exp(-beta * (dist - minDistance))
      }

      // Normalize to get probabilities
      val totalProb                    = unnormalizedProbs.sum
      val probabilities: Array[Double] = if (totalProb > 1e-100) {
        unnormalizedProbs.map(_ / totalProb).toArray
      } else {
        // Fallback: uniform distribution
        Array.fill(centers.length)(1.0 / centers.length)
      }

      // Apply minimum membership threshold
      val adjustedProbs   = probabilities.map(p => math.max(p, minMembership))
      val adjustedSum     = adjustedProbs.sum
      val normalizedProbs = adjustedProbs.map(_ / adjustedSum)

      (point, normalizedProbs)
    }
  }

  /** Compute new cluster centers using soft assignments.
    *
    * Each center is the weighted average of all points, where weights are membership probabilities.
    */
  private def computeSoftCenters(
      memberships: RDD[(BregmanPoint, Array[Double])],
      numClusters: Int,
      pointOps: BregmanPointOps
  ): IndexedSeq[BregmanCenter] = {

    // Compute weighted sums for each cluster
    val clusterSums = memberships.flatMap { case (point, probs) =>
      probs.zipWithIndex.map { case (prob, clusterId) =>
        val weightedPoint = pointOps.scale(point, prob)
        (clusterId, (weightedPoint, prob))
      }
    }.aggregateByKey((pointOps.make(), 0.0))(
      // Sequence operation: add point to accumulator
      { case ((accumulator, totalWeight), (weightedPoint, weight)) =>
        accumulator.add(weightedPoint)
        (accumulator, totalWeight + weight)
      },
      // Combiner operation: merge accumulators
      { case ((acc1, weight1), (acc2, weight2)) =>
        acc1.add(acc2)
        (acc1, weight1 + weight2)
      }
    ).collectAsMap()

    // Convert to centers
    (0 until numClusters).map { clusterId =>
      clusterSums.get(clusterId) match {
        case Some((accumulator, totalWeight)) if totalWeight > pointOps.weightThreshold =>
          pointOps.toCenter(accumulator.asImmutable)
        case _                                                                          =>
          // Handle empty cluster - keep previous center or create default
          logger.warn(s"Cluster $clusterId has insufficient soft membership weight")
          pointOps.toCenter(pointOps.make().asImmutable)
      }
    }
  }

  /** Compute the soft clustering objective function.
    *
    * Objective = Σ_x Σ_c p(c|x) * D_φ(x, μ_c)
    */
  private def computeSoftObjective(
      memberships: RDD[(BregmanPoint, Array[Double])],
      centers: IndexedSeq[BregmanCenter],
      pointOps: BregmanPointOps
  ): Double = {

    val broadcastCenters = memberships.sparkContext.broadcast(centers)

    memberships.map { case (point, probs) =>
      probs.zipWithIndex.map { case (prob, clusterId) =>
        val distance = pointOps.distance(point, broadcastCenters.value(clusterId))
        prob * distance
      }.sum
    }.sum()
  }

  /** Compute hard clustering objective (sum of distances to assigned centers).
    */
  private def computeHardObjective(
      memberships: RDD[(BregmanPoint, Array[Double])],
      centers: IndexedSeq[BregmanCenter],
      pointOps: BregmanPointOps
  ): Double = {

    val broadcastCenters = memberships.sparkContext.broadcast(centers)

    memberships.map { case (point, probs) =>
      val hardAssignment = probs.zipWithIndex.maxBy(_._1)._2
      pointOps.distance(point, broadcastCenters.value(hardAssignment))
    }.sum()
  }

  /** Compute the total movement of centers between iterations.
    */
  private def computeCenterMovement(
      oldCenters: IndexedSeq[BregmanCenter],
      newCenters: IndexedSeq[BregmanCenter],
      pointOps: BregmanPointOps
  ): Double = {

    oldCenters
      .zip(newCenters)
      .map { case (oldCenter, newCenter) =>
        pointOps.distance(pointOps.toPoint(oldCenter), newCenter)
      }
      .sum
  }
}

object BregmanSoftKMeans {

  /** Default configuration for soft clustering.
    */
  def defaultConfig: BregmanSoftKMeansConfig = {
    BregmanSoftKMeansConfig(
      beta = 1.0,
      minMembership = 1e-10,
      maxIterations = 100,
      convergenceThreshold = 1e-6,
      computeObjective = true
    )
  }

  /** Create soft clustering with specified inverse temperature.
    *
    * @param beta
    *   Higher values → sharper assignments, lower values → softer assignments
    */
  def apply(beta: Double): BregmanSoftKMeans = {
    val config = defaultConfig.copy(beta = beta)
    BregmanSoftKMeans(config)
  }

  /** Create very soft clustering (fuzzy assignments).
    */
  def verySoft(beta: Double = 0.1): BregmanSoftKMeans = {
    val config = defaultConfig.copy(
      beta = beta,
      convergenceThreshold = 1e-5, // Slightly looser convergence
      maxIterations = 150
    )
    BregmanSoftKMeans(config)
  }

  /** Create moderately soft clustering (balanced assignments).
    */
  def moderatelySoft(beta: Double = 1.0): BregmanSoftKMeans = {
    BregmanSoftKMeans(defaultConfig.copy(beta = beta))
  }

  /** Create sharp clustering (nearly hard assignments).
    */
  def sharp(beta: Double = 10.0): BregmanSoftKMeans = {
    val config = defaultConfig.copy(
      beta = beta,
      convergenceThreshold = 1e-8, // Tighter convergence for sharp assignments
      minMembership = 1e-12
    )
    BregmanSoftKMeans(config)
  }

  /** Create configuration optimized for mixture model estimation.
    */
  def forMixtureModel(beta: Double = 2.0): BregmanSoftKMeans = {
    val config = defaultConfig.copy(
      beta = beta,
      computeObjective = true,
      convergenceThreshold = 1e-7,
      maxIterations = 200
    )
    BregmanSoftKMeans(config)
  }
}
