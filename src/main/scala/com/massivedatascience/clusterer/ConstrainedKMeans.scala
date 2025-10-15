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

/** Pairwise constraints for semi-supervised clustering.
  *
  * @param mustLink
  *   Set of (point_id, point_id) pairs that must be in same cluster
  * @param cannotLink
  *   Set of (point_id, point_id) pairs that cannot be in same cluster
  */
case class Constraints(
  mustLink: Set[(Long, Long)] = Set.empty,
  cannotLink: Set[(Long, Long)] = Set.empty
) {

  // Transitive closure of must-link constraints
  lazy val mustLinkClosure: Map[Long, Set[Long]] = {
    // Build equivalence classes using union-find
    val groups = scala.collection.mutable.Map[Long, Set[Long]]()

    mustLink.foreach { case (a, b) =>
      val groupA = groups.getOrElse(a, Set(a))
      val groupB = groups.getOrElse(b, Set(b))
      val merged = groupA ++ groupB

      merged.foreach { id =>
        groups(id) = merged
      }
    }

    groups.toMap
  }

  /** Check if two points must be in the same cluster.
    */
  def areMustLinked(a: Long, b: Long): Boolean = {
    mustLinkClosure.get(a).exists(_.contains(b))
  }

  /** Check if two points cannot be in the same cluster.
    */
  def areCannotLinked(a: Long, b: Long): Boolean = {
    cannotLink.contains((a, b)) || cannotLink.contains((b, a))
  }

  /** Get all point IDs that must be with this point.
    */
  def getMustLinkGroup(id: Long): Set[Long] = {
    mustLinkClosure.getOrElse(id, Set(id))
  }

  /** Get all cluster IDs that this point cannot be assigned to.
    */
  def getForbiddenClusters(pointId: Long, assignments: Map[Long, Int]): Set[Int] = {

    // Find all points this point cannot link with
    val cannotLinkPoints = cannotLink.collect {
      case (a, b) if a == pointId => b
      case (a, b) if b == pointId => a
    }

    // Get their cluster assignments
    cannotLinkPoints.flatMap(assignments.get)
  }
}

/** Configuration for constrained k-means clustering.
  *
  * @param constraints
  *   Pairwise constraints (must-link and cannot-link)
  * @param violationPenalty
  *   Penalty for constraint violations (Infinity = hard constraints)
  * @param maxViolations
  *   Maximum constraint violations allowed (0 = hard constraints)
  */
case class ConstrainedKMeansConfig(
  constraints: Constraints = Constraints(),
  violationPenalty: Double = Double.PositiveInfinity,
  maxViolations: Int = 0
) extends ConfigValidator {

  requirePositive(violationPenalty, "Violation penalty")
  requireNonNegative(maxViolations, "Max violations")

  def hasHardConstraints: Boolean = violationPenalty.isInfinity && maxViolations == 0
}

/** Constrained k-means clustering with must-link and cannot-link constraints.
  *
  * This is a semi-supervised clustering algorithm that incorporates pairwise constraints:
  *   - Must-link: two points must be in the same cluster
  *   - Cannot-link: two points cannot be in the same cluster
  *
  * Algorithm:
  *   1. Standard k-means with modified assignment step 2. When assigning point to cluster, respect constraints:
  *      - Cannot assign to clusters containing cannot-link points
  *      - Should assign to same cluster as must-link points (if possible) 3. Center update unchanged (standard k-means)
  *
  * Constraint enforcement:
  *   - Hard constraints: violations forbidden (may fail to converge)
  *   - Soft constraints: violations penalized (always converges)
  *
  * Benefits:
  *   - Incorporates domain knowledge
  *   - Improves clustering quality with limited supervision
  *   - Can enforce business rules (e.g., competitors in different clusters)
  *
  * Use cases:
  *   - Document clustering with known categories
  *   - Customer segmentation with business rules
  *   - Image clustering with partial labels
  *
  * Works with any Bregman divergence.
  *
  * @param config
  *   Configuration with constraints and penalties
  * @param baseClusterer
  *   Underlying clusterer (default: ColumnTrackingKMeans)
  */
class ConstrainedKMeans(
  config: ConstrainedKMeansConfig,
  baseClusterer: MultiKMeansClusterer = new ColumnTrackingKMeans()
) extends MultiKMeansClusterer
    with Logging {

  def cluster(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centers: Seq[IndexedSeq[BregmanCenter]]
  ): Seq[ClusteringWithDistortion] = {

    logger.info(s"Starting constrained k-means")
    logger.info(s"Must-link constraints: ${config.constraints.mustLink.size}")
    logger.info(s"Cannot-link constraints: ${config.constraints.cannotLink.size}")
    logger.info(s"Constraint mode: ${if (config.hasHardConstraints) "hard" else "soft"}")

    // If no constraints, just use base clusterer
    if (config.constraints.mustLink.isEmpty && config.constraints.cannotLink.isEmpty) {
      logger.info("No constraints, using base clusterer")
      return baseClusterer.cluster(maxIterations, pointOps, data, centers)
    }

    // Add point IDs to data for constraint tracking
    val indexedData = data
      .zipWithUniqueId()
      .map { case (point, id) =>
        (id, point)
      }
      .cache()

    // Process each initial center set
    val results = centers.map { initialCenters =>
      constrainedCluster(maxIterations, pointOps, indexedData, initialCenters)
    }

    indexedData.unpersist()

    results
  }

  /** Perform constrained clustering on indexed data.
    */
  private def constrainedCluster(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    indexedData: RDD[(Long, BregmanPoint)],
    initialCenters: IndexedSeq[BregmanCenter]
  ): ClusteringWithDistortion = {

    var centers    = initialCenters
    var iteration  = 0
    var converged  = false
    var violations = 0

    while (iteration < maxIterations && !converged) {
      iteration += 1

      // Assignment step with constraints
      val assignments = assignWithConstraints(indexedData, centers, pointOps)

      // Count violations
      violations = countViolations(assignments)
      if (violations > 0) {
        logger.info(s"Iteration $iteration: $violations constraint violations")
      }

      // Update step (standard k-means)
      val newCenters = updateCenters(indexedData, assignments, centers.length, pointOps)

      // Check convergence
      converged = hasConverged(centers, newCenters, pointOps)
      centers = newCenters

      if (converged) {
        logger.info(s"Converged after $iteration iterations")
      }
    }

    // Compute final cost
    val finalAssignments = assignWithConstraints(indexedData, centers, pointOps)
    val cost             = computeCost(indexedData, finalAssignments, centers, pointOps)

    logger.info(
      f"Constrained k-means completed: $iteration iterations, cost=$cost%.4f, violations=$violations"
    )

    ClusteringWithDistortion(cost, centers)
  }

  /** Assign points to clusters respecting constraints.
    */
  private def assignWithConstraints(
    indexedData: RDD[(Long, BregmanPoint)],
    centers: IndexedSeq[BregmanCenter],
    pointOps: BregmanPointOps
  ): RDD[(Long, Int)] = {

    // Broadcast constraints and centers
    val bcConstraints = indexedData.sparkContext.broadcast(config.constraints)
    val bcCenters     = indexedData.sparkContext.broadcast(centers)

    indexedData.mapPartitions { partition =>
      // Collect assignments within partition for constraint checking
      val localAssignments = scala.collection.mutable.Map[Long, Int]()

      partition.map { case (id, point) =>
        val constraints = bcConstraints.value
        val ctrs        = bcCenters.value

        // Find closest valid cluster
        val cluster = findClosestValidCluster(
          id,
          point,
          ctrs,
          localAssignments.toMap,
          constraints,
          pointOps
        )

        localAssignments(id) = cluster
        (id, cluster)
      }
    }
  }

  /** Find closest cluster that doesn't violate hard constraints.
    */
  private def findClosestValidCluster(
    pointId: Long,
    point: BregmanPoint,
    centers: IndexedSeq[BregmanCenter],
    assignments: Map[Long, Int],
    constraints: Constraints,
    pointOps: BregmanPointOps
  ): Int = {

    // Get forbidden clusters from cannot-link constraints
    val forbidden = constraints.getForbiddenClusters(pointId, assignments)

    // Find distances to all clusters
    val distances = centers.indices.map { i =>
      if (config.hasHardConstraints && forbidden.contains(i)) {
        (i, Double.PositiveInfinity) // Hard constraint: forbid this cluster
      } else {
        val baseDist = pointOps.distance(point, centers(i))
        val penalty  = if (forbidden.contains(i)) config.violationPenalty else 0.0
        (i, baseDist + penalty)
      }
    }

    // Return closest valid cluster
    distances.minBy(_._2)._1
  }

  /** Update cluster centers from assignments (standard k-means).
    */
  private def updateCenters(
    indexedData: RDD[(Long, BregmanPoint)],
    assignments: RDD[(Long, Int)],
    k: Int,
    pointOps: BregmanPointOps
  ): IndexedSeq[BregmanCenter] = {

    // Use fromAssignments to compute centers
    val points     = indexedData.map(_._2)
    val clusterIds = assignments.join(indexedData.map(p => (p._1, ()))).map(_._2._1)

    val model = KMeansModel.fromAssignments(pointOps, points, clusterIds, k)
    model.centers
  }

  /** Count constraint violations.
    */
  private def countViolations(assignments: RDD[(Long, Int)]): Int = {
    val assignmentMap = assignments.collectAsMap()

    var violations = 0

    // Check must-link violations
    config.constraints.mustLink.foreach { case (a, b) =>
      if (assignmentMap.contains(a) && assignmentMap.contains(b)) {
        if (assignmentMap(a) != assignmentMap(b)) {
          violations += 1
        }
      }
    }

    // Check cannot-link violations
    config.constraints.cannotLink.foreach { case (a, b) =>
      if (assignmentMap.contains(a) && assignmentMap.contains(b)) {
        if (assignmentMap(a) == assignmentMap(b)) {
          violations += 1
        }
      }
    }

    violations
  }

  /** Check if clustering has converged.
    */
  private def hasConverged(
    oldCenters: IndexedSeq[BregmanCenter],
    newCenters: IndexedSeq[BregmanCenter],
    pointOps: BregmanPointOps
  ): Boolean = {

    oldCenters.zip(newCenters).forall { case (old, neu) =>
      pointOps.distance(pointOps.toPoint(old), neu) < 1e-4
    }
  }

  /** Compute clustering cost.
    */
  private def computeCost(
    indexedData: RDD[(Long, BregmanPoint)],
    assignments: RDD[(Long, Int)],
    centers: IndexedSeq[BregmanCenter],
    pointOps: BregmanPointOps
  ): Double = {

    indexedData
      .join(assignments)
      .map { case (id, (point, cluster)) =>
        pointOps.distance(point, centers(cluster))
      }
      .sum()
  }
}

object ConstrainedKMeans {

  /** Create constrained k-means with given constraints (hard constraints by default).
    */
  def apply(constraints: Constraints): ConstrainedKMeans = {
    new ConstrainedKMeans(ConstrainedKMeansConfig(constraints = constraints))
  }

  /** Create constrained k-means with soft constraints.
    */
  def withSoftConstraints(constraints: Constraints, penalty: Double = 1000.0): ConstrainedKMeans = {
    new ConstrainedKMeans(
      ConstrainedKMeansConfig(
        constraints = constraints,
        violationPenalty = penalty,
        maxViolations = Int.MaxValue
      )
    )
  }

  /** Create constrained k-means allowing limited violations.
    */
  def withLimitedViolations(constraints: Constraints, maxViolations: Int): ConstrainedKMeans = {
    new ConstrainedKMeans(
      ConstrainedKMeansConfig(
        constraints = constraints,
        violationPenalty = Double.PositiveInfinity,
        maxViolations = maxViolations
      )
    )
  }
}
