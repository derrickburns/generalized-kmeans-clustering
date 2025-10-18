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

import com.massivedatascience.linalg.WeightedVector
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer

/** Batched RDD operations for reducing Spark shuffle and computation overhead.
  *
  * This class combines multiple operations that would otherwise require separate RDD
  * transformations into single, more efficient operations.
  */
object BatchedRDDOps {

  @transient private lazy val logger = LoggerFactory.getLogger(getClass.getName)

  /** Combined assignment and centroid computation in a single RDD pass.
    *
    * This combines three operations that are often done separately:
    *   1. Point assignment to clusters 2. Distortion calculation 3. Centroid accumulation for
    *      cluster updates
    *
    * @param points
    *   RDD of points to process
    * @param centers
    *   Current cluster centers
    * @param pointOps
    *   Distance operations
    * @return
    *   Tuple of (assignments, total distortion, centroid contributions)
    */
  def assignmentAndCentroidsBatch(
      points: RDD[BregmanPoint],
      centers: IndexedSeq[BregmanCenter],
      pointOps: BregmanPointOps
  ): (RDD[ColumnTrackingKMeans.Assignment], Double, Array[WeightedVector]) = {

    require(centers.nonEmpty, "Centers cannot be empty")
    require(pointOps != null, "PointOps cannot be null")

    val numClusters = centers.length
    logger.debug(s"Processing batch assignment for ${numClusters} clusters")

    // Use a single mapPartitions to combine assignment, distortion, and centroid computation
    val batchResults = points.mapPartitions { pointIter =>
      val centroids       = Array.fill(numClusters)(pointOps.make())
      val assignments     = ArrayBuffer[ColumnTrackingKMeans.Assignment]()
      val totalDistortion = pointIter.foldLeft(0.0) { (distortion, point) =>
        // Find closest cluster with optimized distance computation
        val (bestCluster, bestDistance) = pointOps.findClosest(centers, point)

        // Accumulate for assignment
        assignments += ColumnTrackingKMeans.Assignment(bestDistance, bestCluster, 0)

        // Accumulate for centroid computation
        if (bestCluster >= 0 && bestCluster < numClusters) {
          centroids(bestCluster).add(point)
        }

        // Accumulate distortion
        distortion + bestDistance
      }

      logger.debug(
        s"Processed partition with ${assignments.length} points, distortion: ${totalDistortion}"
      )

      // Return combined results
      Iterator((assignments.toSeq, totalDistortion, centroids.map(_.asImmutable)))
    }.persist(StorageLevel.MEMORY_AND_DISK_SER)

    try {
      // Collect assignments
      val allAssignments = batchResults.flatMap(_._1)

      // Sum distortions
      val totalDistortion = batchResults.map(_._2).sum()

      // Aggregate centroids
      val aggregatedCentroids = batchResults.map(_._3).reduce { (c1, c2) =>
        c1.zip(c2).map { case (centroid1, centroid2) =>
          val combined = pointOps.make()
          combined.add(centroid1)
          combined.add(centroid2)
          combined.asImmutable
        }
      }

      logger.info(
        s"Batch processing completed: distortion=${totalDistortion}, clusters=${aggregatedCentroids.length}"
      )

      (allAssignments, totalDistortion, aggregatedCentroids)

    } finally {
      batchResults.unpersist()
    }
  }

  /** Optimized assignment update with movement tracking using existing ColumnTrackingKMeans logic.
    *
    * This leverages the existing incremental assignment logic that avoids redundant distance
    * calculations for stationary centers.
    *
    * @param points
    *   RDD of points
    * @param currentAssignments
    *   Current assignments
    * @param centers
    *   Centers with movement history
    * @param round
    *   Current round number
    * @param pointOps
    *   Distance operations
    * @return
    *   Updated assignments
    */
  def incrementalAssignmentBatch(
      points: RDD[BregmanPoint],
      currentAssignments: RDD[ColumnTrackingKMeans.Assignment],
      centers: IndexedSeq[ColumnTrackingKMeans.CenterWithHistory],
      round: Int,
      pointOps: BregmanPointOps
  ): RDD[ColumnTrackingKMeans.Assignment] = {

    require(centers.nonEmpty, "Centers cannot be empty")
    logger.debug(s"Incremental assignment batch for round ${round}")

    // Identify which centers moved since last round
    val movedCenters      = centers.filter(_.movedSince(round - 1))
    val stationaryCenters = centers.filterNot(_.movedSince(round - 1))

    logger.info(
      s"Round ${round}: ${movedCenters.length} moved, ${stationaryCenters.length} stationary centers"
    )

    // Join points with their current assignments for efficient processing
    val pointsWithAssignments = points.zip(currentAssignments)

    pointsWithAssignments.mapPartitions { iter =>
      iter.map { case (point, assignment) =>
        // Use existing ColumnTrackingKMeans reassignment logic
        ColumnTrackingKMeans.reassignment(pointOps, point, assignment, round, centers)
      }
    }
  }

  /** Batched statistics computation for monitoring convergence.
    *
    * Computes multiple statistics in a single RDD pass:
    *   - Total distortion
    *   - Assignment changes
    *   - Cluster populations
    *   - Point movement statistics
    *
    * @param assignments
    *   Current assignments
    * @param previousAssignments
    *   Previous assignments (optional)
    * @return
    *   Statistics map
    */
  def computeStatsBatch(
      assignments: RDD[ColumnTrackingKMeans.Assignment],
      previousAssignments: Option[RDD[ColumnTrackingKMeans.Assignment]] = None
  ): Map[String, Double] = {

    logger.debug("Computing batch statistics")

    val stats = assignments.mapPartitions { assignmentIter =>
      var totalDistortion = 0.0
      var assignedCount   = 0L
      var unassignedCount = 0L
      val clusterCounts   = scala.collection.mutable.Map[Int, Long]()

      assignmentIter.foreach { assignment =>
        totalDistortion += assignment.distance

        if (assignment.isAssigned) {
          assignedCount += 1
          val cluster = assignment.cluster
          clusterCounts(cluster) = clusterCounts.getOrElse(cluster, 0L) + 1
        } else {
          unassignedCount += 1
        }
      }

      Iterator((totalDistortion, assignedCount, unassignedCount, clusterCounts.toMap))
    }.reduce {
      case (
            (dist1, assigned1, unassigned1, clusters1),
            (dist2, assigned2, unassigned2, clusters2)
          ) =>
        val combinedClusters = (clusters1.keySet ++ clusters2.keySet).map { cluster =>
          cluster -> (clusters1.getOrElse(cluster, 0L) + clusters2.getOrElse(cluster, 0L))
        }.toMap

        (dist1 + dist2, assigned1 + assigned2, unassigned1 + unassigned2, combinedClusters)
    }

    val (totalDistortion, assignedCount, unassignedCount, clusterCounts) = stats
    val totalPoints                                                      = assignedCount + unassignedCount
    val nonEmptyClusters                                                 = clusterCounts.count(_._2 > 0)
    val maxClusterSize                                                   = if (clusterCounts.nonEmpty) clusterCounts.values.max else 0L
    val avgClusterSize                                                   =
      if (nonEmptyClusters > 0) assignedCount.toDouble / nonEmptyClusters else 0.0

    // Compute assignment changes if previous assignments are provided
    val assignmentChanges = previousAssignments.map { prevAssignments =>
      assignments
        .zip(prevAssignments)
        .map { case (curr, prev) => if (curr.cluster != prev.cluster) 1.0 else 0.0 }
        .sum()
    }.getOrElse(0.0)

    val result = Map(
      "totalDistortion"   -> totalDistortion,
      "assignedPoints"    -> assignedCount.toDouble,
      "unassignedPoints"  -> unassignedCount.toDouble,
      "totalPoints"       -> totalPoints.toDouble,
      "nonEmptyClusters"  -> nonEmptyClusters.toDouble,
      "maxClusterSize"    -> maxClusterSize.toDouble,
      "avgClusterSize"    -> avgClusterSize,
      "assignmentChanges" -> assignmentChanges
    )

    logger.info(s"Batch statistics: ${result}")
    result
  }
}
