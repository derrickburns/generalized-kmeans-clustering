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


package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.util.{SparkHelper, XORShiftRandom}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.annotation.tailrec
import scala.collection.{mutable, Map}
import scala.collection.generic.FilterMonadic
import scala.collection.mutable.ArrayBuffer

object ColumnTrackingKMeans {
  private[clusterer] val noCluster = -1
  private[clusterer] val unassigned = Assignment(Infinity, noCluster, -2)

  private[clusterer] case class PointWithDistance(point: BregmanPoint,
    assignment: Assignment, dist: Double)

  /**
   *
   * @param distance the distance to the closest cluster
   * @param cluster the index of the closest cluster, or -1 if no cluster is assigned
   * @param round the round that this assignment was made
   */
  private[clusterer] case class Assignment(distance: Double, cluster: Int, round: Int) {
    def isAssigned = cluster != noCluster

    def isUnassigned = cluster == noCluster
  }

  /**
   *
   * @param center  the centroid of the cluster
   * @param round the round in which his cluster was last moved
   */
  private[clusterer] case class CenterWithHistory(center: BregmanCenter, index: Int, round: Int = -1) {
    @inline def movedSince(r: Int): Boolean = round >= r

    @inline def initialized: Boolean = round >= 0
  }

}

/**
 * A KMeans implementation that tracks which clusters moved and which points are assigned to which
 * clusters and the distance to the closest cluster.
 *
 *
 * @param updateRate for stochastic sampling, the percentage of points to update on each round
 * @param terminationCondition when to terminate the clustering
 */
class ColumnTrackingKMeans(
  updateRate: Double = 1.0,
  terminationCondition: TerminationCondition = DefaultTerminationCondition)
  extends MultiKMeansClusterer with SparkHelper {

  import ColumnTrackingKMeans._

  val addOnly = true

  /**
   * count number of points assigned to each cluster
   *
   * @param currentAssignments the assignments
   * @return a map from cluster index to number of points assigned to that cluster
   */
  private def countByCluster(currentAssignments: RDD[Assignment]) =
    currentAssignments.filter(_.isAssigned).map { p => (p.cluster, p)}.countByKey()

  private def distortion(data: RDD[Assignment]) = data.filter(_.isAssigned).map(_.distance).sum()

  /**
   * Create a K-Means clustering of the input and report on the resulting distortion
   *
   * @param points points to cluster
   * @param centerArrays initial cluster centers
   * @return the distortion of the clustering on the points and the cluster centers (model)
   */
  def cluster(
    pointOps: BregmanPointOps,
    points: RDD[BregmanPoint],
    centerArrays: Array[Array[BregmanCenter]]): (Double, Array[BregmanCenter], Option[RDD[(Int, Double)]]) = {

    require(points.getStorageLevel.useMemory)

    val stats = new TrackingStats(points.sparkContext)

    /**
     * The initial assignments of points to clusters
     *
     * @param points the incoming data
     * @param centers cluster centers
     * @return the assignments
     */
    def initialAssignments(points: RDD[BregmanPoint], centers: Array[CenterWithHistory]) = {
      require(points.getStorageLevel.useMemory)
      points.map(pt => bestAssignment(pt, 0, centers)).setName("initial assignments").persist(StorageLevel.MEMORY_ONLY)
    }


    /**
     * Identify the new cluster assignments for a sample of the points.
     * Persists the new assignments in memory, un-persisting the previous assignments.
     *
     * @param round the number of the round
     * @param currentCenters current clusters
     * @param previousAssignments current assignments
     * @return points and their new assignments
     */
    def updatedAssignments(
      round: Int,
      previousAssignments: RDD[Assignment],
      currentCenters: Array[CenterWithHistory]): RDD[Assignment] = {

      require(previousAssignments.getStorageLevel.useMemory)

      withBroadcast(currentCenters) { bcCenters =>
        points.zip(previousAssignments).mapPartitionsWithIndex {
          (index, assignedPoints) =>
            val rand = new XORShiftRandom(round ^ (index << 16))
            val centers = bcCenters.value
            assignedPoints.map { case (point, current) =>
              if (rand.nextDouble() > updateRate) current
              else reassignment(point, current, round, centers)
            }
        }
      }.setName(s"assignments round $round").persist(StorageLevel.MEMORY_ONLY)
    }

    /**
     * Update the clusters (stochastically if rate < 1.0)
     *
     * @param round the round
     * @param currentAssignments  current assignments
     * @param previousAssignments  previous assignments
     * @param previousCenters  the cluster centers
     * @return the new cluster centers
     */
    def updateCenters(
      round: Int,
      currentAssignments: RDD[Assignment],
      previousAssignments: RDD[Assignment],
      previousCenters: Array[CenterWithHistory]): Array[CenterWithHistory] = {

      require(currentAssignments.getStorageLevel.useMemory)
      require(previousAssignments.getStorageLevel.useMemory)

      val currentCenters = previousCenters.clone()
      if (addOnly) {
        val results: Array[(Int, MutableWeightedVector)] = getCompleteCentroids(points, currentAssignments, previousCenters.length)
        results.foreach { case (index, location) =>
          val change: WeightedVector = location.asImmutable
          val newCenter = CenterWithHistory(pointOps.toCenter(change), index, round)
          currentCenters(index) = newCenter
        }
      } else {
        val changes = getCentroidChanges(points, currentAssignments, previousAssignments, previousCenters.length)
        changes.foreach { case (index, delta) =>
          val c = currentCenters(index)
          val oldPosition = pointOps.toPoint(c.center)
          val x = (if (c.initialized) delta.add(oldPosition) else delta).asImmutable
          currentCenters(index) = CenterWithHistory(pointOps.toCenter(x), index, round)
        }
      }
      currentCenters
    }


    /**
     * Collect and report the statistics about this round
     *
     * @param round the round
     * @param currentCenters the current cluster centers
     * @param previousCenters the previous cluster centers
     * @param currentAssignments the current assignments
     * @param previousAssignments the previous assignments
     */
    def shouldTerminate(
      round: Int,
      currentCenters: Array[CenterWithHistory],
      previousCenters: Array[CenterWithHistory],
      currentAssignments: RDD[Assignment],
      previousAssignments: RDD[Assignment]): Boolean = {

      require(currentAssignments.getStorageLevel.useMemory)
      require(previousAssignments.getStorageLevel.useMemory)

      logInfo("start of stats collection")
      stats.currentRound.setValue(round)

      stats.movement.setValue(0.0)
      stats.relocatedCenters.setValue(0)
      currentCenters.zip(previousCenters).foreach { case (current, previous) =>
        if (current.round == round) {
          stats.movement.add(pointOps.distance(pointOps.toPoint(previous.center), current.center))
          stats.relocatedCenters.add(1)
        }
      }

      stats.reassignedPoints.setValue(0)
      stats.unassignedPoints.setValue(0)
      stats.improvement.setValue(0)
      stats.newlyAssignedPoints.setValue(0)
      currentAssignments.zip(previousAssignments).foreach { case (current, previous) =>
        if (current.isAssigned) {
          if (previous.isAssigned) {
            stats.improvement.add(previous.distance - current.distance)
            if (current.cluster != previous.cluster) stats.reassignedPoints.add(1)
          } else {
            stats.newlyAssignedPoints.add(1)
          }
        } else {
          stats.unassignedPoints.add(1)
        }
      }

      val clusterCounts = countByCluster(currentAssignments)
      val biggest = clusterCounts.maxBy(_._2)
      stats.largestCluster.setValue(biggest._2)
      stats.nonemptyClusters.setValue(clusterCounts.size)
      stats.emptyClusters.setValue(currentCenters.size - clusterCounts.size)
      stats.report()

      logInfo("end of stats collection")
      terminationCondition(stats)
    }

    def changes(to: RDD[Assignment], from: RDD[Assignment]): RDD[Int] = {
      to.zip(from).map { case (curr, prev) =>
        if (curr.cluster != prev.cluster) curr.cluster else -1
      }
    }

    /**
     * Create the centers that changes.
     *
     * This implementation avoids object allocation per BregmanPoint.
     *
     * A previous implementation that uses aggregateByKey on (index, point) tuples was observed
     * to cause to much garbage collection overhead.
     *
     * @param points points
     * @param assignments assignments of points
     * @param numCenters current number of non-empty clusters
     * @return
     */
    def getCompleteCentroids(
      points: RDD[BregmanPoint],
      assignments: RDD[Assignment],
      numCenters: Int): Array[(Int, MutableWeightedVector)] = {

      require(points.getStorageLevel.useMemory)
      require(assignments.getStorageLevel.useMemory)

      points.zipPartitions(assignments) { (x: Iterator[BregmanPoint], y: Iterator[Assignment]) =>
        val centroids = new Array[MutableWeightedVector](numCenters)
        val indexBuffer = new mutable.ArrayBuilder.ofInt
        indexBuffer.sizeHint(numCenters)

        var i = 0
        while (y.hasNext && x.hasNext) {
          val point = x.next()
          val index = y.next().cluster
          if (index != -1) {
            if (centroids(index) == null) {
              centroids(index) = pointOps.getCentroid
              indexBuffer += index
            }
            centroids(index).add(point)
          }
          i = i + 1
        }
        assert(y.hasNext == x.hasNext)
        logInfo(s"partition had $i points")

        val changedClusters = indexBuffer.result()
        logInfo(s"number of clusters changed = ${changedClusters.length}")
        changedClusters.map(index => (index, centroids(index))).iterator
      }.reduceByKey(_.add(_)).collect()
    }

    def getCentroidChanges(
      points: RDD[BregmanPoint],
      assignments: RDD[Assignment],
      previousAssignments: RDD[Assignment],
      numCenters: Int): Array[(Int, MutableWeightedVector)] = {

      require(points.getStorageLevel.useMemory)
      require(assignments.getStorageLevel.useMemory)
      require(previousAssignments.getStorageLevel.useMemory)

      points.zipPartitions(assignments, previousAssignments) {
        (x: Iterator[BregmanPoint], y: Iterator[Assignment], z: Iterator[Assignment]) =>

          val centroids = new Array[MutableWeightedVector](numCenters)
          val indexBuffer = new mutable.ArrayBuilder.ofInt
          indexBuffer.sizeHint(numCenters)

          @inline def centroidAt(index: Int): MutableWeightedVector = {
            if (centroids(index) == null) {
              centroids(index) = pointOps.getCentroid
              indexBuffer += index
            }
            centroids(index)
          }

          while (z.hasNext && y.hasNext && x.hasNext) {
            val point = x.next()
            val current = y.next().cluster
            val previous = z.next().cluster
            if (previous != current) {
              if (previous != -1) centroidAt(previous).sub(point)
              if (current != -1) centroidAt(current).add(point)
            }
          }
          assert(y.hasNext == x.hasNext && z.hasNext == y.hasNext)
          val changedClusters = indexBuffer.result()
          logInfo(s"number of clusters changed = ${changedClusters.length}")
          changedClusters.map(index => (index, centroids(index))).iterator
      }.reduceByKey(_.add(_)).collect()
    }


    /**
     * Find the closest cluster from a given set of clusters
     *
     * @param round the current round
     * @param centers the cluster centers
     * @param point the point
     * @return the assignment of that cluster to the point
     */

    def bestAssignment(
      point: BregmanPoint,
      round: Int,
      centers: FilterMonadic[CenterWithHistory, Seq[CenterWithHistory]],
      initialAssignment: Assignment = unassigned): Assignment = {

      var distance = initialAssignment.distance
      var cluster = initialAssignment.cluster
      for (center <- centers) {
        val dist = pointOps.distance(point, center.center)
        if (dist < distance) {
          cluster = center.index
          distance = dist
        }
      }
      if (cluster != noCluster) Assignment(distance, cluster, round) else unassigned
    }

    /**
     * Find the closest cluster assignment to a given point
     *
     * This implementation is optimized for the cases when:
     *
     * a) one of the clusters that moved is closer than the previous cluster
     *
     * b) the previous cluster did not move.
     *
     * In these case distances to other stationary clusters need not be computed.  As Lloyd's
     * algorithm proceeds, more and more clusters are stationary, so fewer and fewer distnace
     * calculations are needed.
     *
     * @param point point
     * @param assignment the current assignment of the point
     * @param round the current round
     * @param centers the cluster centers
     * @return  the new assignment for the point
     */
    def reassignment(
      point: BregmanPoint,
      assignment: Assignment,
      round: Int,
      centers: Seq[CenterWithHistory]
      ): Assignment = {

      val nonStationaryCenters = centers.withFilter(_.movedSince(assignment.round))
      val stationaryCenters = centers.withFilter(!_.movedSince(assignment.round))
      val closestNonStationary = bestAssignment(point, round, nonStationaryCenters)

      if (!assignment.isAssigned)
        bestAssignment(point, round, stationaryCenters, closestNonStationary)
      else if (closestNonStationary.distance < assignment.distance)
        closestNonStationary
      else if (!centers(assignment.cluster).movedSince(assignment.round))
        assignment
      else
        bestAssignment(point, round, stationaryCenters, closestNonStationary)
    }

    def clusterings(
      points: RDD[BregmanPoint],
      initialCenterSets: Array[Array[BregmanCenter]]): Array[(Double, Array[BregmanCenter], RDD[Assignment])] = {

      require(points.getStorageLevel.useMemory)

      for (initialCenters <- initialCenterSets) yield {
        val centers = initialCenters.zipWithIndex.map { case (c, i) => CenterWithHistory(c, i)}
        val emptyAssignments = nullAssignments(points)
        val (assignments, newCenters) = lloyds(1, centers, initialAssignments(points, centers), emptyAssignments)
        val d = distortion(assignments)
        val finalCenters = newCenters.map(_.center)
        (d, finalCenters, assignments)
      }
    }

    @tailrec
    def lloyds(
      round: Int,
      centers: Array[CenterWithHistory],
      assignments: RDD[Assignment],
      previousAssignments: RDD[Assignment]): (RDD[Assignment], Array[CenterWithHistory]) = {

      require(assignments.getStorageLevel.useMemory)

      val newCenters: Array[CenterWithHistory] = updateCenters(round, assignments, previousAssignments, centers)
      previousAssignments.unpersist()
      val newAssignments = updatedAssignments(round, assignments, newCenters)
      val terminate = shouldTerminate(round, newCenters, centers, newAssignments, assignments)
      if (terminate) (newAssignments, newCenters) else lloyds(round + 1, newCenters, newAssignments, assignments)
    }

    require(updateRate <= 1.0 && updateRate >= 0.0)
    logInfo(s"update rate = $updateRate")
    logInfo(s"runs = ${centerArrays.size}")

    val candidates = clusterings(points, centerArrays)
    val (d, centers, assignments) = candidates.minBy(_._1)
    val best = (d, centers, Some(assignments.map(x => (x.cluster, x.distance)).persist()))
    for ((_, _, a) <- candidates) a.unpersist()
    best
  }


  def nullAssignments(points: RDD[BregmanPoint]): RDD[Assignment] =
    points.map(x => unassigned).persist()
}
