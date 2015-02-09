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

import com.massivedatascience.clusterer.util.XORShiftRandom
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import scala.collection.Map
import scala.collection.generic.FilterMonadic
import scala.collection.mutable.ArrayBuffer

object ColumnTrackingKMeans {
  private[clusterer] val noCluster = -1
  private[clusterer] val unassigned = Assignment(Infinity, noCluster, -2)

  private[clusterer] sealed trait CentroidChange

  private[clusterer] case class Add(point: BregmanPoint) extends CentroidChange

  private[clusterer] case class Sub(point: BregmanPoint) extends CentroidChange

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
    def movedSince(r: Int): Boolean = round >= r

    def initialized: Boolean = round >= 0
  }

}

/**
 * A KMeans implementation that tracks which clusters moved and which points are assigned to which
 * clusters and the distance to the closest cluster.
 *
 *
 * @param updateRate for stochastic sampling, the percentage of points to update on each round
 * @param terminationCondition when to terminate the clusering
 */
class ColumnTrackingKMeans(
  updateRate: Double = 1.0,
  terminationCondition: TerminationCondition = DefaultTerminationCondition)
  extends MultiKMeansClusterer {

  import ColumnTrackingKMeans._

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

    val stats = new TrackingStats(points.sparkContext)

    /**
     * The initial assignments of points to clusters
     *
     * @param points the incoming data
     * @param centers cluster centers
     * @return the assignments
     */
    def initialAssignments(points: RDD[BregmanPoint], centers: Array[CenterWithHistory]) =
      points.map(pt => bestAssignment(pt, 0, centers)).setName("initial assignments").cache()


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

      val sc = points.sparkContext
      val bcCenters = sc.broadcast(currentCenters)

      val currentAssignments = points.zip(previousAssignments).mapPartitionsWithIndex {
        (index, assignedPoints) =>
          val rand = new XORShiftRandom(round ^ (index << 16))
          val centers = bcCenters.value
          assignedPoints.map { case (point, current) =>
            if (rand.nextDouble() > updateRate) unassigned
            else reassignment(point, current, round, centers)
          }
      }
      bcCenters.unpersist(blocking = true)
      currentAssignments.setName(s"assignments round $round").cache()
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
    def updatedCenters(
      round: Int,
      currentAssignments: RDD[Assignment],
      previousAssignments: RDD[Assignment],
      previousCenters: Array[CenterWithHistory]): Array[CenterWithHistory] = {

      val changes = if (updateRate < 1.0)
        getStochasticCentroidChanges(points, currentAssignments)
      else
        getExactCentroidChanges(points, currentAssignments, previousAssignments)

      val currentCenters = previousCenters.clone()
      changes.foreach { case (index, delta) =>
        val c = currentCenters(index)
        val oldPosition = pointOps.toPoint(c.center)
        val x = (if (c.initialized) delta.add(oldPosition) else delta).asImmutable
        currentCenters(index) = CenterWithHistory(pointOps.toCenter(x), index, round)
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

    /**
     * Identify cluster changes.
     *
     * Create an array of the changes per cluster.  Some clusters may not have changes.  They will
     * not be represented in the change set.
     *
     * @param currentAssignments current assignments
     * @param previousAssignments previous assignments

     * @return changes to cluster position
     */
    def getExactCentroidChanges(
      points: RDD[BregmanPoint],
      currentAssignments: RDD[Assignment],
      previousAssignments: RDD[Assignment]): Array[(Int, MutableWeightedVector)] = {

      val pairs = currentAssignments.zip(previousAssignments)

      points.zip(pairs).mapPartitions { pts =>
        val buffer = new ArrayBuffer[(Int, CentroidChange)]
        for ((point, (curr, prev)) <- pts if curr.cluster != prev.cluster) {
          assert(curr.isAssigned)
          if (curr.isAssigned) buffer.append((curr.cluster, Add(point)))
          if (prev.isAssigned) {
            buffer.append((prev.cluster, Sub(point)))
          }
        }
        logInfo(s"buffer size ${buffer.size}")
        buffer.iterator
      }.aggregateByKey(pointOps.getCentroid)(
          (x, y) => y match {
            case Add(p) => x.add(p)
            case Sub(p) => x.sub(p)
          },
          (x, y) => x.add(y)
        ).collect()
    }

    def getStochasticCentroidChanges(
      points: RDD[BregmanPoint],
      assignments: RDD[Assignment]): Array[(Int, MutableWeightedVector)] =

      points.zip(assignments).filter(_._2.isAssigned).map { case (o, p) =>
        (p.cluster, o)
      }.aggregateByKey(pointOps.getCentroid)(_.add(_), _.add(_)).collect()


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

    def showEmpty(
      centersWithHistory: Array[CenterWithHistory],
      points: RDD[BregmanPoint],
      currentAssignments: RDD[Assignment]) = {

      val cp = closestPoints(points, currentAssignments, centersWithHistory)

      val clusterMap = countByCluster(currentAssignments)
      Array.tabulate(centersWithHistory.length) { i =>
        val count = clusterMap.getOrElse(i, 0)
        if (count == 0) {
          val c = centersWithHistory(i)
          val d1 = cp(i).dist

          println(s"center: $i = $c")
          println(s"closest point is ${cp(i)}")
          val closerCluster = cp(i).assignment.cluster
          val d2 = pointOps.distance(cp(i).point, centersWithHistory(closerCluster).center)
          println(s"closest center to that point is $closerCluster =" +
            s"' ${centersWithHistory(closerCluster)} at distance $d2")
          println()
          assert(d1 >= d2, s"closest point to cluster $d1 should be >= to closest cluster " +
            s"to point $d2")
        }
      }
    }

    /**
     * Find the closest point and distance to each cluster
     *
     * @param points the points
     * @param assignments the assignments
     * @param centers the clusters
     * @return a map from cluster index to the closest point to that cluster and the distance
     */
    def closestPoints(
      points: RDD[BregmanPoint],
      assignments: RDD[Assignment],
      centers: Array[CenterWithHistory]): Map[Int, PointWithDistance] = {

      val bcCenters = assignments.sparkContext.broadcast(centers)
      val result = points.zip(assignments).mapPartitions { pts =>
        val bc = bcCenters.value
        pts.flatMap { case (point, a) =>
          bc.map { c => (c.index, PointWithDistance(point, a, pointOps.distance(point, c.center)))
          }
        }
      }.reduceByKeyLocally { (x, y) => if (x.dist < y.dist) x else y}
      bcCenters.unpersist(blocking = true)
      result
    }

    def clusterings(
      points: RDD[BregmanPoint],
      initialCenterSets: Array[Array[BregmanCenter]]): Array[(Double, Array[BregmanCenter], RDD[Assignment])] = {

      for (initialCenters <- initialCenterSets) yield {
        val centers = initialCenters.zipWithIndex.map { case (c, i) => CenterWithHistory(c, i)}
        val emptyAssignments = nullAssignments(points)
        val assignments = lloyds(1, centers, initialAssignments(points, centers), emptyAssignments)
        val d = distortion(assignments)
        val finalCenters = centers.map(_.center)
        (d, finalCenters, assignments)
      }
    }

    @tailrec
    def lloyds(
      round: Int,
      centers: Array[CenterWithHistory],
      assignments: RDD[Assignment],
      previousAssignments: RDD[Assignment]): RDD[Assignment] = {

      val newCenters = updatedCenters(round, assignments, previousAssignments, centers)
      previousAssignments.unpersist(blocking = true)
      val newAssignments = updatedAssignments(round, assignments, newCenters)
      val terminate = shouldTerminate(round, newCenters, centers, newAssignments, assignments)
      if (terminate) newAssignments else lloyds(round + 1, newCenters, newAssignments, assignments)
    }

    require(updateRate <= 1.0 && updateRate >= 0.0)
    logInfo(s"update rate = $updateRate")
    logInfo(s"runs = ${centerArrays.size}")

    val candidates = clusterings(points, centerArrays)
    points.unpersist(blocking = false)
    val (d, centers, assignments) = candidates.minBy(_._1)
    val best = (d, centers, Some(assignments.map(x => (x.cluster, x.distance)).cache()))
    for ((_, _, a) <- candidates) a.unpersist(blocking = true)
    best
  }

  def nullAssignments(points: RDD[BregmanPoint]): RDD[Assignment] =
    points.map(x => unassigned).cache()
}
