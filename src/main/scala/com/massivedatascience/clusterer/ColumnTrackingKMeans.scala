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

import scala.collection.Map
import scala.collection.generic.FilterMonadic
import scala.collection.mutable.ArrayBuffer

object ColumnTrackingKMeans {
  private[clusterer] val noCluster = -1
  private[clusterer] val unassigned = Assignment(Infinity, noCluster, -2)

  private[clusterer] sealed trait CentroidChange

  private[clusterer] case class Add(point: BregmanPoint) extends CentroidChange

  private[clusterer] case class Sub(point: BregmanPoint) extends CentroidChange

  private[clusterer] case class PointWithDistance(point: BregmanPoint, assignment: Assignment, dist: Double)

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

class ColumnTrackingKMeans(
  updateRate: Double = 1.0,
  terminationCondition: TerminationCondition = DefaultTerminationCondition)
  extends MultiKMeansClusterer {


  import ColumnTrackingKMeans._

  /**
   * Augment the stats with information about the clusters
   *
   * @param round the round
   * @param stats  the stats to update
   * @param centersWithHistory the cluster centers
   * @param assignments the assignments
   */
  private def updateRoundStats(
    round: Int,
    stats: TrackingStats,
    centersWithHistory: Array[CenterWithHistory],
    assignments: RDD[Assignment]) {

    val clusterCounts = countByCluster(assignments)
    val biggest = clusterCounts.maxBy(_._2)
    stats.largestCluster.setValue(biggest._2)
    stats.nonemptyClusters.add(clusterCounts.size)
    stats.emptyClusters.add(centersWithHistory.size - clusterCounts.size)
    stats.relocatedCenters.setValue(centersWithHistory.count(c => c.round == round))
  }

  /**
   * count number of points assigned to each cluster
   *
   * @param currentAssignments the assignments
   * @return a map from cluster index to number of points assigned to that cluster
   */
  private def countByCluster(currentAssignments: RDD[Assignment]) =
    currentAssignments.filter(_.isAssigned).map { p => (p.cluster, p)}.countByKey()

  private def updateStats(
    stats: TrackingStats,
    current: Assignment,
    previous: Assignment): Unit = {
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
    centerArrays: Array[Array[BregmanCenter]]): (Double, Array[BregmanCenter]) = {

    /**
     * The initial assignments of points to clusters
     *
     * @param points the incoming data
     * @param centers cluster centers
     * @return the assignments
     */
    def initialAssignments(points: RDD[BregmanPoint], centers: Array[CenterWithHistory]) = {
      val result = points.map(pt => bestAssignment(pt, 0, centers))
      result.setName("initial assignments")
      result.persist()
      result
    }

    /**
     * Identify the new cluster assignments for a sample of the points.
     * Persists the new assignments in memory, un-persisting the previous assignments.
     *
     * @param round the number of the round
     * @param stats statistics on round
     * @param centersWithHistory current clusters
     * @param assignments current assignments
     * @return points and their new assignments
     */
    def updatedAssignments(
      round: Int,
      stats: TrackingStats,
      assignments: RDD[Assignment],
      centersWithHistory: Array[CenterWithHistory]): RDD[Assignment] = {

      val sc = points.sparkContext
      val bcCenters = sc.broadcast(centersWithHistory)

      val newAssignments = points.zip(assignments).mapPartitionsWithIndex { (index, assignedPoints) =>
        val rand = new XORShiftRandom(round ^ (index << 16))
        val centers = bcCenters.value
        assignedPoints.map { case (point, current) =>
          if (rand.nextDouble() > updateRate) unassigned
          else reassignment(point, current, round, stats, centers)
        }
      }
      newAssignments.setName(s"assignments round $round")
      newAssignments.zip(assignments).foreach { case (c, p) => updateStats(stats, c, p)}
      bcCenters.unpersist(blocking = false)
      newAssignments.cache()
      assignments.unpersist(blocking = false)
      newAssignments
    }

    /**
     * Update the clusters (stochastically if rate < 1.0)
     *
     * @param round the round
     * @param stats statistics to keep
     * @param current  current assignments
     * @param previous  current assignments
     * @param centersWithHistory  the current clusters
     * @return
     */
    def updatedCenters(
      round: Int,
      stats: TrackingStats,
      current: RDD[Assignment],
      previous: RDD[Assignment],
      centersWithHistory: Array[CenterWithHistory]): Array[CenterWithHistory] = {

      val changes: Array[(Int, MutableWeightedVector)] = if (updateRate < 1.0)
        getStochasticCentroidChanges(points, current)
      else
        getExactCentroidChanges(points, current, previous)

      changes.map { case (index, delta) =>
        val c = centersWithHistory(index)
        val oldPosition = pointOps.toPoint(c.center)
        val x = if (c.initialized) delta.add(oldPosition) else delta
        centersWithHistory(index) = CenterWithHistory(pointOps.toCenter(x), index, round)
        stats.movement.add(pointOps.distance(oldPosition, centersWithHistory(index).center))
      }
      centersWithHistory
    }

    /**
     * Identify cluster changes.
     *
     * Create an array of the changes per cluster.  Some clusters may not have changes.  They will
     * not be represented in the change set.
     *
     * @param current current assignments
     * @param previous previous assignments

     * @return changes to cluster position
     */
    def getExactCentroidChanges(
      points: RDD[BregmanPoint],
      current: RDD[Assignment],
      previous: RDD[Assignment]): Array[(Int, MutableWeightedVector)] = {

      val pairs = current.zip(previous)

      points.zip(pairs).mapPartitions { pts =>
        val buffer = new ArrayBuffer[(Int, CentroidChange)]
        for ((point, (curr, prev)) <- pts if curr.cluster != prev.cluster) {
          assert(curr.isAssigned)
          if (curr.isAssigned) buffer.append((curr.cluster, Add(point)))
          if (prev.isAssigned) buffer.append((prev.cluster, Sub(point)))
        }
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
      centers: FilterMonadic[CenterWithHistory, Seq[CenterWithHistory]]): Assignment = {

      var bestDist = Infinity
      var bestIndex = noCluster
      for (center <- centers) {
        val dist = pointOps.distance(point, center.center)
        if (dist < bestDist) {
          bestIndex = center.index
          bestDist = dist
        }
      }
      if (bestIndex != noCluster) Assignment(bestDist, bestIndex, round) else unassigned
    }

    /**
     * Find the closest cluster assignment to a given point
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
      stats: TrackingStats,
      centers: Seq[CenterWithHistory]
      ): Assignment = {

      val filteredCenters = centers.withFilter(_.movedSince(assignment.round))
      val closestDirty = bestAssignment(point, round, filteredCenters)
      val newAssignment = if (assignment.isAssigned) {
        if (closestDirty.distance < assignment.distance) {
          if (closestDirty.cluster == assignment.cluster) {
            stats.dirtySame.add(1)
          } else {
            stats.dirtyOther.add(1)
          }
          closestDirty
        } else if (!centers(assignment.cluster).movedSince(assignment.round)) {
          stats.stationary.add(1)
          assignment
        } else {
          best(round, stats, centers, point, assignment, closestDirty)
        }
      } else {
        best(round, stats, centers, point, assignment, closestDirty)
      }
      if (newAssignment.isUnassigned) {
        logWarning(s"cannot find cluster to assign point $assignment")
      }
      newAssignment
    }

    def best(
      round: Int,
      stats: TrackingStats,
      centers: Seq[CenterWithHistory],
      point: BregmanPoint,
      currentAssignment: Assignment,
      closestDirty: Assignment): Assignment = {

      val candidateCenters = centers.withFilter(!_.movedSince(currentAssignment.round))
      val closestClean = bestAssignment(point, round, candidateCenters)
      if (closestDirty.isUnassigned ||
        (closestClean.isAssigned && closestClean.distance < closestDirty.distance)) {
        stats.closestClean.add(1)
        closestClean
      } else {
        stats.closestDirty.add(1)
        closestDirty
      }
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
      points.zip(assignments).mapPartitions { pts =>
        val bc = bcCenters.value
        pts.flatMap { case (point, a) =>
          bc.zipWithIndex.map { case (c, i) => (i, PointWithDistance(point, a, pointOps.distance(point, c.center)))
          }
        }
      }.reduceByKeyLocally { (x, y) => if (x.dist < y.dist) x else y}
    }

    require(updateRate <= 1.0 && updateRate >= 0.0)
    logInfo(s"update rate = $updateRate")
    logInfo(s"runs = ${centerArrays.size}")

    val results = for (centers <- centerArrays) yield {
      var centersWithHistory = centers.zipWithIndex.map { case (c, i) => CenterWithHistory(c, i)}
      var previous = points.map(x => unassigned)
      var current = initialAssignments(points, centersWithHistory)
      var terminate = false
      var round = 1
      do {
        val stats = new TrackingStats(points.sparkContext, round)
        centersWithHistory = updatedCenters(round, stats, current, previous, centersWithHistory)
        previous = current
        current = updatedAssignments(round, stats, current, centersWithHistory)
        updateRoundStats(round, stats, centersWithHistory, current)
        stats.report()
        terminate = terminationCondition(stats)
        round = round + 1
      } while (!terminate)

      val d = distortion(current)
      current.unpersist(blocking = false)
      (d, centersWithHistory.map(_.center))
    }
    points.unpersist(blocking = false)
    results.minBy(_._1)
  }
}
