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
import org.apache.spark.storage.StorageLevel

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

object ColumnTrackingKMeans {
  private[clusterer] val noCluster = -1
  private[clusterer] val unassigned = Assignment(Infinity, noCluster, -2)

  private[clusterer] sealed trait CentroidChange

  private[clusterer] case class Add(point: BregmanPoint) extends CentroidChange

  private[clusterer] case class Sub(point: BregmanPoint) extends CentroidChange

  private[clusterer] case class PointWithDistance(point: BregmanPoint, assignments: RecentAssignments, dist: Double)

  /**
   *
   * @param dist the distance to the closest cluster
   * @param index the index of the closest cluster, or -1 if no cluster is assigned
   * @param round the round that this assignment was made
   */
  private[clusterer] case class Assignment(dist: Double, index: Int, round: Int) {
    def isAssigned = index != noCluster

    def isUnassigned = index == noCluster
  }

  /**
   *
   * @param current  the current cluster (and distance to that cluster) that this point belongs to
   * @param previous the previous cluster (and distance to that cluster) that this point belongs to
   */
  private[clusterer] case class RecentAssignments(current: Assignment, previous: Assignment) {
    def cluster = current.index

    def previousCluster = previous.index

    def isAssigned = current.isAssigned

    def wasPreviouslyAssigned = previous.isAssigned

    def wasReassigned = current.index != previous.index

    def assign(closest: Assignment): RecentAssignments = RecentAssignments(closest, current)

    def distance = current.dist

    def improvement = previous.dist - current.dist

    def round = current.round
  }

  /**
   *
   * @param center  the centroid of the cluster
   * @param round the round in which his cluster was last moved
   */
  private[clusterer] case class CenterWithHistory(center: BregmanCenter, round: Int = -1) {
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
   * @param recentAssignments the fat points
   */
  private def updateRoundStats(
    round: Int,
    stats: TrackingStats,
    centersWithHistory: Array[CenterWithHistory],
    recentAssignments: RDD[RecentAssignments]) {

    val clusterCounts = countByCluster(recentAssignments)
    val biggest = clusterCounts.maxBy(_._2)
    stats.largestCluster.setValue(biggest._2)
    stats.nonemptyClusters.add(clusterCounts.size)
    stats.emptyClusters.add(centersWithHistory.size - clusterCounts.size)
    stats.relocatedCenters.setValue(centersWithHistory.count(c => c.round == round))
  }

  /**
   * count number of points assigned to each cluster
   *
   * @param assignments the assignments
   * @return a map from cluster index to number of points assigned to that cluster
   */
  private def countByCluster(assignments: RDD[RecentAssignments]): Map[Int, Long] =
    assignments.filter(_.isAssigned).map { p => (p.cluster, p)}.countByKey()

  private def updateStats(stats: TrackingStats, p: RecentAssignments): Unit = {
    if (p.isAssigned) {
      if (p.wasPreviouslyAssigned) {
        stats.improvement.add(p.improvement)
        if (p.wasReassigned) stats.reassignedPoints.add(1)
      } else {
        stats.newlyAssignedPoints.add(1)
      }
    } else {
      stats.unassignedPoints.add(1)
    }
  }

  private def distortion(data: RDD[RecentAssignments]) = data.filter(_.isAssigned).map(_.distance).sum()

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
     * The initial assignments
     *
     * @param points the incoming data
     * @param centersWithHistory cluster centers
     * @return the fat points (points with two most recent cluster assignments)
     */
    def initialAssignments(points: RDD[BregmanPoint], centersWithHistory: Array[CenterWithHistory]) = {
      val result = points.map { point =>
        val assignments = RecentAssignments(unassigned, unassigned)
        assignments.copy(current = closestOf(0, centersWithHistory, point, assignments, (_, _) => true))
      }
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
      assignments: RDD[RecentAssignments],
      centersWithHistory: Array[CenterWithHistory]): RDD[RecentAssignments] = {

      val sc = points.sparkContext
      val bcCenters = sc.broadcast(centersWithHistory)

      val result = points.zip(assignments).mapPartitionsWithIndex { (index, points) =>
        val rand = new XORShiftRandom(round ^ (index << 16))
        val centers = bcCenters.value
        points.map { case (point, a) =>
          a.assign(
            if (rand.nextDouble() > updateRate) unassigned
            else reassignment(round, stats, centers, point, a))
        }
      }
      result.setName(s"assignments round $round")
      result.foreach(updateStats(stats, _))
      bcCenters.unpersist(blocking = false)
      result.cache()
      assignments.unpersist(blocking = false)
      result
    }

    /**
     * Update the clusters (stochastically if rate < 1.0)
     *
     * @param round the round
     * @param stats statistics to keep
     * @param assignments  current assignments
     * @param centersWithHistory  the current clusters
     * @return
     */
    def updatedCenters(
      round: Int,
      stats: TrackingStats,
      assignments: RDD[RecentAssignments],
      centersWithHistory: Array[CenterWithHistory]): Array[CenterWithHistory] = {

      val changes = if (updateRate < 1.0)
        getStochasticCentroidChanges(points, assignments)
      else
        getExactCentroidChanges(points, assignments)

      changes.map { case (index, delta) =>
        val c = centersWithHistory(index)
        val oldPosition = pointOps.toPoint(c.center)
        val x = if (c.initialized) delta.add(oldPosition) else delta
        centersWithHistory(index) = CenterWithHistory(pointOps.toCenter(x), round)
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
     * @param assignments points and their assignments
     * @return changes to cluster position
     */
    def getExactCentroidChanges(
      points: RDD[BregmanPoint],
      assignments: RDD[RecentAssignments]): Array[(Int, MutableWeightedVector)] = {

      points.zip(assignments).mapPartitions { pts =>
        val buffer = new ArrayBuffer[(Int, CentroidChange)]
        for ((point, a) <- pts if a.wasReassigned) {
          assert(a.isAssigned)
          if (a.isAssigned) buffer.append((a.cluster, Add(point)))
          if (a.wasPreviouslyAssigned) buffer.append((a.previousCluster, Sub(point)))
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
      assignments: RDD[RecentAssignments]): Array[(Int, MutableWeightedVector)] =

      points.zip(assignments).filter(_._2.isAssigned).map { case (o, p) =>
        (p.cluster, o)
      }.aggregateByKey(pointOps.getCentroid)(_.add(_), _.add(_)).collect()


    /**
     * Find the closest cluster assignment that has moved/not moved since the point was last assigned.
     *
     * @param round the current round
     * @param centers the cluster centers
     * @param point the point
     * @param assignments the assignments for the point
     * @param f filter function
     * @return the assignment
     */

    def closestOf(
      round: Int,
      centers: Array[CenterWithHistory],
      point: BregmanPoint,
      assignments: RecentAssignments,
      f: (CenterWithHistory, Assignment) => Boolean): Assignment = {

      var bestDist = Infinity
      var bestIndex = noCluster
      var i = 0
      val end = centers.length
      while (i < end) {
        val center = centers(i)
        val dist = pointOps.distance(point, center.center)
        if (dist < bestDist) {
          bestIndex = i
          bestDist = dist
        }
        i = i + 1
      }
      if (bestIndex != noCluster) Assignment(bestDist, bestIndex, round) else unassigned
    }

    /**
     * Find the closest cluster assignment to a given point
     *
     * @param round the current round
     * @param centers the cluster centers
     * @param point point
     * @param assignments the recent assignments of the point
     * @return  the new assignment for the point
     */
    def reassignment(
      round: Int,
      stats: TrackingStats,
      centers: Array[CenterWithHistory],
      point: BregmanPoint,
      assignments: RecentAssignments): Assignment = {

      val closestDirty: Assignment = closestOf(round, centers, point, assignments, (x, y) => x.movedSince(y.round))
      val assignment = if (assignments.isAssigned) {
        if (closestDirty.dist < assignments.distance) {
          if (closestDirty.index == assignments.cluster) {
            stats.dirtyOther.add(1)
          } else {
            stats.dirtySame.add(1)

          }
          closestDirty
        } else if (!centers(assignments.cluster).movedSince(assignments.round)) {
          stats.stationary.add(1)
          assignments.current
        } else {
          best(round, stats, centers, point, assignments, closestDirty)
        }
      } else {
        best(round, stats, centers, point, assignments, closestDirty)
      }
      if (assignment.isUnassigned) {
        log.warn("cannot find cluster to assign point {}", assignments)
      }
      assignment
    }

    def best(
      round: Int,
      stats: TrackingStats,
      centers: Array[CenterWithHistory],
      point: BregmanPoint,
      assignments: RecentAssignments,
      closestDirty: Assignment): Assignment = {

      val closestClean = closestOf(round, centers, point, assignments, (x, y) => !x.movedSince(y.round))
      if (closestDirty.isUnassigned ||
        (closestClean.isAssigned && closestClean.dist < closestDirty.dist)) {
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
      assignments: RDD[RecentAssignments]) = {

      val cp = closestPoints(points, assignments, centersWithHistory)

      val clusterMap = countByCluster(assignments)
      Array.tabulate(centersWithHistory.length) { i =>
        val count = clusterMap.getOrElse(i, 0)
        if (count == 0) {
          val c = centersWithHistory(i)
          val d1 = cp(i).dist

          println(s"center: $i = $c")
          println(s"closest point is ${cp(i)}")
          val closerCluster = cp(i).assignments.cluster
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
      assignments: RDD[RecentAssignments],
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
      var centersWithHistory = centers.map(CenterWithHistory(_))
      var recentAssignments = initialAssignments(points, centersWithHistory)
      var terminate = false
      var round = 1
      do {
        val stats = new TrackingStats(points.sparkContext, round)
        centersWithHistory = updatedCenters(round, stats, recentAssignments, centersWithHistory)
        recentAssignments = updatedAssignments(round, stats, recentAssignments, centersWithHistory)
        updateRoundStats(round, stats, centersWithHistory, recentAssignments)
        stats.report()
        terminate = terminationCondition(stats)
        round = round + 1
      } while (!terminate)

      val d = distortion(recentAssignments)
      recentAssignments.unpersist(blocking = false)
      (d, centersWithHistory.map(_.center))
    }
    points.unpersist(blocking = false)
    results.minBy(_._1)
  }
}
