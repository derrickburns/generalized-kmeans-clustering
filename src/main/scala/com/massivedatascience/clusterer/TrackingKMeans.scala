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

/**
 * A KMeans implementation that tracks which clusters moved and which points are assigned to which
 * clusters and the distance to the closest cluster.
 *
 * @param updateRate  percentage of points that are updated on each round
 */
class TrackingKMeans(
  updateRate: Double = 1.0,
  terminationCondition: TerminationCondition = DefaultTerminationCondition)
  extends MultiKMeansClusterer {

  /**
   *
   * @param center  the centroid of the cluster
   * @param round the round in which his cluster was last moved
   */
  case class FatCenter(center: BregmanCenter, round: Int = -1) {
    def movedSince(r: Int): Boolean = round >= r

    def initialized: Boolean = round >= 0
  }

  /**
   *
   * @param location the location of the point
   * @param current  the current cluster (and distance to that cluster) that this point belongs to
   * @param previous the previous cluster (and distance to that cluster) that this point belongs to
   */
  case class FatPoint(location: BregmanPoint, current: Assignment, previous: Assignment) {
    def cluster = current.index

    def previousCluster = previous.index

    def isAssigned = current.isAssigned

    def wasPreviouslyAssigned = previous.isAssigned

    def wasReassigned = current.index != previous.index

    def assign(closest: Assignment): FatPoint = FatPoint(location, closest, current)

    def distance = current.dist

    def improvement = previous.dist - current.dist

    def round = current.round
  }

  type FatCenters = Array[FatCenter]
  val noCluster = -1
  val unassigned = Assignment(Infinity, noCluster, -2)

  /**
   *
   * @param dist the distance to the closest cluster
   * @param index the index of the closest cluster, or -1 if no cluster is assigned
   * @param round the round that this assignment was made
   */
  case class Assignment(dist: Double, index: Int, round: Int) {
    def isAssigned = index != noCluster

    def isUnassigned = index == noCluster
  }

  def cluster(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centerArrays: Array[Array[BregmanCenter]]): (Double, Array[BregmanCenter]) = {

    assert(updateRate <= 1.0 && updateRate >= 0.0)

    def cluster(): (Double, Array[BregmanCenter]) = {

      assert(updateRate <= 1.0 && updateRate >= 0.0)

      logInfo(s"update rate = $updateRate")
      logInfo(s"runs = ${centerArrays.size}")

      val results = for (centers <- centerArrays) yield {
        var fatCenters = centers.map(FatCenter(_))
        var fatPoints = initialFatPoints(data, fatCenters)
        var terminate = false
        var round = 1
        do {
          val stats = new TrackingStats(data.sparkContext, round)
          fatCenters = updatedCenters(round, stats, fatPoints, fatCenters, updateRate)
          fatPoints = reassignedPoints(round, stats, fatCenters, fatPoints, updateRate)
          updateRoundStats(round, stats, fatCenters, fatPoints)
          stats.report()
          terminate = terminationCondition(stats)
          round = round + 1
        } while (!terminate)

        (distortion(fatPoints), fatCenters.map(_.center))
      }
      results.minBy(_._1)
    }

    /**
     * The initial fat points
     *
     * @param data the incoming data
     * @param fatCenters cluster centers
     * @return the fat points (points with two most recent cluster assignments)
     */
    def initialFatPoints(data: RDD[BregmanPoint], fatCenters: FatCenters) = {
      val result = data.map(FatPoint(_, unassigned, unassigned)).map {
        fp => fp.copy(current = closestOf(0, fatCenters, fp, (_, _) => true))
      }
      result.persist()
      data.unpersist()
      result
    }

    /**
     * Augment the stats with information about the clusters
     *
     * @param round the round
     * @param stats  the stats to update
     * @param fatCenters the cluster centers
     * @param fatPoints the fat points
     */
    def updateRoundStats(
      round: Int,
      stats: TrackingStats,
      fatCenters: FatCenters,
      fatPoints: RDD[FatPoint]) {

      val clusterCounts = countByCluster(fatPoints)
      val biggest = clusterCounts.maxBy(_._2)
      stats.largestCluster.setValue(biggest._2)
      stats.nonemptyClusters.add(clusterCounts.size)
      stats.emptyClusters.add(fatCenters.size - clusterCounts.size)
      stats.relocatedCenters.setValue(fatCenters.count(c => c.round == round))
    }

    def updateStats(stats: TrackingStats, p: FatPoint): Unit = {
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

    /**
     * Identify the new cluster assignments for a sample of the points.
     * @param round the number of the round
     * @param stats statistics on round
     * @param fatCenters current clusters
     * @param fatPoints current points
     * @param rate sample rate to use for sampling the points to update
     * @return points and their new assignments
     */
    def reassignedPoints(
      round: Int,
      stats: TrackingStats,
      fatCenters: FatCenters,
      fatPoints: RDD[FatPoint],
      rate: Double): RDD[FatPoint] = {

      val bcCenters = fatPoints.sparkContext.broadcast(fatCenters)
      val result = fatPoints.mapPartitionsWithIndex { (index, points) =>
        val rand = new XORShiftRandom(round ^ (index << 16))
        val centers = bcCenters.value
        points.map { p =>
          p.assign(if (rand.nextDouble() > rate) unassigned else assignment(round, stats, centers, p))
        }
      }

      result.map(updateStats(stats, _))
      bcCenters.unpersist()
      result.persist(StorageLevel.MEMORY_ONLY_SER).count()
      fatPoints.unpersist()
      result
    }

    /**
     * Update the clusters (stochasticly if rate < 1.0)
     *
     * @param round the round
     * @param stats statistics to keep
     * @param fatPoints  the points
     * @param fatCenters  the current clusters
     * @param rate the sample rate for the points
     * @return
     */
    def updatedCenters(
      round: Int,
      stats: TrackingStats,
      fatPoints: RDD[FatPoint],
      fatCenters: FatCenters,
      rate: Double): FatCenters = {

      val changes = if (rate < 1.0)
        getStochasticCentroidChanges(fatPoints)
      else
        getExactCentroidChanges(fatPoints)

      changes.map { case (index, delta) =>
        val c = fatCenters(index)
        val oldPosition = pointOps.toPoint(c.center)
        val x = if (c.initialized) delta.add(oldPosition) else delta
        fatCenters(index) = FatCenter(pointOps.toCenter(x), round)
        stats.movement.add(pointOps.distance(oldPosition, fatCenters(index).center))
      }
      fatCenters
    }

    def distortion(data: RDD[FatPoint]) = data.filter(_.isAssigned).map(_.distance).sum()

    /**
     * Identify cluster changes.
     *
     * Create an array of the changes per cluster.  Some clusters may not have changes.  They will
     * not be represented in the change set.
     *
     * @param points points and their assignments
     * @return changes to cluster position
     */
    def getExactCentroidChanges(points: RDD[FatPoint]): Array[(Int, MutableWeightedVector)] = {
      points.mapPartitions { pts =>
        val buffer = new ArrayBuffer[(Int, (WeightedVector, Boolean))]
        for (p <- pts if p.wasReassigned) {
          assert(p.isAssigned)
          if (p.isAssigned)
            buffer.append((p.cluster, (p.location, true)))

          if (p.wasPreviouslyAssigned)
            buffer.append((p.previousCluster, (p.location, false)))
        }
        buffer.iterator
      }.aggregateByKey(pointOps.getCentroid)(
        (x, y) => if (y._2) x.add(y._1) else x.sub(y._1),
        (x, y) => x.add(y)
      ).collect()
    }

    def getStochasticCentroidChanges(points: RDD[FatPoint]): Array[(Int, MutableWeightedVector)] =
      points.filter(_.isAssigned).map { p =>
        (p.cluster, p.location)
      }.aggregateByKey(pointOps.getCentroid)(_.add(_), _.add(_)).collect()

    /**
     * count number of points assigned to each cluster
     *
     * @param points the points
     * @return a map from cluster index to number of points assigned to that cluster
     */
    def countByCluster(points: RDD[FatPoint]): Map[Int, Long] =
      points.filter(_.isAssigned).map { p => (p.cluster, p)}.countByKey()

    /**
     * Find the closest point and distance to each cluster
     *
     * @param points the points
     * @param centers the clusters
     * @return a map from cluster index to the closest point to that cluster and the distance
     */
    def closestPoints(points: RDD[FatPoint], centers: FatCenters): Map[Int, (FatPoint, Double)] = {
      val bcCenters = points.sparkContext.broadcast(centers)
      points.mapPartitions { pts =>
        val bc = bcCenters.value
        pts.flatMap { p =>
          bc.zipWithIndex.map { case (c, i) =>
            (i, (p, pointOps.distance(p.location, c.center)))
          }
        }
      }.reduceByKeyLocally { (x, y) => if (x._2 < y._2) x else y}
    }

    /**
     * Find the closest cluster assignment that has moved/not moved since the point was last assigned.
     *
     * @param round the current round
     * @param centers the cluster centers
     * @param p the point from which we measure distance
     * @param f filter function
     * @return the assignment
     */
    def closestOf(
      round: Int,
      centers: FatCenters,
      p: FatPoint, f: (FatCenter, FatPoint) => Boolean) = {

      var bestDist = Infinity
      var bestIndex = noCluster
      var i = 0
      val end = centers.length
      while (i < end) {
        val center = centers(i)
        if (f(center, p)) {
          val dist = pointOps.distance(p.location, center.center)
          if (dist < bestDist) {
            bestIndex = i
            bestDist = dist
          }
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
     * @param p the point from which we measure distance
     * @return  the assignment
     */
    def assignment(
      round: Int,
      stats: TrackingStats,
      centers: FatCenters,
      p: FatPoint): Assignment = {

      val closestDirty = closestOf(round, centers, p, (x, y) => x.movedSince(y.round))
      val assignment = if (p.isAssigned) {
        if (closestDirty.dist < p.distance) {
          if (closestDirty.index == p.cluster) {
            stats.dirtyOther.add(1)
          } else {
            stats.dirtySame.add(1)

          }
          closestDirty
        } else if (!centers(p.cluster).movedSince(p.round)) {
          stats.stationary.add(1)
          p.current
        } else {
          best(round, stats, centers, p, closestDirty)
        }
      } else {
        best(round, stats, centers, p, closestDirty)
      }
      if (assignment.isUnassigned) {
        log.warn("cannot find cluster to assign point {}", p)
      }
      assignment
    }

    def best(
      round: Int,
      stats: TrackingStats,
      centers: FatCenters,
      p: FatPoint,
      closestDirty: Assignment): Assignment = {

      val closestClean = closestOf(round, centers, p, (x, y) => !x.movedSince(y.round))
      if (closestDirty.isUnassigned ||
        (closestClean.isAssigned && closestClean.dist < closestDirty.dist)) {
        stats.closestClean.add(1)
        closestClean
      } else {
        stats.closestDirty.add(1)
        closestDirty
      }
    }

    def showEmpty(fatCenters: FatCenters, fatPoints: RDD[FatPoint]) = {
      val cp = closestPoints(fatPoints, fatCenters)

      val clusterMap = countByCluster(fatPoints)
      Array.tabulate(fatCenters.length) { i =>
        val count = clusterMap.getOrElse(i, 0)
        if (count == 0) {
          val c = fatCenters(i)
          val d1 = cp(i)._2

          println(s"center: $i = $c")
          println(s"closest point is ${cp(i)}")
          val closerCluster = cp(i)._1.cluster
          val d2 = pointOps.distance(cp(i)._1.location, fatCenters(closerCluster).center)
          println(s"closest center to that point is $closerCluster =" +
            s"' ${fatCenters(closerCluster)} at distance $d2")
          println()
          assert(d1 >= d2, s"closest point to cluster $d1 should be >= to closest cluster " +
            s"to point $d2")
        }
      }
    }
    cluster()
  }
}
