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
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Accumulator, Logging}

import scala.collection.Map


/**
 *
 * A K-means implementation that caches distances to clusters and that adjusts cluster centers by
 * sampling the points on each iteration.
 *
 * Maintain for each point the index of its closest cluster and the distance to that cluster.
 *
 * Maintain for each cluster the weighted centroid of its points.  Each cluster is initialized with
 * exactly one point from the input point set. Mark each cluster as "moved."
 *
 * 1. Broadcast the moved cluster centers and their indices.
 *
 * 2. For each point, for each moved cluster center, mark the distance to that cluster as "stale."
 * If the cluster is empty, mark the cluster as "empty".
 *
 * 3. Select a set of points to update.  Update the distance to each stale cluster centers.
 *
 * 4. For each selected point, identify the index of the closest cluster center and the distance.
 * If the index has moved, emit two changes: one for the old cluster center and one for the new
 * cluster center, keyed by cluster index:
 *
 * (index, (point, add/subtract))
 *
 * 5. Sum by key the changes for the clusters. Identify the moved clusters and update their
 * locations.
 *
 * 6. If any cluster moved, then go to 1.
 */

class CachingKMeans(ops: BregmanPointOps) extends Serializable with Logging {

  case class Closest(dist: Double, index: Int)

  case class FatPoint(
    point: BregmanPoint,
    assignment: Array[Closest],
    var current: Int,
    distances: Array[Double])

  case class FatCenter(center: BregmanCenter, index: Int, moved: Boolean, nonEmpty: Boolean)

  val unknown = -1.0
  val unassigned = Closest(Unknown, -1)

  def cluster(
    data: RDD[BregmanPoint],
    centers: Array[BregmanCenter],
    maxIterations: Int = 20,
    sampleRate: Double = 1.0)
  : (Double, KMeansModel) = {
    logInfo(s"sample rate = $sampleRate")
    logInfo(s"max iterations = $maxIterations")

    var fatPoints: RDD[FatPoint] = initialFatPoints(data, centers.length).persist()
    data.unpersist()
    var fatCenters: Array[FatCenter] = initialFatCenters(centers)
    var numMoved = fatCenters.count(c => c.moved)
    var numIterations = 0

    while (numMoved > 0 && numIterations < maxIterations) {
      logInfo(s"iteration $numIterations")
      val sc = data.sparkContext
      val bcCenters = sc.broadcast(fatCenters)
      val costDiff = sc.accumulator(0.0)
      val freshPoints = sc.accumulator[Int](0)
      val movedPoints = sc.accumulator[Int](0)
      val improvedPoints = sc.accumulator[Int](0)

      val changes = getNewAssignments(freshPoints, improvedPoints, movedPoints, costDiff, bcCenters,
        fatPoints, sampleRate, numIterations)
      fatPoints = getUpdatedFatPoints(fatPoints, changes)
      fatCenters = getUpdatedFatCenters(getCentroidChanges(bcCenters, fatPoints), fatCenters)
      numMoved = fatCenters.count(c => c.moved)
      logInfo(s"lowered distortion by ${costDiff.value}")
      logInfo(s"relocated = $numMoved centers")
      logInfo(s"improved points = ${improvedPoints.value}")
      logInfo(s"moved points = ${movedPoints.value}")
      logInfo(s"fresh points = ${freshPoints.value}")
      logInfo(s"average improvement per point = ${costDiff.value / improvedPoints.value}")

      bcCenters.unpersist()
      numIterations = numIterations + 1
    }

    (distortion(fatPoints), new KMeansModel(ops, fatCenters.map(_.center)))
  }

  /**
   * The initial fat cluster centers
   *
   * @param centers  the initial center values
   * @return the clusters with their index, whether they moved, and whether they have been 
   *         initialized
   */
  def initialFatCenters(centers: Array[BregmanCenter]): Array[FatCenter] =
    centers.zipWithIndex.map {
      case (c, i) => FatCenter(c, i, moved = true, nonEmpty = false)
    }.toArray

  /**
   * The initial fat points
   *
   * @param data the incoming data
   * @param len the number of cluster centers
   * @return the fat points (points with cluster assignment and distances
   */
  def initialFatPoints(data: RDD[BregmanPoint], len: Int): RDD[FatPoint] =
    data map { p => FatPoint(p, Array(unassigned, unassigned), 0, Array.fill[Double](len)(Unknown))}

  /**
   * Update the cluster assignment of the fat points.  Need to manage the RDD storage to avoid stack
   * overflow.
   *
   * @param fatPoints the old points
   * @param updatedPoints  the new points
   * @return  the new fat points
   */
  def getUpdatedFatPoints(fatPoints: RDD[FatPoint], updatedPoints: RDD[FatPoint]): RDD[FatPoint] = {
    val oldFatPoints = fatPoints
    updatedPoints.persist(StorageLevel.MEMORY_ONLY_SER).count()
    oldFatPoints.unpersist()
    updatedPoints
  }

  /**
   * Identify the new cluster assignments for a sample of the points
   *
   * @param bcCenters current clusters
   * @param fatPoints current points
   * @param sampleRate sample rate to use for sampling the points to update
   * @param seed  the seed value to use for the random number generator
   * @return points and their new assignments
   */
  def getNewAssignments(
    freshPoints: Accumulator[Int],
    improvedPoints: Accumulator[Int],
    movedPoints: Accumulator[Int],
    costDiff: Accumulator[Double],
    bcCenters: Broadcast[Array[FatCenter]],
    fatPoints: RDD[FatPoint],
    sampleRate: Double,
    seed: Int): RDD[FatPoint] = {

    fatPoints.mapPartitionsWithIndex { (index, points) =>
      val rand = new XORShiftRandom(seed ^ index << 16)
      val myFatCenters = bcCenters.value
      points.map { p =>

        if (rand.nextDouble() > sampleRate) {
          markStale(myFatCenters, p.point, p.distances)
          p
        } else {
          val closest = getClosest(updateDistances(myFatCenters, p.point, p.distances))
          p.assignment(p.current) match {
            case Closest(_, -1) => freshPoints.add(1)
            case Closest(d, i) =>
              val diff = d - closest.dist
              if (diff > 0.0) improvedPoints.add(1)
              costDiff += diff
          }
          p.current = 1 - p.current
          p.assignment(p.current) = closest
          if (p.assignment(0).index != p.assignment(1).index) movedPoints.add(1)
          p
        }
      }
    }
  }

  /**
   * Update the clusters given the movement of the centroids
   *
   * @param centroidChanges deltas to the centroids of the clusters
   * @param fatCenters  the current clusters
   * @return
   */
  def getUpdatedFatCenters(
    centroidChanges: Map[Int, MutableWeightedVector],
    fatCenters: Array[FatCenter]): Array[FatCenter] = {

    centroidChanges.map {
      case (index, delta) =>
        val c = fatCenters(index)
        if (delta.weight != 0.0) {
          val x = if (c.nonEmpty) delta.add(c.center) else delta
          val centroid = x.asImmutable
          FatCenter(ops.toCenter(centroid), c.index, moved = true, nonEmpty = true)
        } else {
          c.copy(moved = false)
        }
    }.toArray
  }

  def distortion(data: RDD[FatPoint]): Double = {
    data.mapPartitions {
      points =>
        Array(points.map { p => p.assignment(p.current).dist}.sum).iterator
    }.reduce(_ + _)
  }

  /**
   * Identify cluster changes
   *
   * @param bcCenters current cluster centers
   * @param points points and their assignments
   * @return changes to cluster position
   */
  def getCentroidChanges(
    bcCenters: Broadcast[Array[FatCenter]],
    points: RDD[FatPoint]): Map[Int, MutableWeightedVector] =

    points.mapPartitions { changes =>
      val centers = bcCenters.value.map { _ => ops.getCentroid}

      for (p <- changes if p.assignment(0).index != p.assignment(1).index) {
        if (p.assignment(p.current).index != -1) {
          centers(p.assignment(p.current).index).add(p.point)
        }
        if (p.assignment(1 - p.current).index != -1) {
          centers(p.assignment(1 - p.current).index).sub(p.point)
        }
      }
      centers.zipWithIndex.map { case (l, r) => (r, l)}.iterator
    }.reduceByKeyLocally { case (l, r) => l.add(r)}

  /**
   * Update distances to cluster centers in place
   *
   * @param myFatCenters current clusters
   * @param point   point to update
   * @param distances  current array of distances to clusters
   * @return updated array of distances
   */
  def updateDistances(
    myFatCenters: Array[FatCenter],
    point: BregmanPoint,
    distances: Array[Double]): Array[Double] = {
    for (c <- myFatCenters if c.moved || distances(c.index) == Unknown)
      distances(c.index) = if (c.center.weight > 0.0) ops.distance(point, c.center) else Unknown
    distances
  }

  /**
   * Mark distances to moved cluster centers are stale.
   *
   * @param myFatCenters current clusters
   * @param point   point to update
   * @param distances  current array of distances to clusters
   * @return updated array of distances
   */
  def markStale(myFatCenters: Array[FatCenter], point: BregmanPoint, distances: Array[Double]) = {
    for (c <- myFatCenters if c.moved) distances(c.index) = Unknown
    distances
  }

  /**
   * Return index to closest cluster center
   *
   * @param distances current array of distances to clusters
   * @return Some(index of closest cluster center) or None if no distance is known or cluster is
   *         empty
   */
  def getClosest(distances: Array[Double]): Closest = {
    var i = 0
    val end = distances.length
    var bestDist = Infinity
    var bestIndex = -1
    while (i < end) {
      val d = distances(i)
      if (d != Unknown && d < bestDist) {
        bestDist = d
        bestIndex = i
      }
      i = i + 1
    }
    if (bestIndex != -1) Closest(bestDist, bestIndex) else unassigned
  }
}
