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

import com.massivedatascience.clusterer.ColumnTrackingKMeans._
import com.massivedatascience.clusterer.KMeansSelector.InitialCondition
import com.massivedatascience.clusterer.MultiKMeansClusterer.ClusteringWithDistortion
import com.massivedatascience.linalg.{ MutableWeightedVector, WeightedVector }
import com.massivedatascience.util.{ SparkHelper, XORShiftRandom }
import org.apache.spark.Partitioner._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.joda.time.DateTime

import scala.annotation.tailrec
import scala.collection.generic.FilterMonadic

import org.slf4j.LoggerFactory

object ColumnTrackingKMeans {

  /**
   *
   * @param center  the centroid of the cluster
   * @param round the round in which his cluster was last moved
   */
  private[clusterer] case class CenterWithHistory(index: Int, round: Int, center: BregmanCenter,
      initialized: Boolean) extends Serializable {
    @inline def movedSince(r: Int): Boolean = round > r
  }

  private[clusterer] val noCluster = -1
  private[clusterer] val unassigned = Assignment(Infinity, noCluster, -2)

  private[clusterer] case class Assignment(distance: Double, cluster: Int, round: Int) {
    def isAssigned: Boolean = cluster != noCluster

    def isUnassigned: Boolean = cluster == noCluster
  }

  /**
   * Find the closest cluster from a given set of clusters
   *
   * @param centers the cluster centers
   * @param point the point
   * @return the assignment of that cluster to the point
   */
  def bestAssignment(
    pointOps: BregmanPointOps,
    round: Int,
    point: BregmanPoint,
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
    if (cluster != noCluster) {
      Assignment(distance, cluster, round)
    } else {
      unassigned
    }
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
   * algorithm proceeds, more and more clusters are stationary, so fewer and fewer distance
   * calculations are needed.
   *
   * @param point point
   * @param assignment the current assignment of the point
   * @param round the current round
   * @param centers the cluster centers
   * @return  the new assignment for the point
   */
  def reassignment(
    pointOps: BregmanPointOps,
    point: BregmanPoint,
    assignment: Assignment,
    round: Int,
    centers: Seq[CenterWithHistory]): Assignment = {

    val nonStationaryCenters = centers.withFilter(c => c.movedSince(assignment.round) &&
      c.center.weight > pointOps.weightThreshold)
    val stationaryCenters = centers.withFilter(c => !c.movedSince(assignment.round) &&
      c.center.weight > pointOps.weightThreshold)
    val bestNonStationary = bestAssignment(pointOps, round, point, nonStationaryCenters)

    assignment match {
      case a: Assignment if !a.isAssigned => bestAssignment(pointOps, round, point,
        stationaryCenters, bestNonStationary)
      case a: Assignment if a.distance > bestNonStationary.distance => bestNonStationary
      case a: Assignment if !centers(a.cluster).movedSince(a.round) => a
      case _ => bestAssignment(pointOps, round, point, stationaryCenters, bestNonStationary)
    }
  }

  /**
   * Identify the new cluster assignments for a sample of the points.
   * Persists the new assignments in memory, un-persisting the previous assignments.
   *
   * @param round the number of the round
   * @param bcCenters current clusters
   * @param previousAssignments current assignments
   * @return points and their new assignments
   */
  def updatedAssignments(
    points: RDD[BregmanPoint],
    ops: BregmanPointOps,
    round: Int,
    previousAssignments: RDD[Assignment],
    bcCenters: Broadcast[IndexedSeq[CenterWithHistory]],
    updateRate: Double): RDD[Assignment] = {

    require(previousAssignments.getStorageLevel.useMemory)

    val centers = bcCenters.value
    val r = round
    val pointOps = ops

    points.zip(previousAssignments).mapPartitionsWithIndex[Assignment] {
      (index, assignedPoints) =>
        val rand = new XORShiftRandom(r ^ (index << 16))
        assignedPoints.map {
          case (point, assignment) =>
            if (rand.nextDouble() > updateRate) assignment
            else reassignment(pointOps, point, assignment, round, centers)
        }
    }
  }
}

/**
 * A KMeans implementation that tracks which clusters moved and which points are assigned to which
 * clusters and the distance to the closest cluster.
 *
 * Notably, following each iterator of Lloyd's algorithm, empty clusters are provided new
 * centers using the K-Means-|| algorithm.
 *
 *
 * We maintain for each point two assignments: 1) its current assignment and 2) its previous
 * assignment. With this data, we can determine if a point moves between assignments by comparing
 * the assignments.
 *
 * We maintain for each cluster its index, generation number, centroid.
 *
 * The invariants are:
 *
 * <ol>
 * <li>
 * each cluster is assigned a generation number
 * </li>
 * <li>
 * generation numbers are monotonically increasing
 * </li>
 * <li>
 * all clusters whose centroids change in one Lloyd's round are assigned the same generation #
 * </li>
 * <li>
 * when the membership of a cluster changes, the generation number of the cluster is increased
 * </li>
 * <li>
 * each point is assigned the index of the cluster that is a member of
 * </li>
 * </ol>
 *
 * Initial condition:
 * <ol>
 * <li>
 * Initial cluster centroids are provided. All clusters are assigned generation -1 with the
 * provided cluster centroids
 * </li>
 * <li>
 * All points are assigned to the sentinel cluster (index == -1) with generation -2
 * </li>
 * <li>
 * (Some) points are re-assigned to (non-sentinel) clusters, resulting in the setting of the
 * generation number of those points to -1
 * </li>
 * <li>
 * The current round is set to 0
 * </li>
 * </ol>
 *
 *
 * Lloyd's algorithm can be stated as:
 *
 * <ol>
 * <li>
 * Increase the round
 * </li>
 *
 * <li>
 * If any points were re-assigned (change in generation number), then update the clusters
 * impacted by the re-assignment:
 * <ul>
 * <li>
 * Compute the new cluster centroids for the out-dated clusters
 * </li>
 * <li>
 * Set the generation of the clusters affect to be the value of the round
 * </li>
 * </ul>
 * </li>
 *
 * <li>
 * Increase the round
 * </li>
 *
 * <li>
 * If any centers were updated, then update the assignments of the points:
 * <ul>
 * <li>
 * For each point (or a random sub-set of the points), identify the closest cluster
 * </li>
 * <li>
 * If the closest cluster has a different index or generation number, then update the
 * assignments of the point so that its index is the index of the cluster to which it is assigned
 * and the generation is the round the new assignment is made
 * </li>
 * </ul>
 * </li>
 * </ol>
 *
 */
case class ColumnTrackingKMeans(config: KMeansConfig = DefaultKMeansConfig)
    extends MultiKMeansClusterer with SparkHelper {

  val logger = LoggerFactory.getLogger(getClass.getName)


  private[this] def distortion(data: RDD[Assignment]) =
    data.filter(_.isAssigned).map(_.distance).sum()

  private[this] def backFilledCenters(
    points: RDD[BregmanPoint],
    pointOps: BregmanPointOps,
    round: Int,
    currentAssignments: RDD[Assignment],
    centers: Array[CenterWithHistory]): IndexedSeq[CenterWithHistory] = {

    // adjust centers to fill in empty slots
    val weakClusters = centers.filter(_.center.weight < pointOps.weightThreshold)
    //val weakClusters = centers.filter(_ => myRand.nextDouble() < 0.10)

    if (weakClusters.nonEmpty && round < config.maxRoundsToBackfill) {
      logger.info(s"replacing ${weakClusters.length} empty clusters")
      val strongClusters = centers.filter(!weakClusters.contains(_))
      val bregmanCenters = strongClusters.toIndexedSeq.map(_.center)
      val seed = new DateTime().getMillis
      val incrementer = new KMeansParallel(2, config.fractionOfPointsToWeigh)
      val costs = currentAssignments.map(_.distance)
      val initialCondition = InitialCondition(Seq(bregmanCenters), Seq(costs))
      val newCenters = incrementer.init(pointOps, points, centers.length,
        Some(initialCondition), 1, seed)(0)
      logger.info(s"${newCenters.length} centers returned, dropping ${bregmanCenters.length}")
      val additional = newCenters.drop(bregmanCenters.length)
      val replacements = weakClusters.zip(additional).map {
        case (x, y) => x.copy(round = round,
          center = y, initialized = false)
      }
      logger.info(s"replaced ${replacements.length} clusters")

      strongClusters ++ replacements ++ weakClusters.drop(replacements.length)
    } else {
      centers.toIndexedSeq
    }
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
  private[this] def latestCenters(
    points: RDD[BregmanPoint],
    pointOps: BregmanPointOps,
    round: Int,
    previousCenters: IndexedSeq[CenterWithHistory],
    currentAssignments: RDD[Assignment],
    previousAssignments: RDD[Assignment]): Array[CenterWithHistory] = {

    val centers = previousCenters.toArray
    if (config.addOnly) {
      val results = completeMovedCentroids(points, pointOps, currentAssignments,
        previousAssignments, previousCenters.length)
      for ((index, location) <- results) {
        centers(index) = CenterWithHistory(index, round, pointOps.toCenter(location.asImmutable),
          initialized = true)
      }
    } else {
      val changes = deltasOfMovedCentroids(points, pointOps, currentAssignments,
        previousAssignments, previousCenters.length)
      for ((index, delta) <- changes) {
        val previous = previousCenters(index)
        val location = if (previous.initialized)
          delta.add(pointOps.toPoint(previous.center))
        else
          delta

        centers(index) = CenterWithHistory(index, round, pointOps.toCenter(location.asImmutable),
          initialized = true)
      }
    }
    centers
  }

  /**
   * Computes all centroids, but only returns centroids of changed clusters.
   * *
   * A previous implementation that uses aggregateByKey on (index, point) tuples was observed
   * to cause to much garbage collection overhead.
   *
   * @param points points
   * @param pointOps distance function
   * @param assignments assignments of points
   * @param previousAssignments previous assignments of points
   * @param numCenters current number of non-empty clusters
   * @return complete centroids of clusters that have moved
   */
  private[this] def completeMovedCentroids[T <: WeightedVector](
    points: RDD[T],
    ops: BregmanPointOps,
    assignments: RDD[Assignment],
    previousAssignments: RDD[Assignment],
    numCenters: Int): Array[(Int, MutableWeightedVector)] = {

    require(points.getStorageLevel.useMemory)
    require(assignments.getStorageLevel.useMemory)

    logger.info(s"using $numCenters centers")

    implicit val sc = points.sparkContext

    withBroadcast(ops) { (bcPointOps: Broadcast[BregmanPointOps]) =>
      points.zipPartitions(assignments, previousAssignments) {
        (x: Iterator[T], y: Iterator[Assignment], z: Iterator[Assignment]) =>
          val pointOps = bcPointOps.value
          val centroids: Array[MutableWeightedVector] = Array.tabulate(numCenters)(i => pointOps.make(i))
          val changed = new Array[Boolean](numCenters)
          val indexBuffer = new collection.mutable.ArrayBuilder.ofInt
          indexBuffer.sizeHint(numCenters)

          def update(index: Int, point: T): Unit =
            if (index != -1 && !changed(index)) {
              changed(index) = true
              indexBuffer += index
            }

          while (y.hasNext && x.hasNext && z.hasNext) {
            val point = x.next()
            val current = y.next()
            val previous = z.next()
            val index = current.cluster
            if (index >= 0) centroids(index).add(point)
            if (current.cluster != previous.cluster) {
              update(previous.cluster, point)
              update(current.cluster, point)
            }
          }

          val changedClusters = indexBuffer.result()
          changedClusters.map(index => (index, centroids(index))).iterator
      }.combineByKey[MutableWeightedVector](
        (x: MutableWeightedVector) => x,
        (_: MutableWeightedVector).add(_: MutableWeightedVector),
        (_: MutableWeightedVector).add(_: MutableWeightedVector),
        defaultPartitioner(points),
        mapSideCombine = false).collect()
    }
  }

  /**
   *
   * @param points points
   * @param pointOps distance function
   * @param assignments assignments of points
   * @param previousAssignments previous assignments of points
   * @param numCenters current number of non-empty clusters
   * @return deltas to clusters that have moved
   */
  private[this] def deltasOfMovedCentroids[T <: WeightedVector](
    points: RDD[T],
    pointOps: BregmanPointOps,
    assignments: RDD[Assignment],
    previousAssignments: RDD[Assignment],
    numCenters: Int): Array[(Int, MutableWeightedVector)] = {

    require(points.getStorageLevel.useMemory)
    require(assignments.getStorageLevel.useMemory)
    require(previousAssignments.getStorageLevel.useMemory)

    points.zipPartitions(assignments, previousAssignments) {
      (x: Iterator[T], y: Iterator[Assignment], z: Iterator[Assignment]) =>
        val centroids = IndexedSeq.tabulate(numCenters)(i => pointOps.make(i))

        while (z.hasNext && y.hasNext && x.hasNext) {
          val point = x.next()
          val currentAssignment = y.next()
          val previousAssignment = z.next()
          if (currentAssignment != previousAssignment) {
            val current = currentAssignment.cluster
            val previous = previousAssignment.cluster
            if (previous != -1) centroids(previous).sub(point)
            if (current != -1) centroids(current).add(point)
          }
        }
        centroids.filter(_.nonEmpty).map(x => (x.index, x)).iterator
    }.reduceByKey(_.add(_)).collect()
  }

  /**
   * Create a K-Means clustering of the input and report on the resulting distortion
   *
   * @param points points to cluster
   * @param centerArrays initial cluster centers
   * @return the distortion of the clustering on the points and the cluster centers (model)
   */
  def cluster(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    points: RDD[BregmanPoint],
    centerArrays: Seq[Centers]): Seq[ClusteringWithDistortion] = {

    require(points.getStorageLevel.useMemory)

    implicit val sc = points.sparkContext
    val detector = new ConvergenceDetector(sc)

    @tailrec
    def lloyds(
      round: Int,
      assignments: RDD[Assignment],
      centers: IndexedSeq[CenterWithHistory]): (RDD[Assignment], IndexedSeq[CenterWithHistory]) = {

      require(assignments.getStorageLevel.useMemory)

      val newAssignments = withBroadcast(centers) { bcCenters =>
        sync[Assignment](s"assignments round $round",
          updatedAssignments(points, pointOps, round, assignments, bcCenters, config.updateRate))
      }

      val newCenters = latestCenters(points, pointOps, round + 1, centers, newAssignments,
        assignments)

      val backFilled = backFilledCenters(points, pointOps, round + 1, newAssignments, newCenters)

      detector.update(pointOps, (round + 1) / 2, backFilled, centers, newAssignments, assignments)

      if (round != 0) assignments.unpersist()
      if ((round / 2 + 1) == maxIterations || detector.stable())
        (newAssignments, backFilled)
      else
        lloyds(round + 2, newAssignments, backFilled)
    }

    require(config.updateRate <= 1.0 && config.updateRate >= 0.0)
    logger.info(s"runs = ${centerArrays.size}")

    val u = Assignment(Infinity, noCluster, -2)
    withCached[Assignment, Seq[ClusteringWithDistortion]]("empty assignments", points.map(x => u)) { empty =>
      centerArrays.map { initialCenters =>
        val centers = initialCenters.zipWithIndex.map {
          case (c, i) =>
            CenterWithHistory(i, -1, c, initialized = false)
        }
        val (assignments, centersWithHistory) = lloyds(0, empty, centers)
        assignments.unpersist(blocking = false)
        ClusteringWithDistortion(distortion(assignments), centersWithHistory.map(_.center).toIndexedSeq)
      }
    }
  }
}
