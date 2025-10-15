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
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations under the License.
 */

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.ColumnTrackingKMeans._
import com.massivedatascience.clusterer.MultiKMeansClusterer.ClusteringWithDistortion
import com.massivedatascience.linalg.{MutableWeightedVector, WeightedVector}
import com.massivedatascience.util.{SparkHelper, XORShiftRandom}
import org.apache.spark.Partitioner.defaultPartitioner
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer

object ColumnTrackingKMeans {

  /** Cluster metadata including history of centroid movement. */
  private[clusterer] case class CenterWithHistory(
    index: Int,
    round: Int,
    center: BregmanCenter,
    initialized: Boolean
  ) extends Serializable {
    @inline def movedSince(r: Int): Boolean = round > r
  }

  private[clusterer] val noCluster: Int = -1

  private[clusterer] val PosInf: Double = Double.PositiveInfinity

  private[clusterer] val unassigned: Assignment =
    Assignment(PosInf, noCluster, -2)

  private[clusterer] case class Assignment(distance: Double, cluster: Int, round: Int) {
    def isAssigned: Boolean = cluster != noCluster
    def isUnassigned: Boolean = cluster == noCluster
  }

  /** Find the closest cluster from a given set of centers. */
  def bestAssignment(
    pointOps: BregmanPointOps,
    round: Int,
    point: BregmanPoint,
    centers: Iterable[CenterWithHistory],
    initialAssignment: Assignment = unassigned
  ): Assignment = {
    var bestDistance = initialAssignment.distance
    var bestCluster = initialAssignment.cluster

    centers.foreach { c =>
      val dist = pointOps.distance(point, c.center)
      if (dist < bestDistance) {
        bestDistance = dist
        bestCluster = c.index
      }
    }
    if (bestCluster != noCluster) Assignment(bestDistance, bestCluster, round) else unassigned
  }

  /** Optimized reassignment step that avoids unnecessary distance computations. */
  def reassignment(
    pointOps: BregmanPointOps,
    point: BregmanPoint,
    assignment: Assignment,
    round: Int,
    centers: Seq[CenterWithHistory]
  ): Assignment = {

    val nonStationary = centers.filter(c => c.movedSince(assignment.round) && c.center.weight > pointOps.weightThreshold)
    val stationary = centers.filter(c => !c.movedSince(assignment.round) && c.center.weight > pointOps.weightThreshold)

    val bestNonStationary = bestAssignment(pointOps, round, point, nonStationary)

    assignment match {
      case a if !a.isAssigned =>
        bestAssignment(pointOps, round, point, stationary, bestNonStationary)
      case a if a.distance > bestNonStationary.distance =>
        bestNonStationary
      case a if !centers(a.cluster).movedSince(a.round) =>
        a
      case _ =>
        bestAssignment(pointOps, round, point, stationary, bestNonStationary)
    }
  }

  /** Update cluster assignments, potentially stochastically if updateRate < 1. */
  def updatedAssignments(
    points: RDD[BregmanPoint],
    ops: BregmanPointOps,
    round: Int,
    previousAssignments: RDD[Assignment],
    bcCenters: Broadcast[IndexedSeq[CenterWithHistory]],
    updateRate: Double
  ): RDD[Assignment] = {

    require(previousAssignments.getStorageLevel.useMemory, "previousAssignments must be cached in memory")

    val centers = bcCenters.value
    val r = round
    val pointOps = ops

    points.zip(previousAssignments).mapPartitionsWithIndex { (index, iter) =>
      val rand = new XORShiftRandom(r ^ (index << 16))
      iter.map { case (point, assign) =>
        if (rand.nextDouble() > updateRate) assign
        else reassignment(pointOps, point, assign, round, centers)
      }
    }
  }
}

/** KMeans that tracks which clusters moved and reuses stationary distances to reduce computation. */
case class ColumnTrackingKMeans(config: KMeansConfig = DefaultKMeansConfig)
    extends MultiKMeansClusterer
    with SparkHelper
    with Logging {

  private[this] def distortion(data: RDD[ColumnTrackingKMeans.Assignment]): Double =
    data.filter(_.isAssigned).map(_.distance).sum()

  /** Replace empty or weak clusters with new centers derived from strong ones. */
  private[this] def backFilledCenters(
    points: RDD[BregmanPoint],
    pointOps: BregmanPointOps,
    round: Int,
    currentAssignments: RDD[ColumnTrackingKMeans.Assignment],
    centers: Array[CenterWithHistory]
  ): IndexedSeq[CenterWithHistory] = {

    if (centers.isEmpty) {
      logger.warn("Empty centers array provided to backFilledCenters")
      IndexedSeq.empty
    } else {
      val weakClusters = centers.filter(_.center.weight < pointOps.weightThreshold)
      if (weakClusters.isEmpty || round >= config.maxRoundsToBackfill) {
        centers.toIndexedSeq
      } else {
        val strongClusters = centers.filterNot(weakClusters.contains)
        val replacementsOpt: Option[IndexedSeq[CenterWithHistory]] =
          if (strongClusters.isEmpty) {
            logger.warn("No strong clusters available for backfilling")
            Some(IndexedSeq.empty)
          } else {
            val assignments = currentAssignments.filter(_.isAssigned)
            if (assignments.isEmpty()) {
              logger.warn("No assignments available for backfilling")
              None
            } else {
              val strongIndices = strongClusters.map(_.index).toSet
              val pointsInStrong = points.zipPartitions(assignments) { (pts, assigns) =>
                val buf = new ArrayBuffer[(Int, BregmanPoint)]()
                while (pts.hasNext && assigns.hasNext) {
                  val p = pts.next()
                  val a = assigns.next()
                  if (strongIndices.contains(a.cluster)) buf += ((a.cluster, p))
                }
                buf.iterator
              }.filter(_._1 >= 0).map { case (k, v) => (k, v) }

              if (pointsInStrong.isEmpty()) {
                logger.warn("No points in strong clusters available for backfilling")
                None
              } else {
                try {
                  val rand = new XORShiftRandom(round)
                  val sampleSize = math.min(weakClusters.length, pointsInStrong.count().toInt)
                  val sample = pointsInStrong.takeSample(withReplacement = false, sampleSize, rand.nextLong())

                  val reps = sample.zipWithIndex.collect {
                    case ((_, pt), i) if i < weakClusters.length =>
                      CenterWithHistory(weakClusters(i).index, round, pointOps.toCenter(pt), initialized = false)
                  }.toIndexedSeq

                  Some(reps)
                } catch {
                  case e: Exception =>
                    logger.error(s"Error during backfilling: ${e.getMessage}")
                    Some(IndexedSeq.empty)
                }
              }
            }
          }

        replacementsOpt match {
          case Some(replacements) =>
            logger.info(s"replaced ${replacements.length} clusters")
            val keptWeak = weakClusters.drop(replacements.length)
            (strongClusters ++ replacements ++ keptWeak).toIndexedSeq
          case None =>
            centers.toIndexedSeq
        }
      }
    }
  }

  /** Update cluster centers (incrementally if addOnly=true). */
  private[this] def latestCenters(
    points: RDD[BregmanPoint],
    pointOps: BregmanPointOps,
    round: Int,
    previousCenters: IndexedSeq[CenterWithHistory],
    currentAssignments: RDD[ColumnTrackingKMeans.Assignment],
    previousAssignments: RDD[ColumnTrackingKMeans.Assignment]
  ): Array[CenterWithHistory] = {
    val centers = previousCenters.toArray
    if (config.addOnly) {
      val results = completeMovedCentroids(points, pointOps, currentAssignments, previousAssignments, previousCenters.length)
      results.foreach { case (index, location) =>
        centers(index) = CenterWithHistory(index, round, pointOps.toCenter(location.asImmutable), initialized = true)
      }
    } else {
      val deltas = deltasOfMovedCentroids(points, pointOps, currentAssignments, previousAssignments, previousCenters.length)
      deltas.foreach { case (index, delta) =>
        val prev = previousCenters(index)
        val loc = if (prev.initialized) delta.add(pointOps.toPoint(prev.center)) else delta
        centers(index) = CenterWithHistory(index, round, pointOps.toCenter(loc.asImmutable), initialized = true)
      }
    }
    centers
  }

  /** Computes centroids for clusters that changed membership. */
  private[this] def completeMovedCentroids[T <: WeightedVector](
    points: RDD[T],
    ops: BregmanPointOps,
    assignments: RDD[ColumnTrackingKMeans.Assignment],
    previousAssignments: RDD[ColumnTrackingKMeans.Assignment],
    numCenters: Int
  ): Array[(Int, MutableWeightedVector)] = {
    require(points.getStorageLevel.useMemory)
    require(assignments.getStorageLevel.useMemory)

    implicit val sc = points.sparkContext
    withBroadcast(ops) { bcPointOps =>
      points.zipPartitions(assignments, previousAssignments) { (x, y, z) =>
        val pointOps = bcPointOps.value
        val centroids = Array.tabulate(numCenters)(pointOps.make)
        val changed = new Array[Boolean](numCenters)
        val buf = ArrayBuffer[Int]()

        while (x.hasNext && y.hasNext && z.hasNext) {
          val p = x.next()
          val curr = y.next()
          val prev = z.next()
          val idx = curr.cluster
          if (idx >= 0) centroids(idx).add(p)
          if (curr.cluster != prev.cluster) {
            if (prev.cluster >= 0 && !changed(prev.cluster)) {
              changed(prev.cluster) = true; buf += prev.cluster
            }
            if (curr.cluster >= 0 && !changed(curr.cluster)) {
              changed(curr.cluster) = true; buf += curr.cluster
            }
          }
        }
        buf.distinct.map(i => (i, centroids(i))).iterator
      }.combineByKey(
        (x: MutableWeightedVector) => x,
        (_: MutableWeightedVector).add(_: MutableWeightedVector),
        (_: MutableWeightedVector).add(_: MutableWeightedVector),
        defaultPartitioner(points),
        mapSideCombine = false
      ).collect()
    }
  }

  /** Compute only deltas for moved centroids. */
  private[this] def deltasOfMovedCentroids[T <: WeightedVector](
    points: RDD[T],
    pointOps: BregmanPointOps,
    assignments: RDD[ColumnTrackingKMeans.Assignment],
    previousAssignments: RDD[ColumnTrackingKMeans.Assignment],
    numCenters: Int
  ): Array[(Int, MutableWeightedVector)] = {
    require(points.getStorageLevel.useMemory)
    require(assignments.getStorageLevel.useMemory)
    require(previousAssignments.getStorageLevel.useMemory)

    points.zipPartitions(assignments, previousAssignments) { (x, y, z) =>
      val centroids = IndexedSeq.tabulate(numCenters)(pointOps.make)
      while (x.hasNext && y.hasNext && z.hasNext) {
        val p = x.next()
        val curr = y.next()
        val prev = z.next()
        if (curr != prev) {
          if (prev.cluster >= 0) centroids(prev.cluster).sub(p)
          if (curr.cluster >= 0) centroids(curr.cluster).add(p)
        }
      }
      centroids.filter(_.nonEmpty).map(v => (v.index, v)).iterator
    }.reduceByKey(_.add(_)).collect()
  }

  /** Run full KMeans iterations and return clusterings with distortion. */
  def cluster(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    points: RDD[BregmanPoint],
    centerArrays: Seq[Centers]
  ): Seq[ClusteringWithDistortion] = {
    require(points.getStorageLevel.useMemory)
    implicit val sc = points.sparkContext
    val detector = new ConvergenceDetector(sc)

    @tailrec
    def lloyds(
      round: Int,
      assignments: RDD[ColumnTrackingKMeans.Assignment],
      centers: IndexedSeq[CenterWithHistory]
    ): (RDD[ColumnTrackingKMeans.Assignment], IndexedSeq[CenterWithHistory]) = {
      val newAssignments = withBroadcast(centers) { bcCenters =>
        sync("assignments round " + round,
          ColumnTrackingKMeans.updatedAssignments(points, pointOps, round, assignments, bcCenters, config.updateRate))
      }
      val newCenters = latestCenters(points, pointOps, round + 1, centers, newAssignments, assignments)
      val backFilled = backFilledCenters(points, pointOps, round + 1, newAssignments, newCenters)
      detector.update(pointOps, (round + 1) / 2, backFilled, centers, newAssignments, assignments)

      if (round != 0) assignments.unpersist()
      if ((round / 2 + 1) == maxIterations || detector.stable()) (newAssignments, backFilled)
      else lloyds(round + 2, newAssignments, backFilled)
    }

    val emptyAssign = ColumnTrackingKMeans.Assignment(ColumnTrackingKMeans.PosInf, ColumnTrackingKMeans.noCluster, -2)
    withCached("empty assignments", points.map(_ => emptyAssign)) { empty =>
      centerArrays.map { initialCenters =>
        val centers = initialCenters.zipWithIndex.map { case (c, i) =>
          CenterWithHistory(i, -1, c, initialized = false)
        }
        val (assignments, updatedCenters) = lloyds(0, empty, centers)
        assignments.unpersist(blocking = false)
        ClusteringWithDistortion(distortion(assignments), updatedCenters.map(_.center).toIndexedSeq)
      }
    }
  }
}
