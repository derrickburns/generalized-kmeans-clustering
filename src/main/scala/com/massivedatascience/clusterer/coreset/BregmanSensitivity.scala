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

package com.massivedatascience.clusterer.coreset

import com.massivedatascience.clusterer.{BregmanCenter, BregmanPoint, BregmanPointOps}
import com.massivedatascience.divergence.BregmanDivergence
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory

import scala.util.Random

/** Trait for computing sensitivity scores for Bregman core-set construction.
  *
  * Sensitivity measures how much a point affects the optimal clustering cost. Points with higher sensitivity are more
  * important to include in the core-set.
  */
trait BregmanSensitivity extends Serializable {

  /** Compute the sensitivity score for a single point.
    *
    * @param point
    *   The point to compute sensitivity for
    * @param allPoints
    *   All points in the dataset (for context)
    * @param k
    *   Number of clusters
    * @param pointOps
    *   Bregman point operations
    * @return
    *   Sensitivity score (higher = more important)
    */
  def computeSensitivity(
    point: BregmanPoint,
    allPoints: RDD[BregmanPoint],
    k: Int,
    pointOps: BregmanPointOps
  ): Double

  /** Batch compute sensitivity scores for multiple points. Default implementation calls computeSensitivity for each
    * point.
    */
  def computeBatchSensitivity(
    points: RDD[BregmanPoint],
    k: Int,
    pointOps: BregmanPointOps
  ): RDD[(BregmanPoint, Double)] = {

    points.map(point => (point, computeSensitivity(point, points, k, pointOps)))
  }
}

/** Uniform sensitivity - all points have equal importance. This reduces to simple uniform sampling.
  */
class UniformSensitivity extends BregmanSensitivity {

  def computeSensitivity(
    point: BregmanPoint,
    allPoints: RDD[BregmanPoint],
    k: Int,
    pointOps: BregmanPointOps
  ): Double = {
    1.0 // All points equally important
  }

  override def computeBatchSensitivity(
    points: RDD[BregmanPoint],
    k: Int,
    pointOps: BregmanPointOps
  ): RDD[(BregmanPoint, Double)] = {
    points.map(point => (point, 1.0))
  }
}

/** Distance-based sensitivity using approximate nearest clusters.
  *
  * Points that are far from potential cluster centers have higher sensitivity because they represent outliers or sparse
  * regions that could significantly affect clustering quality.
  */
class DistanceBasedSensitivity(numSampleCenters: Int = 100, seed: Long = 42L) extends BregmanSensitivity {

  @transient private lazy val logger = LoggerFactory.getLogger(getClass.getName)

  def computeSensitivity(
    point: BregmanPoint,
    allPoints: RDD[BregmanPoint],
    k: Int,
    pointOps: BregmanPointOps
  ): Double = {

    // Sample potential cluster centers
    val sampleCenters = samplePotentialCenters(allPoints, numSampleCenters, pointOps)

    if (sampleCenters.isEmpty) {
      logger.warn("No sample centers available, using uniform sensitivity")
      return 1.0
    }

    // Find distance to closest sample center
    val minDistance = sampleCenters.map(center => pointOps.distance(point, center)).min

    // Higher distance = higher sensitivity
    math.max(minDistance, 1e-10) // Avoid zero sensitivity
  }

  override def computeBatchSensitivity(
    points: RDD[BregmanPoint],
    k: Int,
    pointOps: BregmanPointOps
  ): RDD[(BregmanPoint, Double)] = {

    // Sample potential centers once for all points
    val sampleCenters    = samplePotentialCenters(points, numSampleCenters, pointOps)
    val broadcastCenters = points.sparkContext.broadcast(sampleCenters)

    try {
      points.mapPartitions { partitionPoints =>
        val centers = broadcastCenters.value
        partitionPoints.map { point =>
          val sensitivity = if (centers.nonEmpty) {
            val minDistance = centers.map(center => pointOps.distance(point, center)).min
            math.max(minDistance, 1e-10)
          } else {
            1.0
          }
          (point, sensitivity)
        }
      }
    } finally {
      broadcastCenters.unpersist()
    }
  }

  /** Sample potential cluster centers to estimate sensitivity.
    */
  private def samplePotentialCenters(
    points: RDD[BregmanPoint],
    numSamples: Int,
    pointOps: BregmanPointOps
  ): IndexedSeq[BregmanCenter] = {

    val sampledPoints = points.takeSample(withReplacement = false, numSamples, seed)
    sampledPoints.map(pointOps.toCenter).toIndexedSeq
  }
}

/** Density-based sensitivity using local neighborhood analysis.
  *
  * Points in sparse regions have higher sensitivity because they represent important boundary cases or outliers.
  */
class DensityBasedSensitivity(numNeighbors: Int = 50, seed: Long = 42L) extends BregmanSensitivity {

  @transient private lazy val logger = LoggerFactory.getLogger(getClass.getName)

  def computeSensitivity(
    point: BregmanPoint,
    allPoints: RDD[BregmanPoint],
    k: Int,
    pointOps: BregmanPointOps
  ): Double = {

    // Sample neighbors to estimate local density
    val neighbors = allPoints.takeSample(withReplacement = false, numNeighbors, seed)

    if (neighbors.isEmpty) {
      logger.warn("No neighbors available, using uniform sensitivity")
      return 1.0
    }

    // Compute average distance to neighbors (inverse density)
    val avgDistance = neighbors
      .map(neighbor => pointOps.distance(point, pointOps.toCenter(neighbor)))
      .sum / neighbors.length

    // Higher average distance = lower density = higher sensitivity
    math.max(avgDistance, 1e-10)
  }

  override def computeBatchSensitivity(
    points: RDD[BregmanPoint],
    k: Int,
    pointOps: BregmanPointOps
  ): RDD[(BregmanPoint, Double)] = {

    // Sample neighbors once for efficiency
    val allNeighbors       = points.takeSample(withReplacement = false, numNeighbors * 2, seed)
    val broadcastNeighbors = points.sparkContext.broadcast(allNeighbors)

    try {
      points.mapPartitions { partitionPoints =>
        val neighbors = broadcastNeighbors.value
        val random    = new Random(seed)

        partitionPoints.map { point =>
          val sensitivity = if (neighbors.nonEmpty) {
            // Sample subset of neighbors for this point
            val pointNeighbors = random.shuffle(neighbors.toList).take(numNeighbors)
            val avgDistance = pointNeighbors
              .map(neighbor => pointOps.distance(point, pointOps.toCenter(neighbor)))
              .sum / pointNeighbors.length
            math.max(avgDistance, 1e-10)
          } else {
            1.0
          }
          (point, sensitivity)
        }
      }
    } finally {
      broadcastNeighbors.unpersist()
    }
  }
}

/** Hybrid sensitivity combining distance and density measures.
  */
class HybridSensitivity(
  distanceWeight: Double = 0.6,
  densityWeight: Double = 0.4,
  numSampleCenters: Int = 100,
  numNeighbors: Int = 50,
  seed: Long = 42L
) extends BregmanSensitivity {

  require(
    math.abs(distanceWeight + densityWeight - 1.0) < 1e-10,
    s"Weights must sum to 1.0, got: $distanceWeight + $densityWeight = ${distanceWeight + densityWeight}"
  )

  private val distanceSensitivity = new DistanceBasedSensitivity(numSampleCenters, seed)
  private val densitySensitivity  = new DensityBasedSensitivity(numNeighbors, seed)

  def computeSensitivity(
    point: BregmanPoint,
    allPoints: RDD[BregmanPoint],
    k: Int,
    pointOps: BregmanPointOps
  ): Double = {

    val distSens = distanceSensitivity.computeSensitivity(point, allPoints, k, pointOps)
    val densSens = densitySensitivity.computeSensitivity(point, allPoints, k, pointOps)

    distanceWeight * distSens + densityWeight * densSens
  }

  override def computeBatchSensitivity(
    points: RDD[BregmanPoint],
    k: Int,
    pointOps: BregmanPointOps
  ): RDD[(BregmanPoint, Double)] = {

    val distSensitivities = distanceSensitivity.computeBatchSensitivity(points, k, pointOps)
    val densSensitivities = densitySensitivity.computeBatchSensitivity(points, k, pointOps)

    // Join the two sensitivity measures
    distSensitivities.join(densSensitivities).mapValues { case (distSens, densSens) =>
      distanceWeight * distSens + densityWeight * densSens
    }
  }
}

object BregmanSensitivity {

  /** Get the default sensitivity computation for a given divergence.
    */
  def defaultFor(divergence: BregmanDivergence): BregmanSensitivity = {
    // For now, use hybrid sensitivity for all divergences
    // Future: could specialize based on divergence properties
    new HybridSensitivity()
  }

  /** Create a uniform sensitivity (equivalent to random sampling).
    */
  def uniform(): BregmanSensitivity = new UniformSensitivity()

  /** Create a distance-based sensitivity.
    */
  def distanceBased(numSampleCenters: Int = 100, seed: Long = 42L): BregmanSensitivity = {
    new DistanceBasedSensitivity(numSampleCenters, seed)
  }

  /** Create a density-based sensitivity.
    */
  def densityBased(numNeighbors: Int = 50, seed: Long = 42L): BregmanSensitivity = {
    new DensityBasedSensitivity(numNeighbors, seed)
  }

  /** Create a hybrid sensitivity combining distance and density.
    */
  def hybrid(
    distanceWeight: Double = 0.6,
    densityWeight: Double = 0.4,
    numSampleCenters: Int = 100,
    numNeighbors: Int = 50,
    seed: Long = 42L
  ): BregmanSensitivity = {
    new HybridSensitivity(distanceWeight, densityWeight, numSampleCenters, numNeighbors, seed)
  }
}
