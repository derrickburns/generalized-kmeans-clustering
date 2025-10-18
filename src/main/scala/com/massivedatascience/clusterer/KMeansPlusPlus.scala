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
 *
 * This code is a modified version of the original Spark 1.0.2 K-Means implementation.
 */

package com.massivedatascience.clusterer

import com.massivedatascience.linalg.WeightedVector
import com.massivedatascience.util.XORShiftRandom

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer

// Cross-version parallel collections support via compat package
import com.massivedatascience.clusterer.compat._

/** This implements the <a href="http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf">KMeans++ initialization
  * algorithm</a>
  *
  * @param ops
  *   distance function
  */
class KMeansPlusPlus(ops: BregmanPointOps) extends Serializable with Logging {

  /** Select centers in rounds. On each round, select 'perRound' centers, with probability of selection equal to the
    * product of the given weights and distance to the closest cluster center of the previous round.
    *
    * This version allows some centers to be pre-selected.
    *
    * @param seed
    *   a random number seed
    * @param candidateCenters
    *   the candidate centers
    * @param weights
    *   the weights on the candidate centers
    * @param totalRequested
    *   the total number of centers to select
    * @param perRound
    *   the number of centers to add per round
    * @param numPreselected
    *   the number of pre-selected centers
    * @return
    *   an array of at most k cluster centers
    */

  /** Select high-quality initial centers using the K-Means++ algorithm with improved numerical stability.
    *
    * @param seed
    *   random number generator seed
    * @param candidateCenters
    *   sequence of candidate centers
    * @param weights
    *   weights for each candidate center (must be non-negative)
    * @param totalRequested
    *   total number of centers to select
    * @param perRound
    *   number of centers to add in each round
    * @param numPreselected
    *   number of centers that are pre-selected (must be at the start of candidateCenters)
    * @return
    *   sequence of selected centers
    * @throws java.lang.IllegalArgumentException
    *   if inputs are invalid
    */
  def goodCenters(
    seed: Long,
    candidateCenters: IndexedSeq[BregmanCenter],
    weights: IndexedSeq[Double],
    totalRequested: Int,
    perRound: Int,
    numPreselected: Int
  ): IndexedSeq[BregmanCenter] = {

    // Input validation
    require(candidateCenters.nonEmpty, "Candidate centers cannot be empty")
    require(
      candidateCenters.length == weights.length,
      s"Number of candidate centers (${candidateCenters.length}) must match number of weights (${weights.length})"
    )
    require(weights.forall(_ >= 0.0), "Weights must be non-negative")
    require(
      totalRequested > 0 && totalRequested <= candidateCenters.length,
      s"Total requested centers ($totalRequested) must be positive and <= number of candidates (${candidateCenters.length})"
    )
    require(
      numPreselected >= 0 && numPreselected <= totalRequested,
      s"Number of preselected centers ($numPreselected) must be between 0 and total requested ($totalRequested)"
    )
    require(
      perRound > 0 && perRound <= totalRequested,
      s"Centers per round ($perRound) must be positive and <= total requested ($totalRequested)"
    )

    if (candidateCenters.length < totalRequested) {
      logger.warn(
        s"Requested $totalRequested centers but only ${candidateCenters.length} candidates available"
      )
    }

    logger.info(
      s"Starting KMeans++ with ${candidateCenters.length} candidates, " +
        s"requesting $totalRequested centers, $numPreselected preselected"
    )

    // Log weight statistics for debugging
    val totalWeight = weights.sum
    val minWeight   = if (weights.nonEmpty) weights.min else 0.0
    val maxWeight   = if (weights.nonEmpty) weights.max else 0.0
    logger.debug(
      f"Weight statistics: total=$totalWeight%.4f, min=$minWeight%.4f, max=$maxWeight%.4f"
    )

    // Convert candidate centers to points using SELECTION weights for both distance and selection
    val points = reWeightedPoints(candidateCenters, weights)
    val rand    = new XORShiftRandom(seed)
    val centers = new ArrayBuffer[BregmanCenter](totalRequested)

    @tailrec
    def moreCenters(distances: IndexedSeq[Double], iteration: Int = 0): Unit = {
      val needed = totalRequested - centers.length
      if (needed > 0) {
        logger.debug(
          s"Round $iteration: selecting up to $perRound centers from ${distances.length} candidates"
        )

        // Multiply distances by point weights (which are the selection weights from reWeightedPoints)
        // This allows zero selection weights to exclude certain points from selection
        val weightedDistances = points.zip(distances).map { case (p, d) => p.weight * d }

        // Check if we have any valid selection weights
        val totalWeight = weightedDistances.sum
        if (totalWeight <= 0.0) {
          logger.warn("No valid selection weights, falling back to uniform sampling")
          val uniformSample =
            (0 until math.min(perRound, needed)).map(_ => rand.nextInt(candidateCenters.length))
          centers ++= uniformSample.distinct.map(candidateCenters)
        } else {
          val cumulative = cumulativeWeights(weightedDistances)
          val selected = (0 until perRound).par.flatMap { _ =>
            pickWeighted(rand, cumulative).iterator
          }

          val uniqueSelected = selected.seq.toSeq.distinct
          logger.debug(
            s"Selected ${uniqueSelected.size} unique centers from ${selected.size} samples"
          )

          val additionalCenters = uniqueSelected.map(candidateCenters).toIndexedSeq
          val newDistances      = updateDistances(points, distances, additionalCenters)
          centers ++= additionalCenters.take(needed)

          if (additionalCenters.nonEmpty) {
            moreCenters(newDistances, iteration + 1)
          } else {
            logger.warn("No new centers selected, stopping early")
          }
        }
      }
    }

    // Handle pre-selected centers if any
    if (numPreselected > 0) {
      logger.info(s"Using $numPreselected pre-selected centers")
      centers ++= candidateCenters.take(numPreselected)
    } else if (weights.exists(_ > 0.0)) {
      // Only use weighted sampling if we have positive weights
      logger.debug("Selecting initial center using weighted sampling")
      val initialIndex = pickWeighted(rand, cumulativeWeights(weights)).head
      centers += candidateCenters(initialIndex)
      logger.debug(s"Selected initial center at index $initialIndex")
    } else {
      // Fall back to uniform sampling if all weights are zero
      logger.warn("All weights are zero, falling back to uniform sampling for initial center")
      val initialIndex = rand.nextInt(candidateCenters.length)
      centers += candidateCenters(initialIndex)
      logger.debug(s"Selected initial center at index $initialIndex using uniform sampling")
    }

    // Initialize distances for remaining points
    val maxDistances     = IndexedSeq.fill(points.length)(Double.MaxValue)
    val initialDistances = updateDistances(points, maxDistances, centers.toIndexedSeq)

    // Run the main algorithm to select remaining centers
    moreCenters(initialDistances)

    val finalCenters = centers.take(totalRequested).toIndexedSeq
    logger.info(
      s"Selected ${finalCenters.length} centers out of ${candidateCenters.length} candidates"
    )

    // Log some statistics about the selected centers
    if (finalCenters.nonEmpty) {
      val centerIndices = finalCenters.map(c => candidateCenters.indexOf(c))
      val centerWeights =
        centerIndices.map(i => if (i >= 0 && i < weights.length) weights(i) else 0.0)
      logger.debug(s"Selected center weights: ${centerWeights.mkString(", ")}")
    }

    finalCenters
  }

  private[this] def reWeightedPoints(
    candidateCenters: IndexedSeq[BregmanCenter],
    weights: IndexedSeq[Double]
  ): IndexedSeq[KMeansPlusPlus.this.ops.P] = {

    candidateCenters
      .zip(weights)
      .map { case (c, w) =>
        WeightedVector.fromInhomogeneousWeighted(c.inhomogeneous, w)
      }
      .map(ops.toPoint)
  }

  /** Update the distance of each point to its closest cluster center, given the cluster centers that were added.
    *
    * @param points
    *   set of candidate initial cluster centers
    * @param centers
    *   new cluster centers
    * @return
    *   points with their distance to closest to cluster center updated
    */

  private[this] def updateDistances(
    points: IndexedSeq[BregmanPoint],
    distances: IndexedSeq[Double],
    centers: IndexedSeq[BregmanCenter]
  ): IndexedSeq[Double] = {

    val newDistances = points.zip(distances).par.map { case (p, d) =>
      Math.min(ops.pointCost(centers, p), d)
    }
    newDistances.toIndexedSeq
  }

  def cumulativeWeights(weights: IndexedSeq[Double]): IndexedSeq[Double] =
    weights.scanLeft(0.0)(_ + _).tail

  /** Pick a point at random using the alias method for O(1) sampling.
    *
    * The alias method provides constant-time sampling from discrete distributions by pre-computing alias and
    * probability tables. This is more efficient than binary search for repeated sampling from the same distribution.
    *
    * @param rand
    *   random number generator
    * @param weights
    *   the original weights (not cumulative)
    * @return
    *   the index of the chosen point
    */
  private[this] def pickWeightedAlias(
    rand: XORShiftRandom,
    weights: IndexedSeq[Double]
  ): Seq[Int] = {
    require(weights.nonEmpty, "Weights cannot be empty")
    require(weights.exists(_ > 0.0), "At least one weight must be positive")

    val n           = weights.length
    val totalWeight = weights.sum

    if (totalWeight <= 0.0) {
      Seq(rand.nextInt(n))
    } else {
      // Build alias table using Vose's algorithm
      val (alias, prob) = buildAliasTable(weights)

      // Sample using alias method
      val uniformIndex = rand.nextInt(n)
      val uniformProb  = rand.nextDouble()

      val selectedIndex = if (uniformProb < prob(uniformIndex)) {
        uniformIndex
      } else {
        alias(uniformIndex)
      }

      Seq(selectedIndex)
    }
  }

  /** Build alias table for O(1) sampling using Vose's algorithm.
    *
    * @param weights
    *   the probability weights
    * @return
    *   tuple of (alias table, probability table)
    */
  private[this] def buildAliasTable(weights: IndexedSeq[Double]): (Array[Int], Array[Double]) = {
    val n           = weights.length
    val totalWeight = weights.sum

    // Normalize weights to sum to n (required for alias method)
    val normalizedWeights = weights.map(w => n * w / totalWeight)

    val alias = Array.fill(n)(0)
    val prob  = Array.fill(n)(0.0)

    // Separate into small and large probability buckets
    val small = scala.collection.mutable.Queue[Int]()
    val large = scala.collection.mutable.Queue[Int]()

    for (i <- normalizedWeights.indices) {
      prob(i) = normalizedWeights(i)
      if (normalizedWeights(i) < 1.0) {
        small.enqueue(i)
      } else {
        large.enqueue(i)
      }
    }

    // Build alias table
    while (small.nonEmpty && large.nonEmpty) {
      val smallIdx = small.dequeue()
      val largeIdx = large.dequeue()

      alias(smallIdx) = largeIdx
      prob(largeIdx) = prob(largeIdx) + prob(smallIdx) - 1.0

      if (prob(largeIdx) < 1.0) {
        small.enqueue(largeIdx)
      } else {
        large.enqueue(largeIdx)
      }
    }

    // Handle remaining items (should all have probability 1.0)
    while (large.nonEmpty) {
      val idx = large.dequeue()
      prob(idx) = 1.0
    }

    while (small.nonEmpty) {
      val idx = small.dequeue()
      prob(idx) = 1.0
    }

    (alias, prob)
  }

  /** Pick a point at random, weighing the choices by the given cumulative weight vector. This is the legacy method
    * maintained for backward compatibility.
    *
    * @param rand
    *   random number generator
    * @param cumulative
    *   the cumulative weights of the points (must be non-decreasing)
    * @return
    *   the index of the chosen point (always returns a valid index)
    * @throws java.lang.IllegalArgumentException
    *   if cumulative is empty or has non-positive sum
    */
  private[this] def pickWeighted(rand: XORShiftRandom, cumulative: IndexedSeq[Double]): Seq[Int] = {
    require(cumulative.nonEmpty, "Cumulative weights cannot be empty")
    require(cumulative.last > 0.0, "Sum of weights must be positive")

    // For small distributions, use binary search; for large ones, use alias method
    if (cumulative.length <= 32) {
      pickWeightedBinarySearch(rand, cumulative)
    } else {
      // Convert cumulative to weights for alias method
      val weights = cumulative.zipWithIndex.map { case (cum, i) =>
        if (i == 0) cum else cum - cumulative(i - 1)
      }
      pickWeightedAlias(rand, weights)
    }
  }

  /** Binary search implementation for small distributions.
    */
  private[this] def pickWeightedBinarySearch(
    rand: XORShiftRandom,
    cumulative: IndexedSeq[Double]
  ): Seq[Int] = {
    val totalWeight = cumulative.last
    val r           = rand.nextDouble() * totalWeight

    @scala.annotation.tailrec
    def binarySearch(left: Int, right: Int): Int = {
      if (left >= right) {
        left
      } else {
        val mid    = left + (right - left) / 2
        val midVal = cumulative(mid)

        val relTol = 1e-10 * Math.max(Math.abs(r), Math.abs(midVal))
        if (Math.abs(midVal - r) < relTol) {
          (mid + 1).min(cumulative.length - 1)
        } else if (midVal < r) {
          binarySearch(mid + 1, right)
        } else {
          binarySearch(left, mid)
        }
      }
    }

    if (r <= 0.0) {
      // Find first index with non-zero cumulative weight
      val firstNonZero = cumulative.indexWhere(_ > 0.0)
      Seq(if (firstNonZero >= 0) firstNonZero else 0)
    } else if (r >= totalWeight) {
      logger.warn(s"Random value $r exceeds total weight $totalWeight, using last index")
      Seq(cumulative.length - 1)
    } else {
      val idx     = binarySearch(0, cumulative.length - 1)
      val safeIdx = Math.max(0, Math.min(idx, cumulative.length - 1))

      if (safeIdx < 0 || safeIdx >= cumulative.length) {
        logger.error(
          s"Invalid index $safeIdx generated for cumulative weights length ${cumulative.length}"
        )
        Seq(rand.nextInt(cumulative.length))
      } else {
        Seq(safeIdx)
      }
    }
  }
}
