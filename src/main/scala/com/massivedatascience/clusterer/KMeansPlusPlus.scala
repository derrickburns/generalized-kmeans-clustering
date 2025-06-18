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

import org.slf4j.LoggerFactory


/**
 * This implements the
 * <a href="http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf">KMeans++ initialization algorithm</a>
 *
 * @param ops distance function
 */
class KMeansPlusPlus(ops: BregmanPointOps) extends Serializable {

  /**
   * Select centers in rounds.  On each round, select 'perRound' centers, with probability of
   * selection equal to the product of the given weights and distance to the closest cluster center
   * of the previous round.
   *
   * This version allows some centers to be pre-selected.
   *
   * @param seed a random number seed
   * @param candidateCenters  the candidate centers
   * @param weights  the weights on the candidate centers
   * @param totalRequested  the total number of centers to select
   * @param perRound the number of centers to add per round
   * @param numPreselected the number of pre-selected centers
   * @return   an array of at most k cluster centers
   */

  val logger = LoggerFactory.getLogger(getClass.getName)

  /**
   * Select high-quality initial centers using the K-Means++ algorithm with improved numerical stability.
   *
   * @param seed random number generator seed
   * @param candidateCenters sequence of candidate centers
   * @param weights weights for each candidate center (must be non-negative)
   * @param totalRequested total number of centers to select
   * @param perRound number of centers to add in each round
   * @param numPreselected number of centers that are pre-selected (must be at the start of candidateCenters)
   * @return sequence of selected centers
   * @throws IllegalArgumentException if inputs are invalid
   */
  def goodCenters(
    seed: Long,
    candidateCenters: IndexedSeq[BregmanCenter],
    weights: IndexedSeq[Double],
    totalRequested: Int,
    perRound: Int,
    numPreselected: Int): IndexedSeq[BregmanCenter] = {

    // Input validation
    require(candidateCenters.nonEmpty, "Candidate centers cannot be empty")
    require(candidateCenters.length == weights.length, 
      s"Number of candidate centers (${candidateCenters.length}) must match number of weights (${weights.length})")
    require(weights.forall(_ >= 0.0), "Weights must be non-negative")
    require(totalRequested > 0 && totalRequested <= candidateCenters.length,
      s"Total requested centers ($totalRequested) must be positive and <= number of candidates (${candidateCenters.length})")
    require(numPreselected >= 0 && numPreselected <= totalRequested,
      s"Number of preselected centers ($numPreselected) must be between 0 and total requested ($totalRequested)")
    require(perRound > 0 && perRound <= totalRequested,
      s"Centers per round ($perRound) must be positive and <= total requested ($totalRequested)")

    if (candidateCenters.length < totalRequested) {
      logger.warn(s"Requested $totalRequested centers but only ${candidateCenters.length} candidates available")
    }
    
    logger.info(s"Starting KMeans++ with ${candidateCenters.length} candidates, " +
      s"requesting $totalRequested centers, $numPreselected preselected")

    // Log weight statistics for debugging
    val totalWeight = weights.sum
    val minWeight = if (weights.nonEmpty) weights.min else 0.0
    val maxWeight = if (weights.nonEmpty) weights.max else 0.0
    logger.debug(f"Weight statistics: total=$totalWeight%.4f, min=$minWeight%.4f, max=$maxWeight%.4f")
    
    // Pre-compute log-weights for numerical stability
    val logWeights = weights.map { w =>
      if (w > 0.0) math.log(w) else Double.NegativeInfinity
    }
    
    val points = reWeightedPoints(candidateCenters, weights)
    val rand = new XORShiftRandom(seed)
    val centers = new ArrayBuffer[BregmanCenter](totalRequested)

    @tailrec
    def moreCenters(distances: IndexedSeq[Double], iteration: Int = 0): Unit = {
      val needed = totalRequested - centers.length
      if (needed > 0) {
        logger.debug(s"Round $iteration: selecting up to $perRound centers from ${distances.length} candidates")
        
        // Use log-sum-exp trick for numerical stability when computing probabilities
        val logDistances = distances.map(d => if (d > 0.0) math.log(d) else Double.NegativeInfinity)
        val maxLogDist = if (logDistances.nonEmpty) logDistances.max else 0.0
        val logProbs = logDistances.map(_ - maxLogDist) // Subtract max for numerical stability
        
        // Convert back to linear scale with log-sum-exp trick
        val probs = logProbs.map(lp => {
          val expTerm = math.exp(lp)
          if (expTerm.isInfinite || expTerm.isNaN) 0.0 else expTerm
        })
        
        val totalProb = probs.sum
        if (totalProb <= 0.0) {
          logger.warn("No valid probabilities, falling back to uniform sampling")
          val uniformSample = (0 until math.min(perRound, needed)).map(_ => 
            rand.nextInt(candidateCenters.length))
          centers ++= uniformSample.distinct.map(candidateCenters)
        } else {
          val cumulative = cumulativeWeights(probs)
          val selected = (0 until perRound).par.flatMap { _ =>
            pickWeighted(rand, cumulative).iterator
          }
          
          val uniqueSelected = selected.distinct
          logger.debug(s"Selected ${uniqueSelected.size} unique centers from ${selected.size} samples")
          
          val additionalCenters = uniqueSelected.map(candidateCenters).toIndexedSeq
          val newDistances = updateDistances(points, distances, additionalCenters)
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
    val maxDistances = IndexedSeq.fill(points.length)(Double.MaxValue)
    val initialDistances = updateDistances(points, maxDistances, centers)
    
    // Run the main algorithm to select remaining centers
    moreCenters(initialDistances)
    
    val finalCenters = centers.take(totalRequested)
    logger.info(s"Selected ${finalCenters.length} centers out of ${candidateCenters.length} candidates")
    
    // Log some statistics about the selected centers
    if (finalCenters.nonEmpty) {
      val centerIndices = finalCenters.map(c => candidateCenters.indexOf(c))
      val centerWeights = centerIndices.map(i => if (i >= 0) weights(i) else 0.0)
      logger.debug(s"Selected center weights: ${centerWeights.mkString(", ")}")
    }
    
    finalCenters
  }

  private[this] def reWeightedPoints(
    candidateCenters: IndexedSeq[BregmanCenter],
    weights: IndexedSeq[Double]): IndexedSeq[KMeansPlusPlus.this.ops.P] = {

    candidateCenters.zip(weights).map {
      case (c, w) =>
        WeightedVector.fromInhomogeneousWeighted(c.inhomogeneous, w)
    }.map(ops.toPoint)
  }

  /**
   * Update the distance of each point to its closest cluster center, given the cluster
   * centers that were added.
   *
   * @param points set of candidate initial cluster centers
   * @param centers new cluster centers
   * @return  points with their distance to closest to cluster center updated
   */

  private[this] def updateDistances(
    points: IndexedSeq[BregmanPoint],
    distances: IndexedSeq[Double],
    centers: IndexedSeq[BregmanCenter]): IndexedSeq[Double] = {

    val newDistances = points.zip(distances).par.map {
      case (p, d) =>
        Math.min(ops.pointCost(centers, p), d)
    }
    newDistances.toIndexedSeq
  }

  def cumulativeWeights(weights: IndexedSeq[Double]): IndexedSeq[Double] =
    weights.scanLeft(0.0)(_ + _).tail

  /**
   * Pick a point at random, weighing the choices by the given cumulative weight vector.
   *
   * This implementation uses binary search for O(log n) performance and handles
   * floating-point precision issues by:
   * 1. Using a relative tolerance for floating-point comparisons
   * 2. Ensuring the cumulative sum is properly normalized
   * 3. Using a more robust binary search that handles edge cases
   *
   * @param rand  random number generator
   * @param cumulative  the cumulative weights of the points (must be non-decreasing)
   * @return the index of the chosen point (always returns a valid index)
   * @throws IllegalArgumentException if cumulative is empty or has non-positive sum
   */
  private[this] def pickWeighted(rand: XORShiftRandom, cumulative: IndexedSeq[Double]): Seq[Int] = {
    require(cumulative.nonEmpty, "Cumulative weights cannot be empty")
    require(cumulative.last > 0.0, "Sum of weights must be positive")
    
    // Generate a random value in [0, totalWeight)
    val totalWeight = cumulative.last
    val r = rand.nextDouble() * totalWeight
    
    // Use binary search to find the insertion point
    // This is more numerically stable than indexWhere with floating-point comparison
    @scala.annotation.tailrec
    def binarySearch(left: Int, right: Int): Int = {
      if (left >= right) {
        left
      } else {
        val mid = left + (right - left) / 2
        // Use relative tolerance for floating-point comparison
        val midVal = cumulative(mid)
        
        // Check if we've found the exact match (within floating-point tolerance)
        val relTol = 1e-10 * Math.max(Math.abs(r), Math.abs(midVal))
        if (Math.abs(midVal - r) < relTol) {
          // If exact match, return next index to maintain uniform distribution
          (mid + 1).min(cumulative.length - 1)
        } else if (midVal < r) {
          binarySearch(mid + 1, right)
        } else {
          binarySearch(left, mid)
        }
      }
    }
    
    // Handle edge cases and perform the search
    if (r <= 0.0) {
      // Handle case where r is very close to 0
      Seq(0)
    } else if (r >= totalWeight) {
      // This should theoretically never happen due to how r is generated,
      // but we handle it defensively
      logger.warn(s"Random value $r exceeds total weight $totalWeight, using last index")
      Seq(cumulative.length - 1)
    } else {
      val idx = binarySearch(0, cumulative.length - 1)
      // Ensure we don't return -1 or an out-of-bounds index
      val safeIdx = Math.max(0, Math.min(idx, cumulative.length - 1))
      
      // Verify the selected index is valid
      if (safeIdx < 0 || safeIdx >= cumulative.length) {
        logger.error(s"Invalid index $safeIdx generated for cumulative weights length ${cumulative.length}")
        // Fall back to uniform sampling as a last resort
        Seq(rand.nextInt(cumulative.length))
      } else {
        Seq(safeIdx)
      }
    }
  }
}
