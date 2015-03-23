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
import org.apache.spark.Logging

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer

/**
 * This implements the
 * <a href="http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf">KMeans++ initialization algorithm</a>
 *
 * @param ops distance function
 */
class KMeansPlusPlus(ops: BregmanPointOps) extends Serializable with Logging {

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

  def goodCenters(
    seed: Long,
    candidateCenters: IndexedSeq[BregmanCenter],
    weights: IndexedSeq[Double],
    totalRequested: Int,
    perRound: Int,
    numPreselected: Int): IndexedSeq[BregmanCenter] = {

    require(candidateCenters.length > 0)
    require(totalRequested > 0 && totalRequested <= candidateCenters.length)
    require(numPreselected >= 0 && numPreselected <= totalRequested)
    require(perRound <= totalRequested)

    if (candidateCenters.length < totalRequested)
      logWarning(s"# requested $totalRequested exceeds number of points ${candidateCenters.length}")

    val points = reWeightedPoints(candidateCenters, weights)
    val rand = new XORShiftRandom(seed)
    val centers = new ArrayBuffer[BregmanCenter](totalRequested)

    @tailrec
    def moreCenters(distances: IndexedSeq[Double]): Unit = {
      val needed = totalRequested - centers.length
      if (needed > 0) {
        val cumulative = cumulativeWeights(points.zip(distances).map { case (p, d) => p.weight * d })
        val selected = (0 until perRound).par.flatMap { _ =>
          pickWeighted(rand, cumulative).iterator
        }
        val additionalCenters = selected.map(candidateCenters(_)).toIndexedSeq
        val newDistances = updateDistances(points, distances, additionalCenters)
        centers ++= additionalCenters.take(needed)
        if (additionalCenters.nonEmpty) moreCenters(newDistances)
      }
    }

    centers ++= (if (numPreselected == 0) {
      pickWeighted(rand, cumulativeWeights(weights)).map(candidateCenters(_))
    } else {
      candidateCenters.take(numPreselected)
    })

    val maxDistances = IndexedSeq.fill(candidateCenters.length)(Double.MaxValue)
    val initialDistances = updateDistances(points, maxDistances, centers)
    moreCenters(initialDistances)
    centers.take(totalRequested)
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
   * @param rand  random number generator
   * @param cumulative  the cumulative weights of the points
   * @return the index of the point chosen
   */
  private[this] def pickWeighted(rand: XORShiftRandom, cumulative: IndexedSeq[Double]): Seq[Int] = {
    val r = rand.nextDouble() * cumulative.last
    val index = cumulative.indexWhere(x => x > r)
    if (index == -1) Seq[Int]() else Seq(index)
  }
}
