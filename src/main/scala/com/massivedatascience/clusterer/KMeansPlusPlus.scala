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

import com.massivedatascience.clusterer.util.XORShiftRandom
import org.apache.spark.{Logging, SparkContext}

import scala.collection.mutable.ArrayBuffer

/**
 *
 * The KMeans++ initialization algorithm
 *
 * @param ops distance function
 */
class KMeansPlusPlus(ops: BregmanPointOps) extends Serializable with Logging {

  /**
   * K-means++ on the weighted point set `points`. This first does the K-means++
   * initialization procedure and then rounds of Lloyd's algorithm.
   */

  def cluster(
    sc: SparkContext,
    seed: Int,
    points: Array[BregmanCenter],
    weights: Array[Double],
    k: Int,
    maxIterations: Int): Array[BregmanCenter] = {
    val centers = getCenters(sc, seed, points, weights, k, 1)
    val pts = sc.parallelize(points.map(ops.toPoint))
    new MultiKMeans(ops, maxIterations).cluster(pts, Array(centers))._2.centers
  }

  /**
   * Select centers in rounds.  On each round, select 'perRound' centers, with probability of
   * selection equal to the product of the given weights and distance to the closest cluster center
   * of the previous round.
   *
   * @param sc the Spark context
   * @param seed a random number seed
   * @param candidateCenters  the candidate centers
   * @param weights  the weights on the candidate centers
   * @param k  the total number of centers to select
   * @param perRound the number of centers to add per round
   * @return   an array of at most k cluster centers
   */
  def getCenters(
    sc: SparkContext,
    seed: Int,
    candidateCenters: Array[BregmanCenter],
    weights: Array[Double],
    k: Int,
    perRound: Int): Array[BregmanCenter] = {

    require(candidateCenters.length > 0)
    require(k > 0)
    require(perRound > 0)

    if (candidateCenters.length < k)
      logWarning(s"# of clusters requested $k exceeds number of points ${candidateCenters.length}")

    val points = candidateCenters.map(ops.toPoint)
    val centers = new ArrayBuffer[BregmanCenter](k)
    val rand = new XORShiftRandom(seed)
    val newCenter = pickWeighted(rand, weights).map(candidateCenters(_))
    centers += newCenter.get
    logInfo(s"starting kMeansPlusPlus initialization on ${candidateCenters.length} points")

    var distances = Array.fill(candidateCenters.length)(Double.MaxValue)
    distances = updateDistances(points, distances, Array(newCenter.get))
    var more = true
    while (centers.length < k && more) {
      val selected = (0 until perRound).flatMap {_ =>
        pickWeighted(rand, points.zip(distances).map{ case (p,d) => p.weight * d})
      }
      val newCenters = selected.toArray.map(candidateCenters(_))
      distances = updateDistances(points, distances, newCenters)
      logInfo(s"chose ${newCenters.length} new centers")
      for (c <- newCenters) {
        logInfo(s"  chose center $c")
        centers += c
      }
      more = selected.nonEmpty
    }
    val result = centers.take(k)
    logInfo(s"completed kMeansPlusPlus with ${result.length} centers of $k requested")
    result.toArray
  }

  /**
   * Update the distance of each point to its closest cluster center, given the cluster
   * centers that were added.
   *
   * @param points set of candidate initial cluster centers
   * @param centers new cluster centers
   * @return  points with their distance to closest to cluster center updated
   */

  def updateDistances(
    points: Array[BregmanPoint],
    distances: Array[Double],
    centers: Array[BregmanCenter]): Array[Double] = {
    
    points.zip(distances).map { case (p, d) =>
      val dist = ops.pointCost(centers, p)
      if(dist < d) dist else d
    }
  }

  /**
   * Pick a point at random, weighing the choices by the given weight vector.
   * Return -1 if all weights are 0.0
   *
   * @param rand  random number generator
   * @param weights  the weights of the points
   * @return the index of the point chosen
   */
  def pickWeighted(rand: XORShiftRandom, weights: IndexedSeq[Double]): Option[Int] = {
    require(weights.nonEmpty)

    var cumulative = 0.0
    val rangeAndIndices = weights map { z =>
      cumulative = cumulative + z
      cumulative
    }
    val r = rand.nextDouble() * rangeAndIndices.last
    val index = rangeAndIndices.indexWhere(x => x > r)
    if( index == -1 ) None else Some(index)
  }
}
