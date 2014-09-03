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

package com.rincaro.clusterer.base

import com.rincaro.clusterer.util.XORShiftRandom
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, Logging}
import org.joda.time.DateTime

import scala.reflect.ClassTag
import scala.util.Random
import scala.Option


/**
 * A re-write of the KMeansPlusPlus initialization method of the Spark Mllib clusterer.
 *
 * @param pointOps the distance function
 * @tparam T  user data type
 * @tparam P  point type
 * @tparam C  center type
 */
class KMeansPlusPlus[T, P <: FP[T] : ClassTag, C <: FP[T] : ClassTag](pointOps: PointOps[P,C,T]) extends Serializable with Logging {

  val kmeans = new GeneralizedKMeans[T, P, C](pointOps)

  /**
   * K-means++ on the weighted point set `points`. This first does the K-means++
   * initialization procedure and then rounds of Lloyd's algorithm.
   *
   * Unlike the Spark implementation, this one uses the parallel K-means clustering on the initial maxIterations * 2 * k
   * centers.
   */

  def cluster(
               sc: SparkContext,
               seed: Int,
               points: Array[C],
               weights: Array[Double],
               k: Int,
               maxIterations: Int,
               numPartitions: Int): Array[C] = {
    val centers: Array[C] = getCenters(sc, seed, points, weights, k, numPartitions)
    val (newCenters, _) = kmeans.cluster(sc.parallelize(points.map(pointOps.centerToPoint)), maxIterations, Array(centers))
    newCenters(0)
  }


  /**
   * Select at most k samples to be cluster centers by randomly sampling the points, using a probability distribution
   * computed as the product of the given weights and distance to the closest cluster center.
   *
   * @param sc the Spark context
   * @param seed a random number seed
   * @param points  the candidate points
   * @param weights  the weights on the candidate points
   * @param k  the number of desired centers
   * @param numPartitions the number of data partitions to use
   * @return   an array of at most k cluster centers
   */
  def getCenters(sc: SparkContext, seed: Int, points: Array[C], weights: Array[Double], k: Int, numPartitions: Int): Array[C] = {
    assert(points.length > 0)
    assert(k > 0)
    assert(numPartitions > 0)

    if (points.length < k) log.warn("number of clusters requested {} exceeds number of points {}", k, points.length)
    val centers = new Array[C](k)
    val rand = new XORShiftRandom(seed)
    val index = pickWeighted(rand, weights)
    if (index == -1) throw new IllegalArgumentException("all weights are zero")
    centers(0) = points(index)

    val allPoints = points.map(pointOps.centerToPoint).zipWithIndex.zip(weights).toArray
    val localRandom = new XORShiftRandom(rand.nextInt())
    val indexedPointsWithWeights = sc.parallelize(allPoints, numPartitions).cache()

    var numCenters = 1
    var sequential = true

    log.info("starting kMeansPlusPlus initialization on {} points", points.length)

    for (i <- 1 until k) {
      log.info("selecting point {} of {}", i, k)
      val start = new DateTime()
      val sample = sampleAll(sc, centers, allPoints, sequential, localRandom, indexedPointsWithWeights, numCenters)
      val end = new DateTime()
      val duration = end.getMillis - start.getMillis
      log.info("step took {} milliseconds using {} selection algorithm", duration, if (sequential) "sequential" else "parallel")

      if (sample.length > 0) {
        val index = sample(0)._1._2
        centers(numCenters) = points(index)
        numCenters = numCenters + 1
        sequential = sequential && duration < 2000
        log.info("centers({}) = point({})", i, index)
      } else {
        log.warn("could not find a new cluster center on iteration {}", k)
      }
    }

    indexedPointsWithWeights.unpersist()
    log.info("completed kMeansPlusPlus initialization with {} centers of {} requested", numCenters, k)
    if (numCenters < k) centers.slice(0, numCenters) else centers
  }

  /**
   * Pick a point at random, weighing the choices by the given weight vector.  Return -1 if all weights are 0.0
   *
   * @param rand  random number generator
   * @param weights  the weights of the points
   * @return the index of the point chosen
   */
  private def pickWeighted(rand: Random, weights: Array[Double]): Int = {
    val r = rand.nextDouble() * weights.sum
    var i = 0
    var curWeight = 0.0
    while (i < weights.length && curWeight < r) {
      assert(weights(i) >= 0.0)
      curWeight += weights(i)
      i += 1
    }
    i - 1
  }

  /**
   * Pick the next center with a probability proportional to cost under current centers multiplied by the weights
   *
   * @param sc  the Spark context
   * @param centers  the current centers
   * @param allPoints   the points
   * @param sequential  whether to use the sequential algorithm or the parallel algorithm
   * @param rand  random number generator
   * @param indexedPointsWithWeights   points with their original indices and weights
   * @param numCenters  number of centers in the current array of cluster centers
   * @return
   */
  private def sampleAll(sc: SparkContext,
                        centers: Array[C],
                        allPoints: Array[((P, Int), Double)],
                        sequential: Boolean,
                        rand: XORShiftRandom,
                        indexedPointsWithWeights: RDD[((P, Int), Double)], numCenters: Int): Array[((P, Int), Double)] = {
    val seed = rand.nextInt()
    val pts = if (sequential) allPoints else samplePartitions(sc, centers, seed, indexedPointsWithWeights, numCenters)
    val (sum, finalIndexedPointsWithWeightRanges) = getRanges(pts.iterator, centers, numCenters, sequential)
    findNextCenter(rand, sum, finalIndexedPointsWithWeightRanges).toArray
  }

  /**
   * Collect at most one point per partition according to the probability distribution of the weights times the distance
   * from the closest cluster center.
   *
   * @param sc  the Spark context
   * @param centers  an array of cluster centers
   * @param seed  the seed for a random number generator
   * @param indexedPointsWithWeights   points with their original indices and weights
   * @param numCenters  the number of cluster centers in the centers array

   * @return  an array of points and the indices and their composite weights
   */
  def samplePartitions(
                        sc: SparkContext,
                        centers: Array[C],
                        seed: Int,
                        indexedPointsWithWeights: RDD[((P, Int), Double)],
                        numCenters: Int): Array[((P, Int), Double)] = {
    val bcActiveCenters = sc.broadcast(centers)

    log.info("using parallel algorithm to select point {}", numCenters)
    val result = indexedPointsWithWeights.mapPartitionsWithIndex {
      (partitionIndex, points) => {
        val bcCenters = bcActiveCenters.value
        val rand = new XORShiftRandom(seed ^ (numCenters << 16) ^ (partitionIndex + 1))
        val (sum, x) = getRanges(points, bcCenters, numCenters, scale=true)
        findNextCenter(rand, sum, x)
      }
    }.collect()
    bcActiveCenters.unpersist()
    result
  }

  /**
   *
   * Select a (point,index) randomly, with probability weighted by the given weights
   *
   * @param rand random number generator
   * @param sum  sum of the weights
   * @param weightedIndexedPoints points with their indices and weight ranges

   * @return
   */
  def findNextCenter(rand: XORShiftRandom, sum: Double, weightedIndexedPoints: Iterator[(Double, Double, (P, Int), Double)]): Iterator[((P, Int), Double)] = {
    val r = rand.nextDouble() * sum
    val results = for (w <- weightedIndexedPoints if w._1 <= r && r < w._2) yield (w._3, sum)
    val result = results.toArray
    if (result.length == 0) {
      log.info("no point selected, sum = {}", sum)
    }
    result.toIterator
  }

  def getRanges(pointAndWeights: Iterator[((P, Int), Double)], centers: Array[C], i: Int, scale: Boolean) = {
    var cumulativeWeight = 0.0
    val indexedPointsWithWeightRanges: Iterator[(Double, Double, (P, Int), Double)] = for (z <- pointAndWeights) yield {
      val weight = if (scale) z._2 * kmeans.pointCost(centers, z._1._1, Some(i)) else z._2
      val from = cumulativeWeight
      cumulativeWeight = cumulativeWeight + weight
      val result = (from, cumulativeWeight, z._1, z._2)
      result
    }
    val a = indexedPointsWithWeightRanges.toArray
    (cumulativeWeight, a.iterator)
  }
}