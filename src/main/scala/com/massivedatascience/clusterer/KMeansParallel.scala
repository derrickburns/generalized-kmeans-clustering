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
 * This code is a modified version of the original Spark 1.0.2 implementation.
 */

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.util.XORShiftRandom
import org.apache.spark.Logging
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer


class KMeansParallel(
  pointOps: BregmanPointOps,
  k: Int,
  runs: Int,
  initializationSteps: Int)
  extends KMeansInitializer with Logging {

  /**
   * Initialize `runs` sets of cluster centers using the k-means|| algorithm by Bahmani et al.
   * (Bahmani et al., Scalable K-Means++, VLDB 2012). This is a variant of k-means++ that tries
   * to find  dissimilar cluster centers by starting with a random center and then doing
   * passes where more centers are chosen with probability proportional to their squared distance
   * to the current cluster set. It results in a provable approximation to an optimal clustering.
   *
   * The original paper can be found at http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf.
   *
   * @param data the RDD of points
   * @param seed the random number generator seed
   * @return
   */
  def init(data: RDD[BregmanPoint], seed: Int): Array[Array[BregmanCenter]] = {
    logDebug(s"k-means parallel on ${data.count()} points")

    // randomly select one center per run, putting each into a separate array buffer
    val sc = data.sparkContext
    val sample = data.takeSample(withReplacement = true, runs, seed).toSeq.map(pointOps.toCenter)
    val centers = Array.tabulate(runs)(r => ArrayBuffer(sample(r)))

    // add at most 2k points per step
    for (step <- 0 until initializationSteps) {
      if (log.isInfoEnabled) showCenters(centers, step)
      val centerArrays = centers.map(_.toArray)
      val bcCenters = sc.broadcast(centerArrays)
      for ((r, p) <- choose(data, seed, step, bcCenters)) {
        centers(r) += pointOps.toCenter(p)
      }
      bcCenters.unpersist()
    }

    val bcCenters = sc.broadcast(centers.map(_.toArray))
    val result = finalCenters(data, bcCenters, seed)
    bcCenters.unpersist()
    result
  }

  def showCenters(centers: Array[ArrayBuffer[BregmanCenter]], step: Int) {
    logInfo(s"step $step")
    for (run <- 0 until runs) {
      logInfo(s"final: run $run has ${centers.length} centers")
    }
  }

  /**
   * Randomly choose at most 2 * k  additional cluster centers by weighting them by their distance
   * to the current closest cluster
   *
   * @param data  the RDD of points
   * @param seed  random generator seed
   * @param step  which step of the selection process
   * @return  array of (run, point)
   */
  def choose(
    data: RDD[BregmanPoint],
    seed: Int,
    step: Int,
    bcCenters: Broadcast[Array[Array[BregmanCenter]]]): Array[(Int, BregmanPoint)] = {
    // compute the weighted distortion for each run

    val sumCosts = data.flatMap { point =>
      val centers = bcCenters.value
      Array.tabulate(runs)(r => (r, point.weight * pointOps.pointCost(centers(r), point)))
    }.reduceByKeyLocally(_ + _)

    // choose points in proportion to ratio of weighted cost to weighted distortion
    data.mapPartitionsWithIndex { (index, points) =>
      val centers = bcCenters.value
      val range = 0 until runs
      val rand = new XORShiftRandom(seed ^ (step << 16) ^ index)
      points.flatMap { p =>
        range.filter { r =>
          rand.nextDouble() < 2.0 * p.weight * pointOps.pointCost(centers(r), p) * k / sumCosts(r)
        }.map((_, p))
      }
    }.collect()
  }

  /**
   * Reduce sets of candidate cluster centers to at most k points per set using KMeansPlusPlus.
   * Weight the points by the distance to the closest cluster center.
   *
   * @param data  original points
   * @param bcCenters  array of sets of candidate centers
   * @param seed  random number seed
   * @return  array of sets of cluster centers
   */
  def finalCenters(
    data: RDD[BregmanPoint],
    bcCenters: Broadcast[Array[Array[BregmanCenter]]], seed: Int): Array[Array[BregmanCenter]] = {
    // for each (run, cluster) compute the sum of the weights of the points in the cluster
    val weightMap = data.flatMap { point =>
      val centers = bcCenters.value
      Array.tabulate(runs)(r => ((r, pointOps.findClosestCluster(centers(r), point)), point.weight))
    }.reduceByKeyLocally(_ + _)

    val centers = bcCenters.value
    val kmeansPlusPlus = new KMeansPlusPlus(pointOps)
    val trackingKmeans = new MultiKMeans(pointOps, 30)
    val sc = data.sparkContext

    Array.tabulate(runs) { r =>
      val myCenters = centers(r)
      logInfo(s"run $r has ${myCenters.length} centers")
      val weights = Array.tabulate(myCenters.length)(i => weightMap.getOrElse((r, i), 0.0))
      val kx = if (k > myCenters.length) myCenters.length else k
      val initial = kmeansPlusPlus.getCenters(sc, seed, myCenters, weights, kx, 1)
      val parallelCenters = sc.parallelize(myCenters.map(pointOps.toPoint))
      trackingKmeans.cluster(parallelCenters, Array(initial))._2.centers
    }
  }
}
