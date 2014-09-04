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

import org.apache.spark.Logging
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


/**
 * K-means clustering with support for multiple parallel runs and a k-means++ like initialization
 * mode (the k-means|| algorithm by Bahmani et al). When multiple concurrent runs are requested,
 * they are executed together with joint passes over the data for efficiency.
 *
 * This is an iterative algorithm that will make multiple passes over the data, so any RDDs given
 * to it should be cached by the user.
 */

class GeneralizedKMeans[T, P <: FP[T] : ClassTag, C <: FP[T] : ClassTag](pointOps: PointOps[P,C,T]) extends Serializable with Logging {

  /**
   * Train a K-means model on the given set of points; `data` should be cached for high
   * performance, because this is an iterative algorithm.
   */

  def run(data: RDD[P], initializer: KMeansInitializer[T, P,C], maxIterations: Int = 20 ): GeneralizedKMeansModel[T, P, C] = {
    val seed = 0
    val centers: Array[Array[C]] = initializer.init(data, seed)
    val (newCenters: Array[Array[C]], costs: Array[Double]) = cluster(data, maxIterations, centers)
    new GeneralizedKMeansModel(pointOps, newCenters(costs.zipWithIndex.min._2))
  }

  def cluster(data: RDD[P], maxIterations: Int, centers: Array[Array[C]])  = {
    val runs = centers.length
    val active = Array.fill(runs)(true)
    val costs = Array.fill(runs)(0.0)
    var activeRuns = new ArrayBuffer[Int] ++ (0 until runs)
    var iteration = 0

    /*
     * Execute iterations of Lloyd's algorithm until all runs have converged.
     */

    while (iteration < maxIterations && activeRuns.nonEmpty) {
      // remove the empty clusters
      log.info("iteration {}", iteration)

      val activeCenters = activeRuns.map(r => centers(r)).toArray

      if (log.isInfoEnabled) for (r <- 0 until activeCenters.length) log.info("run {} has {} centers", activeRuns(r), activeCenters(r).length)

      // Find the sum and count of points mapping to each center
      val (centroids: Array[((Int, Int), Centroid)], runDistortion: Array[Double]) = getCentroids(data, activeCenters)

      if (log.isInfoEnabled) for (run <- activeRuns) log.info("run {} distortion {}", run, runDistortion(run))

      for (run <- activeRuns) active(run) = false

      for (((runIndex: Int, clusterIndex: Int), cn: Centroid) <- centroids) {
        val run = activeRuns(runIndex)
        if (cn.weight > 0.0) {
          val centroid = pointOps.centroidToPoint(cn)
          active(run) = active(run) || pointOps.centerMoved(centroid, centers(run)(clusterIndex))
          centers(run)(clusterIndex) = pointOps.pointToCenter(centroid)
        } else {
          active(run) = true
          centers(run)(clusterIndex) = null.asInstanceOf[C]
        }
      }

      // filter out null centers
      for (r <- activeRuns) {
        centers(r) = centers(r).filter(_ != null)
      }

      // update distortions and print log message if run completed during this iteration
      for ((run, runIndex) <- activeRuns.zipWithIndex) {
        costs(run) = runDistortion(runIndex)
        if (!active(run))
          log.info("run {} finished in {} iterations", run, iteration + 1)
      }
      activeRuns = activeRuns.filter(active(_))
      iteration += 1
    }
    (centers, costs)

  }

  /**
   * Return the index of the closest point in `centers` to `point`, as well as its distance.
   */
  def findClosest(centers: Array[C], point: P, count: Option[Int] = None): (Int, Double) = {
    var bestDistance = 1.0 / 0.0
    var bestIndex = 0
    var i = 0
    val end = count.getOrElse(centers.length)
    while (i < end) {
      val distance = pointOps.distance(point, centers(i))
      if (distance < bestDistance) {
        bestIndex = i
        bestDistance = distance
      }
      i = i + 1
    }
    (bestIndex, bestDistance)
  }

  /**
   * Return the K-means cost of a given point against the given cluster centers.
   */
  def pointCost(centers: Array[C], point: P, count: Option[Int] = None): Double =
    findClosest(centers, point, count)._2

  def getCentroids(data: RDD[P], activeCenters: Array[Array[C]]): (Array[((Int, Int), Centroid)], Array[Double]) = {
    val runDistortion = activeCenters.map(_ => data.sparkContext.accumulator(0.0))
    val bcActiveCenters = data.sparkContext.broadcast(activeCenters)
    val result = data.mapPartitions {
      (points: Iterator[P]) =>
        val bcCenters = bcActiveCenters.value
        val centers = bcCenters.map {
          _.map { _ => new Centroid}
        }
        for (
          point <- points;
          (clusters: Array[C], run) <- bcCenters.zipWithIndex
        ) {
          val (cluster, cost) = findClosest(clusters, point)
          log.trace("cost: {} closest: {}", cost, cluster)
          log.trace("cluster: {}", clusters(cluster))

          runDistortion(run) += cost
          centers(run)(cluster).add(point)
        }

        val contribution =
          for (
            (clusters, run) <- bcCenters.zipWithIndex;
            (contrib, cluster) <- clusters.zipWithIndex
          ) yield {
            ((run, cluster), centers(run)(cluster))
          }

        contribution.iterator
    }.reduceByKey { case (x, y) => x.add(y)}.collect()
    bcActiveCenters.unpersist()
    (result, runDistortion.map(x => x.localValue))
  }
}
