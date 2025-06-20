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
 *
 */

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.MultiKMeansClusterer.ClusteringWithDistortion
import com.massivedatascience.linalg.{ MutableWeightedVector, WeightedVector }
import org.apache.spark.rdd.RDD
import org.apache.spark.util.DoubleAccumulator

import scala.collection.mutable.ArrayBuffer

import org.slf4j.LoggerFactory

/**
 * A K-Means clustering implementation that performs multiple K-means clusterings simultaneously,
 * returning the one with the lowest cost.
 *
 */

//scalastyle:off
@deprecated("use ColumnTrackingKMeans", "1.2.0")
class MultiKMeans extends MultiKMeansClusterer {

  val logger = LoggerFactory.getLogger(getClass.getName)

  def cluster(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    c: Seq[IndexedSeq[BregmanCenter]]): Seq[ClusteringWithDistortion] = {

    val centers = c.map(_.toArray).toArray

    def cluster(): Seq[ClusteringWithDistortion] = {
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
        logger.info(s"iteration $iteration")

        val activeCenters = activeRuns.map(r => centers(r)).toArray

        if (logger.isInfoEnabled) {
          for (r <- 0 until activeCenters.length)
            logger.info(s"run ${activeRuns(r)} has ${activeCenters(r).length} centers")
        }

        // Find the sum and count of points mapping to each center
        val (centroids: Array[((Int, Int), WeightedVector)], runDistortion) = getCentroids(data, activeCenters)

        if (logger.isInfoEnabled) {
          for (run <- activeRuns) logger.info(s"run $run distortion ${runDistortion(run)}")
        }

        for (run <- activeRuns) active(run) = false

        for (((runIndex: Int, clusterIndex: Int), cn: MutableWeightedVector) <- centroids) {
          val run = activeRuns(runIndex)
          if (cn.weight <= pointOps.weightThreshold) {
            active(run) = true
            // Mark center for removal instead of setting to null
            centers(run)(clusterIndex) = null.asInstanceOf[BregmanCenter]
            logger.warn(s"Run $run, cluster $clusterIndex has insufficient weight (${cn.weight}), marking for removal")
          } else {
            val centroid = cn.asImmutable
            active(run) = active(run) || pointOps.centerMoved(pointOps.toPoint(centroid), centers(run)(clusterIndex))
            centers(run)(clusterIndex) = pointOps.toCenter(centroid)
          }
        }

        // filter out null centers
        for (r <- activeRuns) centers(r) = centers(r).filter(_ != null)

        // update distortions and print log message if run completed during this iteration
        for ((run, runIndex) <- activeRuns.zipWithIndex) {
          costs(run) = runDistortion(runIndex)
          if (!active(run)) logger.info(s"run $run finished in ${iteration + 1} iterations")
        }
        activeRuns = activeRuns.filter(active(_))
        iteration += 1
      }
      costs.zip(centers).map { case (x, y) => ClusteringWithDistortion(x, y.toIndexedSeq) }
    }

    def getCentroids(
      data: RDD[BregmanPoint],
      activeCenters: Array[Array[BregmanCenter]]): (Array[((Int, Int), WeightedVector)], Array[Double]) = {

      val sc = data.sparkContext
      val runDistortion:Array[DoubleAccumulator] = Array.fill(activeCenters.length)(sc.doubleAccumulator("distortion"))
      val bcActiveCenters = sc.broadcast(activeCenters)
      val result = data.mapPartitions[((Int, Int), WeightedVector)] { points =>
        val bcCenters = bcActiveCenters.value
        val centers = bcCenters.map(c => Array.fill(c.length)(pointOps.make()))
        for (point <- points; (clusters, run) <- bcCenters.zipWithIndex) {
          val (cluster, cost) = pointOps.findClosest(clusters, point)
          runDistortion(run).add(cost)
          centers(run)(cluster).add(point)
        }

        val contribution = for (
          (clusters, run) <- bcCenters.zipWithIndex;
          (contrib, cluster) <- clusters.zipWithIndex
        ) yield {
          ((run, cluster), centers(run)(cluster).asImmutable)
        }

        contribution.iterator
      }.aggregateByKey(pointOps.make())(
        (x, y) => x.add(y),
        (x, y) => x.add(y)
      ).map(x => (x._1, x._2.asImmutable)).collect()
      bcActiveCenters.unpersist()
      (result, runDistortion.map(x => x.value.doubleValue))
    }

    cluster()
  }

}
//scalastyle:on
