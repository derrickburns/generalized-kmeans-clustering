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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.massivedatascience.clusterer

import com.massivedatascience.linalg.MutableWeightedVector
import org.apache.spark.Logging
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import scala.collection.Map

/**
 * A simple k-means implementation that re-computes the closest cluster centers on each iteration
 * and that recomputes each cluster on each iteration.
 *
 * @param pointOps distance function
 */
class SingleKMeans(pointOps: BregmanPointOps) extends Serializable with Logging {

  def cluster(
    data: RDD[BregmanPoint],
    centers: Array[BregmanCenter],
    maxIterations: Int = 20): (Double, KMeansModel) = {

    var active = true
    var iteration = 0
    var activeCenters = centers

    while (active && iteration < maxIterations) {
      logInfo(s"iteration $iteration number of centers ${activeCenters.length}")
      active = false
      for ((clusterIndex: Int, cn: MutableWeightedVector) <- getCentroids(data, activeCenters)) {
        val centroid = cn.asImmutable
        if (centroid.weight == 0.0) {
          active = true
          activeCenters(clusterIndex) = null.asInstanceOf[BregmanCenter]
        } else {
          active = active || pointOps.centerMoved(pointOps.toPoint(centroid), activeCenters(clusterIndex))
          activeCenters(clusterIndex) = pointOps.toCenter(centroid)
        }
      }
      activeCenters = activeCenters.filter(_ != null)
      iteration += 1
    }
    (pointOps.distortion(data, activeCenters), new KMeansModel(pointOps, activeCenters))
  }

  def getCentroids(
    data: RDD[BregmanPoint],
    activeCenters: Array[BregmanCenter]): Map[Int, MutableWeightedVector] = {

    val bcActiveCenters = data.sparkContext.broadcast(activeCenters)
    val result = data.mapPartitions { points =>
      val bcCenters = bcActiveCenters.value
      val centers = IndexedSeq.fill(bcCenters.length)(pointOps.make)
      for (point <- points) centers(pointOps.findClosestCluster(bcCenters, point)).add(point)
      centers.zipWithIndex.map(_.swap).iterator
    }.reduceByKeyLocally { case (x, y) => x.add(y)}
    bcActiveCenters.unpersist()
    result
  }
}
