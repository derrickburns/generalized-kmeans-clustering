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

import org.apache.spark.SparkContext._
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

class KMeansModel(pointOps: BregmanPointOps, centers: Array[BregmanCenter]) extends Serializable {
  lazy val k: Int = centers.length

  /** Returns the cluster centers.  N.B. These are in the embedded space where the clustering
    * takes place, which may be different from the space of the input vectors!
    */
  lazy val weightedClusterCenters: Array[WeightedVector] = centers.map(pointOps.toPoint)

  /** Returns the cluster index that a given point belongs to. */
  def predictWeighted(point: WeightedVector): Int = pointOps.findClosestCluster(centers, pointOps.toPoint(point))

  def predictClusterAndDistanceWeighted(point: WeightedVector): (Int, Double) =
    pointOps.findClosest(centers, pointOps.toPoint(point))

  /** Maps given points to their cluster indices. */
  def predictWeighted(points: RDD[WeightedVector]): RDD[Int] = {
    points.map(p => pointOps.findClosestCluster(centers, pointOps.toPoint(p)))
  }

  def computeCostWeighted(data: RDD[WeightedVector]): Double =
    data.map(p => pointOps.findClosest(centers, pointOps.toPoint(p))._2).sum()

  def clusterCenters: Array[Vector] = weightedClusterCenters.map(_.inhomogeneous)

  /** Returns the cluster index that a given point belongs to. */
  def predict(point: Vector): Int = predictWeighted(ImmutableInhomogeneousVector(point))

  /** Returns the cluster index that a given point belongs to. */
  def predictClusterAndDistance(point: Vector): (Int, Double) =
    predictClusterAndDistanceWeighted(ImmutableInhomogeneousVector(point))

  /** Maps given points to their cluster indices. */
  def predict(points: RDD[Vector]): RDD[Int] =
    predictWeighted(points.map(ImmutableInhomogeneousVector.apply))

  /** Maps given points to their cluster indices. */
  def predict(points: JavaRDD[Vector]): JavaRDD[java.lang.Integer] =
    predict(points.rdd).toJavaRDD().asInstanceOf[JavaRDD[java.lang.Integer]]

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  def computeCost(data: RDD[Vector]): Double =
    computeCostWeighted(data.map(ImmutableInhomogeneousVector.apply))
}
