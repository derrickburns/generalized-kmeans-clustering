/*
 * Licensed to the Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Derrick R. Burns licenses this file to You under the Apache License, Version 2.0
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

package com.massivedatascience.clusterer.base

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD


class KMeansModel(specific: GeneralizedKMeansModel[_, _]) {

  val k: Int = specific.k

  /** Returns the cluster index that a given point belongs to. */
  def predict(point: Vector): Int = specific.predict(point)


  /** Maps given points to their cluster indices. */
  def predict(points: RDD[Vector]): RDD[Int] = specific.predict(points)


  /** Maps given points to their cluster indices. */
  def predict(points: JavaRDD[Vector]): JavaRDD[java.lang.Integer] = specific.predict(points)

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  def computeCost(data: RDD[Vector]): Double = specific.computeCost(data)

  def clusterCenters: Array[Vector] = specific.clusterCenters
}
