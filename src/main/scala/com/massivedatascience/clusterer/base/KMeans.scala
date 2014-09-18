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

import com.massivedatascience.clusterer.metrics.FastEuclideanOps
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging

import scala.reflect.ClassTag

import org.apache.spark.mllib.linalg.Vector


/**
 * This object offers a Java callable interface that this API compatible
 * with org.apache.spark.mllib.clustering.KMeans.
 *
 */
object KMeans extends Logging {
  // Initialization mode names
  val RANDOM = "random"
  val K_MEANS_PARALLEL = "k-means||"

  def train(data: RDD[Vector], k: Int, maxIterations: Int, runs: Int, mode: String): KMeansModel =
    new KMeansModel(doTrain(new FastEuclideanOps)(data, k, maxIterations, runs, mode)._2)

  /**
   * Trains a k-means model using specified parameters and the default values for unspecified.
   */
  def train(data: RDD[Vector], k: Int, maxIterations: Int): KMeansModel =
    new KMeansModel(doTrain(new FastEuclideanOps)(data, k, maxIterations)._2)


  /**
   * Trains a k-means model using specified parameters and the default values for unspecified.
   */
  def train(data: RDD[Vector], k: Int, maxIterations: Int, runs: Int): KMeansModel =
    new KMeansModel(doTrain(new FastEuclideanOps)(data, k, maxIterations, runs)._2)


  def doTrain[P <: FP, C <: FP](pointOps: PointOps[P, C])(
    raw: RDD[Vector],
    k: Int = 2,
    maxIterations: Int = 20,
    runs: Int = 1,
    initializationMode: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5)(implicit ctag: ClassTag[C], ptag: ClassTag[P])
  : (Double, GeneralizedKMeansModel[P, C]) = {

    val initializer = if (initializationMode == RANDOM) {
      new KMeansRandom(pointOps, k, runs)
    } else {
      new KMeansParallel(pointOps, k, runs, initializationSteps)
    }
    val data = (raw map { vals => pointOps.vectorToPoint(vals)}).cache()
    val centers = initializer.init(data, 0)
    new GeneralizedKMeans(pointOps, maxIterations).cluster(data, centers)
  }
}