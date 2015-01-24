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

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD


object KMeans extends Logging  {
  // Initialization mode names
  val RANDOM = "random"
  val K_MEANS_PARALLEL = "k-means||"

  val RELATIVE_ENTROPY = "DENSE_KL_DIVERGENCE"
  val DISCRETE_KL = "DISCRETE_DENSE_KL_DIVERGENCE"
  val SPARSE_SMOOTHED_KL = "SPARSE_SMOOTHED_KL_DIVERGENCE"
  val DISCRETE_SMOOTHED_KL = "DISCRETE_DENSE_SMOOTHED_KL_DIVERGENCE"
  val GENERALIZED_SYMMETRIZED_KL = "GENERALIZED_SYMMETRIZED_KL"
  val EUCLIDEAN = "DENSE_EUCLIDEAN"
  val SPARSE_EUCLIDEAN = "SPARSE_EUCLIDEAN"
  val LOGISTIC_LOSS = "LOGISTIC_LOSS"
  val GENERALIZED_I = "GENERALIZED_I_DIVERGENCE"
  val LOW_DIMENSIONAL_RI = "LOW_DIMENSIONAL_RI"
  val MEDIUM_DIMENSIONAL_RI = "MEDIUM_DIMENSIONAL_RI"
  val HIGH_DIMENSIONAL_RI = "HIGH_DIMENSIONAL_RI"


  private def getPointOps(name: String): BregmanPointOps = {
    name match {
      case EUCLIDEAN => DenseSquaredEuclideanPointOps
      case RELATIVE_ENTROPY => DenseKLPointOps
      case DISCRETE_KL => DiscreteDenseKLPointOps
      case DISCRETE_SMOOTHED_KL => DiscreteDenseSmoothedKLPointOps
      case SPARSE_SMOOTHED_KL => SparseRealKLPointOps
      case SPARSE_EUCLIDEAN => SparseSquaredEuclideanPointOps
      case LOGISTIC_LOSS => LogisticLossPointOps
      case GENERALIZED_I => GeneralizedIPointOps
      case GENERALIZED_SYMMETRIZED_KL => GeneralizedSymmetrizedKLPointOps
      case LOW_DIMENSIONAL_RI => LowDimensionalRISquaredEuclideanPointOps
      case MEDIUM_DIMENSIONAL_RI => MediumDimensionalRISquaredEuclideanPointOps
      case HIGH_DIMENSIONAL_RI => HighDimensionalRISquaredEuclideanPointOps

      case _ => DenseSquaredEuclideanPointOps
    }
  }

  def train(data: RDD[Vector], k: Int, maxIterations: Int, runs: Int, mode: String): KMeansModel =
    doTrain(data, k, maxIterations, runs, mode)._2

  /**
   * Trains a k-means model using specified parameters and the default values for unspecified.
   */
  def train(data: RDD[Vector], k: Int, maxIterations: Int): KMeansModel =
    doTrain(data, k, maxIterations)._2

  /**
   * Trains a k-means model using specified parameters and the default values for unspecified.
   */
  def train( data: RDD[Vector], k: Int, maxIterations: Int, runs: Int): KMeansModel =
    doTrain(data, k, maxIterations, runs)._2

  def train( data: RDD[Vector], k: Int, maxIterations: Int, runs: Int, mode: String,
    distanceMetric: String): KMeansModel =

    doTrain(data, k, maxIterations, runs, initializationMode = mode, distanceMetric = distanceMetric)._2


  def doTrain(
    raw: RDD[Vector],
    k: Int = 2,
    maxIterations: Int = 20,
    runs: Int = 1,
    initializationMode: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    epsilon: Double = 1e-4,
    distanceMetric: String = EUCLIDEAN)
  : (Double, KMeansModel) = {

    val pointOps = getPointOps(distanceMetric)
    val initializer = if (initializationMode == RANDOM) {
      new KMeansRandom(pointOps, k, runs)
    } else {
      new KMeansParallel(pointOps, k, runs, initializationSteps)
    }
    val data = (raw map {
      pointOps.inhomogeneousToPoint(_, 1.0)
    }).cache()
    val centers = initializer.init(data, 0)
    val (cost, finalCenters) = new MultiKMeans(pointOps, maxIterations).cluster(data, centers)
    (cost, new KMeansModel(pointOps, finalCenters))
  }
}
