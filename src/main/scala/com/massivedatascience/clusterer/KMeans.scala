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

import com.massivedatascience.clusterer.util.HaarWavelet
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.rdd.RDD


object KMeans extends Logging  {
  // Initialization mode names
  val RANDOM = "random"
  val K_MEANS_PARALLEL = "k-means||"

  // point ops
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


  /**
   *
   * @param raw input data
   * @param k  number of clusters desired
   * @param maxIterations maximum number of iterations of Lloyd's algorithm
   * @param runs number of parallel clusterings to run
   * @param mode initialization algorithm to use
   * @param initializationSteps number of steps of the initialization algorithm
   * @param distanceFunction the distance functions to use
   * @return (distortion, K-Means model)
   */
  def train(
    raw: RDD[Vector],
    k: Int = 2,
    maxIterations: Int = 20,
    runs: Int = 1,
    mode: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunction: String = EUCLIDEAN)
  : KMeansModel = {

    val pointOps: BregmanPointOps = distanceFunction match {
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

    val initializer: KMeansInitializer = mode match {
      case RANDOM => new KMeansRandom(pointOps, k, runs, 0)
      case K_MEANS_PARALLEL => new KMeansParallel(pointOps, k, runs, initializationSteps, 0)
      case _ => new KMeansRandom(pointOps, k, runs, 0)
    }
    train(pointOps, maxIterations)(raw, initializer)._2
  }

  def train(pointOps: BregmanPointOps, maxIterations: Int = 30)(
    raw: RDD[Vector],
    initializer: KMeansInitializer,
    kMeans: MultiKMeansClusterer = new MultiKMeans(pointOps, maxIterations))
  : (Double, KMeansModel) = {

    val (data, centers) = initializer.init(raw)
    val (cost, finalCenters) = kMeans.cluster(data, centers)
    (cost, new KMeansModel(pointOps, finalCenters))
  }
  
  def recursivelyTrain(pointOps: BregmanPointOps, maxIterations: Int = 30)(
    raw: RDD[Vector],
    initializer: KMeansInitializer,
    embedding : Embedding = HaarEmbedding,
    depth: Int = 0,
    initializationSteps: Int = 5,
    kMeans: MultiKMeansClusterer = new MultiKMeans(pointOps, maxIterations)
    ) : (Double, KMeansModel) = {

    def recurse(data: RDD[Vector], remaining: Int) : (Double, KMeansModel) = {
      val currentInitializer = if (remaining > 0) {
        val downData = data.map{embedding.embed}
        downData.cache()
        val (downCost, model) = recurse(downData, remaining - 1)
        val assignments = model.predict(downData)
        downData.unpersist(blocking = false)
        new SampleInitializer(pointOps, assignments)
      } else {
        initializer
      }
      train(pointOps, maxIterations)(data,currentInitializer,kMeans)
    }

    recurse(raw, depth)
  }

  object HaarEmbedding extends Embedding {
    def embed(raw: Vector): Vector = Vectors.dense(HaarWavelet.average(raw.toArray))
  }



}
