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

import com.massivedatascience.linalg.WeightedVector
import com.massivedatascience.transforms.Embeddings
import org.apache.spark.rdd.RDD


object Examples {

  import com.massivedatascience.clusterer.KMeans._

  /**
   * Iteratively train using low dimensional embedding of the high dimensional sparse input data
   * using the same distance function.
   *
   * @param raw  input data
   * @param k  number of clusters desired
   * @param maxIterations maximum number of iterations per clustering run
   * @param runs number of different clusterings to perform
   * @param mode  initialization algorithm to use
   * @param initializationSteps  number of initialization steps to perform
   * @param distanceFunctionName distance function
   * @param clustererName name of the clustering implementation to use
   * @param embeddingNames  names of the embeddings to use
   * @return
   */
  def sparseTrain(
    raw: RDD[WeightedVector],
    k: Int,
    maxIterations: Int = 20,
    runs: Int = 1,
    mode: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionName: String = BregmanPointOps.EUCLIDEAN,
    clustererName: String = MultiKMeansClusterer.COLUMN_TRACKING,
    embeddingNames: Seq[String] = Seq(Embeddings.LOW_DIMENSIONAL_RI, Embeddings.MEDIUM_DIMENSIONAL_RI,
      Embeddings.HIGH_DIMENSIONAL_RI)): KMeansModel = {

    val distances = Array.fill(embeddingNames.length)(distanceFunctionName)
    KMeans.trainWithResults(raw, k, maxIterations, runs, mode, initializationSteps, distances,
      clustererName, embeddingNames)
  }

  def timeSeriesTrain(
    raw: RDD[WeightedVector],
    k: Int,
    maxIterations: Int = 20,
    runs: Int = 1,
    initializerName: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionName: String = BregmanPointOps.EUCLIDEAN,
    clustererName: String = MultiKMeansClusterer.COLUMN_TRACKING,
    embeddingName: String = Embeddings.HAAR_EMBEDDING): KMeansModel = {

    val dim = raw.first().homogeneous.toArray.length
    require(dim > 0)
    val maxDepth = Math.floor(Math.log(dim) / Math.log(2.0)).toInt
    val target = Math.max(maxDepth - 4, 0)
    KMeans.trainViaSubsampling(raw, k, maxIterations, runs, initializerName, initializationSteps,
      distanceFunctionName, clustererName, embeddingName, depth = target)
  }
}
