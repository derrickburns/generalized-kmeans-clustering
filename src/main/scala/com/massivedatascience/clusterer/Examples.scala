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


import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD


object Examples {

  import KMeans._

  def sparseTrain(
    raw: RDD[Vector],
    k: Int,
    maxIterations: Int = 20,
    runs: Int = 1,
    mode: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionName: String = EUCLIDEAN,
    kMeansImplName: String = COLUMN_TRACKING,
    embeddingNames: List[String] = List(LOW_DIMENSIONAL_RI, MEDIUM_DIMENSIONAL_RI, HIGH_DIMENSIONAL_RI)): KMeansModel = {

    KMeans.train(raw, k, maxIterations, runs, mode, initializationSteps, distanceFunctionName, kMeansImplName, embeddingNames)
  }

  def timeSeriesTrain(
    raw: RDD[Vector],
    k: Int,
    maxIterations: Int = 20,
    runs: Int = 1,
    initializerName: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionName: String = EUCLIDEAN,
    kMeansImplName: String = COLUMN_TRACKING,
    embeddingName: String = HAAR_EMBEDDING): KMeansModel = {

    val dim = raw.first().toArray.length
    require(dim > 0)
    val maxDepth = Math.floor(Math.log(dim) / Math.log(2.0)).toInt
    val target = Math.max(maxDepth - 4, 0)
    KMeans.trainViaSubsampling(raw, k, maxIterations, runs, initializerName, initializationSteps, distanceFunctionName, kMeansImplName, embeddingName, depth = target)
  }
}
