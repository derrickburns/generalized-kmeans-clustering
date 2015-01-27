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


object KMeans extends Logging {
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

  val TRACKING = "TRACKING"
  val SIMPLE = "SIMPLE"
  val COLUMN_TRACKING = "COLUMN_TRACKING"

  val IDENTITY_EMBEDDING = "IDENTITY"
  val HAAR_EMBEDDING = "HAAR"
  val LOW_DIMENSIONAL_RI = "LOW_DIMENSIONAL_RI"
  val MEDIUM_DIMENSIONAL_RI = "MEDIUM_DIMENSIONAL_RI"
  val HIGH_DIMENSIONAL_RI = "HIGH_DIMENSIONAL_RI"

  /**
   * 
   * Train a K-Means model using Lloyd's algorithm.  
   *
   *
   * @param data input data
   * @param k  number of clusters desired
   * @param maxIterations maximum number of iterations of Lloyd's algorithm
   * @param runs number of parallel clusterings to run
   * @param mode initialization algorithm to use
   * @param initializationSteps number of steps of the initialization algorithm
   * @param distanceFunctionName the distance functions to use
   * @param kMeansImplName which k-means implementation to use
   * @param embeddingNames sequence of embeddings to use, from lowest dimension to greatest
   * @return K-Means model
   */
  def train(
    data: RDD[Vector],
    k: Int,
    maxIterations: Int = 20,
    runs: Int = 1,
    mode: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionName: String = EUCLIDEAN,
    kMeansImplName : String = SIMPLE,
    embeddingNames : List[String] = List(IDENTITY_EMBEDDING))
  : KMeansModel = {

    val ops = getPointOps(distanceFunctionName)
    val initializer = getInitializer(mode, k, runs, initializationSteps)
    val kMeansImpl = getKMeansImpl(kMeansImplName, maxIterations)
    val embeddings = embeddingNames.map(getEmbedding)

    reSampleTrain(ops)(data, initializer, kMeansImpl, embeddings)._2
  }
  
  /**
   * 
   * Train a K-Means model by recursively sub-sampling the data via the provided embedding.
   *
   * @param data input data
   * @param k  number of clusters desired
   * @param maxIterations maximum number of iterations of Lloyd's algorithm
   * @param runs number of parallel clusterings to run
   * @param initializerName initialization algorithm to use
   * @param initializationSteps number of steps of the initialization algorithm
   * @param distanceFunctionName the distance functions to use
   * @param kMeansImplName which k-means implementation to use
   * @param embeddingName embedding to use recursively
   * @param depth number of times to recurse                     
   * @return K-Means model
   */
  def trainViaSubsampling(
    data: RDD[Vector],
    k: Int,
    maxIterations: Int = 20,
    runs: Int = 1,
    initializerName: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionName: String = EUCLIDEAN,
    kMeansImplName : String = COLUMN_TRACKING,
    embeddingName : String = HAAR_EMBEDDING,
    depth: Int = 2)
  : KMeansModel = {

    val ops = getPointOps(distanceFunctionName)
    val initializer = getInitializer(initializerName, k, runs, initializationSteps)
    val kMeansImpl = getKMeansImpl(kMeansImplName, maxIterations)
    val embedding = getEmbedding(embeddingName)

    val samples = subsample(data, depth, embedding)
    recursivelyTrain(ops)(samples, initializer, kMeansImpl)._2
  }

  def getPointOps(distanceFunction: String): BasicPointOps = {
    distanceFunction match {
      case EUCLIDEAN => DenseSquaredEuclideanPointOps
      case RELATIVE_ENTROPY => DenseKLPointOps
      case DISCRETE_KL => DiscreteDenseKLPointOps
      case DISCRETE_SMOOTHED_KL => DiscreteDenseSmoothedKLPointOps
      case SPARSE_SMOOTHED_KL => SparseRealKLPointOps
      case SPARSE_EUCLIDEAN => SparseSquaredEuclideanPointOps
      case LOGISTIC_LOSS => LogisticLossPointOps
      case GENERALIZED_I => GeneralizedIPointOps
      case GENERALIZED_SYMMETRIZED_KL => GeneralizedSymmetrizedKLPointOps
      case _ => DenseSquaredEuclideanPointOps
    }
  }

  private def getEmbedding(embeddingName: String ) : Embedding = {
    embeddingName match {
      case IDENTITY_EMBEDDING => IdentityEmbedding
      case LOW_DIMENSIONAL_RI => new RandomIndexEmbedding(64, 0.01)
      case MEDIUM_DIMENSIONAL_RI => new RandomIndexEmbedding(256, 0.01)
      case HIGH_DIMENSIONAL_RI => new RandomIndexEmbedding(1024, 0.01)
      case HAAR_EMBEDDING => HaarEmbedding
      case _ => IdentityEmbedding
    }
  }


  private def getKMeansImpl(kmeansImpl: String, maxIterations: Int): MultiKMeansClusterer = {
    kmeansImpl match {
      case SIMPLE => new MultiKMeans(maxIterations)
      case TRACKING => new TrackingKMeans( terminationCondition = { s: BasicStats => s.getRound > maxIterations ||
        s.getNonEmptyClusters == 0 ||
        s.getMovement / s.getNonEmptyClusters < 1.0E-5
      })
      case COLUMN_TRACKING => new ColumnTrackingKMeans( terminationCondition = { s: BasicStats => s.getRound > maxIterations ||
        s.getNonEmptyClusters == 0 ||
        s.getMovement / s.getNonEmptyClusters < 1.0E-5
      })
      case _ => new MultiKMeans(maxIterations)
    }
  }

  private def getInitializer(mode: String, k: Int, runs: Int, initializationSteps: Int): KMeansInitializer = {
    mode match {
      case RANDOM => new KMeansRandom(k, runs, 0)
      case K_MEANS_PARALLEL => new KMeansParallel(k, runs, initializationSteps, 0)
      case _ => new KMeansRandom(k, runs, 0)
    }
  }

  def simpleTrain(pointOps: BregmanPointOps)(
    raw: RDD[Vector],
    initializer: KMeansInitializer,
    kMeans: MultiKMeansClusterer = new MultiKMeans(30))
  : (Double, KMeansModel) = {

    val (data, centers) = initializer.init(pointOps, raw)
    val (cost, finalCenters) = kMeans.cluster(pointOps, data, centers)
    (cost, new KMeansModel(pointOps, finalCenters))
  }

  def subSampleTrain(pointOps: BregmanPointOps)(
    raw: RDD[Vector],
    initializer: KMeansInitializer,
    kMeans: MultiKMeansClusterer = new MultiKMeans(30),
    depth: Int = 4,
    embedding: Embedding = HaarEmbedding): (Double, KMeansModel) = {

    val samples = subsample(raw, depth, embedding)
    recursivelyTrain(pointOps)(samples, initializer, kMeans)
  }

  def reSampleTrain(pointOps: BregmanPointOps)(
    raw: RDD[Vector],
    initializer: KMeansInitializer,
    kMeans: MultiKMeansClusterer = new MultiKMeans(30),
    embeddings: List[Embedding]
    ): (Double, KMeansModel) = {

    val samples = resample(raw, embeddings)
    recursivelyTrain(pointOps)(samples, initializer, kMeans)
  }

  private def recursivelyTrain(pointOps: BregmanPointOps)(
    raw: List[RDD[Vector]],
    initializer: KMeansInitializer,
    kMeans: MultiKMeansClusterer = new MultiKMeans(30)
    ): (Double, KMeansModel) = {

    def recurse(data: List[RDD[Vector]]): (Double, KMeansModel) = {
      val currentInitializer = if (data.tail.nonEmpty) {
        val downData = data.head
        downData.cache()
        val (downCost, model) = recurse(data.tail)
        val assignments = model.predict(downData)
        downData.unpersist(blocking = false)
        new SampleInitializer(assignments)
      } else {
        initializer
      }
      simpleTrain(pointOps)(data.head, currentInitializer, kMeans)
    }

    recurse(raw)
  }

  private def subsample(raw: RDD[Vector], depth: Int = 0, embedding: Embedding = HaarEmbedding): List[RDD[Vector]] = {
    if (depth == 0) {
      List(raw)
    } else {
      val x = subsample(raw, depth - 1)
      x.head.map (embedding.embed) :: x
    }
  }

  /**
   *
   * @param raw data set to embed
   * @param embeddings  list of embedding from smallest to largest
   * @return
   */

  private def resample(raw: RDD[Vector], embeddings: List[Embedding] = List(IdentityEmbedding)): List[RDD[Vector]] = {
    if (embeddings.isEmpty) {
      List[RDD[Vector]]()
    } else {
      raw.map (embeddings.head.embed) :: resample(raw, embeddings.tail)
    }
  }
}

