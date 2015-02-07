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
    kMeansImplName: String = COLUMN_TRACKING,
    embeddingNames: List[String] = List(IDENTITY_EMBEDDING))
  : KMeansModel = {

    val ops = getPointOps(distanceFunctionName)
    val initializer = getInitializer(mode, k, runs, initializationSteps)
    val kMeansImpl = getClustererImpl(kMeansImplName, maxIterations)
    val embeddings = embeddingNames.map(getEmbedding)

    reSampleTrain(ops, kMeansImpl)(data, initializer, embeddings)._1
  }

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
  def trainWithResults(
    data: RDD[Vector],
    k: Int,
    maxIterations: Int = 20,
    runs: Int = 1,
    mode: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionName: String = EUCLIDEAN,
    kMeansImplName: String = COLUMN_TRACKING,
    embeddingNames: List[String] = List(IDENTITY_EMBEDDING))
  : (KMeansModel, KMeansResults) = {

    val ops = getPointOps(distanceFunctionName)
    val initializer = getInitializer(mode, k, runs, initializationSteps)
    val kMeansImpl = getClustererImpl(kMeansImplName, maxIterations)
    val embeddings = embeddingNames.map(getEmbedding)

    reSampleTrain(ops, kMeansImpl)(data, initializer, embeddings)
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
   * @param clustererName which k-means implementation to use
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
    clustererName: String = COLUMN_TRACKING,
    embeddingName: String = HAAR_EMBEDDING,
    depth: Int = 2)
  : (KMeansModel, KMeansResults) = {

    val pointOps = getPointOps(distanceFunctionName)
    val initializer = getInitializer(initializerName, k, runs, initializationSteps)
    val clusterer = getClustererImpl(clustererName, maxIterations)
    val embedding = getEmbedding(embeddingName)

    logInfo(s"k = $k")
    logInfo(s"maxIterations = $maxIterations")
    logInfo(s"runs = $runs")
    logInfo(s"initializer = $initializerName")
    logInfo(s"initialization steps = $initializationSteps")
    logInfo(s"distance function = $distanceFunctionName")
    logInfo(s"clusterer implementation = $clustererName")
    logInfo(s"embedding = $embeddingName")
    logInfo(s"depth = $depth")

    val samples = subsample(data, depth, embedding)
    iterativelyTrain(pointOps, clusterer)(samples, initializer)
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
      case _ => throw new RuntimeException(s"unknown distance function $distanceFunction")
    }
  }

  def getEmbedding(embeddingName: String): Embedding = {
    embeddingName match {
      case IDENTITY_EMBEDDING => IdentityEmbedding
      case LOW_DIMENSIONAL_RI => new RandomIndexEmbedding(64, 0.01)
      case MEDIUM_DIMENSIONAL_RI => new RandomIndexEmbedding(256, 0.01)
      case HIGH_DIMENSIONAL_RI => new RandomIndexEmbedding(1024, 0.01)
      case HAAR_EMBEDDING => HaarEmbedding
      case _ => throw new RuntimeException(s"unknown embedding name $embeddingName")
    }
  }


  def getClustererImpl(clustererName: String, maxIterations: Int): MultiKMeansClusterer = {
    clustererName match {
      case SIMPLE => new MultiKMeans(maxIterations)
      case TRACKING => new TrackingKMeans(terminationCondition = { s: BasicStats => s.getRound > maxIterations ||
        s.getNonEmptyClusters == 0 ||
        s.getMovement / s.getNonEmptyClusters < 1.0E-5
      })
      case COLUMN_TRACKING => new ColumnTrackingKMeans(terminationCondition = { s: BasicStats => s.getRound > maxIterations ||
        s.getNonEmptyClusters == 0 ||
        s.getMovement / s.getNonEmptyClusters < 1.0E-5
      })
      case _ => throw new RuntimeException(s"unknown clusterer $clustererName")
    }
  }

  def getInitializer(initializerName: String, k: Int, runs: Int, initializationSteps: Int): KMeansInitializer = {
    initializerName match {
      case RANDOM => new KMeansRandom(k, runs, 0)
      case K_MEANS_PARALLEL => new KMeansParallel(k, runs, initializationSteps, 0)
      case _ => throw new RuntimeException(s"unknown initializer $initializerName")
    }
  }

  def simpleTrain(
    pointOps: BregmanPointOps,
    clusterer: MultiKMeansClusterer = new MultiKMeans(30))(
    raw: RDD[Vector],
    initializer: KMeansInitializer): (KMeansModel, KMeansResults) = {

    val (bregmanPoints, initialCenters) = initializer.init(pointOps, raw)
    bregmanPoints.setName("Bregman points")
    val (cost, finalCenters, assignmentOpt) = clusterer.cluster(pointOps, bregmanPoints, initialCenters)
    val assignments = assignmentOpt.getOrElse {
      bregmanPoints.map(p => pointOps.findClosest(finalCenters, p)).setName("assignments")
    }
    (new KMeansModel(pointOps, finalCenters), new KMeansResults(cost, assignments))
  }

  def subSampleTrain(
    pointOps: BregmanPointOps,
    clusterer: MultiKMeansClusterer = new MultiKMeans(30))(
    raw: RDD[Vector],
    initializer: KMeansInitializer,
    depth: Int = 4,
    embedding: Embedding = HaarEmbedding): (KMeansModel, KMeansResults) = {

    val samples = subsample(raw, depth, embedding)
    iterativelyTrain(pointOps, clusterer)(samples, initializer)
  }

  def reSampleTrain(pointOps: BregmanPointOps, clusterer: MultiKMeansClusterer = new MultiKMeans(30))(
    raw: RDD[Vector],
    initializer: KMeansInitializer,
    embeddings: List[Embedding]
    ): (KMeansModel, KMeansResults) = {

    val samples = resample(raw, embeddings)
    iterativelyTrain(pointOps, clusterer)(samples, initializer)
  }

  private def iterativelyTrain(pointOps: BregmanPointOps, clusterer: MultiKMeansClusterer)(
    raw: List[RDD[Vector]],
    initializer: KMeansInitializer): (KMeansModel, KMeansResults) = {

    require(raw.nonEmpty)
    val train = simpleTrain(pointOps, clusterer) _
    raw.tail.foldLeft(train(raw.head, initializer)) { case ((_, clustering), data) =>
      data.cache()
      val result = train(data, new SampleInitializer(clustering.assignments.map(_._1)))
      data.unpersist(blocking = false)
      clustering.assignments.unpersist(blocking = false)
      result
    }
  }

  /**
   * Returns sub-sampled data from lowest dimension to highest dimension, repeatedly applying
   * the same embedding.
   *
   * @param raw full resolution data
   * @param depth  number of levels of sub-sampling, 0 means no sub-sampling
   * @param embedding embedding to use iteratively
   * @return
   */
  private def subsample(
    raw: RDD[Vector],
    depth: Int = 0,
    embedding: Embedding = HaarEmbedding): List[RDD[Vector]] =
    (0 until depth).foldLeft(List(raw)) { case (data, e) => data.head.map(embedding.embed) :: data}

  /**
   * Returns sub-sampled data from lowest dimension to highest dimension
   *
   * @param raw data set to embed
   * @param embeddings  list of embedding from smallest to largest
   * @return
   */

  private def resample(
    raw: RDD[Vector],
    embeddings: List[Embedding] = List(IdentityEmbedding)): List[RDD[Vector]] =
    embeddings.map(x => raw.map(x.embed))
}
