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
import org.apache.spark.storage.StorageLevel


object KMeans extends Logging {
  // Initialization mode names
  val RANDOM = "random"
  val K_MEANS_PARALLEL = "k-means||"

  // point ops
  val RELATIVE_ENTROPY = "DENSE_KL_DIVERGENCE"
  val DISCRETE_KL = "DISCRETE_DENSE_KL_DIVERGENCE"
  val SPARSE_SMOOTHED_KL = "SPARSE_SMOOTHED_KL_DIVERGENCE"
  val DISCRETE_SMOOTHED_KL = "DISCRETE_DENSE_SMOOTHED_KL_DIVERGENCE"
  val SIMPLEX_SMOOTHED_KL = "SIMPLEX_SMOOTHED_KL"
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
   * @param distanceFunctionNames the distance functions to use
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
    distanceFunctionNames: Seq[String] = Seq(EUCLIDEAN),
    kMeansImplName: String = COLUMN_TRACKING,
    embeddingNames: List[String] = List(IDENTITY_EMBEDDING))
  : KMeansModel = {

    implicit val kMeansImpl = getClustererImpl(kMeansImplName, maxIterations)

    val ops = distanceFunctionNames.map(getPointOps)
    val initializer = getInitializer(mode, k, runs, initializationSteps)
    val embeddings = embeddingNames.map(getEmbedding)
    val results = reSampleTrain(data, initializer, ops, embeddings)
    results._2.assignments.unpersist(blocking = false)
    results._1
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
   * @param distanceFunctionNames the distance functions to use
   * @param kMeansImplName which k-means implementation to use
   * @param embeddingNames sequence of embeddings to use, from lowest dimension to greatest
   * @return K-Means model and a clustering of the input data
   */
  def trainWithResults(
    data: RDD[Vector],
    k: Int,
    maxIterations: Int = 20,
    runs: Int = 1,
    mode: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionNames: Seq[String] = Seq(EUCLIDEAN),
    kMeansImplName: String = COLUMN_TRACKING,
    embeddingNames: Seq[String] = Seq(IDENTITY_EMBEDDING))
  : (KMeansModel, KMeansResults) = {

    require(distanceFunctionNames.length == embeddingNames.length)

    implicit val kMeansImpl = getClustererImpl(kMeansImplName, maxIterations)

    val ops = distanceFunctionNames.map(x => getPointOps(x))
    val initializer = getInitializer(mode, k, runs, initializationSteps)
    val embeddings = embeddingNames.map(getEmbedding)

    reSampleTrain(data, initializer, ops, embeddings)
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

    implicit val clusterer = getClustererImpl(clustererName, maxIterations)

    val distanceFunc = getPointOps(distanceFunctionName)
    val initializer = getInitializer(initializerName, k, runs, initializationSteps)
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
    val ops = Array.fill(depth + 1)(distanceFunc)
    val results = iterativelyTrain(ops, samples, initializer)
    samples.reverse.tail.map(_.unpersist())
    results
  }

  def getPointOps(distanceFunction: String): BasicPointOps = {
    distanceFunction match {
      case EUCLIDEAN => DenseSquaredEuclideanPointOps
      case RELATIVE_ENTROPY => DenseKLPointOps
      case DISCRETE_KL => DiscreteDenseKLPointOps
      case SIMPLEX_SMOOTHED_KL => DiscreteDenseSimplexSmoothedKLPointOps
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
      case TRACKING => new TrackingKMeans(terminationCondition = { s: BasicStats =>
        s.getRound > maxIterations ||
          s.getNonEmptyClusters == 0 ||
          s.getMovement / s.getNonEmptyClusters < 1.0E-5
      })
      case COLUMN_TRACKING => new ColumnTrackingKMeans(terminationCondition = { s: BasicStats =>
        s.getRound > maxIterations ||
          s.getNonEmptyClusters == 0 ||
          s.getMovement / s.getNonEmptyClusters < 1.0E-5
      })
      case _ => throw new RuntimeException(s"unknown clusterer $clustererName")
    }
  }

  def getInitializer(initializerName: String, k: Int, runs: Int, initializationSteps: Int)(
    implicit clusterer: MultiKMeansClusterer): KMeansInitializer = {
    initializerName match {
      case RANDOM => new KMeansRandom(k, runs, 0)
      case K_MEANS_PARALLEL => new KMeansParallel(k, runs, initializationSteps, 0, clusterer)
      case _ => throw new RuntimeException(s"unknown initializer $initializerName")
    }
  }

  def simpleTrain(distanceFunc: BregmanPointOps, raw: RDD[Vector], initializer: KMeansInitializer)(
    implicit clusterer: MultiKMeansClusterer): (KMeansModel, KMeansResults) = {

    val (bregmanPoints, initialCenters) = initializer.init(distanceFunc, raw)
    bregmanPoints.setName("Bregman points")
    require(bregmanPoints.getStorageLevel.useMemory)
    logInfo("completed initialization of cluster centers")
    val (cost, finalCenters, assignmentOpt) =
      clusterer.cluster(distanceFunc, bregmanPoints, initialCenters)
    logInfo("completed clustering")
    val assignments = assignmentOpt.getOrElse {
      bregmanPoints.map(p => distanceFunc.findClosest(finalCenters, p)).setName("assignments").cache()
    }
    logInfo("completed assignments")

    bregmanPoints.unpersist()
    (new KMeansModel(distanceFunc, finalCenters), new KMeansResults(cost, assignments))
  }

  def subSampleTrain(
    pointOps: BregmanPointOps,
    raw: RDD[Vector],
    initializer: KMeansInitializer,
    depth: Int = 4,
    embedding: Embedding = HaarEmbedding)(
    implicit clusterer: MultiKMeansClusterer): (KMeansModel, KMeansResults) = {

    val samples = subsample(raw, depth, embedding)
    val results = iterativelyTrain(Array.fill(depth + 1)(pointOps), samples, initializer)
    samples.reverse.tail.map(_.unpersist())
    results
  }

  def reSampleTrain(
    raw: RDD[Vector],
    initializer: KMeansInitializer,
    ops: Seq[BregmanPointOps],
    embeddings: Seq[Embedding]
    )(implicit clusterer: MultiKMeansClusterer): (KMeansModel, KMeansResults) = {

    require(ops.length == embeddings.length)

    val samples = resample(raw, embeddings)
    val results = iterativelyTrain(ops, samples, initializer)
    samples.map(_.unpersist())
    results
  }

  private def iterativelyTrain(
    pointOps: Seq[BregmanPointOps],
    dataSets: Seq[RDD[Vector]],
    initializer: KMeansInitializer)(
    implicit clusterer: MultiKMeansClusterer): (KMeansModel, KMeansResults) = {

    require(dataSets.nonEmpty)
    dataSets.zip(pointOps).tail.foldLeft(simpleTrain(pointOps.head, dataSets.head, initializer)) {
      case ((_, KMeansResults(_, assignments)), (data, op)) =>
        data.cache()
        val result = simpleTrain(op, data, new SampleInitializer(assignments.map(_._1)))
        data.unpersist(blocking = false)
        assignments.unpersist(blocking = false)
        result
    }
  }

  /**
   * Returns sub-sampled data from lowest dimension to highest dimension, repeatedly applying
   * the same embedding.
   *
   * All data that is embedded is cached.
   *
   * @param dataSet full resolution data
   * @param depth  number of levels of sub-sampling, 0 means no sub-sampling
   * @param embedding embedding to use iteratively
   * @return
   */
  private def subsample(
    dataSet: RDD[Vector],
    depth: Int = 0,
    embedding: Embedding = HaarEmbedding): List[RDD[Vector]] = {
    val subs = (0 until depth).foldLeft(List(dataSet)) {
      case (data, e) => {
        data.head.map(embedding.embed).cache() :: data
      }
    }
    subs
  }

  /**
   * Returns sub-sampled data from lowest dimension to highest dimension
   *
   * @param dataSet data set to embed
   * @param embeddings  list of embedding from smallest to largest
   * @return
   */

  private def resample(
    dataSet: RDD[Vector],
    embeddings: Seq[Embedding] = Seq(IdentityEmbedding)): Seq[RDD[Vector]] = {
    embeddings.map(x => dataSet.map(x.embed).cache())
  }

}
