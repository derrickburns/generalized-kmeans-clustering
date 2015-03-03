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
import com.massivedatascience.transforms.{ Embedding, HaarEmbedding, IdentityEmbedding }
import com.massivedatascience.util.SparkHelper
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
 * A helper object that creates K-Means Models using the underlying classes in this package.
 */
object KMeans extends SparkHelper {

  private val defaultMaxIterations = 20
  private val defaultNumRuns = 1

  /**
   * The standard configuration for a clusterer that runs Lloyd's algorithm
   * @param numClusters number of clusters desired
   * @param runs number of parallel runs to attempt
   * @param seed random number seed
   * @param maxIterations maximum number of iterations of Lloyd'a algorithm to execute
   */
  case class RunConfig(numClusters: Int, runs: Int, seed: Int, maxIterations: Int)

  /**
   *
   * Train a K-Means model using Lloyd's algorithm.
   *
   * @param data input data
   * @param k  number of clusters desired
   * @param maxIterations maximum number of iterations of Lloyd's algorithm
   * @param runs number of parallel clusterings to run
   * @param mode initialization algorithm to use
   * @param distanceFunctionNames the distance functions to use
   * @param clustererName which k-means implementation to use
   * @param embeddingNames sequence of embeddings to use, from lowest dimension to greatest
   * @return K-Means model
   */
  def train(
    data: RDD[Vector],
    k: Int,
    maxIterations: Int = KMeans.defaultMaxIterations,
    runs: Int = KMeans.defaultNumRuns,
    mode: String = KMeansInitializer.K_MEANS_PARALLEL,
    distanceFunctionNames: Seq[String] = Seq(BregmanPointOps.EUCLIDEAN),
    clustererName: String = MultiKMeansClusterer.COLUMN_TRACKING,
    embeddingNames: List[String] = List(Embedding.IDENTITY_EMBEDDING)): KMeansModel = {

    implicit val kMeansImpl = MultiKMeansClusterer(clustererName)

    val runConfig = RunConfig(k, runs, 0, maxIterations)

    withCached("weighted vectors", data.map(x => WeightedVector(x))) { data =>
      val ops = distanceFunctionNames.map(BregmanPointOps.apply)
      val initializer = KMeansInitializer(mode)
      val embeddings = embeddingNames.map(Embedding.apply)
      reSampleTrain(runConfig, data, initializer, ops, embeddings)
    }
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
   * @param distanceFunctionNames the distance functions to use
   * @param clustererName which k-means implementation to use
   * @param embeddingNames sequence of embeddings to use, from lowest dimension to greatest
   * @return K-Means model
   */
  def trainWeighted(
    data: RDD[WeightedVector],
    k: Int,
    maxIterations: Int = KMeans.defaultMaxIterations,
    runs: Int = KMeans.defaultNumRuns,
    mode: String = KMeansInitializer.K_MEANS_PARALLEL,
    distanceFunctionNames: Seq[String] = Seq(BregmanPointOps.EUCLIDEAN),
    clustererName: String = MultiKMeansClusterer.COLUMN_TRACKING,
    embeddingNames: List[String] = List(Embedding.IDENTITY_EMBEDDING)): KMeansModel = {

    implicit val kMeansImpl = MultiKMeansClusterer(clustererName)

    val runConfig = RunConfig(k, runs, 0, maxIterations)
    val ops = distanceFunctionNames.map(BregmanPointOps.apply)
    val initializer = KMeansInitializer(mode)
    val embeddings = embeddingNames.map(Embedding.apply)
    reSampleTrain(runConfig, data, initializer, ops, embeddings)
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
   * @param distanceFunctionNames the distance functions to use
   * @param clustererName which k-means implementation to use
   * @param embeddingNames sequence of embeddings to use, from lowest dimension to greatest
   * @return K-Means model and a clustering of the input data
   */
  def trainWithResults(
    data: RDD[WeightedVector],
    k: Int,
    maxIterations: Int = KMeans.defaultMaxIterations,
    runs: Int = KMeans.defaultNumRuns,
    mode: String = KMeansInitializer.K_MEANS_PARALLEL,
    distanceFunctionNames: Seq[String] = Seq(BregmanPointOps.EUCLIDEAN),
    clustererName: String = MultiKMeansClusterer.COLUMN_TRACKING,
    embeddingNames: Seq[String] = Seq(Embedding.IDENTITY_EMBEDDING)): KMeansModel = {

    require(distanceFunctionNames.length == embeddingNames.length)

    implicit val kMeansImpl = MultiKMeansClusterer(clustererName)

    val runConfig = RunConfig(k, runs, 0, maxIterations)
    val ops = distanceFunctionNames.map(BregmanPointOps.apply)
    val initializer = KMeansInitializer(mode)
    val embeddings = embeddingNames.map(Embedding.apply)

    reSampleTrain(runConfig, data, initializer, ops, embeddings)
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
   * @param distanceFunctionName the distance functions to use
   * @param clustererName which k-means implementation to use
   * @param embeddingName embedding to use recursively
   * @param depth number of times to recurse
   * @return K-Means model
   */
  def trainViaSubsampling(
    data: RDD[WeightedVector],
    k: Int,
    maxIterations: Int = KMeans.defaultMaxIterations,
    runs: Int = KMeans.defaultNumRuns,
    initializerName: String = KMeansInitializer.K_MEANS_PARALLEL,
    distanceFunctionName: String = BregmanPointOps.EUCLIDEAN,
    clustererName: String = MultiKMeansClusterer.COLUMN_TRACKING,
    embeddingName: String = Embedding.HAAR_EMBEDDING,
    depth: Int = 2): KMeansModel = {

    implicit val kMeansImpl = MultiKMeansClusterer(clustererName)

    val runConfig = RunConfig(k, runs, 0, maxIterations)
    val distanceFunc = BregmanPointOps(distanceFunctionName)
    val initializer = KMeansInitializer(initializerName)
    val embedding = Embedding(embeddingName)

    logInfo(s"k = $k")
    logInfo(s"maxIterations = $maxIterations")
    logInfo(s"runs = $runs")
    logInfo(s"initializer = $initializerName")
    logInfo(s"distance function = $distanceFunctionName")
    logInfo(s"clusterer implementation = $clustererName")
    logInfo(s"embedding = $embeddingName")
    logInfo(s"depth = $depth")

    val samples = subsample(data, distanceFunc, depth, embedding)
    val names = Array.tabulate(samples.length)(i => s"data embedded at depth $i")
    withCached(names, samples) { samples =>
      val ops = Seq.fill(samples.length)(distanceFunc)
      iterativelyTrain(runConfig, ops, samples, initializer)
    }
  }

  def simpleTrain(
    runConfig: RunConfig,
    distanceFunc: BregmanPointOps,
    bregmanPoints: RDD[BregmanPoint],
    initializer: KMeansInitializer)(
      implicit clusterer: MultiKMeansClusterer): KMeansModel = {

    require(bregmanPoints.getStorageLevel.useMemory)
    val initialCenters = initializer.init(distanceFunc, bregmanPoints, runConfig.numClusters, None,
      runConfig.runs, runConfig.seed)
    val (_, finalCenters) = clusterer.best(runConfig.maxIterations, distanceFunc, bregmanPoints, initialCenters)
    new KMeansModel(distanceFunc, finalCenters)
  }

  def subSampleTrain(
    runConfig: RunConfig,
    pointOps: BregmanPointOps,
    raw: RDD[WeightedVector],
    initializer: KMeansInitializer,
    depth: Int = 4,
    embedding: Embedding = HaarEmbedding)(
      implicit clusterer: MultiKMeansClusterer): KMeansModel = {

    val samples = subsample(raw, pointOps, depth, embedding)
    val names = Array.tabulate(depth)(i => s"data embedded at depth $i")
    withCached(names, samples) { samples =>
      iterativelyTrain(runConfig, Seq.fill(depth + 1)(pointOps), samples, initializer)
    }
  }

  def reSampleTrain(
    runConfig: RunConfig,
    raw: RDD[WeightedVector],
    initializer: KMeansInitializer,
    ops: Seq[BregmanPointOps],
    embeddings: Seq[Embedding])(implicit clusterer: MultiKMeansClusterer): KMeansModel = {

    require(ops.length == embeddings.length)

    val samples = resample(raw, ops, embeddings)
    val names = embeddings.map(embedding => s"data embedded with $embedding")
    withCached(names, samples) { samples =>
      iterativelyTrain(runConfig, ops, samples, initializer)
    }
  }

  /**
   * Iteratively train using low dimensional embedding of the high dimensional sparse input data
   * using the same distance function.
   *
   * @param raw  input data
   * @param k  number of clusters desired
   * @param maxIterations maximum number of iterations per clustering run
   * @param runs number of different clusterings to perform
   * @param mode  initialization algorithm to use
   * @param distanceFunctionName distance function
   * @param clustererName name of the clustering implementation to use
   * @param embeddingNames  names of the embeddings to use
   * @return
   */
  def sparseTrain(
    raw: RDD[WeightedVector],
    k: Int,
    maxIterations: Int = KMeans.defaultMaxIterations,
    runs: Int = KMeans.defaultNumRuns,
    mode: String = KMeansInitializer.K_MEANS_PARALLEL,
    distanceFunctionName: String = BregmanPointOps.EUCLIDEAN,
    clustererName: String = MultiKMeansClusterer.COLUMN_TRACKING,
    embeddingNames: Seq[String] = Seq(Embedding.LOW_DIMENSIONAL_RI, Embedding.MEDIUM_DIMENSIONAL_RI,
      Embedding.HIGH_DIMENSIONAL_RI)): KMeansModel = {

    val distances = Array.fill(embeddingNames.length)(distanceFunctionName)
    trainWithResults(raw, k, maxIterations, runs, mode, distances,
      clustererName, embeddingNames)
  }

  def timeSeriesTrain(
    raw: RDD[WeightedVector],
    k: Int,
    maxIterations: Int = KMeans.defaultMaxIterations,
    runs: Int = KMeans.defaultNumRuns,
    initializerName: String = KMeansInitializer.K_MEANS_PARALLEL,
    distanceFunctionName: String = BregmanPointOps.EUCLIDEAN,
    clustererName: String = MultiKMeansClusterer.COLUMN_TRACKING,
    embeddingName: String = Embedding.HAAR_EMBEDDING): KMeansModel = {

    val dim = raw.first().homogeneous.toArray.length
    require(dim > 0)
    val maxDepth = Math.floor(Math.log(dim) / Math.log(2.0)).toInt
    val target = Math.max(maxDepth - 4, 0)
    trainViaSubsampling(raw, k, maxIterations, runs, initializerName,
      distanceFunctionName, clustererName, embeddingName, depth = target)
  }

  private[this] def iterativelyTrain(
    runConfig: RunConfig,
    pointOps: Seq[BregmanPointOps],
    dataSets: Seq[RDD[BregmanPoint]],
    initializer: KMeansInitializer)(
      implicit clusterer: MultiKMeansClusterer): KMeansModel = {

    require(dataSets.nonEmpty)

    withCached("original", dataSets.head) { original =>
      val remaining = dataSets.zip(pointOps).tail
      remaining.foldLeft(simpleTrain(runConfig, pointOps.head, original, initializer)) {
        case (model, (data, op)) =>
          withNamed("assignments", model.predictBregman(data)) { assignments =>
            simpleTrain(runConfig, op, data, new SampleInitializer(assignments))
          }
      }
    }
  }

  /**
   * Returns sub-sampled data from lowest dimension to highest dimension, repeatedly applying
   * the same embedding.  Data is returned cached.
   *
   * @param input input data set to embed
   * @param depth  number of levels of sub-sampling, 0 means no sub-sampling
   * @param embedding embedding to use iteratively
   * @return
   */
  private[this] def subsample(
    input: RDD[WeightedVector],
    ops: BregmanPointOps,
    depth: Int = 0,
    embedding: Embedding = HaarEmbedding): List[RDD[BregmanPoint]] = {
    val subs = (0 until depth).foldLeft(List(input)) {
      case (data, e) => data.head.map(embedding.embed) :: data
    }
    subs.map(_.map(ops.toPoint))
  }

  /**
   * Returns sub-sampled data from lowest dimension to highest dimension.
   *
   * @param input input data set to embed
   * @param embeddings  list of embedding from smallest to largest
   * @return
   */

  private[this] def resample(
    input: RDD[WeightedVector],
    ops: Seq[BregmanPointOps],
    embeddings: Seq[Embedding] = Seq(IdentityEmbedding)): Seq[RDD[BregmanPoint]] = {

    embeddings.zip(ops).map { case (x, o) => input.map(x.embed).map(o.toPoint) }
  }
}
