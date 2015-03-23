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

import com.massivedatascience.clusterer.MultiKMeansClusterer.ClusteringWithDistortion
import com.massivedatascience.linalg.WeightedVector
import com.massivedatascience.transforms.{ Embedding, HaarEmbedding, IdentityEmbedding }
import com.massivedatascience.transforms.Embedding._
import com.massivedatascience.util.SparkHelper
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
 * A helper object that creates K-Means Models using the underlying classes in this package.
 */
object KMeans extends SparkHelper with Logging {

  private val defaultMaxIterations = 20
  private val defaultNumRuns = 1

  /**
   * The standard configuration for a clusterer that runs Lloyd's algorithm
   * @param numClusters number of clusters desired
   * @param runs number of parallel runs to attempt
   * @param seed random number seed
   * @param maxIterations maximum number of iterations of Lloyd'a algorithm to execute
   */
  case class RunConfig(numClusters: Int, runs: Int, seed: Int, maxIterations: Int) {
    override def toString: String =
      s"RunConfig(numClusters=$numClusters, runs=$runs, seed=$seed, maxIterations=$maxIterations)"
  }

  /**
   *
   * Train a K-Means model using Lloyd's algorithm using a signature that is similar to
   * the one provided for the Spark 1.1.0 K-Means Batch clusterer.
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
    mode: String = KMeansSelector.K_MEANS_PARALLEL,
    distanceFunctionNames: Seq[String] = Seq(BregmanPointOps.EUCLIDEAN),
    clustererName: String = MultiKMeansClusterer.COLUMN_TRACKING,
    embeddingNames: List[String] = List(Embedding.IDENTITY_EMBEDDING)): KMeansModel = {

    val clusterer = MultiKMeansClusterer(clustererName)
    val runConfig = RunConfig(k, runs, 0, maxIterations)

    withCached[WeightedVector, KMeansModel]("weighted vectors", data.map(x => WeightedVector(x))) { data =>
      val ops = distanceFunctionNames.map(BregmanPointOps.apply)
      val initializer = KMeansSelector(mode)
      val embeddings = embeddingNames.map(Embedding.apply)
      trainWeighted(runConfig, data, initializer, ops, embeddings, clusterer)
    }
  }

  /**
   *
   * Train a K-Means model using Lloyd's algorithm on WeightedVectors
   *
   * @param data input data
   * @param runConfig run configuration
   * @param pointOps the distance functions to use
   * @param initializer initialization algorithm to use
   * @param embeddings sequence of embeddings to use, from lowest dimension to greatest
   * @param clusterer which k-means implementation to use
   * @return K-Means model
   */

  def trainWeighted(
    runConfig: RunConfig,
    data: RDD[WeightedVector],
    initializer: KMeansSelector,
    pointOps: Seq[BregmanPointOps],
    embeddings: Seq[Embedding],
    clusterer: MultiKMeansClusterer): KMeansModel = {

    require(pointOps.length == embeddings.length)

    val samples = resample(data, pointOps, embeddings)
    val names = embeddings.map(embedding => s"data embedded with $embedding")
    withCached(names, samples) { samples =>
      iterativelyTrain(runConfig, pointOps, samples, initializer, clusterer)
    }
  }

  /**
   *
   * Train a K-Means model by recursively sub-sampling the data via the provided embedding.
   *
   * @param runConfig run configuration
   * @param data input data
   * @param pointOps the distance functions to use
   * @param initializer initialization algorithm to use
   * @param embedding embedding to use recursively
   * @param clusterer which k-means implementation to use
   * @param depth number of times to recurse
   * @return K-Means model
   */
  def trainViaSubsampling(
    runConfig: RunConfig,
    data: RDD[WeightedVector],
    pointOps: BregmanPointOps,
    initializer: KMeansSelector,
    embedding: Embedding,
    clusterer: MultiKMeansClusterer,
    depth: Int = 2): KMeansModel = {

    val samples = subsample(data, pointOps, depth, embedding)
    val names = Array.tabulate(samples.length)(i => s"data embedded at depth $i")
    withCached(names, samples) { samples =>
      iterativelyTrain(runConfig, Seq.fill(samples.length)(pointOps), samples, initializer, clusterer)
    }
  }

  /**
   *
   * @param runConfig run configuration
   * @param data input data
   * @param pointOps the distance functions to use
   * @param initializer initialization algorithm to use
   * @param clusterer which k-means implementation to use
   * @return K-Means model
   */
  def simpleTrain(
    runConfig: RunConfig,
    data: RDD[BregmanPoint],
    pointOps: BregmanPointOps,
    initializer: KMeansSelector,
    clusterer: MultiKMeansClusterer): KMeansModel = {

    logInfo(s"runConfig = $runConfig")
    logInfo(s"initializer = $initializer")
    logInfo(s"distance function = $pointOps")
    logInfo(s"clusterer implementation = $clusterer")

    require(data.getStorageLevel.useMemory)
    val initialCenters = initializer.init(pointOps, data, runConfig.numClusters, None,
      runConfig.runs, runConfig.seed)
    val ClusteringWithDistortion(_, finalCenters) = clusterer.best(runConfig.maxIterations, pointOps, data, initialCenters)
    new KMeansModel(pointOps, finalCenters)
  }

  /**
   * Iteratively train using low dimensional embedding of the high dimensional sparse input data
   * using the same distance function.
   *
   * @param runConfig run configuration
   * @param data  input data
   * @param initializer  initialization algorithm to use
   * @param pointOps distance function
   * @param clusterer  clustering implementation to use
   * @param embeddings  embeddings to use
   * @return  KMeansModel
   */
  def sparseTrain(
    runConfig: RunConfig,
    data: RDD[WeightedVector],
    initializer: KMeansSelector,
    pointOps: BregmanPointOps,
    clusterer: MultiKMeansClusterer,
    embeddings: Seq[Embedding] = Seq(Embedding(LOW_DIMENSIONAL_RI), Embedding(MEDIUM_DIMENSIONAL_RI),
      Embedding(HIGH_DIMENSIONAL_RI))): KMeansModel = {

    val distances = Seq.fill(embeddings.length)(pointOps)
    trainWeighted(runConfig, data, initializer, distances, embeddings, clusterer)
  }

  /**
   * Re-sample data recursively
   *
   * @param runConfig run configuration
   * @param data  input data
   * @param initializer  initialization algorithm to use
   * @param pointOps distance function
   * @param clusterer  clustering implementation to use
   * @param embedding  embedding to use
   * @return  KMeansModel
   */
  def timeSeriesTrain(
    runConfig: RunConfig,
    data: RDD[WeightedVector],
    initializer: KMeansSelector,
    pointOps: BregmanPointOps,
    clusterer: MultiKMeansClusterer,
    embedding: Embedding = Embedding(HAAR_EMBEDDING)): KMeansModel = {

    val dim = data.first().homogeneous.toArray.length
    require(dim > 0)
    val maxDepth = Math.floor(Math.log(dim) / Math.log(2.0)).toInt
    val target = Math.max(maxDepth - 4, 0)
    trainViaSubsampling(runConfig, data, pointOps, initializer,
      embedding, clusterer, depth = target)
  }

  /**
   * Train on a series of data sets, where the data sets were derived from the same
   * original data set via embeddings. Use the cluster assignments of one stage to
   * initialize the clusters of the next stage.
   *
   * @param runConfig run configuration
   * @param dataSets  input data sets to use
   * @param initializer  initialization algorithm to use
   * @param pointOps distance function
   * @param clusterer  clustering implementation to use
   * @return
   */
  def iterativelyTrain(
    runConfig: RunConfig,
    pointOps: Seq[BregmanPointOps],
    dataSets: Seq[RDD[BregmanPoint]],
    initializer: KMeansSelector,
    clusterer: MultiKMeansClusterer): KMeansModel = {

    require(dataSets.nonEmpty)
    require(pointOps.length == dataSets.length)

    val remaining = dataSets.zip(pointOps)
    remaining match {
      case Seq((initialData, ops)) =>
        simpleTrain(runConfig, initialData, ops, initializer, clusterer)

      case Seq((initialData, ops), y @ _*) =>
        y.foldLeft(simpleTrain(runConfig, initialData, ops, initializer, clusterer)) {
          case (model, (data, op)) =>
            withNamed("assignments", model.predictBregman(data)) { assignments =>
              simpleTrain(runConfig, data, op, new AssignmentSelector(assignments), clusterer)
            }
        }
    }
  }

  /**
   * Sub-sampled data from lowest dimension to highest dimension, repeatedly applying
   * the same embedding.  Data is returned cached.
   *
   * @param input input data set to embed
   * @param pointOps distance function
   * @param depth  number of levels of sub-sampling, 0 means no sub-sampling
   * @param embedding embedding to use iteratively
   * @return
   */
  private[this] def subsample(
    input: RDD[WeightedVector],
    pointOps: BregmanPointOps,
    depth: Int = 0,
    embedding: Embedding = HaarEmbedding): List[RDD[BregmanPoint]] = {
    val subs = (0 until depth).foldLeft(List(input)) {
      case (data @ List(first, _), e) => first.map(embedding.embed) :: data
    }
    subs.map(_.map(pointOps.toPoint))
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
