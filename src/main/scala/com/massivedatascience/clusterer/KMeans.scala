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

import com.massivedatascience.clusterer.util.SparkHelper
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector


object KMeans extends SparkHelper {
  // Initialization mode names
  val RANDOM = "random"
  val K_MEANS_PARALLEL = "k-means||"

  // point ops
  val RELATIVE_ENTROPY = "DENSE_KL_DIVERGENCE"
  val DISCRETE_KL = "DISCRETE_DENSE_KL_DIVERGENCE"
  val SPARSE_SMOOTHED_KL = "SPARSE_SMOOTHED_KL_DIVERGENCE"
  val DISCRETE_SMOOTHED_KL = "DISCRETE_DENSE_SMOOTHED_KL_DIVERGENCE"
  val SIMPLEX_SMOOTHED_KL = "SIMPLEX_SMOOTHED_KL"
  val EUCLIDEAN = "EUCLIDEAN"
  val LOGISTIC_LOSS = "LOGISTIC_LOSS"
  val GENERALIZED_I = "GENERALIZED_I_DIVERGENCE"

  val TRACKING = "TRACKING"
  val SIMPLE = "SIMPLE"
  val COLUMN_TRACKING = "COLUMN_TRACKING"

  val IDENTITY_EMBEDDING = "IDENTITY"
  val HAAR_EMBEDDING = "HAAR"
  val SYMMETRIZING_KL_EMBEDDING = "SYMMETRIZING_KL_EMBEDDING"

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

    withCached("weighted vectors", data.map(x => ImmutableInhomogeneousVector.apply(x))) { data =>
      val ops = distanceFunctionNames.map(getPointOps)
      val initializer = getInitializer(mode, k, runs, initializationSteps)
      val embeddings = embeddingNames.map(getEmbedding)
      val results = reSampleTrain(data, initializer, ops, embeddings)
      results._2.assignments.unpersist(blocking = false)
      results._1
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
   * @param initializationSteps number of steps of the initialization algorithm
   * @param distanceFunctionNames the distance functions to use
   * @param kMeansImplName which k-means implementation to use
   * @param embeddingNames sequence of embeddings to use, from lowest dimension to greatest
   * @return K-Means model
   */
  def trainWeighted(
    data: RDD[WeightedVector],
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

    sideEffect(reSampleTrain(data, initializer, ops, embeddings)) { case (_, y) =>
      y.assignments.unpersist(blocking = false)
    }._1
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
    data: RDD[WeightedVector],
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
    data: RDD[WeightedVector],
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

    val samples = subsample(data, distanceFunc, depth, embedding)
    val names = Array.tabulate(samples.length)(i => s"data embedded at depth $i")
    withCached(names, samples) { samples =>
      val ops = Array.fill(samples.length)(distanceFunc)
      iterativelyTrain(ops, samples, initializer)
    }
  }

  def getPointOps(distanceFunction: String): BasicPointOps = {
    distanceFunction match {
      case EUCLIDEAN => SquaredEuclideanPointOps
      case RELATIVE_ENTROPY => DenseKLPointOps
      case DISCRETE_KL => DiscreteDenseKLPointOps
      case SIMPLEX_SMOOTHED_KL => DiscreteDenseSimplexSmoothedKLPointOps
      case DISCRETE_SMOOTHED_KL => DiscreteDenseSmoothedKLPointOps
      case SPARSE_SMOOTHED_KL => SparseRealKLPointOps
      case LOGISTIC_LOSS => LogisticLossPointOps
      case GENERALIZED_I => GeneralizedIPointOps
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
      case SYMMETRIZING_KL_EMBEDDING => SymmetrizingKLEmbedding
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
        s.getRound > maxIterations * 2 ||
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

  def simpleTrain(distanceFunc: BregmanPointOps, bregmanPoints: RDD[BregmanPoint], initializer: KMeansInitializer)(
    implicit clusterer: MultiKMeansClusterer): (KMeansModel, KMeansResults) = {

    val initialCenters = initializer.init(distanceFunc, bregmanPoints)
    require(bregmanPoints.getStorageLevel.useMemory)
    logInfo("completed initialization of cluster centers")

    val (cost, finalCenters, assignmentOpt) =
      clusterer.cluster(distanceFunc, bregmanPoints, initialCenters)
    logInfo("completed clustering")

    val assignments = assignmentOpt.getOrElse(
      sync("cluster assignments", bregmanPoints.map(p => distanceFunc.findClosest(finalCenters, p)))
    )
    logInfo("completed assignments")

    (new KMeansModel(distanceFunc, finalCenters), new KMeansResults(cost, assignments))
  }

  def subSampleTrain(
    pointOps: BregmanPointOps,
    raw: RDD[WeightedVector],
    initializer: KMeansInitializer,
    depth: Int = 4,
    embedding: Embedding = HaarEmbedding)(
    implicit clusterer: MultiKMeansClusterer): (KMeansModel, KMeansResults) = {

    val samples = subsample(raw, pointOps, depth, embedding)
    val names = Array.tabulate(depth)(i => s"data embedded at depth $i")
    withCached(names, samples) { samples =>
      iterativelyTrain(Array.fill(depth + 1)(pointOps), samples, initializer)
    }
  }

  def reSampleTrain(
    raw: RDD[WeightedVector],
    initializer: KMeansInitializer,
    ops: Seq[BregmanPointOps],
    embeddings: Seq[Embedding]
    )(implicit clusterer: MultiKMeansClusterer): (KMeansModel, KMeansResults) = {

    require(ops.length == embeddings.length)

    val samples = resample(raw, ops, embeddings)
    val names = embeddings.map(embedding => s"data embedded with $embedding")
    withCached(names, samples) { samples =>
      iterativelyTrain(ops, samples, initializer)
    }
  }

  private def iterativelyTrain(
    pointOps: Seq[BregmanPointOps],
    dataSets: Seq[RDD[BregmanPoint]],
    initializer: KMeansInitializer)(
    implicit clusterer: MultiKMeansClusterer): (KMeansModel, KMeansResults) = {

    require(dataSets.nonEmpty)

    withCached("original", dataSets.head) { original =>
      val remaining = dataSets.zip(pointOps).tail
      remaining.foldLeft(simpleTrain(pointOps.head, original, initializer)) {
        case ((_, KMeansResults(_, assignments)), (data, op)) =>
          sideEffect(simpleTrain(op, data, new SampleInitializer(assignments.map(_._1)))) { result =>
            assignments.unpersist(blocking = false)
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
  private def subsample(
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

  private def resample(
    input: RDD[WeightedVector],
    ops: Seq[BregmanPointOps],
    embeddings: Seq[Embedding] = Seq(IdentityEmbedding)): Seq[RDD[BregmanPoint]] = {

    embeddings.zip(ops).map { case (x, o) => input.map(x.embed).map(o.toPoint)}
  }
}
