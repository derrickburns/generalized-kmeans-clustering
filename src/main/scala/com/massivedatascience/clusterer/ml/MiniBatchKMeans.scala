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

package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{ DefaultParamsReadable, DefaultParamsWritable, Identifiable }
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

import scala.util.Random

/** Mini-Batch K-Means clustering with pluggable Bregman divergences.
  *
  * Mini-Batch K-Means is a variant of K-Means that uses random mini-batches of the data to update
  * cluster centers, dramatically reducing computation time while achieving similar clustering
  * quality.
  *
  * ==Algorithm==
  *
  * The algorithm proceeds as follows:
  *   1. '''Initialization''': Select initial centers using k-means|| or random sampling 2.
  *      '''Mini-Batch Sampling''': Sample `batchSize` points uniformly at random 3.
  *      '''Assignment''': Assign each mini-batch point to the nearest center 4. '''Incremental
  *      Update''': Update centers using a streaming/online update rule:
  *      - Track per-center counts
  *      - Compute learning rate: η = 1 / (count + 1)
  *      - Update: center = (1 - η) * center + η * new_point 5. '''Convergence''': Repeat until
  *        `maxIter` batches or early stopping
  *
  * ==Comparison with Standard K-Means==
  *
  * | Aspect             | Standard K-Means   | Mini-Batch K-Means         |
  * |:-------------------|:-------------------|:---------------------------|
  * | Data per iteration | All data           | `batchSize` samples        |
  * | Center update      | Complete recompute | Incremental average        |
  * | Memory             | O(n)               | O(batchSize)               |
  * | Convergence        | Monotonic decrease | Stochastic (may oscillate) |
  * | Best for           | Small-medium data  | Large-scale data           |
  *
  * ==Example Usage==
  *
  * '''Basic mini-batch clustering:'''
  * {{{
  * val mbKmeans = new MiniBatchKMeans()
  *   .setK(10)
  *   .setBatchSize(1000)
  *   .setMaxIter(100)
  *
  * val model = mbKmeans.fit(largeDataset)
  * val predictions = model.transform(largeDataset)
  * }}}
  *
  * '''With early stopping:'''
  * {{{
  * val mbKmeans = new MiniBatchKMeans()
  *   .setK(50)
  *   .setBatchSize(2048)
  *   .setMaxIter(500)
  *   .setMaxNoImprovement(20)  // Stop if no improvement for 20 batches
  *
  * val model = mbKmeans.fit(veryLargeDataset)
  * }}}
  *
  * @see
  *   [[GeneralizedKMeans]] for standard batch K-Means
  * @see
  *   [[StreamingKMeans]] for true streaming/online K-Means
  *
  * @note
  *   For reproducible results, set the seed using `setSeed()`.
  * @note
  *   Mini-batch K-Means may not converge to the same solution as batch K-Means, but typically
  *   achieves similar quality with much less computation.
  *
  * Reference: Sculley (2010): "Web-Scale K-Means Clustering"
  */
class MiniBatchKMeans(override val uid: String)
    extends Estimator[GeneralizedKMeansModel]
    with MiniBatchKMeansParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("minibatch_kmeans"))

  // Parameter setters

  /** Sets the number of clusters (k). Must be > 1. Default: 2. */
  def setK(value: Int): this.type = set(k, value)

  /** Sets the mini-batch size. Default: 1024. */
  def setBatchSize(value: Int): this.type = set(batchSize, value)

  /** Sets the Bregman divergence. Default: "squaredEuclidean". */
  def setDivergence(value: String): this.type = set(divergence, value)

  /** Sets the smoothing parameter. Default: 1e-10. */
  def setSmoothing(value: Double): this.type = set(smoothing, value)

  /** Sets the features column name. Default: "features". */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** Sets the prediction column name. Default: "prediction". */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** Sets the optional weight column. */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /** Sets the maximum number of iterations (mini-batches). Default: 100. */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** Sets the convergence tolerance. Default: 1e-4. */
  def setTol(value: Double): this.type = set(tol, value)

  /** Sets the random seed. */
  def setSeed(value: Long): this.type = set(seed, value)

  /** Sets the max iterations without improvement for early stopping. Set to 0 to disable early
    * stopping. Default: 10.
    */
  def setMaxNoImprovement(value: Int): this.type = set(maxNoImprovement, value)

  /** Sets the reassignment ratio for empty clusters. Default: 0.01. */
  def setReassignmentRatio(value: Double): this.type = set(reassignmentRatio, value)

  /** Sets the initialization mode. Default: "k-means||". */
  def setInitMode(value: String): this.type = set(initMode, value)

  /** Sets the number of k-means|| initialization steps. Default: 2. */
  def setInitSteps(value: Int): this.type = set(initSteps, value)

  override def fit(dataset: Dataset[_]): GeneralizedKMeansModel = {
    transformSchema(dataset.schema, logging = true)

    val df          = dataset.toDF()
    val numFeatures = df.select($(featuresCol)).first().getAs[Vector](0).size
    val totalPoints = df.count()

    logInfo(
      s"Training MiniBatchKMeans with k=${$(k)}, batchSize=${$(batchSize)}, " +
        s"maxIter=${$(maxIter)}, divergence=${$(divergence)}, numFeatures=$numFeatures, " +
        s"totalPoints=$totalPoints"
    )

    // Validate input data domain requirements
    ClusteringOps.validateDomain(
      df,
      $(featuresCol),
      $(divergence),
      maxSamples = 1000
    )

    // Create kernel
    val kernel = ClusteringOps.createKernel($(divergence), $(smoothing))

    // Initialize centers
    val initialCenters = initializeCenters(df, $(featuresCol), kernel)
    logInfo(s"Initialized ${initialCenters.length} centers using ${$(initMode)}")

    // Run mini-batch iterations
    val startTime = System.currentTimeMillis()
    val result    = runMiniBatch(df, initialCenters, kernel, totalPoints)
    val elapsed   = System.currentTimeMillis() - startTime

    logInfo(
      s"Training completed: iterations=${result.iterations}, " +
        s"converged=${result.converged}, elapsed=${elapsed}ms"
    )

    // Create model
    val model = new GeneralizedKMeansModel(uid, result.centers, kernel.name)
    copyValues(model)

    // Attach training summary
    val summary = TrainingSummary.fromLloydResult(
      algorithm = "MiniBatchKMeans",
      result = result,
      k = $(k),
      dim = numFeatures,
      numPoints = totalPoints,
      assignmentStrategy = s"MiniBatch(batchSize=${$(batchSize)})",
      divergence = $(divergence),
      elapsedMillis = elapsed
    )
    model.trainingSummary = Some(summary)

    logInfo(s"Training summary:\n${summary.convergenceReport}")

    model
  }

  /** Run mini-batch k-means iterations.
    */
  private def runMiniBatch(
      df: DataFrame,
      initialCenters: Array[Array[Double]],
      kernel: ClusteringKernel,
      totalPoints: Long
  ): LloydResult = {

    val rand    = new Random($(seed))
    val kVal    = $(k)
    val dim     = initialCenters.head.length
    val batchSz = $(batchSize)

    // Mutable state for centers and counts
    val centers      = initialCenters.map(_.clone())
    val centerCounts = Array.fill(kVal)(0L)

    // Track distortion history
    val distortionHistory = scala.collection.mutable.ArrayBuffer[Double]()

    // Early stopping state
    var bestDistortion     = Double.PositiveInfinity
    var noImprovementCount = 0
    var converged          = false
    var iteration          = 0

    // Compute sampling fraction for mini-batches
    val sampleFraction = math.min(1.0, batchSz.toDouble / totalPoints * 1.5)

    while (iteration < $(maxIter) && !converged) {
      // Sample mini-batch
      val miniBatch = df
        .select($(featuresCol))
        .sample(withReplacement = false, sampleFraction, rand.nextLong())
        .limit(batchSz)
        .collect()
        .map(_.getAs[Vector](0))

      if (miniBatch.nonEmpty) {
        // Assign each point to nearest center and update incrementally
        var batchDistortion = 0.0
        val newAssignments  = Array.fill(kVal)(scala.collection.mutable.ArrayBuffer[Vector]())

        for (point <- miniBatch) {
          // Find nearest center
          var minDist   = Double.PositiveInfinity
          var minCenter = 0
          var i         = 0
          while (i < kVal) {
            val dist = kernel.divergence(point, Vectors.dense(centers(i)))
            if (dist < minDist) {
              minDist = dist
              minCenter = i
            }
            i += 1
          }
          batchDistortion += minDist
          newAssignments(minCenter) += point
        }

        // Update centers using incremental averaging
        for (clusterId <- 0 until kVal) {
          val assigned = newAssignments(clusterId)
          if (assigned.nonEmpty) {
            for (point <- assigned) {
              centerCounts(clusterId) += 1
              val eta = 1.0 / centerCounts(clusterId)

              // center = (1 - eta) * center + eta * point
              var d = 0
              while (d < dim) {
                centers(clusterId)(d) = (1.0 - eta) * centers(clusterId)(d) + eta * point(d)
                d += 1
              }
            }
          }
        }

        // Handle empty clusters: reassign random points
        val emptyClusters = (0 until kVal).filter(i => centerCounts(i) == 0)
        if (emptyClusters.nonEmpty && miniBatch.length >= emptyClusters.length) {
          val shuffled = rand.shuffle(miniBatch.toSeq)
          for ((clusterId, idx) <- emptyClusters.zipWithIndex if idx < shuffled.length) {
            centers(clusterId) = shuffled(idx).toArray
            centerCounts(clusterId) = 1
            logInfo(s"Reassigned empty cluster $clusterId with random point")
          }
        }

        // Track distortion
        val avgDistortion = batchDistortion / miniBatch.length
        distortionHistory += avgDistortion

        // Early stopping check
        if (avgDistortion < bestDistortion - $(tol)) {
          bestDistortion = avgDistortion
          noImprovementCount = 0
        } else {
          noImprovementCount += 1
        }

        if ($(maxNoImprovement) > 0 && noImprovementCount >= $(maxNoImprovement)) {
          converged = true
          logInfo(s"Early stopping: no improvement for ${$(maxNoImprovement)} iterations")
        }

        if (iteration % 10 == 0) {
          logInfo(
            s"Iteration $iteration: avgDistortion=$avgDistortion, noImprovementCount=$noImprovementCount"
          )
        }
      }

      iteration += 1
    }

    LloydResult(
      centers = centers,
      iterations = iteration,
      distortionHistory = distortionHistory.toArray,
      movementHistory = Array.empty, // Not tracked in mini-batch
      converged = converged,
      emptyClusterEvents = 0         // Handled inline
    )
  }


  /** Initialize cluster centers. */
  private def initializeCenters(
      df: DataFrame,
      featuresCol: String,
      kernel: ClusteringKernel
  ): Array[Array[Double]] = {
    $(initMode) match {
      case "random"    => initializeRandom(df, featuresCol, $(k), $(seed))
      case "k-means||" => initializeKMeansPP(df, featuresCol, $(k), $(initSteps), $(seed), kernel)
      case _           =>
        throw new IllegalArgumentException(
          s"Unknown init mode: '${$(initMode)}'. Valid options: random, k-means||"
        )
    }
  }

  /** Random initialization. */
  private def initializeRandom(
      df: DataFrame,
      featuresCol: String,
      k: Int,
      seed: Long
  ): Array[Array[Double]] = {
    val fraction = math.min(1.0, (k * 10.0) / df.count().toDouble)
    df.select(featuresCol)
      .sample(withReplacement = false, fraction, seed)
      .limit(k)
      .collect()
      .map(_.getAs[Vector](0).toArray)
  }

  /** K-means++ initialization. */
  private def initializeKMeansPP(
      df: DataFrame,
      featuresCol: String,
      k: Int,
      steps: Int,
      seed: Long,
      kernel: ClusteringKernel
  ): Array[Array[Double]] = {
    val rand     = new Random(seed)
    val bcKernel = df.sparkSession.sparkContext.broadcast(kernel)

    val allPoints = df.select(featuresCol).collect()
    require(allPoints.nonEmpty, "Dataset is empty")

    val firstCenter = allPoints(rand.nextInt(allPoints.length)).getAs[Vector](0).toArray
    var centers     = Array(firstCenter)

    for (_ <- 1 until math.min(k, steps + 1)) {
      val bcCenters = df.sparkSession.sparkContext.broadcast(centers)

      val distanceUDF = udf { (features: Vector) =>
        val ctrs    = bcCenters.value
        val kern    = bcKernel.value
        var minDist = Double.PositiveInfinity
        var i       = 0
        while (i < ctrs.length) {
          val dist = kern.divergence(features, Vectors.dense(ctrs(i)))
          if (dist < minDist) minDist = dist
          i += 1
        }
        minDist
      }

      val withDistances =
        df.select(featuresCol).withColumn("distance", distanceUDF(col(featuresCol)))
      val numToSample   = math.min(k - centers.length, 2 * k)
      val samples       = withDistances
        .sample(withReplacement = false, numToSample.toDouble / df.count(), rand.nextLong())
        .collect()
        .map(_.getAs[Vector](0).toArray)

      centers = centers ++ samples.take(k - centers.length)
      bcCenters.destroy()
    }

    centers.take(k)
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def copy(extra: ParamMap): MiniBatchKMeans = defaultCopy(extra)
}

/** Parameters for MiniBatchKMeans. */
trait MiniBatchKMeansParams
    extends Params
    with HasFeaturesCol
    with HasPredictionCol
    with HasMaxIter
    with HasTol
    with HasSeed {

  /** Number of clusters (k). */
  final val k   = new IntParam(this, "k", "Number of clusters", ParamValidators.gt(1))
  def getK: Int = $(k)

  /** Mini-batch size. */
  final val batchSize   = new IntParam(
    this,
    "batchSize",
    "Number of samples per mini-batch",
    ParamValidators.gt(0)
  )
  def getBatchSize: Int = $(batchSize)

  /** Bregman divergence. */
  final val divergence      = new Param[String](
    this,
    "divergence",
    "Bregman divergence kernel",
    ParamValidators.inArray(
      Array(
        "squaredEuclidean",
        "kl",
        "itakuraSaito",
        "generalizedI",
        "logistic",
        "l1",
        "manhattan",
        "spherical",
        "cosine"
      )
    )
  )
  def getDivergence: String = $(divergence)

  /** Smoothing parameter. */
  final val smoothing      = new DoubleParam(
    this,
    "smoothing",
    "Smoothing parameter for divergences",
    ParamValidators.gt(0.0)
  )
  def getSmoothing: Double = $(smoothing)

  /** Weight column. */
  final val weightCol       = new Param[String](this, "weightCol", "Weight column name")
  def getWeightCol: String  = $(weightCol)
  def hasWeightCol: Boolean = isSet(weightCol)

  /** Max iterations without improvement for early stopping. */
  final val maxNoImprovement   = new IntParam(
    this,
    "maxNoImprovement",
    "Max iterations without improvement (0 to disable)",
    ParamValidators.gtEq(0)
  )
  def getMaxNoImprovement: Int = $(maxNoImprovement)

  /** Reassignment ratio for empty clusters. */
  final val reassignmentRatio      = new DoubleParam(
    this,
    "reassignmentRatio",
    "Fraction of mini-batch to reassign for empty clusters",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true)
  )
  def getReassignmentRatio: Double = $(reassignmentRatio)

  /** Initialization mode. */
  final val initMode      = new Param[String](
    this,
    "initMode",
    "Initialization mode",
    ParamValidators.inArray(Array("random", "k-means||"))
  )
  def getInitMode: String = $(initMode)

  /** K-means|| initialization steps. */
  final val initSteps   = new IntParam(
    this,
    "initSteps",
    "Number of k-means|| initialization steps",
    ParamValidators.gt(0)
  )
  def getInitSteps: Int = $(initSteps)

  /** Validates schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val featuresType = schema($(featuresCol)).dataType
    require(
      featuresType.typeName == "vector",
      s"Features column must be of type Vector, got $featuresType"
    )
    schema
  }

  setDefault(
    k                 -> 2,
    batchSize         -> 1024,
    divergence        -> "squaredEuclidean",
    smoothing         -> 1e-10,
    maxIter           -> 100,
    tol               -> 1e-4,
    maxNoImprovement  -> 10,
    reassignmentRatio -> 0.01,
    initMode          -> "k-means||",
    initSteps         -> 2,
    featuresCol       -> "features",
    predictionCol     -> "prediction"
  )
}

object MiniBatchKMeans extends DefaultParamsReadable[MiniBatchKMeans] {
  override def load(path: String): MiniBatchKMeans = super.load(path)
}
