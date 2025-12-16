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
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructType}

import scala.collection.mutable.ArrayBuffer

/** Parameters for DPMeans and DPMeansModel.
  *
  * DPMeans uses a distance threshold (lambda) instead of a fixed number of clusters.
  * New clusters are created when a point is farther than lambda from all existing centers.
  */
trait DPMeansParams
    extends Params
    with HasFeaturesCol
    with HasPredictionCol
    with HasMaxIter
    with HasTol
    with HasSeed {

  /** Distance threshold for creating new clusters.
    *
    * When a point's distance to the nearest center exceeds lambda, a new cluster
    * is created with that point as the center. Controls the granularity of clustering:
    *   - Small lambda → more, smaller clusters
    *   - Large lambda → fewer, larger clusters
    *
    * The optimal lambda depends on your data scale and desired cluster granularity.
    * A good starting point is the median or 75th percentile of pairwise distances
    * in a sample of your data.
    *
    * Must be > 0.
    * Default: 1.0
    */
  final val lambda = new DoubleParam(
    this,
    "lambda",
    "Distance threshold for creating new clusters",
    ParamValidators.gt(0.0)
  )

  def getLambda: Double = $(lambda)

  /** Maximum number of clusters to create.
    *
    * Provides an upper bound on the number of clusters to prevent runaway cluster creation
    * with a too-small lambda. Set to 0 for no limit.
    *
    * Must be >= 0.
    * Default: 100
    */
  final val maxK = new IntParam(
    this,
    "maxK",
    "Maximum number of clusters (0 for unlimited)",
    ParamValidators.gtEq(0)
  )

  def getMaxK: Int = $(maxK)

  /** Bregman divergence kernel name. Supported values:
    *   - "squaredEuclidean" (default)
    *   - "kl" (Kullback-Leibler)
    *   - "itakuraSaito"
    *   - "generalizedI"
    *   - "logistic"
    *   - "l1" or "manhattan"
    *   - "spherical" or "cosine"
    */
  final val divergence = new Param[String](
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

  /** Smoothing parameter for divergences with domain constraints.
    * Must be > 0.
    * Default: 1e-10
    */
  final val smoothing = new DoubleParam(
    this,
    "smoothing",
    "Smoothing parameter for divergences with domain constraints",
    ParamValidators.gt(0.0)
  )

  def getSmoothing: Double = $(smoothing)

  /** Optional weight column name. */
  final val weightCol = new Param[String](this, "weightCol", "Weight column name")

  def getWeightCol: String = $(weightCol)

  def hasWeightCol: Boolean = isSet(weightCol)

  /** Optional distance column name for transform output. */
  final val distanceCol = new Param[String](this, "distanceCol", "Distance column name")

  def getDistanceCol: String = $(distanceCol)

  def hasDistanceCol: Boolean = isSet(distanceCol)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val featuresType = schema($(featuresCol)).dataType
    require(
      featuresType.typeName == "vector",
      s"Features column must be of type Vector, got $featuresType"
    )

    if (hasWeightCol) {
      val weightType = schema($(weightCol)).dataType
      require(weightType == DoubleType, s"Weight column must be of type Double, got $weightType")
    }

    schema
  }

  setDefault(
    lambda           -> 1.0,
    maxK             -> 100,
    divergence       -> "squaredEuclidean",
    smoothing        -> 1e-10,
    maxIter          -> 20,
    tol              -> 1e-4,
    featuresCol      -> "features",
    predictionCol    -> "prediction"
  )
}

/** DP-Means clustering with automatic cluster count determination.
  *
  * DP-Means is a Bayesian nonparametric extension of K-Means that automatically
  * determines the number of clusters based on a distance threshold (lambda).
  * Instead of specifying k, you specify how far apart clusters should be.
  *
  * ==Algorithm==
  *
  * The algorithm proceeds as follows:
  *   1. '''Initialization''': Start with first point as the only center
  *   2. '''Assignment''': For each point:
  *      - If distance to nearest center > lambda: create new cluster
  *      - Otherwise: assign to nearest center
  *   3. '''Update''': Recompute centers as centroids of assigned points
  *   4. '''Convergence''': Repeat until no new clusters are created and centers stabilize
  *
  * ==Choosing Lambda==
  *
  * The lambda parameter controls cluster granularity:
  *   - '''Small lambda''': More clusters, tighter groupings
  *   - '''Large lambda''': Fewer clusters, looser groupings
  *
  * Practical guidelines:
  *   - Use the median pairwise distance in a sample as a starting point
  *   - For normalized data (unit vectors), try lambda in [0.5, 2.0]
  *   - For standardized data (zero mean, unit variance), try lambda in [1.0, 5.0]
  *
  * ==Example Usage==
  *
  * {{{
  * val dpmeans = new DPMeans()
  *   .setLambda(2.0)           // Distance threshold
  *   .setMaxK(50)              // Upper bound on clusters
  *   .setMaxIter(20)
  *
  * val model = dpmeans.fit(dataset)
  * println(s"Found ${model.getK} clusters")
  * val predictions = model.transform(dataset)
  * }}}
  *
  * @see Kulis & Jordan (2012): "Revisiting k-means: New Algorithms via Bayesian Nonparametrics"
  * @see [[GeneralizedKMeans]] for fixed-k clustering
  */
class DPMeans(override val uid: String)
    extends Estimator[DPMeansModel]
    with DPMeansParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("dpmeans"))

  def setLambda(value: Double): this.type = set(lambda, value)

  def setMaxK(value: Int): this.type = set(maxK, value)

  def setDivergence(value: String): this.type = set(divergence, value)

  def setSmoothing(value: Double): this.type = set(smoothing, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setTol(value: Double): this.type = set(tol, value)

  def setSeed(value: Long): this.type = set(seed, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setDistanceCol(value: String): this.type = set(distanceCol, value)

  override def fit(dataset: Dataset[_]): DPMeansModel = {
    val df = dataset.toDF()
    validateAndTransformSchema(df.schema)

    val featCol = $(featuresCol)
    val lambdaVal = $(lambda)
    val maxKVal = $(maxK)
    val maxIterVal = $(maxIter)
    val tolVal = $(tol)
    val smooth = $(smoothing)

    val kernel = createKernel($(divergence), smooth)

    logInfo(s"Starting DP-Means with lambda=$lambdaVal, maxK=$maxKVal, maxIter=$maxIterVal")

    // Initialize with first point
    val firstPoint = df.select(featCol).head().getAs[Vector](0).toArray
    var centers = ArrayBuffer[Array[Double]](firstPoint)

    val spark = df.sparkSession
    var iter = 0
    var converged = false
    var newClustersCreated = true

    while (iter < maxIterVal && !converged) {
      iter += 1
      logInfo(s"DP-Means iteration $iter with ${centers.length} clusters")

      // Broadcast current centers
      val centersArray = centers.toArray
      val bcCenters = spark.sparkContext.broadcast(centersArray)
      val bcKernel = spark.sparkContext.broadcast(kernel)

      // Assign points and compute distances
      val assignUDF = udf { (features: Vector) =>
        val point = features.toArray
        val ctrs = bcCenters.value
        val kern = bcKernel.value

        var minIdx = 0
        var minDist = kern.divergence(Vectors.dense(point), Vectors.dense(ctrs(0)))

        var i = 1
        while (i < ctrs.length) {
          val d = kern.divergence(Vectors.dense(point), Vectors.dense(ctrs(i)))
          if (d < minDist) {
            minDist = d
            minIdx = i
          }
          i += 1
        }

        (minIdx, minDist)
      }

      val assigned = df
        .withColumn("_assign", assignUDF(col(featCol)))
        .withColumn("cluster", col("_assign._1"))
        .withColumn("_dist", col("_assign._2"))
        .drop("_assign")
        .cache()

      // Find the point that is furthest from its nearest center (if > lambda)
      val furthestOutlier = assigned
        .filter(col("_dist") > lambdaVal)
        .orderBy(col("_dist").desc)
        .select(featCol, "_dist")
        .head(1)

      bcCenters.unpersist()
      bcKernel.unpersist()

      // Create at most one new cluster per iteration (the furthest outlier)
      newClustersCreated = false
      if (furthestOutlier.nonEmpty && (maxKVal == 0 || centers.length < maxKVal)) {
        val newCenter = furthestOutlier.head.getAs[Vector](0).toArray
        val dist = furthestOutlier.head.getDouble(1)
        centers += newCenter
        newClustersCreated = true
        logInfo(f"Created new cluster (distance=$dist%.4f > lambda=$lambdaVal%.4f), total now ${centers.length}")
      }

      // Update centers based on assignments
      if (!newClustersCreated) {
        // Only update centers if no new clusters were created (stabilization phase)
        val weightColOpt = if (hasWeightCol) Some($(weightCol)) else None
        val updater = createUpdater($(divergence))

        val newCentersArray = updater.update(
          assigned.select(col(featCol), col("cluster")).toDF(featCol, "cluster"),
          featCol,
          weightColOpt,
          centers.length,
          kernel
        )

        // Check convergence (max center movement)
        val maxMovement = centers.zip(newCentersArray).map { case (old, neu) =>
          kernel.divergence(Vectors.dense(old), Vectors.dense(neu))
        }.maxOption.getOrElse(0.0)

        logInfo(f"Iteration $iter: max center movement = $maxMovement%.6f")

        if (maxMovement < tolVal) {
          converged = true
          logInfo(s"Converged after $iter iterations (movement $maxMovement < tol $tolVal)")
        }

        centers = ArrayBuffer.from(newCentersArray)
      }

      assigned.unpersist()
    }

    if (!converged && !newClustersCreated) {
      logWarning(s"Did not converge after $maxIterVal iterations")
    }

    val finalCenters = centers.toArray.map(Vectors.dense)
    logInfo(s"DP-Means completed: found ${finalCenters.length} clusters")

    val model = new DPMeansModel(uid, finalCenters)
    copyValues(model)
  }

  private def createKernel(divergence: String, smoothing: Double): BregmanKernel = {
    divergence match {
      case "squaredEuclidean"     => new SquaredEuclideanKernel()
      case "kl"                   => new KLDivergenceKernel(smoothing)
      case "itakuraSaito"         => new ItakuraSaitoKernel(smoothing)
      case "generalizedI"         => new GeneralizedIDivergenceKernel(smoothing)
      case "logistic"             => new LogisticLossKernel(smoothing)
      case "l1" | "manhattan"     => new L1Kernel()
      case "spherical" | "cosine" => new SphericalKernel()
      case _ =>
        throw new IllegalArgumentException(
          s"Unknown divergence: '$divergence'. " +
            s"Valid options: squaredEuclidean, kl, itakuraSaito, generalizedI, logistic, l1, manhattan, spherical, cosine"
        )
    }
  }

  private def createUpdater(divergence: String): UpdateStrategy = {
    divergence match {
      case "l1" | "manhattan" => new MedianUpdateStrategy()
      case _                  => new GradMeanUDAFUpdate()
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): DPMeans = defaultCopy(extra)
}

/** Model produced by DPMeans.
  *
  * Contains the cluster centers determined by the DP-Means algorithm.
  * The number of clusters is automatically determined based on the lambda threshold.
  */
class DPMeansModel private[ml] (
    override val uid: String,
    val clusterCenters: Array[Vector]
) extends org.apache.spark.ml.Model[DPMeansModel]
    with DPMeansParams
    with DefaultParamsWritable
    with Logging {

  /** Number of clusters found. */
  def getK: Int = clusterCenters.length

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()
    val featCol = $(featuresCol)
    val predCol = $(predictionCol)
    val smooth = $(smoothing)

    val kernel = createKernel($(divergence), smooth)
    val centersArray = clusterCenters.map(_.toArray)

    val spark = df.sparkSession
    val bcCenters = spark.sparkContext.broadcast(centersArray)
    val bcKernel = spark.sparkContext.broadcast(kernel)

    val predictUDF = udf { (features: Vector) =>
      val point = features.toArray
      val ctrs = bcCenters.value
      val kern = bcKernel.value

      var minIdx = 0
      var minDist = kern.divergence(Vectors.dense(point), Vectors.dense(ctrs(0)))

      var i = 1
      while (i < ctrs.length) {
        val d = kern.divergence(Vectors.dense(point), Vectors.dense(ctrs(i)))
        if (d < minDist) {
          minDist = d
          minIdx = i
        }
        i += 1
      }

      minIdx
    }

    var result = df.withColumn(predCol, predictUDF(col(featCol)))

    if (hasDistanceCol) {
      val distUDF = udf { (features: Vector, cluster: Int) =>
        val point = features.toArray
        val center = bcCenters.value(cluster)
        bcKernel.value.divergence(Vectors.dense(point), Vectors.dense(center))
      }
      result = result.withColumn($(distanceCol), distUDF(col(featCol), col(predCol)))
    }

    result
  }

  private def createKernel(divergence: String, smoothing: Double): BregmanKernel = {
    divergence match {
      case "squaredEuclidean"     => new SquaredEuclideanKernel()
      case "kl"                   => new KLDivergenceKernel(smoothing)
      case "itakuraSaito"         => new ItakuraSaitoKernel(smoothing)
      case "generalizedI"         => new GeneralizedIDivergenceKernel(smoothing)
      case "logistic"             => new LogisticLossKernel(smoothing)
      case "l1" | "manhattan"     => new L1Kernel()
      case "spherical" | "cosine" => new SphericalKernel()
      case _                      => new SquaredEuclideanKernel()
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): DPMeansModel = {
    val copied = new DPMeansModel(uid, clusterCenters)
    copyValues(copied, extra).asInstanceOf[DPMeansModel]
  }
}

object DPMeans extends DefaultParamsReadable[DPMeans] {
  override def load(path: String): DPMeans = super.load(path)
}
