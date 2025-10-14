package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/**
 * Model fitted by GeneralizedKMeans.
 *
 * @param uid unique identifier
 * @param clusterCenters cluster centers (k x d array)
 * @param kernelName name of the Bregman kernel used during training
 */
class GeneralizedKMeansModel(
    override val uid: String,
    val clusterCenters: Array[Array[Double]],
    val kernelName: String)
    extends Model[GeneralizedKMeansModel]
    with GeneralizedKMeansParams
    with DefaultParamsWritable
    with Logging {

  def this(clusterCenters: Array[Array[Double]], kernelName: String) =
    this(Identifiable.randomUID("gkmeans"), clusterCenters, kernelName)

  /**
   * Number of clusters.
   */
  def numClusters: Int = clusterCenters.length

  /**
   * Dimensionality of features.
   */
  def numFeatures: Int = clusterCenters.headOption.map(_.length).getOrElse(0)

  /**
   * Get cluster centers as ML Vector array.
   */
  def clusterCentersAsVectors: Array[Vector] = clusterCenters.map(Vectors.dense)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val df = dataset.toDF()

    // Create kernel (must match training kernel)
    val kernel = createKernel(kernelName, $(smoothing))

    logInfo(s"Transforming dataset with ${clusterCenters.length} clusters using kernel $kernelName")

    // Broadcast centers and kernel
    val bcCenters = df.sparkSession.sparkContext.broadcast(clusterCenters)
    val bcKernel = df.sparkSession.sparkContext.broadcast(kernel)

    // UDF to find nearest cluster
    val predictUDF = udf { (features: Vector) =>
      val ctrs = bcCenters.value
      val kern = bcKernel.value
      var minDist = Double.PositiveInfinity
      var minIdx = 0
      var i = 0
      while (i < ctrs.length) {
        val center = Vectors.dense(ctrs(i))
        val dist = kern.divergence(features, center)
        if (dist < minDist) {
          minDist = dist
          minIdx = i
        }
        i += 1
      }
      minIdx
    }

    // UDF to compute distance to assigned cluster
    val distanceUDF = udf { (features: Vector, clusterId: Int) =>
      val center = Vectors.dense(bcCenters.value(clusterId))
      bcKernel.value.divergence(features, center)
    }

    // Add prediction column
    val withPrediction = df.withColumn($(predictionCol), predictUDF(col($(featuresCol))))

    // Optionally add distance column
    val result = if (hasDistanceCol) {
      withPrediction.withColumn(
        $(distanceCol),
        distanceUDF(col($(featuresCol)), col($(predictionCol))))
    } else {
      withPrediction
    }

    // Note: Don't destroy broadcast here because DataFrame is lazy
    // The broadcast will be automatically cleaned up by Spark

    result
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): GeneralizedKMeansModel = {
    val copied = new GeneralizedKMeansModel(uid, clusterCenters, kernelName)
    copyValues(copied, extra)
  }

  /**
   * Compute the sum of squared distances from each point to its assigned center.
   */
  def computeCost(dataset: Dataset[_]): Double = {
    val df = dataset.toDF()
    val kernel = createKernel(kernelName, $(smoothing))

    val bcCenters = df.sparkSession.sparkContext.broadcast(clusterCenters)
    val bcKernel = df.sparkSession.sparkContext.broadcast(kernel)

    val costUDF = udf { (features: Vector) =>
      val ctrs = bcCenters.value
      val kern = bcKernel.value
      var minDist = Double.PositiveInfinity
      var i = 0
      while (i < ctrs.length) {
        val center = Vectors.dense(ctrs(i))
        val dist = kern.divergence(features, center)
        if (dist < minDist) {
          minDist = dist
        }
        i += 1
      }
      minDist
    }

    val cost = df
      .select($(featuresCol))
      .agg(sum(costUDF(col($(featuresCol)))).as("cost"))
      .first()
      .getDouble(0)

    cost
  }

  /**
   * Predict cluster for a single feature vector.
   */
  def predict(features: Vector): Int = {
    val kernel = createKernel(kernelName, $(smoothing))
    var minDist = Double.PositiveInfinity
    var minIdx = 0
    var i = 0
    while (i < clusterCenters.length) {
      val center = Vectors.dense(clusterCenters(i))
      val dist = kernel.divergence(features, center)
      if (dist < minDist) {
        minDist = dist
        minIdx = i
      }
      i += 1
    }
    minIdx
  }

  /**
   * Create Bregman kernel based on kernel name.
   */
  private def createKernel(kernelName: String, smoothing: Double): BregmanKernel = {
    kernelName match {
      case "SquaredEuclidean" => new SquaredEuclideanKernel()
      case name if name.startsWith("KL(") => new KLDivergenceKernel(smoothing)
      case name if name.startsWith("ItakuraSaito(") => new ItakuraSaitoKernel(smoothing)
      case name if name.startsWith("GeneralizedI(") => new GeneralizedIDivergenceKernel(smoothing)
      case name if name.startsWith("LogisticLoss(") => new LogisticLossKernel(smoothing)
      case _ => throw new IllegalArgumentException(s"Unknown kernel: $kernelName")
    }
  }

  override def toString: String = {
    s"GeneralizedKMeansModel: uid=$uid, k=$numClusters, features=$numFeatures, kernel=$kernelName"
  }
}

object GeneralizedKMeansModel extends DefaultParamsReadable[GeneralizedKMeansModel] {
  override def load(path: String): GeneralizedKMeansModel = super.load(path)
}

/**
 * Summary of GeneralizedKMeans training.
 *
 * @param predictions DataFrame with predictions
 * @param predictionCol name of prediction column
 * @param featuresCol name of features column
 * @param numClusters number of clusters
 * @param numFeatures number of features
 * @param numIter number of iterations run
 * @param converged whether the algorithm converged
 * @param distortionHistory distortion at each iteration
 * @param movementHistory max center movement at each iteration
 */
class GeneralizedKMeansSummary(
    val predictions: DataFrame,
    val predictionCol: String,
    val featuresCol: String,
    val numClusters: Int,
    val numFeatures: Int,
    val numIter: Int,
    val converged: Boolean,
    val distortionHistory: Array[Double],
    val movementHistory: Array[Double]) extends Serializable {

  /**
   * Number of data points.
   */
  lazy val numPoints: Long = predictions.count()

  /**
   * Final distortion (sum of distances to assigned centers).
   */
  lazy val finalDistortion: Double = distortionHistory.last

  /**
   * Cluster sizes.
   */
  lazy val clusterSizes: Array[Long] = {
    predictions
      .groupBy(predictionCol)
      .count()
      .orderBy(predictionCol)
      .collect()
      .map(_.getLong(1))
  }

  /**
   * Mean silhouette coefficient (approximate).
   * Note: Computing true silhouette is expensive for large datasets.
   */
  def silhouette: Double = {
    // TODO: Implement silhouette computation
    // This requires computing distances between all points
    // For now, return placeholder
    0.0
  }

  override def toString: String = {
    s"""GeneralizedKMeansSummary:
       |  numClusters: $numClusters
       |  numFeatures: $numFeatures
       |  numPoints: $numPoints
       |  numIter: $numIter
       |  converged: $converged
       |  finalDistortion: $finalDistortion
       |  clusterSizes: ${clusterSizes.mkString("[", ", ", "]")}
       |""".stripMargin
  }
}
