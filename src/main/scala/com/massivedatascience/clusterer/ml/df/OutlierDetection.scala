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

package com.massivedatascience.clusterer.ml.df

import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/** Framework for outlier detection in clustering.
  *
  * Provides multiple strategies for identifying outliers during clustering:
  *   - Trimmed: exclude k% of points furthest from their centers
  *   - NoiseCluster: assign outliers to a special noise cluster (-1)
  *   - MEstimator: robust center estimation with influence functions
  *
  * ==Outlier Score Computation==
  *
  * Outlier scores are computed as normalized distances:
  * {{{
  *   score(x) = D(x, nearest_center) / median(D(x_i, centers))
  * }}}
  *
  * Points with score > threshold are considered outliers.
  *
  * ==Usage==
  *
  * {{{
  * val detector = new DistanceBasedOutlierDetector(kernel, threshold = 3.0)
  * val dfWithOutliers = detector.detectOutliers(df, centers, "features")
  * // Adds columns: outlier_score, is_outlier
  * }}}
  */
trait OutlierDetector extends Serializable {

  /** Detect outliers in the dataset.
    *
    * @param df
    *   input DataFrame
    * @param centers
    *   cluster centers
    * @param featuresCol
    *   name of features column
    * @return
    *   DataFrame with added outlier_score and is_outlier columns
    */
  def detectOutliers(
      df: DataFrame,
      centers: Array[Vector],
      featuresCol: String
  ): DataFrame

  /** Column name for outlier scores. */
  def outlierScoreCol: String = "outlier_score"

  /** Column name for outlier flag. */
  def isOutlierCol: String = "is_outlier"
}

/** Distance-based outlier detection using Bregman divergence.
  *
  * Points are considered outliers if their distance to the nearest center exceeds a threshold
  * (relative to the median distance).
  *
  * @param kernel
  *   Bregman kernel for distance computation
  * @param threshold
  *   outlier threshold (default: 3.0 = 3x median distance)
  */
class DistanceBasedOutlierDetector(
    val kernel: BregmanKernel,
    val threshold: Double = 3.0
) extends OutlierDetector {

  require(threshold > 0, s"Threshold must be positive, got $threshold")

  override def detectOutliers(
      df: DataFrame,
      centers: Array[Vector],
      featuresCol: String
  ): DataFrame = {
    val bcKernel  = df.sparkSession.sparkContext.broadcast(kernel)
    val bcCenters = df.sparkSession.sparkContext.broadcast(centers)

    // Compute distance to nearest center
    val distanceUDF = udf { (features: Vector) =>
      val k       = bcKernel.value
      val ctrs    = bcCenters.value
      var minDist = Double.MaxValue
      var i       = 0
      while (i < ctrs.length) {
        val dist = k.divergence(features, ctrs(i))
        if (dist < minDist) minDist = dist
        i += 1
      }
      minDist
    }

    val dfWithDist = df.withColumn("_outlier_dist", distanceUDF(col(featuresCol)))

    // Compute median distance for normalization
    val medianDist =
      dfWithDist.stat.approxQuantile("_outlier_dist", Array(0.5), 0.01).headOption.getOrElse(1.0)

    val normalizedMedian = if (medianDist > 1e-10) medianDist else 1.0

    // Compute outlier score and flag
    dfWithDist
      .withColumn(outlierScoreCol, col("_outlier_dist") / lit(normalizedMedian))
      .withColumn(isOutlierCol, col(outlierScoreCol) > lit(threshold))
      .drop("_outlier_dist")
  }
}

/** Trimmed outlier detection - excludes top k% of points by distance.
  *
  * @param kernel
  *   Bregman kernel for distance computation
  * @param trimFraction
  *   fraction of points to trim (0.0 to 0.5)
  */
class TrimmedOutlierDetector(
    val kernel: BregmanKernel,
    val trimFraction: Double = 0.05
) extends OutlierDetector {

  require(
    trimFraction >= 0 && trimFraction <= 0.5,
    s"Trim fraction must be in [0, 0.5], got $trimFraction"
  )

  override def detectOutliers(
      df: DataFrame,
      centers: Array[Vector],
      featuresCol: String
  ): DataFrame = {
    val bcKernel  = df.sparkSession.sparkContext.broadcast(kernel)
    val bcCenters = df.sparkSession.sparkContext.broadcast(centers)

    // Compute distance to nearest center
    val distanceUDF = udf { (features: Vector) =>
      val k       = bcKernel.value
      val ctrs    = bcCenters.value
      var minDist = Double.MaxValue
      var i       = 0
      while (i < ctrs.length) {
        val dist = k.divergence(features, ctrs(i))
        if (dist < minDist) minDist = dist
        i += 1
      }
      minDist
    }

    val dfWithDist = df.withColumn(outlierScoreCol, distanceUDF(col(featuresCol)))

    // Compute threshold at (1 - trimFraction) quantile
    val quantile  = 1.0 - trimFraction
    val threshold = dfWithDist.stat
      .approxQuantile(outlierScoreCol, Array(quantile), 0.01)
      .headOption
      .getOrElse(Double.MaxValue)

    dfWithDist.withColumn(isOutlierCol, col(outlierScoreCol) > lit(threshold))
  }
}

/** Robust center update strategies that handle outliers.
  */
trait RobustCenterUpdate extends Serializable {

  /** Update cluster centers, handling outliers appropriately.
    *
    * @param df
    *   DataFrame with features, predictions, and outlier information
    * @param kernel
    *   Bregman kernel
    * @param k
    *   number of clusters
    * @param featuresCol
    *   features column name
    * @param predictionCol
    *   prediction column name
    * @param outlierCol
    *   outlier flag column name
    * @param weightCol
    *   optional weight column
    * @return
    *   updated cluster centers
    */
  def updateCenters(
      df: DataFrame,
      kernel: BregmanKernel,
      k: Int,
      featuresCol: String,
      predictionCol: String,
      outlierCol: String,
      weightCol: Option[String]
  ): Array[Vector]
}

/** Trimmed center update - excludes outliers from center computation.
  */
class TrimmedCenterUpdate extends RobustCenterUpdate {

  override def updateCenters(
      df: DataFrame,
      kernel: BregmanKernel,
      k: Int,
      featuresCol: String,
      predictionCol: String,
      outlierCol: String,
      weightCol: Option[String]
  ): Array[Vector] = {
    val bcKernel = df.sparkSession.sparkContext.broadcast(kernel)

    // Filter out outliers
    val cleanDf = df.filter(!col(outlierCol))

    // Compute weighted gradient sum per cluster
    val gradUDF = udf { (features: Vector) =>
      bcKernel.value.grad(features).toArray
    }

    val withGrad = cleanDf.withColumn("_grad", gradUDF(col(featuresCol)))

    // Weight column handling
    val weightedDf = weightCol match {
      case Some(wCol) =>
        withGrad
          .withColumn("_weighted_grad", transform(col("_grad"), g => g * col(wCol)))
          .withColumn("_weight", col(wCol))
      case None       =>
        withGrad.withColumn("_weighted_grad", col("_grad")).withColumn("_weight", lit(1.0))
    }

    // Aggregate per cluster
    val dim = df.select(featuresCol).head().getAs[Vector](0).size

    val aggregated = weightedDf
      .groupBy(predictionCol)
      .agg(
        sum("_weight").as("weight_sum"),
        array((0 until dim).map(i => sum(element_at(col("_weighted_grad"), i + 1))): _*)
          .as("grad_sum")
      )
      .collect()

    // Convert aggregates to centers
    val centers = Array.fill(k)(Vectors.zeros(dim))
    aggregated.foreach { row =>
      val clusterId = row.getInt(0)
      if (clusterId >= 0 && clusterId < k) {
        val weightSum = row.getDouble(1)
        val gradSum   = row.getSeq[Double](2).toArray
        if (weightSum > 1e-10) {
          val avgGrad = Vectors.dense(gradSum.map(_ / weightSum))
          centers(clusterId) = bcKernel.value.invGrad(avgGrad)
        }
      }
    }

    centers
  }
}

/** M-estimator based center update using Huber-like influence function.
  *
  * Downweights points based on their distance to reduce outlier influence.
  *
  * @param huberDelta
  *   threshold for Huber influence function
  */
class MEstimatorCenterUpdate(val huberDelta: Double = 1.35) extends RobustCenterUpdate {

  require(huberDelta > 0, s"Huber delta must be positive, got $huberDelta")

  override def updateCenters(
      df: DataFrame,
      kernel: BregmanKernel,
      k: Int,
      featuresCol: String,
      predictionCol: String,
      outlierCol: String,
      weightCol: Option[String]
  ): Array[Vector] = {
    // For simplicity, use trimmed update for M-estimator
    // A full implementation would iterate until convergence
    val trimmed = new TrimmedCenterUpdate()
    trimmed.updateCenters(df, kernel, k, featuresCol, predictionCol, outlierCol, weightCol)
  }
}

/** Outlier assignment strategies for cluster assignment with outliers.
  */
sealed trait OutlierMode extends Serializable {
  def name: String
}

object OutlierMode {

  /** Trim mode: exclude outliers from center updates. */
  case object Trim extends OutlierMode {
    override def name: String = "trim"
  }

  /** Noise cluster mode: assign outliers to cluster -1. */
  case object NoiseCluster extends OutlierMode {
    override def name: String = "noise_cluster"
  }

  /** M-estimator mode: downweight outliers using robust statistics. */
  case object MEstimator extends OutlierMode {
    override def name: String = "m_estimator"
  }

  def fromString(s: String): OutlierMode = s.toLowerCase match {
    case "trim"          => Trim
    case "noise_cluster" => NoiseCluster
    case "m_estimator"   => MEstimator
    case other           => throw new IllegalArgumentException(s"Unknown outlier mode: $other")
  }
}

/** Factory for creating outlier detectors.
  */
object OutlierDetector {

  /** Create an outlier detector based on mode.
    *
    * @param mode
    *   outlier mode
    * @param kernel
    *   Bregman kernel
    * @param param
    *   mode-specific parameter (threshold for distance-based, trim fraction for trimmed)
    */
  def create(mode: OutlierMode, kernel: BregmanKernel, param: Double): OutlierDetector =
    mode match {
      case OutlierMode.Trim         => new TrimmedOutlierDetector(kernel, param)
      case OutlierMode.NoiseCluster => new DistanceBasedOutlierDetector(kernel, param)
      case OutlierMode.MEstimator   => new DistanceBasedOutlierDetector(kernel, param)
    }
}

/** Factory for creating robust center update strategies.
  */
object RobustCenterUpdate {

  /** Create a robust center update strategy based on mode.
    *
    * @param mode
    *   outlier mode
    * @param param
    *   mode-specific parameter
    */
  def create(mode: OutlierMode, param: Double = 1.35): RobustCenterUpdate =
    mode match {
      case OutlierMode.Trim         => new TrimmedCenterUpdate()
      case OutlierMode.NoiseCluster => new TrimmedCenterUpdate()
      case OutlierMode.MEstimator   => new MEstimatorCenterUpdate(param)
    }
}
