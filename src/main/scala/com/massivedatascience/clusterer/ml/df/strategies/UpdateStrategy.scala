package com.massivedatascience.clusterer.ml.df.strategies

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import com.massivedatascience.clusterer.ml.df.kernels.{ BregmanKernel, ClusteringKernel }

/** Strategy for updating cluster centers.
  *
  * Computes new centers based on assigned points using gradient-based aggregation.
  */
trait UpdateStrategy extends Serializable {

  /** Compute new cluster centers from assignments.
    *
    * @param assigned
    *   DataFrame with "cluster" column
    * @param featuresCol
    *   name of features column
    * @param weightCol
    *   optional weight column
    * @param k
    *   number of clusters
    * @param kernel
    *   Bregman kernel
    * @return
    *   new cluster centers (may have fewer than k if some clusters are empty)
    */
  def update(
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      k: Int,
      kernel: ClusteringKernel
  ): Array[Array[Double]]
}

/** Gradient mean update strategy using UDAF.
  *
  * Computes μ = invGrad(∑_i w_i · grad(x_i) / ∑_i w_i) for each cluster.
  *
  * Requires a [[BregmanKernel]] (which has grad/invGrad). Passing a non-Bregman kernel (e.g.,
  * L1Kernel) will throw an [[IllegalArgumentException]].
  */
private[df] class GradMeanUDAFUpdate extends UpdateStrategy with Logging {

  override def update(
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      k: Int,
      kernel: ClusteringKernel
  ): Array[Array[Double]] = {

    require(
      kernel.isInstanceOf[BregmanKernel],
      s"GradMeanUDAFUpdate requires a BregmanKernel (with grad/invGrad), " +
        s"but got ${kernel.getClass.getSimpleName} (${kernel.name}). " +
        s"Use MedianUpdateStrategy for non-Bregman kernels like L1."
    )
    val bregmanKernel = kernel.asInstanceOf[BregmanKernel]

    logDebug(s"GradMeanUDAFUpdate: computing centers for k=$k clusters")

    val spark    = assigned.sparkSession
    val bcKernel = spark.sparkContext.broadcast(bregmanKernel)

    // UDF to compute gradient
    val gradUDF = udf { (features: Vector) =>
      bcKernel.value.grad(features)
    }

    // Add gradient column
    val withGrad = assigned.withColumn("gradient", gradUDF(col(featuresCol)))

    // Add weight column if not present
    val withWeight      = weightCol match {
      case Some(col) => withGrad
      case None      => withGrad.withColumn("weight", lit(1.0))
    }
    val actualWeightCol = weightCol.getOrElse("weight")

    // Aggregate using RDD for weighted vector sum
    val withWeightRDD = withWeight.select("cluster", "gradient", actualWeightCol).rdd.map { row =>
      val cluster = row.getInt(0)
      val grad    = row.getAs[Vector](1)
      val weight  = row.getDouble(2)
      (cluster, (grad, weight))
    }

    val clusterSums = withWeightRDD
      .aggregateByKey((Array.empty[Double], 0.0))(
        seqOp = { case ((gradSum, weightSum), (grad, weight)) =>
          val gArr       = grad.toArray
          val newGradSum = if (gradSum.isEmpty) {
            gArr.map(_ * weight)
          } else {
            gradSum.zip(gArr).map { case (sum, g) => sum + g * weight }
          }
          (newGradSum, weightSum + weight)
        },
        combOp = { case ((gradSum1, weightSum1), (gradSum2, weightSum2)) =>
          val newGradSum = if (gradSum1.isEmpty) {
            gradSum2
          } else if (gradSum2.isEmpty) {
            gradSum1
          } else {
            gradSum1.zip(gradSum2).map { case (a, b) => a + b }
          }
          (newGradSum, weightSum1 + weightSum2)
        }
      )
      .collectAsMap()

    // Compute centers: invGrad(weighted mean gradient)
    // Skip clusters with zero total weight to avoid division by zero
    val centers = (0 until k).flatMap { clusterId =>
      clusterSums.get(clusterId).flatMap { case (gradSum, weightSum) =>
        if (weightSum > 0.0) {
          val meanGrad = gradSum.map(_ / weightSum)
          Some(bregmanKernel.invGrad(Vectors.dense(meanGrad)).toArray)
        } else {
          // Cluster has only zero-weight points - skip it
          // (will be handled by EmptyClusterHandler)
          logWarning(s"Cluster $clusterId has zero total weight, skipping center computation")
          None
        }
      }
    }.toArray

    logDebug(s"GradMeanUDAFUpdate: computed ${centers.length} non-empty centers")

    centers
  }
}

/** Median update strategy for K-Medians clustering.
  *
  * Computes component-wise weighted median for each cluster instead of gradient-based mean. More
  * robust to outliers than mean-based methods.
  *
  * Note: This should be paired with L1Kernel (Manhattan distance).
  */
private[df] class MedianUpdateStrategy extends UpdateStrategy with Logging {

  override def update(
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      k: Int,
      kernel: ClusteringKernel
  ): Array[Array[Double]] = {

    logDebug(s"MedianUpdateStrategy: computing medians for k=$k clusters")

    val numFeatures = assigned.select(featuresCol).first().getAs[Vector](0).size

    // Add weight column if not present
    val withWeight      = weightCol match {
      case Some(col) => assigned
      case None      => assigned.withColumn("weight", lit(1.0))
    }
    val actualWeightCol = weightCol.getOrElse("weight")

    // For each cluster, compute component-wise median
    val centers = (0 until k).flatMap { clusterId =>
      val clusterData = withWeight.filter(col("cluster") === clusterId)
      val count       = clusterData.count()

      if (count == 0) {
        // Empty cluster
        None
      } else {
        // Compute median for each dimension
        val medians = (0 until numFeatures).map { dim =>
          computeWeightedMedian(clusterData, featuresCol, actualWeightCol, dim)
        }

        Some(medians.toArray)
      }
    }.toArray

    logDebug(s"MedianUpdateStrategy: computed ${centers.length} non-empty centers")

    centers
  }

  /** Compute weighted median of a specific dimension across a DataFrame.
    *
    * @param df
    *   DataFrame with features and weights
    * @param featuresCol
    *   name of features column
    * @param weightCol
    *   name of weight column
    * @param dimension
    *   which dimension to compute median for
    * @return
    *   weighted median value
    */
  private def computeWeightedMedian(
      df: DataFrame,
      featuresCol: String,
      weightCol: String,
      dimension: Int
  ): Double = {

    // Extract dimension values with weights
    val dimUDF = udf { (features: Vector) =>
      features(dimension)
    }

    val values = df
      .select(dimUDF(col(featuresCol)).alias("value"), col(weightCol).alias("weight"))
      .rdd
      .map { row =>
        (row.getDouble(0), row.getDouble(1))
      }
      .collect()

    if (values.isEmpty) {
      return 0.0
    }

    // Sort by value
    val sorted      = values.sortBy(_._1)
    val totalWeight = sorted.map(_._2).sum
    val halfWeight  = totalWeight / 2.0

    // Find weighted median
    var cumWeight = 0.0
    var i         = 0
    while (i < sorted.length && cumWeight < halfWeight) {
      cumWeight += sorted(i)._2
      i += 1
    }

    // Return median value
    if (i == 0) {
      sorted(0)._1
    } else if (i >= sorted.length) {
      sorted.last._1
    } else {
      // If we landed exactly on half weight, average the two middle values
      if (math.abs(cumWeight - halfWeight) < 1e-10 && i < sorted.length - 1) {
        (sorted(i - 1)._1 + sorted(i)._1) / 2.0
      } else {
        sorted(i - 1)._1
      }
    }
  }
}
