package com.massivedatascience.clusterer.ml.df.strategies

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import com.massivedatascience.clusterer.ml.df.kernels.ClusteringKernel

/** Strategy for checking convergence.
  */
trait ConvergenceCheck extends Serializable {

  /** Check convergence and compute statistics.
    *
    * @param oldCenters
    *   previous centers
    * @param newCenters
    *   new centers
    * @param assigned
    *   DataFrame with assignments
    * @param featuresCol
    *   name of features column
    * @param weightCol
    *   optional weight column
    * @param kernel
    *   Bregman kernel
    * @return
    *   (max center movement, total distortion)
    */
  def check(
      oldCenters: Array[Array[Double]],
      newCenters: Array[Array[Double]],
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: ClusteringKernel
  ): (Double, Double)
}

/** Movement-based convergence check.
  *
  * Computes maximum L2 movement of any center and total distortion.
  */
private[df] class MovementConvergence extends ConvergenceCheck with Logging {

  override def check(
      oldCenters: Array[Array[Double]],
      newCenters: Array[Array[Double]],
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: ClusteringKernel
  ): (Double, Double) = {

    // Compute max movement (only for centers that exist in both arrays)
    val minLength = math.min(oldCenters.length, newCenters.length)
    val movement  = if (minLength > 0) {
      (0 until minLength).map { i =>
        val diff = oldCenters(i).zip(newCenters(i)).map { case (a, b) => a - b }
        math.sqrt(diff.map(d => d * d).sum)
      }.max
    } else {
      0.0
    }

    // Compute total distortion
    val spark      = assigned.sparkSession
    val bcCenters  = spark.sparkContext.broadcast(newCenters)
    val bcKernel   = spark.sparkContext.broadcast(kernel)
    val numCenters = newCenters.length

    val distortionUDF = udf { (features: Vector, clusterId: Int) =>
      // Only compute distortion for valid cluster IDs
      if (clusterId >= 0 && clusterId < numCenters) {
        val center = Vectors.dense(bcCenters.value(clusterId))
        bcKernel.value.divergence(features, center)
      } else {
        // Point assigned to dropped cluster - use 0.0 distortion
        // (these points will be reassigned in the next iteration)
        0.0
      }
    }

    val actualWeightCol = weightCol.getOrElse("weight")
    val withWeight      = if (assigned.columns.contains(actualWeightCol)) {
      assigned
    } else {
      assigned.withColumn(actualWeightCol, lit(1.0))
    }

    val distortion = withWeight
      .withColumn("distortion", distortionUDF(col(featuresCol), col("cluster")))
      .agg(sum(col("distortion") * col(actualWeightCol)).as("total"))
      .collect()(0)
      .getDouble(0)

    (movement, distortion)
  }
}
