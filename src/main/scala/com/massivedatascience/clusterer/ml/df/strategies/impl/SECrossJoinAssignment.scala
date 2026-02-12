package com.massivedatascience.clusterer.ml.df.strategies.impl

import com.massivedatascience.clusterer.ml.df.kernels.ClusteringKernel
import com.massivedatascience.clusterer.ml.df.strategies.AssignmentStrategy
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/** Squared Euclidean cross-join assignment strategy.
  *
  * Uses DataFrame cross-join with expression-based distance computation. Much faster than UDF for
  * Squared Euclidean, but only works with SE kernel.
  */
private[df] class SECrossJoinAssignment extends AssignmentStrategy with Logging {

  override def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: ClusteringKernel
  ): DataFrame = {

    require(
      kernel.supportsExpressionOptimization,
      s"SECrossJoinAssignment only works with Squared Euclidean kernel, got ${kernel.name}"
    )

    logDebug(s"SECrossJoinAssignment: assigning ${centers.length} clusters")

    val spark = df.sparkSession

    // Create centers DataFrame using RDD
    val centersRDD = spark.sparkContext.parallelize(
      centers.toIndexedSeq.zipWithIndex.map { case (center, idx) =>
        (idx, Vectors.dense(center))
      }
    )
    val centersDF  = spark.createDataFrame(centersRDD).toDF("clusterId", "center")

    // Cross-join and compute squared distances using expressions
    val joined = df.crossJoin(centersDF)

    // Compute squared Euclidean distance using array operations
    val distanceUDF = udf { (features: Vector, center: Vector) =>
      val fArr = features.toArray
      val cArr = center.toArray
      var sum  = 0.0
      var i    = 0
      while (i < fArr.length) {
        val diff = fArr(i) - cArr(i)
        sum += diff * diff
        i += 1
      }
      sum * 0.5
    }

    val withDistances = joined.withColumn("distance", distanceUDF(col(featuresCol), col("center")))

    // Find minimum distance cluster for each point
    // Use window function for efficiency
    import org.apache.spark.sql.expressions.Window
    val partitionCols = df.columns.map(col)
    val windowSpec    = Window.partitionBy(partitionCols: _*)

    val withRank = withDistances
      .withColumn("rank", row_number().over(windowSpec.orderBy("distance")))
      .filter(col("rank") === 1)
      .drop("center", "distance", "rank")
      .withColumnRenamed("clusterId", "cluster")

    withRank
  }
}
