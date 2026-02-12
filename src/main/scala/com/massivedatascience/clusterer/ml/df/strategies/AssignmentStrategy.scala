package com.massivedatascience.clusterer.ml.df.strategies

import com.massivedatascience.clusterer.ml.df.kernels.ClusteringKernel
import org.apache.spark.sql.DataFrame

/** Strategy for assigning points to clusters.
  *
  * Takes a DataFrame with features and produces a DataFrame with cluster assignments.
  *
  * ==Available Implementations==
  *
  *   - [[impl.BroadcastUDFAssignment]]: Generic UDF-based (works with all kernels)
  *   - [[impl.SECrossJoinAssignment]]: Fast path for Squared Euclidean
  *   - [[impl.ChunkedBroadcastAssignment]]: Memory-efficient for large k×dim
  *   - [[impl.AdaptiveBroadcastAssignment]]: Adapts to executor memory
  *   - [[impl.AutoAssignment]]: Automatically selects best strategy
  */
trait AssignmentStrategy extends Serializable {

  /** Assign each point to the nearest cluster center.
    *
    * @param df
    *   input DataFrame
    * @param featuresCol
    *   name of features column
    * @param weightCol
    *   optional weight column
    * @param centers
    *   current cluster centers
    * @param kernel
    *   clustering kernel
    * @return
    *   DataFrame with additional "cluster" column (Int)
    */
  def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: ClusteringKernel
  ): DataFrame
}
