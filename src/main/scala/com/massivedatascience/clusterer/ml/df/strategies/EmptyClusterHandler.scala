package com.massivedatascience.clusterer.ml.df.strategies

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import com.massivedatascience.clusterer.ml.df.BregmanKernel

/** Strategy for handling empty clusters.
  */
trait EmptyClusterHandler extends Serializable {

  /** Handle empty clusters by reseeding or dropping.
    *
    * @param assigned
    *   DataFrame with "cluster" assignments
    * @param featuresCol
    *   name of features column
    * @param weightCol
    *   optional weight column
    * @param centers
    *   computed centers (may have fewer than k)
    * @param originalDF
    *   original DataFrame for reseeding
    * @param kernel
    *   Bregman kernel
    * @return
    *   (final centers with k elements, number of empty clusters handled)
    */
  def handle(
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      originalDF: DataFrame,
      kernel: BregmanKernel
  ): (Array[Array[Double]], Int)
}

/** Reseed empty clusters with random points.
  */
private[df] class ReseedRandomHandler(seed: Long = System.currentTimeMillis())
    extends EmptyClusterHandler
    with Logging {

  override def handle(
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      originalDF: DataFrame,
      kernel: BregmanKernel
  ): (Array[Array[Double]], Int) = {

    val k = centers.length

    // Check if we have all k centers
    if (centers.length == k) {
      return (centers, 0)
    }

    val numEmpty = k - centers.length
    logWarning(s"ReseedRandomHandler: reseeding $numEmpty empty clusters")

    // Sample random points to fill empty clusters
    val fraction = math.min(1.0, (numEmpty * 10.0) / originalDF.count().toDouble)
    val samples  = originalDF
      .sample(withReplacement = false, fraction, seed)
      .select(featuresCol)
      .limit(numEmpty)
      .collect()
      .map(_.getAs[Vector](0).toArray)

    val finalCenters = centers ++ samples
    (finalCenters.take(k), numEmpty)
  }
}

/** Drop empty clusters (return fewer than k centers).
  */
private[df] class DropEmptyClustersHandler extends EmptyClusterHandler with Logging {

  override def handle(
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      originalDF: DataFrame,
      kernel: BregmanKernel
  ): (Array[Array[Double]], Int) = {

    (centers, 0) // Just return what we have
  }
}
