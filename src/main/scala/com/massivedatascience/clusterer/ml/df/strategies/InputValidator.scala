package com.massivedatascience.clusterer.ml.df.strategies

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import com.massivedatascience.clusterer.ml.df.kernels.ClusteringKernel

/** Input validation strategy.
  */
trait InputValidator extends Serializable {

  /** Validate input DataFrame.
    *
    * @param df
    *   input DataFrame
    * @param featuresCol
    *   name of features column
    * @param weightCol
    *   optional weight column
    * @param kernel
    *   Bregman kernel
    * @throws java.lang.IllegalArgumentException
    *   if validation fails
    */
  def validate(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: ClusteringKernel
  ): Unit
}

/** Standard input validator.
  */
private[df] class StandardInputValidator extends InputValidator with Logging {

  override def validate(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: ClusteringKernel
  ): Unit = {

    require(
      df.columns.contains(featuresCol),
      s"Features column '$featuresCol' not found in DataFrame"
    )

    weightCol.foreach { col =>
      require(df.columns.contains(col), s"Weight column '$col' not found in DataFrame")
    }

    // Check feature column type - try to get first row to validate it's a Vector
    val firstRow = df.select(featuresCol).first()
    require(firstRow.get(0).isInstanceOf[Vector], s"Features column must contain Vector type")

    // Basic count check
    val count = df.count()
    if (count == 0) {
      throw new IllegalArgumentException("Input DataFrame is empty")
    }

    logInfo(s"Validation passed for $count points with kernel ${kernel.name}")
  }
}
