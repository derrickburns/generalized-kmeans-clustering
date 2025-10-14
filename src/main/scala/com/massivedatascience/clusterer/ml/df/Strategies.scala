package com.massivedatascience.clusterer.ml.df

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
 * Strategy for assigning points to clusters.
 *
 * Takes a DataFrame with features and produces a DataFrame with cluster assignments.
 */
trait AssignmentStrategy extends Serializable {
  /**
   * Assign each point to the nearest cluster center.
   *
   * @param df          input DataFrame
   * @param featuresCol name of features column
   * @param weightCol   optional weight column
   * @param centers     current cluster centers
   * @param kernel      Bregman kernel
   * @return DataFrame with additional "cluster" column (Int)
   */
  def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: BregmanKernel): DataFrame
}

/**
 * Broadcast UDF assignment strategy.
 *
 * Broadcasts centers to executors and uses UDF to compute assignments.
 * Works with any Bregman divergence but may be slower than expression-based approaches.
 */
class BroadcastUDFAssignment extends AssignmentStrategy with Logging {

  override def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: BregmanKernel): DataFrame = {

    logDebug(s"BroadcastUDFAssignment: assigning ${centers.length} clusters")

    val spark = df.sparkSession
    val bcCenters = spark.sparkContext.broadcast(centers)
    val bcKernel = spark.sparkContext.broadcast(kernel)

    val assignUDF = udf { (features: Vector) =>
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

    df.withColumn("cluster", assignUDF(col(featuresCol)))
  }
}

/**
 * Squared Euclidean cross-join assignment strategy.
 *
 * Uses DataFrame cross-join with expression-based distance computation.
 * Much faster than UDF for Squared Euclidean, but only works with SE kernel.
 */
class SECrossJoinAssignment extends AssignmentStrategy with Logging {

  override def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: BregmanKernel): DataFrame = {

    require(kernel.supportsExpressionOptimization,
      s"SECrossJoinAssignment only works with Squared Euclidean kernel, got ${kernel.name}")

    logDebug(s"SECrossJoinAssignment: assigning ${centers.length} clusters")

    val spark = df.sparkSession

    // Create centers DataFrame using RDD
    val centersRDD = spark.sparkContext.parallelize(
      centers.zipWithIndex.map { case (center, idx) =>
        (idx, Vectors.dense(center))
      }
    )
    val centersDF = spark.createDataFrame(centersRDD).toDF("clusterId", "center")

    // Cross-join and compute squared distances using expressions
    val joined = df.crossJoin(centersDF)

    // Compute squared Euclidean distance using array operations
    val distanceUDF = udf { (features: Vector, center: Vector) =>
      val fArr = features.toArray
      val cArr = center.toArray
      var sum = 0.0
      var i = 0
      while (i < fArr.length) {
        val diff = fArr(i) - cArr(i)
        sum += diff * diff
        i += 1
      }
      sum * 0.5
    }

    val withDistances = joined
      .withColumn("distance", distanceUDF(col(featuresCol), col("center")))

    // Find minimum distance cluster for each point
    // Use window function for efficiency
    import org.apache.spark.sql.expressions.Window
    val windowSpec = Window.partitionBy(df.columns.map(col): _*)

    val withRank = withDistances
      .withColumn("rank", row_number().over(windowSpec.orderBy("distance")))
      .filter(col("rank") === 1)
      .drop("center", "distance", "rank")
      .withColumnRenamed("clusterId", "cluster")

    withRank
  }
}

/**
 * Auto-selecting assignment strategy.
 *
 * Chooses between BroadcastUDF and SECrossJoin based on kernel and data size.
 */
class AutoAssignment extends AssignmentStrategy with Logging {

  private val broadcastStrategy = new BroadcastUDFAssignment()
  private val seStrategy = new SECrossJoinAssignment()

  override def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: BregmanKernel): DataFrame = {

    if (kernel.supportsExpressionOptimization) {
      logInfo("AutoAssignment: using SECrossJoin for Squared Euclidean")
      seStrategy.assign(df, featuresCol, weightCol, centers, kernel)
    } else {
      logInfo(s"AutoAssignment: using BroadcastUDF for ${kernel.name}")
      broadcastStrategy.assign(df, featuresCol, weightCol, centers, kernel)
    }
  }
}

/**
 * Strategy for updating cluster centers.
 *
 * Computes new centers based on assigned points using gradient-based aggregation.
 */
trait UpdateStrategy extends Serializable {
  /**
   * Compute new cluster centers from assignments.
   *
   * @param assigned    DataFrame with "cluster" column
   * @param featuresCol name of features column
   * @param weightCol   optional weight column
   * @param k           number of clusters
   * @param kernel      Bregman kernel
   * @return new cluster centers (may have fewer than k if some clusters are empty)
   */
  def update(
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      k: Int,
      kernel: BregmanKernel): Array[Array[Double]]
}

/**
 * Gradient mean update strategy using UDAF.
 *
 * Computes μ = invGrad(∑_i w_i · grad(x_i) / ∑_i w_i) for each cluster.
 */
class GradMeanUDAFUpdate extends UpdateStrategy with Logging {

  override def update(
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      k: Int,
      kernel: BregmanKernel): Array[Array[Double]] = {

    logDebug(s"GradMeanUDAFUpdate: computing centers for k=$k clusters")

    val spark = assigned.sparkSession
    val bcKernel = spark.sparkContext.broadcast(kernel)

    // UDF to compute gradient
    val gradUDF = udf { (features: Vector) =>
      bcKernel.value.grad(features)
    }

    // Add gradient column
    val withGrad = assigned.withColumn("gradient", gradUDF(col(featuresCol)))

    // Add weight column if not present
    val withWeight = weightCol match {
      case Some(col) => withGrad
      case None => withGrad.withColumn("weight", lit(1.0))
    }
    val actualWeightCol = weightCol.getOrElse("weight")

    // Aggregate using RDD for weighted vector sum
    val withWeightRDD = withWeight
      .select("cluster", "gradient", actualWeightCol)
      .rdd
      .map { row =>
        val cluster = row.getInt(0)
        val grad = row.getAs[Vector](1)
        val weight = row.getDouble(2)
        (cluster, (grad, weight))
      }

    val clusterSums = withWeightRDD
      .aggregateByKey((Array.empty[Double], 0.0))(
        seqOp = { case ((gradSum, weightSum), (grad, weight)) =>
          val gArr = grad.toArray
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
    val centers = (0 until k).flatMap { clusterId =>
      clusterSums.get(clusterId).map { case (gradSum, weightSum) =>
        val meanGrad = gradSum.map(_ / weightSum)
        kernel.invGrad(Vectors.dense(meanGrad)).toArray
      }
    }.toArray

    logDebug(s"GradMeanUDAFUpdate: computed ${centers.length} non-empty centers")

    centers
  }
}

/**
 * Strategy for handling empty clusters.
 */
trait EmptyClusterHandler extends Serializable {
  /**
   * Handle empty clusters by reseeding or dropping.
   *
   * @param assigned    DataFrame with "cluster" assignments
   * @param featuresCol name of features column
   * @param weightCol   optional weight column
   * @param centers     computed centers (may have fewer than k)
   * @param originalDF  original DataFrame for reseeding
   * @param kernel      Bregman kernel
   * @return (final centers with k elements, number of empty clusters handled)
   */
  def handle(
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      originalDF: DataFrame,
      kernel: BregmanKernel): (Array[Array[Double]], Int)
}

/**
 * Reseed empty clusters with random points.
 */
class ReseedRandomHandler(seed: Long = System.currentTimeMillis())
    extends EmptyClusterHandler with Logging {

  override def handle(
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      originalDF: DataFrame,
      kernel: BregmanKernel): (Array[Array[Double]], Int) = {

    val k = centers.length

    // Check if we have all k centers
    if (centers.length == k) {
      return (centers, 0)
    }

    val numEmpty = k - centers.length
    logWarning(s"ReseedRandomHandler: reseeding $numEmpty empty clusters")

    // Sample random points to fill empty clusters
    val fraction = math.min(1.0, (numEmpty * 10.0) / originalDF.count().toDouble)
    val samples = originalDF
      .sample(withReplacement = false, fraction, seed)
      .select(featuresCol)
      .limit(numEmpty)
      .collect()
      .map(_.getAs[Vector](0).toArray)

    val finalCenters = centers ++ samples
    (finalCenters.take(k), numEmpty)
  }
}

/**
 * Drop empty clusters (return fewer than k centers).
 */
class DropEmptyClustersHandler extends EmptyClusterHandler with Logging {

  override def handle(
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      originalDF: DataFrame,
      kernel: BregmanKernel): (Array[Array[Double]], Int) = {

    (centers, 0)  // Just return what we have
  }
}

/**
 * Strategy for checking convergence.
 */
trait ConvergenceCheck extends Serializable {
  /**
   * Check convergence and compute statistics.
   *
   * @param oldCenters  previous centers
   * @param newCenters  new centers
   * @param assigned    DataFrame with assignments
   * @param featuresCol name of features column
   * @param weightCol   optional weight column
   * @param kernel      Bregman kernel
   * @return (max center movement, total distortion)
   */
  def check(
      oldCenters: Array[Array[Double]],
      newCenters: Array[Array[Double]],
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: BregmanKernel): (Double, Double)
}

/**
 * Movement-based convergence check.
 *
 * Computes maximum L2 movement of any center and total distortion.
 */
class MovementConvergence extends ConvergenceCheck with Logging {

  override def check(
      oldCenters: Array[Array[Double]],
      newCenters: Array[Array[Double]],
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: BregmanKernel): (Double, Double) = {

    // Compute max movement
    val movement = oldCenters.zip(newCenters).map { case (old, neu) =>
      val diff = old.zip(neu).map { case (a, b) => a - b }
      math.sqrt(diff.map(d => d * d).sum)
    }.max

    // Compute total distortion
    val spark = assigned.sparkSession
    val bcCenters = spark.sparkContext.broadcast(newCenters)
    val bcKernel = spark.sparkContext.broadcast(kernel)

    val distortionUDF = udf { (features: Vector, clusterId: Int) =>
      val center = Vectors.dense(bcCenters.value(clusterId))
      bcKernel.value.divergence(features, center)
    }

    val actualWeightCol = weightCol.getOrElse("weight")
    val withWeight = if (assigned.columns.contains(actualWeightCol)) {
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

/**
 * Input validation strategy.
 */
trait InputValidator extends Serializable {
  /**
   * Validate input DataFrame.
   *
   * @param df          input DataFrame
   * @param featuresCol name of features column
   * @param weightCol   optional weight column
   * @param kernel      Bregman kernel
   * @throws IllegalArgumentException if validation fails
   */
  def validate(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: BregmanKernel): Unit
}

/**
 * Standard input validator.
 */
class StandardInputValidator extends InputValidator with Logging {

  override def validate(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: BregmanKernel): Unit = {

    require(df.columns.contains(featuresCol),
      s"Features column '$featuresCol' not found in DataFrame")

    weightCol.foreach { col =>
      require(df.columns.contains(col),
        s"Weight column '$col' not found in DataFrame")
    }

    // Check feature column type - try to get first row to validate it's a Vector
    val firstRow = df.select(featuresCol).first()
    require(firstRow.get(0).isInstanceOf[Vector],
      s"Features column must contain Vector type")

    // Basic count check
    val count = df.count()
    if (count == 0) {
      throw new IllegalArgumentException("Input DataFrame is empty")
    }

    logInfo(s"Validation passed for $count points with kernel ${kernel.name}")
  }
}
