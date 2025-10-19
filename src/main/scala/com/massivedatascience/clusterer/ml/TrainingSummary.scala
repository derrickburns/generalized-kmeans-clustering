package com.massivedatascience.clusterer.ml

import org.apache.spark.sql.{ DataFrame, Row, SparkSession }
import org.apache.spark.sql.types._
import java.time.Instant

/** Summary of a clustering model's training process.
  *
  * Provides diagnostic information about how the model was trained, including convergence metrics,
  * iteration history, and performance statistics. Available via `model.summary` after training.
  *
  * Note: Summary is only available for models that were trained in the current session. Models
  * loaded from disk have `hasSummary = false` and will throw `NoSuchElementException` when
  * accessing `summary`.
  *
  * @param algorithm
  *   algorithm name (e.g., "GeneralizedKMeans", "XMeans")
  * @param k
  *   number of clusters requested
  * @param effectiveK
  *   actual number of non-empty clusters in final model
  * @param dim
  *   feature dimensionality
  * @param numPoints
  *   number of training points
  * @param iterations
  *   number of iterations performed
  * @param converged
  *   whether algorithm converged (vs hitting max iterations)
  * @param distortionHistory
  *   total distortion (sum of squared distances) at each iteration
  * @param movementHistory
  *   maximum center movement (L2 distance) at each iteration
  * @param assignmentStrategy
  *   strategy used for point-to-center assignment (e.g., "SECrossJoin", "BroadcastUDF")
  * @param divergence
  *   divergence function used (e.g., "squaredEuclidean", "kl")
  * @param elapsedMillis
  *   total training time in milliseconds
  * @param trainedAt
  *   timestamp when training completed
  */
case class TrainingSummary(
    algorithm: String,
    k: Int,
    effectiveK: Int,
    dim: Int,
    numPoints: Long,
    iterations: Int,
    converged: Boolean,
    distortionHistory: Array[Double],
    movementHistory: Array[Double],
    assignmentStrategy: String,
    divergence: String,
    elapsedMillis: Long,
    trainedAt: Instant = Instant.now()
) {

  /** Final distortion (last entry in history) */
  def finalDistortion: Double = if (distortionHistory.nonEmpty) distortionHistory.last else 0.0

  /** Average time per iteration in milliseconds */
  def avgIterationMillis: Double = if (iterations > 0) elapsedMillis.toDouble / iterations else 0.0

  /** Whether any clusters were empty in final result */
  def hasEmptyClusters: Boolean = effectiveK < k

  /** Convert summary to DataFrame for easy analysis.
    *
    * Returns a single-row DataFrame with all summary fields as columns.
    */
  def toDF(spark: SparkSession): DataFrame = {
    val schema = StructType(
      Seq(
        StructField("algorithm", StringType, nullable = false),
        StructField("k", IntegerType, nullable = false),
        StructField("effectiveK", IntegerType, nullable = false),
        StructField("dim", IntegerType, nullable = false),
        StructField("numPoints", LongType, nullable = false),
        StructField("iterations", IntegerType, nullable = false),
        StructField("converged", BooleanType, nullable = false),
        StructField("finalDistortion", DoubleType, nullable = false),
        StructField("assignmentStrategy", StringType, nullable = false),
        StructField("divergence", StringType, nullable = false),
        StructField("elapsedMillis", LongType, nullable = false),
        StructField("avgIterationMillis", DoubleType, nullable = false),
        StructField("hasEmptyClusters", BooleanType, nullable = false),
        StructField("trainedAt", TimestampType, nullable = false),
        StructField("distortionHistory", ArrayType(DoubleType), nullable = false),
        StructField("movementHistory", ArrayType(DoubleType), nullable = false)
      )
    )

    val row = Row(
      algorithm,
      k,
      effectiveK,
      dim,
      numPoints,
      iterations,
      converged,
      finalDistortion,
      assignmentStrategy,
      divergence,
      elapsedMillis,
      avgIterationMillis,
      hasEmptyClusters,
      java.sql.Timestamp.from(trainedAt),
      distortionHistory.toSeq,
      movementHistory.toSeq
    )

    spark.createDataFrame(spark.sparkContext.parallelize(Seq(row)), schema)
  }

  /** Pretty-print summary for console output */
  override def toString: String = {
    val convergenceMsg = if (converged) "converged" else "max iterations reached"
    val emptyMsg       = if (hasEmptyClusters) s" (${k - effectiveK} empty)" else ""

    s"""TrainingSummary($algorithm)
       |  Clusters: $effectiveK/$k$emptyMsg
       |  Iterations: $iterations ($convergenceMsg)
       |  Final distortion: ${f"$finalDistortion%.6f"}
       |  Divergence: $divergence
       |  Assignment strategy: $assignmentStrategy
       |  Training time: ${elapsedMillis}ms (${f"$avgIterationMillis%.1f"}ms/iter)
       |  Data: $numPoints points × $dim dimensions
       |  Trained at: $trainedAt""".stripMargin
  }

  /** Summary of convergence behavior for logging */
  def convergenceReport: String = {
    val distortionChange =
      if (distortionHistory.length >= 2) {
        val initial    = distortionHistory.head
        val finalValue = distortionHistory.last
        val pct        = if (initial > 0) 100.0 * (initial - finalValue) / initial else 0.0
        f"${initial}%.3f → ${finalValue}%.3f (${pct}%.1f%% reduction)"
      } else "N/A"

    val maxMovement = if (movementHistory.nonEmpty) f"${movementHistory.max}%.6f" else "N/A"
    val finalMove   = if (movementHistory.nonEmpty) f"${movementHistory.last}%.6f" else "N/A"

    s"""Convergence Report:
       |  Distortion: $distortionChange
       |  Max center movement: $maxMovement
       |  Final center movement: $finalMove
       |  Iterations: $iterations""".stripMargin
  }
}

object TrainingSummary {

  /** Create summary from LloydResult and additional metadata.
    *
    * @param algorithm
    *   algorithm name
    * @param result
    *   Lloyd's algorithm result
    * @param k
    *   requested number of clusters
    * @param dim
    *   feature dimensionality
    * @param numPoints
    *   number of training points
    * @param assignmentStrategy
    *   assignment strategy used
    * @param divergence
    *   divergence function name
    * @param elapsedMillis
    *   training time in milliseconds
    * @return
    *   training summary
    */
  def fromLloydResult(
      algorithm: String,
      result: com.massivedatascience.clusterer.ml.df.LloydResult,
      k: Int,
      dim: Int,
      numPoints: Long,
      assignmentStrategy: String,
      divergence: String,
      elapsedMillis: Long
  ): TrainingSummary = {
    TrainingSummary(
      algorithm = algorithm,
      k = k,
      effectiveK = result.centers.length,
      dim = dim,
      numPoints = numPoints,
      iterations = result.iterations,
      converged = result.converged,
      distortionHistory = result.distortionHistory,
      movementHistory = result.movementHistory,
      assignmentStrategy = assignmentStrategy,
      divergence = divergence,
      elapsedMillis = elapsedMillis,
      trainedAt = Instant.now()
    )
  }
}
