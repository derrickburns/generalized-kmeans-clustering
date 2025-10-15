package com.massivedatascience.clusterer.ml

import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.sql.types.{DoubleType, StructType}

/**
 * Params for GeneralizedKMeans and GeneralizedKMeansModel.
 */
trait GeneralizedKMeansParams extends Params
    with HasFeaturesCol
    with HasPredictionCol
    with HasMaxIter
    with HasTol
    with HasSeed {

  /**
   * Number of clusters to create (k).
   * Must be > 1.
   * Default: 2
   */
  final val k = new IntParam(this, "k", "Number of clusters", ParamValidators.gt(1))

  def getK: Int = $(k)

  /**
   * Bregman divergence kernel name.
   * Supported values:
   * - "squaredEuclidean" (default)
   * - "kl" (Kullback-Leibler)
   * - "itakuraSaito"
   * - "generalizedI"
   * - "logistic"
   * - "l1" or "manhattan" (L1 for K-Medians)
   * Default: "squaredEuclidean"
   */
  final val divergence = new Param[String](
    this,
    "divergence",
    "Bregman divergence kernel",
    ParamValidators.inArray(Array(
      "squaredEuclidean",
      "kl",
      "itakuraSaito",
      "generalizedI",
      "logistic",
      "l1",
      "manhattan"
    )))

  def getDivergence: String = $(divergence)

  /**
   * Smoothing parameter for divergences that need it (KL, IS, etc.).
   * Must be > 0.
   * Default: 1e-10
   */
  final val smoothing = new DoubleParam(
    this,
    "smoothing",
    "Smoothing parameter for divergences",
    ParamValidators.gt(0.0))

  def getSmoothing: Double = $(smoothing)

  /**
   * Name of the optional weight column.
   * If not set, all instances are treated equally with weight 1.0.
   * Default: None
   */
  final val weightCol = new Param[String](this, "weightCol", "Weight column name")

  def getWeightCol: String = $(weightCol)

  def hasWeightCol: Boolean = isSet(weightCol)

  /**
   * Assignment strategy: "broadcast", "crossJoin", or "auto".
   * - broadcast: Use UDF with broadcasted centers (works with all kernels)
   * - crossJoin: Use expression-based cross-join (only for squaredEuclidean)
   * - auto: Automatically select based on kernel
   * Default: "auto"
   */
  final val assignmentStrategy = new Param[String](
    this,
    "assignmentStrategy",
    "Assignment strategy",
    ParamValidators.inArray(Array("broadcast", "crossJoin", "auto")))

  def getAssignmentStrategy: String = $(assignmentStrategy)

  /**
   * Empty cluster handling strategy: "reseedRandom" or "drop".
   * - reseedRandom: Reseed empty clusters with random points
   * - drop: Drop empty clusters (return fewer than k)
   * Default: "reseedRandom"
   */
  final val emptyClusterStrategy = new Param[String](
    this,
    "emptyClusterStrategy",
    "Empty cluster handling strategy",
    ParamValidators.inArray(Array("reseedRandom", "drop")))

  def getEmptyClusterStrategy: String = $(emptyClusterStrategy)

  /**
   * Checkpoint interval (iterations between checkpoints).
   * Set to 0 to disable checkpointing.
   * Must be >= 0.
   * Default: 10
   */
  final val checkpointInterval = new IntParam(
    this,
    "checkpointInterval",
    "Checkpoint interval",
    ParamValidators.gtEq(0))

  def getCheckpointInterval: Int = $(checkpointInterval)

  /**
   * Checkpoint directory. If not set, uses Spark's default.
   * Default: None
   */
  final val checkpointDir = new Param[String](this, "checkpointDir", "Checkpoint directory")

  def getCheckpointDir: String = $(checkpointDir)

  def hasCheckpointDir: Boolean = isSet(checkpointDir)

  /**
   * Initialization mode: "random" or "k-means||".
   * - random: Random selection of k points
   * - k-means||: K-means|| parallel initialization
   * Default: "k-means||"
   */
  final val initMode = new Param[String](
    this,
    "initMode",
    "Initialization mode",
    ParamValidators.inArray(Array("random", "k-means||")))

  def getInitMode: String = $(initMode)

  /**
   * Number of steps for k-means|| initialization.
   * Only used when initMode = "k-means||".
   * Must be > 0.
   * Default: 2
   */
  final val initSteps = new IntParam(
    this,
    "initSteps",
    "Number of steps for k-means|| initialization",
    ParamValidators.gt(0))

  def getInitSteps: Int = $(initSteps)

  /**
   * Optional column name for distances to cluster centers in transform output.
   * If set, adds a column with distance from point to assigned center.
   * Default: None
   */
  final val distanceCol = new Param[String](this, "distanceCol", "Distance column name")

  def getDistanceCol: String = $(distanceCol)

  def hasDistanceCol: Boolean = isSet(distanceCol)

  /**
   * Validates that the features column has the correct type.
   */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val featuresType = schema($(featuresCol)).dataType
    require(
      featuresType.typeName == "vector",
      s"Features column must be of type Vector, got $featuresType")

    if (hasWeightCol) {
      val weightType = schema($(weightCol)).dataType
      require(
        weightType == DoubleType,
        s"Weight column must be of type Double, got $weightType")
    }

    schema
  }

  setDefault(
    k -> 2,
    divergence -> "squaredEuclidean",
    smoothing -> 1e-10,
    maxIter -> 20,
    tol -> 1e-4,
    assignmentStrategy -> "auto",
    emptyClusterStrategy -> "reseedRandom",
    checkpointInterval -> 10,
    initMode -> "k-means||",
    initSteps -> 2,
    featuresCol -> "features",
    predictionCol -> "prediction"
  )
}
