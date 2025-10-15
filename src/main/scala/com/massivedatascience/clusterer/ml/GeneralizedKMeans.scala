package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import scala.util.Random

/**
 * Generalized K-Means clustering with pluggable Bregman divergences.
 *
 * This estimator implements Lloyd's algorithm for clustering with any Bregman divergence.
 * Unlike MLlib's KMeans (which only supports Squared Euclidean distance), this supports:
 * - Squared Euclidean (L2)
 * - Kullback-Leibler divergence (for probability distributions)
 * - Itakura-Saito divergence (for spectral data)
 * - Generalized I-divergence (for count data)
 * - Logistic loss (for binary probabilities)
 *
 * Example usage:
 * {{{
 *   val kmeans = new GeneralizedKMeans()
 *     .setK(5)
 *     .setDivergence("kl")
 *     .setMaxIter(20)
 *     .setFeaturesCol("features")
 *     .setPredictionCol("cluster")
 *
 *   val model = kmeans.fit(dataset)
 *   val predictions = model.transform(dataset)
 * }}}
 */
class GeneralizedKMeans(override val uid: String)
    extends Estimator[GeneralizedKMeansModel]
    with GeneralizedKMeansParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("gkmeans"))

  // Parameter setters

  /** Sets the number of clusters to create (k must be > 1). Default: 2. */
  def setK(value: Int): this.type = set(k, value)

  /**
   * Sets the Bregman divergence function.
   * Options: "squaredEuclidean", "kl", "itakuraSaito", "generalizedI", "logistic".
   * Default: "squaredEuclidean".
   */
  def setDivergence(value: String): this.type = set(divergence, value)

  /** Sets the smoothing parameter for divergences that need it (KL, Itakura-Saito, etc). Default: 1e-10. */
  def setSmoothing(value: Double): this.type = set(smoothing, value)

  /** Sets the features column name. Default: "features". */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** Sets the prediction output column name. Default: "prediction". */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** Sets the optional weight column for weighted clustering. */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /** Sets the maximum number of iterations. Default: 20. */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** Sets the convergence tolerance (max center movement). Default: 1e-4. */
  def setTol(value: Double): this.type = set(tol, value)

  /** Sets the random seed for reproducibility. Default: current time. */
  def setSeed(value: Long): this.type = set(seed, value)

  /**
   * Sets the assignment strategy for mapping points to clusters.
   * Options: "auto" (default), "broadcast", "crossJoin".
   * "auto" chooses the best strategy based on the divergence.
   */
  def setAssignmentStrategy(value: String): this.type = set(assignmentStrategy, value)

  /**
   * Sets the strategy for handling empty clusters.
   * Options: "reseedRandom" (default), "dropEmpty".
   * "reseedRandom" fills empty clusters with random points.
   */
  def setEmptyClusterStrategy(value: String): this.type = set(emptyClusterStrategy, value)

  /** Sets the checkpoint interval (0 to disable). Default: 10. */
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** Sets the checkpoint directory for intermediate results. */
  def setCheckpointDir(value: String): this.type = set(checkpointDir, value)

  /**
   * Sets the initialization algorithm.
   * Options: "k-means||" (default), "random".
   * "k-means||" is the parallel variant of k-means++.
   */
  def setInitMode(value: String): this.type = set(initMode, value)

  /** Sets the number of steps for k-means|| initialization. Default: 2. */
  def setInitSteps(value: Int): this.type = set(initSteps, value)

  /** Sets an optional column name to store distances to assigned centers. */
  def setDistanceCol(value: String): this.type = set(distanceCol, value)

  override def fit(dataset: Dataset[_]): GeneralizedKMeansModel = {
    transformSchema(dataset.schema, logging = true)

    val df = dataset.toDF()
    val numFeatures = df.select($(featuresCol)).first().getAs[Vector](0).size

    logInfo(s"Training GeneralizedKMeans with k=${$(k)}, maxIter=${$(maxIter)}, " +
      s"divergence=${$(divergence)}, numFeatures=$numFeatures")

    // Create kernel
    val kernel = createKernel($(divergence), $(smoothing))

    // Initialize centers
    val initialCenters = initializeCenters(df, $(featuresCol), getWeightColOpt, kernel)

    logInfo(s"Initialized ${initialCenters.length} centers using ${$(initMode)}")

    // Create strategies
    val assigner = createAssignmentStrategy($(assignmentStrategy))
    val updater = createUpdateStrategy($(divergence))
    val emptyHandler = createEmptyClusterHandler($(emptyClusterStrategy), $(seed))
    val convergence = new MovementConvergence()
    val validator = new StandardInputValidator()

    // Create config
    val config = LloydsConfig(
      k = $(k),
      maxIter = $(maxIter),
      tol = $(tol),
      kernel = kernel,
      assigner = assigner,
      updater = updater,
      emptyHandler = emptyHandler,
      convergence = convergence,
      validator = validator,
      checkpointInterval = $(checkpointInterval),
      checkpointDir = if (hasCheckpointDir) Some($(checkpointDir)) else None
    )

    // Run Lloyd's algorithm
    val iterator = new DefaultLloydsIterator()
    val result = iterator.run(df, $(featuresCol), getWeightColOpt, initialCenters, config)

    logInfo(s"Training completed: iterations=${result.iterations}, " +
      s"converged=${result.converged}, finalDistortion=${result.distortionHistory.last}")

    // Create model
    val model = new GeneralizedKMeansModel(uid, result.centers, kernel.name)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): GeneralizedKMeans = defaultCopy(extra)

  /**
   * Get weight column as Option.
   */
  private def getWeightColOpt: Option[String] = {
    if (hasWeightCol) Some($(weightCol)) else None
  }

  /**
   * Create Bregman kernel based on divergence name.
   */
  private def createKernel(divName: String, smooth: Double): BregmanKernel = {
    divName match {
      case "squaredEuclidean" => new SquaredEuclideanKernel()
      case "kl" => new KLDivergenceKernel(smooth)
      case "itakuraSaito" => new ItakuraSaitoKernel(smooth)
      case "generalizedI" => new GeneralizedIDivergenceKernel(smooth)
      case "logistic" => new LogisticLossKernel(smooth)
      case "l1" | "manhattan" => new L1Kernel()
      case _ => throw new IllegalArgumentException(s"Unknown divergence: $divName")
    }
  }

  /**
   * Create assignment strategy.
   */
  private def createAssignmentStrategy(strategy: String): AssignmentStrategy = {
    strategy match {
      case "broadcast" => new BroadcastUDFAssignment()
      case "crossJoin" => new SECrossJoinAssignment()
      case "auto" => new AutoAssignment()
      case _ => throw new IllegalArgumentException(s"Unknown assignment strategy: $strategy")
    }
  }

  /**
   * Create update strategy based on divergence.
   * L1/Manhattan distance uses MedianUpdateStrategy, others use GradMeanUDAFUpdate.
   */
  private def createUpdateStrategy(divName: String): UpdateStrategy = {
    divName match {
      case "l1" | "manhattan" => new MedianUpdateStrategy()
      case _ => new GradMeanUDAFUpdate()
    }
  }

  /**
   * Create empty cluster handler.
   */
  private def createEmptyClusterHandler(strategy: String, seed: Long): EmptyClusterHandler = {
    strategy match {
      case "reseedRandom" => new ReseedRandomHandler(seed)
      case "drop" => new DropEmptyClustersHandler()
      case _ => throw new IllegalArgumentException(s"Unknown empty cluster strategy: $strategy")
    }
  }

  /**
   * Initialize cluster centers.
   */
  private def initializeCenters(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: BregmanKernel): Array[Array[Double]] = {

    $(initMode) match {
      case "random" => initializeRandom(df, featuresCol, $(k), $(seed))
      case "k-means||" => initializeKMeansPlusPlus(df, featuresCol, weightCol, $(k), $(initSteps), $(seed), kernel)
      case _ => throw new IllegalArgumentException(s"Unknown init mode: ${$(initMode)}")
    }
  }

  /**
   * Random initialization: sample k random points.
   */
  private def initializeRandom(
      df: DataFrame,
      featuresCol: String,
      k: Int,
      seed: Long): Array[Array[Double]] = {

    val fraction = math.min(1.0, (k * 10.0) / df.count().toDouble)
    df.select(featuresCol)
      .sample(withReplacement = false, fraction, seed)
      .limit(k)
      .collect()
      .map(_.getAs[Vector](0).toArray)
  }

  /**
   * K-means|| initialization (simplified version).
   *
   * This is a simplified implementation. A full implementation would use
   * the parallel k-means++ algorithm with oversampling.
   */
  private def initializeKMeansPlusPlus(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      k: Int,
      steps: Int,
      seed: Long,
      kernel: BregmanKernel): Array[Array[Double]] = {

    val rand = new Random(seed)
    val bcKernel = df.sparkSession.sparkContext.broadcast(kernel)

    // Step 1: Select first center uniformly at random
    val allPoints = df.select(featuresCol).collect()
    require(allPoints.nonEmpty, "Dataset is empty")

    val firstCenter = allPoints(rand.nextInt(allPoints.length))
      .getAs[Vector](0)
      .toArray

    var centers = Array(firstCenter)

    // Steps 2-k: Iteratively select centers with probability proportional to distance^2
    for (step <- 1 until math.min(k, steps + 1)) {
      val bcCenters = df.sparkSession.sparkContext.broadcast(centers)

      // Compute distances to nearest center
      val distanceUDF = udf { (features: Vector) =>
        val ctrs = bcCenters.value
        val kern = bcKernel.value
        var minDist = Double.PositiveInfinity
        var i = 0
        while (i < ctrs.length) {
          val center = Vectors.dense(ctrs(i))
          val dist = kern.divergence(features, center)
          if (dist < minDist) {
            minDist = dist
          }
          i += 1
        }
        minDist
      }

      val withDistances = df
        .select(featuresCol)
        .withColumn("distance", distanceUDF(col(featuresCol)))

      // Sample proportional to distance^2
      val numToSample = math.min(k - centers.length, 2 * k)
      val samples = withDistances
        .sample(withReplacement = false, numToSample.toDouble / df.count(), rand.nextLong())
        .collect()
        .map(_.getAs[Vector](0).toArray)

      centers = centers ++ samples.take(k - centers.length)

      bcCenters.destroy()

      logInfo(s"K-means|| step $step: selected ${centers.length} centers")
    }

    // If we have more than k centers, run one iteration of Lloyd's to reduce
    if (centers.length > k) {
      logInfo(s"Reducing ${centers.length} centers to $k using Lloyd's iteration")
      val assigner = new BroadcastUDFAssignment()
      val assigned = assigner.assign(df, featuresCol, weightCol, centers, kernel)
      val updater = new GradMeanUDAFUpdate()
      centers = updater.update(assigned, featuresCol, weightCol, k, kernel)
    }

    centers.take(k)
  }
}

object GeneralizedKMeans extends DefaultParamsReadable[GeneralizedKMeans] {
  override def load(path: String): GeneralizedKMeans = super.load(path)
}
