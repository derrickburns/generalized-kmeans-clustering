package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{ DefaultParamsReadable, DefaultParamsWritable, Identifiable }
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import scala.util.Random

/** Generalized K-Means clustering with pluggable Bregman divergences.
  *
  * This estimator implements Lloyd's algorithm for clustering with any Bregman divergence. Unlike
  * MLlib's KMeans (which only supports Squared Euclidean distance), this supports multiple
  * distance measures suitable for different data types.
  *
  * ==Supported Divergences==
  *
  * | Divergence | Use Case | Domain Requirement |
  * |------------|----------|-------------------|
  * | `squaredEuclidean` | General clustering, continuous data | Any finite values |
  * | `kl` | Probability distributions, topic models, TF-IDF | Non-negative values |
  * | `itakuraSaito` | Spectral analysis, audio, power spectra | Strictly positive values |
  * | `generalizedI` | Count data, non-negative matrices | Non-negative values |
  * | `logistic` | Binary probabilities, proportions | Values in [0, 1] |
  * | `l1` / `manhattan` | Outlier-robust clustering (K-Medians) | Any finite values |
  * | `spherical` / `cosine` | Text embeddings, document vectors | Non-zero vectors |
  *
  * ==Domain Validation==
  *
  * For divergences with domain requirements (KL, Itakura-Saito, etc.), the algorithm validates
  * input data at fit time. Use `setSmoothing()` to add a small constant to avoid numerical issues
  * with zeros or near-zero values.
  *
  * ==Algorithm==
  *
  * The algorithm proceeds as follows:
  *   1. '''Initialization''': Select initial centers using k-means|| (default) or random sampling
  *   2. '''Assignment''': Assign each point to the nearest center using the specified divergence
  *   3. '''Update''': Recompute centers as the Bregman centroid of assigned points
  *   4. '''Convergence''': Repeat until centers move less than `tol` or `maxIter` is reached
  *
  * ==Performance Considerations==
  *
  * - For '''Squared Euclidean''', the `crossJoin` assignment strategy uses vectorized Spark SQL
  *   operations and is significantly faster than the UDF-based approach.
  * - For '''large k × dim''', consider using checkpointing to avoid lineage explosion.
  * - For '''high-dimensional sparse data''', consider using `spherical` divergence which
  *   normalizes vectors and focuses on directional similarity.
  *
  * ==Example Usage==
  *
  * '''Basic clustering with Squared Euclidean:'''
  * {{{
  * val kmeans = new GeneralizedKMeans()
  *   .setK(5)
  *   .setMaxIter(20)
  *   .setFeaturesCol("features")
  *
  * val model = kmeans.fit(dataset)
  * val predictions = model.transform(dataset)
  * }}}
  *
  * '''Clustering probability distributions with KL divergence:'''
  * {{{
  * val kmeans = new GeneralizedKMeans()
  *   .setK(10)
  *   .setDivergence("kl")
  *   .setSmoothing(1e-6)  // Add smoothing for numerical stability
  *   .setMaxIter(50)
  *
  * val model = kmeans.fit(probabilityData)
  * }}}
  *
  * '''Clustering text embeddings with cosine similarity:'''
  * {{{
  * val kmeans = new GeneralizedKMeans()
  *   .setK(20)
  *   .setDivergence("spherical")  // or "cosine"
  *   .setMaxIter(30)
  *
  * val model = kmeans.fit(embeddingsData)
  * }}}
  *
  * '''Weighted clustering:'''
  * {{{
  * val kmeans = new GeneralizedKMeans()
  *   .setK(5)
  *   .setWeightCol("importance")
  *
  * val model = kmeans.fit(weightedData)
  * }}}
  *
  * @see [[GeneralizedKMeansModel]] for the trained model
  * @see [[BisectingKMeans]] for hierarchical divisive clustering
  * @see [[SoftKMeans]] for fuzzy/soft clustering with membership probabilities
  * @see [[StreamingKMeans]] for online/streaming clustering
  *
  * @note For reproducible results, set the seed using `setSeed()`.
  * @note Empty clusters are handled according to `emptyClusterStrategy` (default: reseed with random points).
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

  /** Sets the Bregman divergence function. Options: "squaredEuclidean", "kl", "itakuraSaito",
    * "generalizedI", "logistic". Default: "squaredEuclidean".
    */
  def setDivergence(value: String): this.type = set(divergence, value)

  /** Sets the smoothing parameter for divergences that need it (KL, Itakura-Saito, etc). Default:
    * 1e-10.
    */
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

  /** Sets the assignment strategy for mapping points to clusters. Options: "auto" (default),
    * "broadcast", "crossJoin". "auto" chooses the best strategy based on the divergence.
    */
  def setAssignmentStrategy(value: String): this.type = set(assignmentStrategy, value)

  /** Sets the strategy for handling empty clusters. Options: "reseedRandom" (default), "dropEmpty".
    * "reseedRandom" fills empty clusters with random points.
    */
  def setEmptyClusterStrategy(value: String): this.type = set(emptyClusterStrategy, value)

  /** Sets the checkpoint interval (0 to disable). Default: 10. */
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** Sets the checkpoint directory for intermediate results. */
  def setCheckpointDir(value: String): this.type = set(checkpointDir, value)

  /** Sets the initialization algorithm. Options: "k-means||" (default), "random". "k-means||" is
    * the parallel variant of k-means++.
    */
  def setInitMode(value: String): this.type = set(initMode, value)

  /** Sets the number of steps for k-means|| initialization. Default: 2. */
  def setInitSteps(value: Int): this.type = set(initSteps, value)

  /** Sets an optional column name to store distances to assigned centers. */
  def setDistanceCol(value: String): this.type = set(distanceCol, value)

  override def fit(dataset: Dataset[_]): GeneralizedKMeansModel = {
    transformSchema(dataset.schema, logging = true)

    val df          = dataset.toDF()
    val numFeatures = df.select($(featuresCol)).first().getAs[Vector](0).size

    logInfo(
      s"Training GeneralizedKMeans with k=${$(k)}, maxIter=${$(maxIter)}, " +
        s"divergence=${$(divergence)}, numFeatures=$numFeatures"
    )

    // Validate input data domain requirements for the selected divergence
    com.massivedatascience.util.DivergenceDomainValidator.validateDataFrame(
      df,
      $(featuresCol),
      $(divergence),
      maxSamples = Some(1000) // Check first 1000 rows for performance
    )

    // Create kernel
    val kernel = createKernel($(divergence), $(smoothing))

    // Initialize centers
    val initialCenters = initializeCenters(df, $(featuresCol), getWeightColOpt, kernel)

    logInfo(s"Initialized ${initialCenters.length} centers using ${$(initMode)}")

    // Create strategies
    val assigner     = createAssignmentStrategy($(assignmentStrategy))
    val updater      = createUpdateStrategy($(divergence))
    val emptyHandler = createEmptyClusterHandler($(emptyClusterStrategy), $(seed))
    val convergence  = new MovementConvergence()
    val validator    = new StandardInputValidator()

    logInfo(
      s"Strategy selection: assignment=${assigner.getClass.getSimpleName}, " +
        s"update=${updater.getClass.getSimpleName}, emptyHandler=${emptyHandler.getClass.getSimpleName}"
    )

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
    val iterator  = new DefaultLloydsIterator()
    val startTime = System.currentTimeMillis()
    val result    = iterator.run(df, $(featuresCol), getWeightColOpt, initialCenters, config)
    val elapsed   = System.currentTimeMillis() - startTime

    logInfo(
      s"Training completed: iterations=${result.iterations}, " +
        s"converged=${result.converged}, finalDistortion=${result.distortionHistory.lastOption.getOrElse(0.0)}, " +
        s"elapsed=${elapsed}ms"
    )

    // Create model
    val model = new GeneralizedKMeansModel(uid, result.centers, kernel.name)
    copyValues(model)

    // Attach training summary
    val summary = TrainingSummary.fromLloydResult(
      algorithm = "GeneralizedKMeans",
      result = result,
      k = $(k),
      dim = numFeatures,
      numPoints = df.count(),
      assignmentStrategy = assigner.getClass.getSimpleName,
      divergence = $(divergence),
      elapsedMillis = elapsed
    )
    model.trainingSummary = Some(summary)

    logInfo(s"Training summary:\n${summary.convergenceReport}")

    model
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): GeneralizedKMeans = defaultCopy(extra)

  /** Get weight column as Option.
    */
  private def getWeightColOpt: Option[String] = {
    if (hasWeightCol) Some($(weightCol)) else None
  }

  /** Create Bregman kernel based on divergence name.
    */
  private def createKernel(divName: String, smooth: Double): BregmanKernel = {
    divName match {
      case "squaredEuclidean"     => new SquaredEuclideanKernel()
      case "kl"                   => new KLDivergenceKernel(smooth)
      case "itakuraSaito"         => new ItakuraSaitoKernel(smooth)
      case "generalizedI"         => new GeneralizedIDivergenceKernel(smooth)
      case "logistic"             => new LogisticLossKernel(smooth)
      case "l1" | "manhattan"     => new L1Kernel()
      case "spherical" | "cosine" => new SphericalKernel()
      case _                      =>
        throw new IllegalArgumentException(
          s"Unknown divergence: '$divName'. " +
            s"Valid options: squaredEuclidean, kl, itakuraSaito, generalizedI, logistic, l1, manhattan, spherical, cosine"
        )
    }
  }

  /** Create assignment strategy.
    */
  private def createAssignmentStrategy(strategy: String): AssignmentStrategy = {
    strategy match {
      case "broadcast" => new BroadcastUDFAssignment()
      case "crossJoin" => new SECrossJoinAssignment()
      case "auto"      => new AutoAssignment()
      case _           =>
        throw new IllegalArgumentException(
          s"Unknown assignment strategy: '$strategy'. Valid options: auto, broadcast, crossJoin"
        )
    }
  }

  /** Create update strategy based on divergence. L1/Manhattan distance uses MedianUpdateStrategy,
    * others use GradMeanUDAFUpdate.
    */
  private def createUpdateStrategy(divName: String): UpdateStrategy = {
    divName match {
      case "l1" | "manhattan" => new MedianUpdateStrategy()
      case _                  => new GradMeanUDAFUpdate()
    }
  }

  /** Create empty cluster handler.
    */
  private def createEmptyClusterHandler(strategy: String, seed: Long): EmptyClusterHandler = {
    strategy match {
      case "reseedRandom" => new ReseedRandomHandler(seed)
      case "drop"         => new DropEmptyClustersHandler()
      case _              =>
        throw new IllegalArgumentException(
          s"Unknown empty cluster strategy: '$strategy'. Valid options: reseedRandom, drop"
        )
    }
  }

  /** Initialize cluster centers.
    */
  private def initializeCenters(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: BregmanKernel
  ): Array[Array[Double]] = {

    $(initMode) match {
      case "random"    => initializeRandom(df, featuresCol, $(k), $(seed))
      case "k-means||" =>
        initializeKMeansPlusPlus(df, featuresCol, weightCol, $(k), $(initSteps), $(seed), kernel)
      case _           =>
        throw new IllegalArgumentException(
          s"Unknown init mode: '${$(initMode)}'. Valid options: random, k-means||"
        )
    }
  }

  /** Random initialization: sample k random points.
    */
  private def initializeRandom(
      df: DataFrame,
      featuresCol: String,
      k: Int,
      seed: Long
  ): Array[Array[Double]] = {

    val fraction = math.min(1.0, (k * 10.0) / df.count().toDouble)
    df.select(featuresCol)
      .sample(withReplacement = false, fraction, seed)
      .limit(k)
      .collect()
      .map(_.getAs[Vector](0).toArray)
  }

  /** K-means|| initialization (simplified version).
    *
    * This is a simplified implementation. A full implementation would use the parallel k-means++
    * algorithm with oversampling.
    */
  private def initializeKMeansPlusPlus(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      k: Int,
      steps: Int,
      seed: Long,
      kernel: BregmanKernel
  ): Array[Array[Double]] = {

    val rand     = new Random(seed)
    val bcKernel = df.sparkSession.sparkContext.broadcast(kernel)

    // Step 1: Select first center uniformly at random
    val allPoints = df.select(featuresCol).collect()
    require(
      allPoints.nonEmpty,
      s"Dataset is empty. Cannot initialize k-means|| with k=$k on an empty dataset."
    )

    val firstCenter = allPoints(rand.nextInt(allPoints.length)).getAs[Vector](0).toArray

    var centers = Array(firstCenter)

    // Steps 2-k: Iteratively select centers with probability proportional to distance^2
    for (step <- 1 until math.min(k, steps + 1)) {
      val bcCenters = df.sparkSession.sparkContext.broadcast(centers)

      // Compute distances to nearest center
      val distanceUDF = udf { (features: Vector) =>
        val ctrs    = bcCenters.value
        val kern    = bcKernel.value
        var minDist = Double.PositiveInfinity
        var i       = 0
        while (i < ctrs.length) {
          val center = Vectors.dense(ctrs(i))
          val dist   = kern.divergence(features, center)
          if (dist < minDist) {
            minDist = dist
          }
          i += 1
        }
        minDist
      }

      val withDistances =
        df.select(featuresCol).withColumn("distance", distanceUDF(col(featuresCol)))

      // Sample proportional to distance^2
      val numToSample = math.min(k - centers.length, 2 * k)
      val samples     = withDistances
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
      val updater  = new GradMeanUDAFUpdate()
      centers = updater.update(assigned, featuresCol, weightCol, k, kernel)
    }

    centers.take(k)
  }
}

object GeneralizedKMeans extends DefaultParamsReadable[GeneralizedKMeans] {
  override def load(path: String): GeneralizedKMeans = super.load(path)
}
