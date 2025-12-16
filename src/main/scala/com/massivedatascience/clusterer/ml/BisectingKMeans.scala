package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{ DefaultParamsReadable, DefaultParamsWritable, Identifiable }
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Parameters for Bisecting K-Means clustering.
  */
trait BisectingKMeansParams extends GeneralizedKMeansParams {

  /** Minimum divisible cluster size. Clusters with fewer points than this will not be split. Must
    * be >= 1. Default: 1
    */
  final val minDivisibleClusterSize = new IntParam(
    this,
    "minDivisibleClusterSize",
    "Minimum size for a cluster to be divisible",
    ParamValidators.gtEq(1)
  )

  def getMinDivisibleClusterSize: Int = $(minDivisibleClusterSize)

  setDefault(minDivisibleClusterSize -> 1)
}

/** Bisecting K-Means clustering with pluggable Bregman divergences.
  *
  * A hierarchical divisive clustering algorithm that produces a dendrogram-like structure.
  * Unlike standard k-means which starts with k random centers, bisecting k-means builds
  * clusters incrementally by recursive splitting.
  *
  * ==Algorithm==
  *
  *   1. Start with all points in one cluster
  *   2. Select the largest divisible cluster (size ≥ minDivisibleClusterSize)
  *   3. Split it into two using k=2 clustering with the selected divergence
  *   4. Repeat until reaching the target k clusters
  *
  * ==Advantages==
  *
  *   - '''More deterministic:''' Less sensitive to initialization than standard k-means
  *   - '''Balanced clusters:''' Natural handling of imbalanced cluster sizes
  *   - '''Faster for large k:''' Only splits locally, O(log k) full passes vs O(k) for k-means
  *   - '''Hierarchical structure:''' Implicit dendrogram can be extracted from split order
  *   - '''Quality:''' Often produces higher quality clusters than random initialization
  *
  * ==Divergences==
  *
  * Supports all Bregman divergences:
  *   - `squaredEuclidean` (default): Standard bisecting k-means
  *   - `kl`: Hierarchical topic/probability modeling
  *   - `itakuraSaito`: Spectral analysis with nested structure
  *   - `spherical`/`cosine`: Text/embedding hierarchical clustering
  *   - `l1`: Robust hierarchical clustering with median-based splits
  *
  * ==Example Usage==
  *
  * {{{
  * val bisecting = new BisectingKMeans()
  *   .setK(10)
  *   .setDivergence("squaredEuclidean")
  *   .setMaxIter(20)
  *   .setMinDivisibleClusterSize(5)  // Don't split clusters smaller than 5
  *
  * val model = bisecting.fit(dataset)
  * val predictions = model.transform(dataset)
  *
  * // Training summary shows number of splits performed
  * println(s"Performed ${model.summary.iterations} splits")
  * }}}
  *
  * ==When to Use==
  *
  * Prefer bisecting k-means when:
  *   - You need reproducible results
  *   - Cluster sizes are expected to be imbalanced
  *   - You want to analyze hierarchical structure
  *   - k is large (e.g., k > 50)
  *
  * @see [[GeneralizedKMeans]] for standard k-means with multiple initialization methods
  * @see [[GeneralizedKMeansModel]] for prediction and model persistence
  *
  * @param uid
  *   Unique identifier
  */
class BisectingKMeans(override val uid: String)
    extends Estimator[GeneralizedKMeansModel]
    with BisectingKMeansParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("bisecting_kmeans"))

  // Parameter setters

  /** Sets the number of leaf clusters to create (k must be > 1). Default: 2. */
  def setK(value: Int): this.type = set(k, value)

  /** Sets the Bregman divergence function. */
  def setDivergence(value: String): this.type = set(divergence, value)

  /** Sets the smoothing parameter. */
  def setSmoothing(value: Double): this.type = set(smoothing, value)

  /** Sets the features column name. */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** Sets the prediction output column name. */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** Sets the optional weight column. */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /** Sets the maximum number of iterations per split. */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** Sets the convergence tolerance. */
  def setTol(value: Double): this.type = set(tol, value)

  /** Sets the random seed. */
  def setSeed(value: Long): this.type = set(seed, value)

  /** Sets the minimum divisible cluster size. */
  def setMinDivisibleClusterSize(value: Int): this.type = set(minDivisibleClusterSize, value)

  /** Sets the optional distance column. */
  def setDistanceCol(value: String): this.type = set(distanceCol, value)

  override def fit(dataset: Dataset[_]): GeneralizedKMeansModel = {
    transformSchema(dataset.schema, logging = true)

    val df = dataset.toDF()

    logInfo(
      s"Training BisectingKMeans with k=${$(k)}, maxIter=${$(maxIter)}, " +
        s"divergence=${$(divergence)}, minDivisibleClusterSize=${$(minDivisibleClusterSize)}"
    )

    // Validate input data domain requirements for the selected divergence
    com.massivedatascience.util.DivergenceDomainValidator.validateDataFrame(
      df,
      $(featuresCol),
      $(divergence),
      maxSamples = Some(1000)
    )

    // Create kernel
    val kernel = createKernel($(divergence), $(smoothing))

    // Bisecting algorithm with timing
    val startTime                 = System.currentTimeMillis()
    val (finalCenters, numSplits) = bisectWithHistory(df, $(featuresCol), getWeightColOpt, kernel)
    val elapsedMillis             = System.currentTimeMillis() - startTime

    logInfo(s"BisectingKMeans completed with ${finalCenters.length} clusters in $numSplits splits")

    // Create model
    val model = new GeneralizedKMeansModel(uid, finalCenters, kernel.name)
    copyValues(model)

    // Create training summary
    // Note: For bisecting k-means, "iterations" represents the number of splits performed
    val numFeatures = finalCenters.headOption.map(_.length).getOrElse(0)
    val summary     = TrainingSummary(
      algorithm = "BisectingKMeans",
      k = $(k),
      effectiveK = finalCenters.length,
      dim = numFeatures,
      numPoints = df.count(),
      iterations = numSplits,
      converged = finalCenters.length == $(k), // Converged if we reached target k
      distortionHistory =
        Array.empty[Double],                   // Not tracked for bisecting (would need per-split tracking)
      movementHistory = Array.empty[Double],   // Not tracked for bisecting
      assignmentStrategy = "Bisecting",
      divergence = $(divergence),
      elapsedMillis = elapsedMillis
    )

    model.trainingSummary = Some(summary)
    model
  }

  /** Get weight column as Option.
    */
  private def getWeightColOpt: Option[String] = {
    if (hasWeightCol) Some($(weightCol)) else None
  }

  /** Bisecting clustering algorithm with split tracking.
    *
    * @return
    *   Tuple of (cluster centers, number of splits performed)
    */
  private def bisectWithHistory(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: BregmanKernel
  ): (Array[Array[Double]], Int) = {

    val targetK = $(k)
    val spark   = df.sparkSession

    // Start with all data in cluster 0
    var clusteredDF    = df.withColumn("cluster", lit(0))
    var clusterCenters = Map[Int, Array[Double]](
      0 -> computeCenter(clusteredDF.filter(col("cluster") === 0), featuresCol, weightCol, kernel)
    )
    var nextClusterId  = 1
    var numSplits      = 0

    // Cache for performance
    clusteredDF.cache()

    // Bisect until we reach target k
    while (clusterCenters.size < targetK) {
      // Find largest cluster to split
      val clusterSizes = clusteredDF
        .groupBy("cluster")
        .count()
        .collect()
        .map(row => (row.getInt(0), row.getLong(1)))
        .toMap

      val divisibleClusters = clusterSizes.filter { case (_, size) =>
        size >= $(minDivisibleClusterSize)
      }

      val largestClusterId = if (divisibleClusters.nonEmpty) {
        Some(divisibleClusters.maxBy(_._2)._1)
      } else {
        None
      }

      largestClusterId match {
        case None =>
          logWarning(
            s"No clusters large enough to split (minSize=${$(minDivisibleClusterSize)}). " +
              s"Stopping with ${clusterCenters.size} clusters."
          )
          clusteredDF.unpersist()
          return ((0 until clusterCenters.size).map(i => clusterCenters(i)).toArray, numSplits)

        case Some(clusterId) =>
          val clusterSize = clusterSizes(clusterId)
          logInfo(
            s"Splitting cluster $clusterId (size=$clusterSize) " +
              s"into clusters $clusterId and $nextClusterId"
          )

          // Extract data for this cluster
          val clusterData = clusteredDF.filter(col("cluster") === clusterId)

          // Split into 2 using k-means with k=2
          val (center1, center2) = splitCluster(clusterData, featuresCol, weightCol, kernel)

          // Reassign points in this cluster to the two new centers
          val bcCenter1 = spark.sparkContext.broadcast(center1)
          val bcCenter2 = spark.sparkContext.broadcast(center2)
          val bcKernel  = spark.sparkContext.broadcast(kernel)

          val assignUDF = udf { (features: Vector, oldCluster: Int) =>
            if (oldCluster == clusterId) {
              val c1    = Vectors.dense(bcCenter1.value)
              val c2    = Vectors.dense(bcCenter2.value)
              val dist1 = bcKernel.value.divergence(features, c1)
              val dist2 = bcKernel.value.divergence(features, c2)
              if (dist1 < dist2) clusterId else nextClusterId
            } else {
              oldCluster
            }
          }

          // Update cluster assignments
          val oldClusteredDF = clusteredDF
          val oldClusterCol  = oldClusteredDF("cluster") // Capture before dropping
          clusteredDF = oldClusteredDF
            .withColumn(
              "cluster_new",
              assignUDF(col(featuresCol), oldClusterCol)
            )
            .drop("cluster")
            .withColumnRenamed("cluster_new", "cluster")
          clusteredDF.cache()
          oldClusteredDF.unpersist()

          // Update centers and increment split count
          clusterCenters = clusterCenters + (clusterId -> center1) + (nextClusterId -> center2)
          nextClusterId += 1
          numSplits += 1
      }
    }

    clusteredDF.unpersist()

    logInfo(s"Bisecting completed with ${clusterCenters.size} clusters after $numSplits splits")

    // Return centers in order with split count
    ((0 until clusterCenters.size).map(i => clusterCenters(i)).toArray, numSplits)
  }

  /** Split a cluster into two using k=2 clustering.
    *
    * @param clusterData
    *   DataFrame containing points in the cluster
    * @param featuresCol
    *   Name of features column
    * @param weightCol
    *   Optional weight column
    * @param kernel
    *   Bregman kernel
    * @return
    *   Tuple of (center1, center2)
    */
  private def splitCluster(
      clusterData: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: BregmanKernel
  ): (Array[Double], Array[Double]) = {

    // Drop the "cluster" column if it exists to avoid conflicts with assignment strategy
    val cleanData = if (clusterData.columns.contains("cluster")) {
      clusterData.drop("cluster")
    } else {
      clusterData
    }

    // Initialize with two random points
    val sample = cleanData.select(featuresCol).limit(2).collect()
    if (sample.length < 2) {
      // Can't split - return same center twice
      val center = computeCenter(cleanData, featuresCol, weightCol, kernel)
      return (center, center)
    }

    var centers = Array(
      sample(0).getAs[Vector](0).toArray,
      sample(1).getAs[Vector](0).toArray
    )

    // Create strategies for k=2 clustering
    val assigner = createAssignmentStrategy("auto")
    val updater  = createUpdateStrategy($(divergence))

    // Run Lloyd's for a few iterations
    var iteration = 0
    var converged = false

    while (iteration < $(maxIter) && !converged) {
      // Assignment step
      val assigned = assigner.assign(cleanData, featuresCol, weightCol, centers, kernel)

      // Update step
      val newCenters = updater.update(assigned, featuresCol, weightCol, 2, kernel)

      // Check convergence
      if (newCenters.length == 2) {
        val movements = centers.zip(newCenters).map { case (old, new_) =>
          math.sqrt(old.zip(new_).map { case (a, b) => val d = a - b; d * d }.sum)
        }
        converged = movements.max < $(tol)
        centers = newCenters
      }

      iteration += 1
    }

    if (centers.length >= 2) {
      (centers(0), centers(1))
    } else {
      // Fallback: split didn't work, return same center
      val center = computeCenter(cleanData, featuresCol, weightCol, kernel)
      (center, center)
    }
  }

  /** Compute the center of a cluster.
    */
  private def computeCenter(
      data: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: BregmanKernel
  ): Array[Double] = {

    val updater = createUpdateStrategy($(divergence))
    val centers = updater.update(
      data.withColumn("cluster", lit(0)),
      featuresCol,
      weightCol,
      1,
      kernel
    )

    if (centers.nonEmpty) centers(0) else Array.empty[Double]
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
      case _                      => throw new IllegalArgumentException(s"Unknown divergence: $divName")
    }
  }

  /** Create assignment strategy.
    */
  private def createAssignmentStrategy(strategy: String): AssignmentStrategy = {
    strategy match {
      case "broadcast" => new BroadcastUDFAssignment()
      case "crossJoin" => new SECrossJoinAssignment()
      case "auto"      => new AutoAssignment()
      case _           => throw new IllegalArgumentException(s"Unknown assignment strategy: $strategy")
    }
  }

  /** Create update strategy based on divergence.
    */
  private def createUpdateStrategy(divName: String): UpdateStrategy = {
    divName match {
      case "l1" | "manhattan" => new MedianUpdateStrategy()
      case _                  => new GradMeanUDAFUpdate()
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): BisectingKMeans = defaultCopy(extra)
}

object BisectingKMeans extends DefaultParamsReadable[BisectingKMeans] {
  override def load(path: String): BisectingKMeans = super.load(path)
}
