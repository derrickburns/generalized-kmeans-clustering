package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Parameters for Bisecting K-Means clustering.
  */
trait BisectingKMeansParams extends GeneralizedKMeansParams {

  /** Minimum divisible cluster size. Clusters with fewer points than this will not be split. Must be >= 1. Default: 1
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
  * This is a hierarchical divisive clustering algorithm that:
  *   1. Starts with all points in one cluster 2. Repeatedly selects the largest cluster and splits it into two using
  *      k=2 clustering 3. Continues until reaching target k clusters
  *
  * Benefits over standard k-means:
  *   - More deterministic (less sensitive to initialization)
  *   - Better handling of imbalanced cluster sizes
  *   - Often faster for large k (only splits locally)
  *   - Generally higher quality than random initialization
  *
  * Example usage:
  * {{{
  *   val bisecting = new BisectingKMeans()
  *     .setK(10)
  *     .setDivergence("squaredEuclidean")
  *     .setMaxIter(20)
  *     .setMinDivisibleClusterSize(5)
  *
  *   val model = bisecting.fit(dataset)
  *   val predictions = model.transform(dataset)
  * }}}
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

    // Create kernel
    val kernel = createKernel($(divergence), $(smoothing))

    // Bisecting algorithm
    val finalCenters = bisect(df, $(featuresCol), getWeightColOpt, kernel)

    logInfo(s"BisectingKMeans completed with ${finalCenters.length} clusters")

    // Create model
    val model = new GeneralizedKMeansModel(uid, finalCenters, kernel.name)
    copyValues(model)
  }

  /** Get weight column as Option.
    */
  private def getWeightColOpt: Option[String] = {
    if (hasWeightCol) Some($(weightCol)) else None
  }

  /** Bisecting clustering algorithm.
    *
    * @param df
    *   Input DataFrame
    * @param featuresCol
    *   Name of features column
    * @param weightCol
    *   Optional weight column
    * @param kernel
    *   Bregman kernel
    * @return
    *   Array of cluster centers
    */
  private def bisect(
    df: DataFrame,
    featuresCol: String,
    weightCol: Option[String],
    kernel: BregmanKernel
  ): Array[Array[Double]] = {

    val targetK = $(k)
    val spark   = df.sparkSession

    // Start with all data in cluster 0
    var clusteredDF = df.withColumn("cluster", lit(0))
    var clusterCenters = Map[Int, Array[Double]](
      0 -> computeCenter(clusteredDF.filter(col("cluster") === 0), featuresCol, weightCol, kernel)
    )
    var nextClusterId = 1

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

      val divisibleClusters = clusterSizes
        .filter { case (_, size) => size >= $(minDivisibleClusterSize) }

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
          return clusterCenters.values.toArray

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
          clusteredDF = oldClusteredDF.withColumn(
            "cluster",
            assignUDF(col(featuresCol), col("cluster"))
          )
          clusteredDF.cache()
          oldClusteredDF.unpersist()

          // Update centers
          clusterCenters = clusterCenters + (clusterId -> center1) + (nextClusterId -> center2)
          nextClusterId += 1
      }
    }

    clusteredDF.unpersist()

    logInfo(s"Bisecting completed with ${clusterCenters.size} clusters")

    // Return centers in order
    (0 until clusterCenters.size).map(i => clusterCenters(i)).toArray
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

    // Initialize with two random points
    val sample = clusterData.select(featuresCol).limit(2).collect()
    if (sample.length < 2) {
      // Can't split - return same center twice
      val center = computeCenter(clusterData, featuresCol, weightCol, kernel)
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
      val assigned = assigner.assign(clusterData, featuresCol, weightCol, centers, kernel)

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
      val center = computeCenter(clusterData, featuresCol, weightCol, kernel)
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
      case "squaredEuclidean" => new SquaredEuclideanKernel()
      case "kl"               => new KLDivergenceKernel(smooth)
      case "itakuraSaito"     => new ItakuraSaitoKernel(smooth)
      case "generalizedI"     => new GeneralizedIDivergenceKernel(smooth)
      case "logistic"         => new LogisticLossKernel(smooth)
      case "l1" | "manhattan" => new L1Kernel()
      case _                  => throw new IllegalArgumentException(s"Unknown divergence: $divName")
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
