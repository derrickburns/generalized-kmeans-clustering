package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.StreamingQuery

/** Parameters for Streaming K-Means clustering.
  */
trait StreamingKMeansParams extends GeneralizedKMeansParams {

  /** Decay factor for exponential forgetting.
    *
    * If decayFactor = 1.0, all batches are weighted equally (no forgetting).
    * If decayFactor = 0.0, only the current batch matters (complete forgetting).
    * Values between 0 and 1 provide exponential decay of old data.
    *
    * Default: 1.0 (no forgetting)
    */
  final val decayFactor = new DoubleParam(
    this,
    "decayFactor",
    "Decay factor for exponential forgetting (0.0 = complete forgetting, 1.0 = no forgetting)",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = true, upperInclusive = true)
  )

  def getDecayFactor: Double = $(decayFactor)

  /** Time unit for decay: "batches" or "points".
    *
    * If "batches", decay is applied per batch regardless of batch size.
    * If "points", decay is scaled by number of points in batch.
    *
    * Default: "batches"
    */
  final val timeUnit = new Param[String](
    this,
    "timeUnit",
    "Time unit for decay (batches or points)",
    ParamValidators.inArray(Array("batches", "points"))
  )

  def getTimeUnit: String = $(timeUnit)

  /** Half-life for decay in number of batches or points.
    *
    * If set, overrides decayFactor with: decayFactor = 0.5^(1/halfLife)
    * Half-life is the time it takes for a data point's weight to decay to 50%.
    *
    * Default: None (use explicit decayFactor)
    */
  final val halfLife = new DoubleParam(
    this,
    "halfLife",
    "Half-life for decay in number of batches or points",
    ParamValidators.gt(0.0)
  )

  def getHalfLife: Double = $(halfLife)

  setDefault(
    decayFactor -> 1.0,
    timeUnit -> "batches"
  )

  /** Compute effective decay factor, using half-life if set.
    */
  protected def getEffectiveDecayFactor: Double = {
    if (isSet(halfLife)) {
      math.pow(0.5, 1.0 / $(halfLife))
    } else {
      $(decayFactor)
    }
  }
}

/** Streaming K-Means clustering for incremental updates.
  *
  * This implementation uses the mini-batch K-Means algorithm with exponential forgetting
  * to enable real-time clustering on streaming data. The model is updated incrementally
  * as new batches arrive.
  *
  * Update Rule (for each cluster):
  * {{{
  * c_{t+1} = [(c_t * n_t * α) + (x_t * m_t)] / [n_t * α + m_t]
  * n_{t+1} = n_t * α + m_t
  * }}}
  *
  * Where:
  * - c_t: current center
  * - n_t: current weight (number of points assigned)
  * - x_t: mean of new points assigned to this cluster
  * - m_t: number of new points assigned to this cluster
  * - α: decay factor (applies exponential forgetting)
  *
  * Example usage:
  * {{{
  * // Initialize with batch data
  * val kmeans = new StreamingKMeans()
  *   .setK(3)
  *   .setDecayFactor(0.9)
  *   .setMaxIter(10)
  *
  * val initialModel = kmeans.fit(batchDF)
  *
  * // Create streaming updater
  * val updater = initialModel.createStreamingUpdater()
  *
  * // Update on streaming data
  * val query = updater.updateOn(streamingDF)
  *
  * // Get current model anytime
  * val currentModel = updater.currentModel
  * }}}
  *
  * @param uid unique identifier
  */
class StreamingKMeans(override val uid: String)
    extends GeneralizedKMeans(uid)
    with StreamingKMeansParams {

  def this() = this(Identifiable.randomUID("streamingKMeans"))

  // Override to return StreamingKMeansModel
  override def fit(dataset: Dataset[_]): StreamingKMeansModel = {
    val baseModel = super.fit(dataset).asInstanceOf[GeneralizedKMeansModel]

    val model = new StreamingKMeansModel(
      uid = Identifiable.randomUID("streamingKMeansModel"),
      initialCenters = baseModel.clusterCenters,
      kernelNameForParent = baseModel.kernelName, // Use parent's kernel name format
      divergenceName = get(divergence).getOrElse("squaredEuclidean"),
      smoothingValue = get(smoothing).getOrElse(1e-10),
      decayFactorValue = getEffectiveDecayFactor,
      timeUnitValue = $(timeUnit)
    )
    model.setParent(this)
    model
  }

  // Parameter setters
  def setDecayFactor(value: Double): this.type = set(decayFactor, value)
  def setTimeUnit(value: String): this.type = set(timeUnit, value)
  def setHalfLife(value: Double): this.type = set(halfLife, value)

  override def copy(extra: ParamMap): StreamingKMeans = defaultCopy(extra)
}

object StreamingKMeans extends DefaultParamsReadable[StreamingKMeans] {
  override def load(path: String): StreamingKMeans = super.load(path)
}

/** Model for Streaming K-Means.
  *
  * Maintains mutable cluster centers and weights that can be updated incrementally.
  * Provides methods to create a streaming updater for real-time updates.
  */
class StreamingKMeansModel(
  override val uid: String,
  initialCenters: Array[Array[Double]],
  kernelNameForParent: String, // Kernel name in parent's format (e.g., "SquaredEuclidean")
  val divergenceName: String,   // Divergence name in our format (e.g., "squaredEuclidean")
  val smoothingValue: Double,
  val decayFactorValue: Double,
  val timeUnitValue: String
) extends GeneralizedKMeansModel(uid, initialCenters, kernelNameForParent)
    with Logging {

  // Mutable state for streaming updates
  @transient private var centerArrays: Array[Vector] = initialCenters.map(Vectors.dense)
  @transient private var clusterWeights: Array[Double] = Array.fill(initialCenters.length)(1.0)
  @transient private lazy val kernel: BregmanKernel = createKernel(divergenceName, smoothingValue)

  /** Get current cluster centers as Vectors (defensive copy).
    */
  def currentCenters: Array[Vector] = centerArrays.map(_.copy)

  /** Get current cluster weights (defensive copy).
    */
  def currentWeights: Array[Double] = clusterWeights.clone()

  /** Sync mutable centers back to immutable clusterCenters array.
    */
  private def syncCenters(): Unit = {
    // Update parent's clusterCenters by creating new object
    // This is a hack but necessary since parent has immutable val
    var i = 0
    while (i < clusterCenters.length) {
      val newCenter = centerArrays(i).toArray
      var j = 0
      while (j < newCenter.length) {
        clusterCenters(i)(j) = newCenter(j)
        j += 1
      }
      i += 1
    }
  }

  /** Create Bregman kernel from divergence name.
    */
  private def createKernel(divName: String, smooth: Double): BregmanKernel = {
    divName match {
      case "squaredEuclidean" => new SquaredEuclideanKernel()
      case "kl"               => new KLDivergenceKernel(smooth)
      case "itakuraSaito"     => new ItakuraSaitoKernel(smooth)
      case "generalizedI"     => new GeneralizedIDivergenceKernel(smooth)
      case "l1" | "manhattan" => new L1Kernel()
      case _ =>
        logWarning(s"Unknown divergence $divName, using squared Euclidean")
        new SquaredEuclideanKernel()
    }
  }

  /** Update the model with a new batch of data.
    *
    * This method performs one iteration of mini-batch K-Means with exponential forgetting:
    * 1. Assign each point to nearest cluster
    * 2. Compute new cluster means from this batch
    * 3. Merge with existing centers using decay factor
    * 4. Handle dying clusters by splitting largest cluster
    *
    * @param batchDF DataFrame with features column
    * @return this (for chaining)
    */
  def update(batchDF: Dataset[_]): this.type = {
    val df = batchDF.toDF()
    val featCol = $(featuresCol)
    val weightColOpt = if (hasWeightCol) Some($(weightCol)) else None

    // Assign each point to nearest cluster
    val predictUDF = udf { (features: Vector) =>
      predict(features)
    }

    val assigned = df.withColumn("cluster", predictUDF(col(featCol)))

    // Compute statistics for each cluster from this batch using RDD operations
    // (simpler than complex SQL aggregations on Vector structs)
    val dim = centerArrays(0).size

    val stats = if (weightColOpt.isDefined) {
      val wCol = weightColOpt.get
      // Weighted statistics
      assigned
        .select("cluster", featCol, wCol)
        .rdd
        .map { row =>
          val cluster = row.getInt(0)
          val features = row.getAs[Vector](1)
          val weight = row.getDouble(2)
          (cluster, (features, weight))
        }
        .groupByKey()
        .mapValues { points =>
          val totalWeight = points.map(_._2).sum
          val weightedSum = Array.fill(dim)(0.0)
          points.foreach { case (features, weight) =>
            var i = 0
            while (i < dim) {
              weightedSum(i) += features(i) * weight
              i += 1
            }
          }
          val centroid = weightedSum.map(_ / totalWeight)
          (totalWeight, centroid)
        }
        .collect()
        .map { case (cluster, (count, centroid)) => (cluster, count, centroid) }
    } else {
      // Unweighted statistics
      assigned
        .select("cluster", featCol)
        .rdd
        .map { row =>
          val cluster = row.getInt(0)
          val features = row.getAs[Vector](1)
          (cluster, features)
        }
        .groupByKey()
        .mapValues { points =>
          val count = points.size.toDouble
          val sum = Array.fill(dim)(0.0)
          points.foreach { features =>
            var i = 0
            while (i < dim) {
              sum(i) += features(i)
              i += 1
            }
          }
          val centroid = sum.map(_ / count)
          (count, centroid)
        }
        .collect()
        .map { case (cluster, (count, centroid)) => (cluster, count, centroid) }
    }

    // Compute decay factor
    val totalNewPoints = stats.map(_._2).sum

    val discount = timeUnitValue match {
      case "batches" => decayFactorValue
      case "points" => math.pow(decayFactorValue, totalNewPoints)
    }

    // Apply decay to existing weights
    var i = 0
    while (i < clusterWeights.length) {
      clusterWeights(i) *= discount
      i += 1
    }

    // Update centers using mini-batch rule
    stats.foreach { case (clusterId, batchCount, batchCentroid) =>
      // Update rule: c_{t+1} = [(c_t * n_t) + (x_t * m_t)] / [n_t + m_t]
      val oldWeight = clusterWeights(clusterId)
      val newWeight = oldWeight + batchCount
      val lambda = batchCount / math.max(newWeight, 1e-16)

      clusterWeights(clusterId) = newWeight

      // Update center: c = (1-λ)*c + λ*x
      val oldCenter = centerArrays(clusterId).toArray
      val newCenter = new Array[Double](oldCenter.length)

      var j = 0
      while (j < newCenter.length) {
        newCenter(j) = (1.0 - lambda) * oldCenter(j) + lambda * batchCentroid(j)
        j += 1
      }

      centerArrays(clusterId) = Vectors.dense(newCenter)

      logInfo(f"Cluster $clusterId updated: weight=$newWeight%.1f, center=[${newCenter.take(5).mkString(", ")}...]")
    }

    // Handle dying clusters
    handleDyingClusters()

    // Sync mutable state to parent's immutable clusterCenters
    syncCenters()

    this
  }

  /** Split the largest cluster if the smallest cluster is dying.
    */
  private def handleDyingClusters(): Unit = {
    val maxWeight = clusterWeights.max
    val minWeight = clusterWeights.min

    if (minWeight < 1e-8 * maxWeight) {
      val largest = clusterWeights.indexOf(maxWeight)
      val smallest = clusterWeights.indexOf(minWeight)

      logInfo(f"Cluster $smallest is dying (weight=$minWeight%.2e). Splitting cluster $largest (weight=$maxWeight%.2f).")

      val newWeight = (maxWeight + minWeight) / 2.0
      clusterWeights(largest) = newWeight
      clusterWeights(smallest) = newWeight

      val largestCenter = centerArrays(largest).toArray
      val perturbation = 1e-14

      val l = largestCenter.map(x => x + perturbation * math.max(math.abs(x), 1.0))
      val s = largestCenter.map(x => x - perturbation * math.max(math.abs(x), 1.0))

      centerArrays(largest) = Vectors.dense(l)
      centerArrays(smallest) = Vectors.dense(s)
    }
  }

  /** Override predict to use current centers.
    */
  override def predict(features: Vector): Int = {
    var bestCluster = 0
    var minDistance = kernel.divergence(features, centerArrays(0))

    var i = 1
    while (i < centerArrays.length) {
      val distance = kernel.divergence(features, centerArrays(i))
      if (distance < minDistance) {
        minDistance = distance
        bestCluster = i
      }
      i += 1
    }

    bestCluster
  }

  /** Override clusterCentersAsVectors to return current dynamic centers.
    */
  override def clusterCentersAsVectors: Array[Vector] = currentCenters

  /** Create a streaming updater for real-time updates.
    *
    * The updater maintains a reference to this model and updates it incrementally
    * as streaming batches arrive.
    *
    * @return StreamingKMeansUpdater
    */
  def createStreamingUpdater(): StreamingKMeansUpdater = {
    new StreamingKMeansUpdater(this)
  }

  override def copy(extra: ParamMap): StreamingKMeansModel = {
    // Sync current state before copying
    syncCenters()

    val copied = new StreamingKMeansModel(
      uid,
      clusterCenters,
      kernelName, // Parent's kernel name
      divergenceName,
      smoothingValue,
      decayFactorValue,
      timeUnitValue
    )

    // Copy weights too
    System.arraycopy(clusterWeights, 0, copied.clusterWeights, 0, clusterWeights.length)

    copyValues(copied, extra)
    if (parent != null) copied.setParent(parent)
    copied
  }
}

/** Streaming updater for incremental model updates.
  *
  * Provides a convenient interface for updating a StreamingKMeansModel
  * on streaming data sources using foreachBatch.
  *
  * Example:
  * {{{
  * val updater = model.createStreamingUpdater()
  * val query = updater.updateOn(streamingDF)
  * query.awaitTermination()
  *
  * // Get current model anytime
  * val current = updater.currentModel
  * }}}
  */
class StreamingKMeansUpdater(private val model: StreamingKMeansModel) extends Logging {

  private val batchCounter = new java.util.concurrent.atomic.AtomicLong(0L)

  /** Get the current model (returns the same mutable instance).
    */
  def currentModel: StreamingKMeansModel = model

  /** Start updating the model on streaming data.
    *
    * Uses foreachBatch to update the model incrementally as batches arrive.
    *
    * @param streamingDF Streaming DataFrame with features column
    * @param checkpointLocation Optional checkpoint location for fault tolerance
    * @return StreamingQuery
    */
  def updateOn(streamingDF: Dataset[_], checkpointLocation: Option[String] = None): StreamingQuery = {
    def processBatch(batchDF: Dataset[_], batchId: Long): Unit = {
      val count = batchDF.count()
      if (count > 0) {
        logInfo(s"Processing batch $batchId with $count records")
        model.update(batchDF)
        batchCounter.incrementAndGet()
        ()
      } else {
        logInfo(s"Skipping empty batch $batchId")
        ()
      }
    }

    val query = streamingDF
      .writeStream
      .foreachBatch(processBatch _)

    checkpointLocation match {
      case Some(path) => query.option("checkpointLocation", path)
      case None => query
    }

    query.start()
  }

  /** Get the number of batches processed so far.
    */
  def batchesProcessed: Long = batchCounter.get()

  /** Predict on a batch DataFrame using current model.
    */
  def transform(df: Dataset[_]): DataFrame = {
    model.transform(df)
  }
}

object StreamingKMeansModel extends MLReadable[StreamingKMeansModel] {

  override def read: MLReader[StreamingKMeansModel] = new StreamingKMeansModelReader

  override def load(path: String): StreamingKMeansModel = super.load(path)

  private class StreamingKMeansModelReader extends MLReader[StreamingKMeansModel] {

    override def load(path: String): StreamingKMeansModel = {
      val spark = sparkSession
      import spark.implicits._

      val centersPath = s"$path/centers"
      val metadataPath = s"$path/metadata"

      // Load centers
      val centersDF = spark.read.parquet(centersPath)
      val centers = centersDF.as[(Int, Array[Double])].collect()
        .sortBy(_._1)
        .map { case (_, arr) => arr }

      // Load metadata
      val metadataDF = spark.read.parquet(metadataPath)
      val metadata = metadataDF.as[(String, String, String, Double, Double, String)].head()
      val (uid, kernelName, divergence, smoothing, decayFactor, timeUnit) = metadata

      new StreamingKMeansModel(uid, centers, kernelName, divergence, smoothing, decayFactor, timeUnit)
    }
  }
}
