package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.StreamingQuery

/** Parameters for Streaming K-Means clustering.
  */
trait StreamingKMeansParams extends GeneralizedKMeansParams {

  /** Decay factor for exponential forgetting.
    *
    * If decayFactor = 1.0, all batches are weighted equally (no forgetting). If decayFactor = 0.0,
    * only the current batch matters (complete forgetting). Values between 0 and 1 provide
    * exponential decay of old data.
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
    * If "batches", decay is applied per batch regardless of batch size. If "points", decay is
    * scaled by number of points in batch.
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
    * If set, overrides decayFactor with: decayFactor = 0.5^(1/halfLife) Half-life is the time it
    * takes for a data point's weight to decay to 50%.
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
    timeUnit    -> "batches"
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

/** Streaming K-Means clustering for incremental updates on streaming data.
  *
  * Implements mini-batch K-Means with exponential forgetting for real-time clustering. The model is
  * updated incrementally as new batches arrive, adapting to concept drift in the data distribution
  * over time.
  *
  * ==Algorithm==
  *
  * For each incoming batch, the algorithm:
  *   1. Assigns new points to nearest clusters 2. Computes batch-local cluster statistics (mean,
  *      count) 3. Applies exponential decay to existing cluster weights 4. Merges batch statistics
  *      with decayed historical statistics 5. Handles dying clusters by splitting the largest
  *      cluster
  *
  * '''Update Rule (for each cluster):'''
  * {{{
  * c_{t+1} = [(c_t * n_t * α) + (x_t * m_t)] / [n_t * α + m_t]
  * n_{t+1} = n_t * α + m_t
  * }}}
  *
  * Where:
  *   - c_t: current center
  *   - n_t: current effective weight (decayed historical count)
  *   - x_t: mean of new points assigned to this cluster
  *   - m_t: count of new points assigned to this cluster
  *   - α: decay factor (controls rate of forgetting)
  *
  * ==Decay Factor==
  *
  * The `decayFactor` controls how quickly old data is forgotten:
  *   - '''α = 1.0:''' No forgetting (all batches weighted equally) - good for stationary data
  *   - '''α = 0.9:''' Moderate forgetting - good for slowly drifting data
  *   - '''α = 0.5:''' Strong forgetting - good for rapidly changing data
  *   - '''α = 0.0:''' Complete forgetting (only current batch matters)
  *
  * Alternatively, use `halfLife` to specify how many batches/points until a data point's influence
  * decays to 50%.
  *
  * ==Divergences==
  *
  * Supports all Bregman divergences:
  *   - `squaredEuclidean` (default): Standard streaming k-means
  *   - `kl`: Streaming topic modeling for probability distributions
  *   - `spherical`/`cosine`: Streaming clustering for embeddings/text
  *
  * ==Example Usage==
  *
  * {{{
  * // Step 1: Initialize model on batch data
  * val streaming = new StreamingKMeans()
  *   .setK(5)
  *   .setDecayFactor(0.9)  // 90% retention per batch
  *   .setMaxIter(10)       // Iterations for initial training
  *
  * val initialModel = streaming.fit(batchDF)
  *
  * // Step 2: Create streaming updater
  * val updater = initialModel.createStreamingUpdater()
  *
  * // Step 3: Update on streaming data
  * val query = updater.updateOn(streamingDF, Some("/path/to/checkpoint"))
  *
  * // Step 4: Access current model anytime (thread-safe)
  * val currentModel = updater.currentModel
  * val predictions = currentModel.transform(newData)
  *
  * // Monitor progress
  * println(s"Processed ${updater.batchesProcessed} batches")
  * }}}
  *
  * ==Use Cases==
  *
  *   - '''Real-time anomaly detection:''' Track cluster centers, flag distant points
  *   - '''IoT sensor clustering:''' Adapt to changing sensor behavior
  *   - '''User behavior clustering:''' Track evolving user segments
  *   - '''Log analysis:''' Cluster streaming logs with concept drift
  *
  * ==Model Persistence==
  *
  * The streaming model can be saved and loaded, preserving cluster weights for seamless
  * continuation after restarts. This enables fault-tolerant streaming pipelines.
  *
  * @see
  *   [[StreamingKMeansModel]] for the updateable model
  * @see
  *   [[StreamingKMeansUpdater]] for the streaming update interface
  * @see
  *   [[GeneralizedKMeans]] for batch clustering
  *
  * @param uid
  *   unique identifier
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

    // Copy training summary from base model
    model.trainingSummary = baseModel.trainingSummary

    model.setParent(this)
    model
  }

  // Parameter setters
  def setDecayFactor(value: Double): this.type = set(decayFactor, value)
  def setTimeUnit(value: String): this.type    = set(timeUnit, value)
  def setHalfLife(value: Double): this.type    = set(halfLife, value)

  override def copy(extra: ParamMap): StreamingKMeans = defaultCopy(extra)
}

object StreamingKMeans extends DefaultParamsReadable[StreamingKMeans] {
  override def load(path: String): StreamingKMeans = super.load(path)
}

/** Model for Streaming K-Means.
  *
  * Maintains mutable cluster centers and weights that can be updated incrementally. Provides
  * methods to create a streaming updater for real-time updates.
  */
class StreamingKMeansModel(
    override val uid: String,
    initialCenters: Array[Array[Double]],
    kernelNameForParent: String, // Kernel name in parent's format (e.g., "SquaredEuclidean")
    val divergenceName: String,  // Divergence name in our format (e.g., "squaredEuclidean")
    val smoothingValue: Double,
    val decayFactorValue: Double,
    val timeUnitValue: String
) extends GeneralizedKMeansModel(uid, initialCenters, kernelNameForParent)
    with MLWritable
    with Logging
    with HasTrainingSummary
    with CentroidModelHelpers {

  // Mutable state for streaming updates (array contents are mutated, not references)
  @transient private val centerArrays: Array[Vector]   = initialCenters.map(Vectors.dense)
  @transient private val clusterWeights: Array[Double] = Array.fill(initialCenters.length)(1.0)
  @transient private lazy val kernel: ClusteringKernel =
    createKernel(divergenceName, smoothingValue)

  /** Get current cluster centers as Vectors (defensive copy).
    */
  def currentCenters: Array[Vector] = centerArrays.map(_.copy)

  // For CentroidModelHelpers — return live centers
  override def clusterCentersAsVectors: Array[Vector] = currentCenters

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
      var j         = 0
      while (j < newCenter.length) {
        clusterCenters(i)(j) = newCenter(j)
        j += 1
      }
      i += 1
    }
  }

  /** Create Bregman kernel from divergence name.
    */
  private def createKernel(divName: String, smooth: Double): ClusteringKernel = {
    ClusteringOps.createKernel(divName, smooth)
  }

  /** Update the model with a new batch of data.
    *
    * This method performs one iteration of mini-batch K-Means with exponential forgetting:
    *   1. Assign each point to nearest cluster 2. Compute new cluster means from this batch 3.
    *      Merge with existing centers using decay factor 4. Handle dying clusters by splitting
    *      largest cluster
    *
    * @param batchDF
    *   DataFrame with features column
    * @return
    *   this (for chaining)
    */
  def update(batchDF: Dataset[_]): this.type = {
    val df           = batchDF.toDF()
    val featCol      = $(featuresCol)
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
          val cluster  = row.getInt(0)
          val features = row.getAs[Vector](1)
          val weight   = row.getDouble(2)
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
          val centroid    = weightedSum.map(_ / totalWeight)
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
          val cluster  = row.getInt(0)
          val features = row.getAs[Vector](1)
          (cluster, features)
        }
        .groupByKey()
        .mapValues { points =>
          val count    = points.size.toDouble
          val sum      = Array.fill(dim)(0.0)
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
      case "points"  => math.pow(decayFactorValue, totalNewPoints)
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
      val lambda    = batchCount / math.max(newWeight, 1e-16)

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

      logInfo(
        f"Cluster $clusterId updated: weight=$newWeight%.1f, center=[${newCenter.take(5).mkString(", ")}...]"
      )
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
      val largest  = clusterWeights.indexOf(maxWeight)
      val smallest = clusterWeights.indexOf(minWeight)

      logInfo(
        f"Cluster $smallest is dying (weight=$minWeight%.2e). Splitting cluster $largest (weight=$maxWeight%.2f)."
      )

      val newWeight = (maxWeight + minWeight) / 2.0
      clusterWeights(largest) = newWeight
      clusterWeights(smallest) = newWeight

      val largestCenter = centerArrays(largest).toArray
      val perturbation  = 1e-14

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

  /** Create a streaming updater for real-time updates.
    *
    * The updater maintains a reference to this model and updates it incrementally as streaming
    * batches arrive.
    *
    * @return
    *   StreamingKMeansUpdater
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

  override def write: MLWriter = new StreamingKMeansModel.StreamingKMeansModelWriter(this)
}

/** Streaming updater for incremental model updates.
  *
  * Provides a convenient interface for updating a StreamingKMeansModel on streaming data sources
  * using foreachBatch.
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
    * @param streamingDF
    *   Streaming DataFrame with features column
    * @param checkpointLocation
    *   Optional checkpoint location for fault tolerance
    * @return
    *   StreamingQuery
    */
  def updateOn(
      streamingDF: Dataset[_],
      checkpointLocation: Option[String] = None
  ): StreamingQuery = {
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

    val query = streamingDF.writeStream.foreachBatch(processBatch _)

    checkpointLocation match {
      case Some(path) => query.option("checkpointLocation", path)
      case None       => query
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

  private class StreamingKMeansModelWriter(instance: StreamingKMeansModel)
      extends MLWriter
      with Logging {

    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession

      logInfo(s"Saving StreamingKMeansModel to $path")

      // Sync mutable state before saving
      instance.syncCenters()

      // Prepare centers data with weights: (center_id, weight, vector)
      // For streaming K-Means, we need to save the cluster weights!
      val currentCenters = instance.currentCenters
      val currentWeights = instance.currentWeights
      val centersData    = currentCenters.indices.map { i =>
        val weight = currentWeights(i)
        val vector = currentCenters(i)
        (i, weight, vector)
      }

      // Write centers with deterministic ordering
      val centersHash = writeCenters(spark, path, centersData)
      logInfo(s"Centers saved with SHA-256: $centersHash")

      // Collect all model parameters (explicitly typed to avoid Any inference)
      val params: Map[String, Any] = Map(
        "k"             -> instance.numClusters,
        "featuresCol"   -> instance.getOrDefault(instance.featuresCol),
        "predictionCol" -> instance.getOrDefault(instance.predictionCol),
        "divergence"    -> instance.divergenceName,
        "smoothing"     -> instance.smoothingValue,
        "decayFactor"   -> instance.decayFactorValue,
        "timeUnit"      -> instance.timeUnitValue,
        "kernelName"    -> instance.kernelName // Parent's kernel name format
      )

      val k   = instance.numClusters
      val dim = currentCenters.headOption.map(_.size).getOrElse(0)

      // Build metadata object (explicitly typed to avoid Any inference)
      implicit val formats          = DefaultFormats
      val metaObj: Map[String, Any] = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "StreamingKMeansModel",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "divergence"         -> instance.divergenceName,
        "k"                  -> k,
        "dim"                -> dim,
        "uid"                -> instance.uid,
        "params"             -> params,
        "centers"            -> Map[String, Any](
          "count"           -> k,
          "ordering"        -> "center_id ASC (0..k-1)",
          "storage"         -> "parquet",
          "includesWeights" -> true // Important: weights are stored in weight column
        ),
        "checksums"          -> Map[String, String](
          "centersParquetSHA256" -> centersHash
        )
      )

      // Serialize to JSON
      val json = Serialization.write(metaObj)(formats)

      // Write metadata
      val metadataHash = writeMetadata(path, json)
      logInfo(s"Metadata saved with SHA-256: $metadataHash")
      logInfo(s"StreamingKMeansModel successfully saved to $path (includes cluster weights)")
    }
  }

  private class StreamingKMeansModelReader extends MLReader[StreamingKMeansModel] with Logging {

    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): StreamingKMeansModel = {
      val spark = sparkSession

      logInfo(s"Loading StreamingKMeansModel from $path")

      // Read metadata
      val metaStr          = readMetadata(path)
      implicit val formats = DefaultFormats
      val metaJ            = JsonMethods.parse(metaStr)

      // Extract and validate layout version
      val layoutVersion = (metaJ \ "layoutVersion").extract[Int]
      val k             = (metaJ \ "k").extract[Int]
      val dim           = (metaJ \ "dim").extract[Int]
      val uid           = (metaJ \ "uid").extract[String]
      val divergence    = (metaJ \ "divergence").extract[String]

      logInfo(
        s"Model metadata: layoutVersion=$layoutVersion, k=$k, dim=$dim, divergence=$divergence"
      )

      // Read centers with weights
      val centersDF = readCenters(spark, path)
      val rows      = centersDF.collect()

      // Validate metadata
      validateMetadata(layoutVersion, k, dim, rows.length)

      // Extract centers and weights (sorted by center_id)
      val sortedRows = rows.sortBy(_.getInt(0))
      val centers    = sortedRows.map { row =>
        row.getAs[Vector]("vector").toArray
      }
      val weights    = sortedRows.map { row =>
        row.getDouble(1) // weight column
      }

      // Extract parameters
      val paramsJ     = metaJ \ "params"
      val smoothing   = (paramsJ \ "smoothing").extract[Double]
      val decayFactor = (paramsJ \ "decayFactor").extract[Double]
      val timeUnit    = (paramsJ \ "timeUnit").extract[String]
      val kernelName  = (paramsJ \ "kernelName").extract[String]

      // Reconstruct model
      val model = new StreamingKMeansModel(
        uid,
        centers,
        kernelName,
        divergence,
        smoothing,
        decayFactor,
        timeUnit
      )

      // Restore cluster weights (critical for streaming!)
      System.arraycopy(weights, 0, model.clusterWeights, 0, weights.length)
      logInfo(s"Restored cluster weights: ${weights.mkString("[", ", ", "]")}")

      // Set parameters
      model.set(model.featuresCol, (paramsJ \ "featuresCol").extract[String])
      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])

      logInfo(s"StreamingKMeansModel successfully loaded from $path")
      model
    }
  }
}
