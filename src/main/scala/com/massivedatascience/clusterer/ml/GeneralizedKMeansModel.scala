package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Model fitted by GeneralizedKMeans.
  *
  * Represents a trained clustering model that can:
  *   - Transform new data by assigning cluster labels
  *   - Predict clusters for individual feature vectors
  *   - Compute clustering cost (sum of distances to centers)
  *   - Integrate with Spark ML Pipelines
  *
  * Example usage:
  * {{{
  *   // After fitting
  *   val predictions = model.transform(testData)
  *   val cost = model.computeCost(testData)
  *
  *   // Single point prediction
  *   val cluster = model.predict(featureVector)
  *
  *   // Access cluster information
  *   println(s"Found \${model.numClusters} clusters")
  *   model.clusterCentersAsVectors.foreach(println)
  * }}}
  *
  * @param uid
  *   unique identifier for this model instance
  * @param clusterCenters
  *   cluster centers as k x d array (k clusters, d dimensions)
  * @param kernelName
  *   name of the Bregman kernel used during training (e.g., "SquaredEuclidean", "KL")
  */
class GeneralizedKMeansModel(
    override val uid: String,
    val clusterCenters: Array[Array[Double]],
    val kernelName: String
) extends Model[GeneralizedKMeansModel]
    with GeneralizedKMeansParams
    with MLWritable
    with Logging {

  /** Training summary (only available for models trained in current session).
    *
    * Contains diagnostic information about the training process including convergence metrics,
    * iteration history, and performance statistics.
    *
    * Note: This field is not persisted. Models loaded from disk will have trainingSummary = None.
    */
  @transient private[ml] var trainingSummary: Option[TrainingSummary] = None

  /** Constructs a model with a random UID.
    *
    * @param clusterCenters
    *   cluster centers as k x d array
    * @param kernelName
    *   name of the Bregman kernel
    */
  def this(clusterCenters: Array[Array[Double]], kernelName: String) =
    this(Identifiable.randomUID("gkmeans"), clusterCenters, kernelName)

  /** Number of clusters (k).
    */
  def numClusters: Int = clusterCenters.length

  /** Dimensionality of features (d).
    */
  def numFeatures: Int = clusterCenters.headOption.map(_.length).getOrElse(0)

  /** Get cluster centers as Spark ML Vector array.
    *
    * @return
    *   array of k cluster center vectors
    */
  def clusterCentersAsVectors: Array[Vector] = clusterCenters.map(Vectors.dense)

  /** Get training summary.
    *
    * Contains diagnostic information about the training process including convergence metrics,
    * iteration history, and performance statistics.
    *
    * @throws NoSuchElementException
    *   if summary is not available (e.g., model was loaded from disk)
    * @return
    *   training summary
    */
  def summary: TrainingSummary = trainingSummary.getOrElse(
    throw new NoSuchElementException(
      "Training summary not available. Summary is only available for models trained " +
        "in the current session, not for models loaded from disk."
    )
  )

  /** Check if training summary is available.
    *
    * Summary is only available for models trained in the current session. Models loaded from disk
    * will have hasSummary = false.
    *
    * @return
    *   true if summary is available
    */
  def hasSummary: Boolean = trainingSummary.isDefined

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val df = dataset.toDF()

    // Create kernel (must match training kernel)
    val kernel = createKernel(kernelName, $(smoothing))

    logInfo(s"Transforming dataset with ${clusterCenters.length} clusters using kernel $kernelName")

    // Broadcast centers and kernel
    val bcCenters = df.sparkSession.sparkContext.broadcast(clusterCenters)
    val bcKernel  = df.sparkSession.sparkContext.broadcast(kernel)

    // UDF to find nearest cluster
    val predictUDF = udf { (features: Vector) =>
      val ctrs    = bcCenters.value
      val kern    = bcKernel.value
      var minDist = Double.PositiveInfinity
      var minIdx  = 0
      var i       = 0
      while (i < ctrs.length) {
        val center = Vectors.dense(ctrs(i))
        val dist   = kern.divergence(features, center)
        if (dist < minDist) {
          minDist = dist
          minIdx = i
        }
        i += 1
      }
      minIdx
    }

    // UDF to compute distance to assigned cluster
    val distanceUDF = udf { (features: Vector, clusterId: Int) =>
      val center = Vectors.dense(bcCenters.value(clusterId))
      bcKernel.value.divergence(features, center)
    }

    // Add prediction column
    val withPrediction = df.withColumn($(predictionCol), predictUDF(col($(featuresCol))))

    // Optionally add distance column
    val result = if (hasDistanceCol) {
      withPrediction.withColumn(
        $(distanceCol),
        distanceUDF(col($(featuresCol)), col($(predictionCol)))
      )
    } else {
      withPrediction
    }

    // Note: Don't destroy broadcast here because DataFrame is lazy
    // The broadcast will be automatically cleaned up by Spark

    result
  }

  /** Check and transform the input schema.
    *
    * @param schema
    *   input schema
    * @return
    *   output schema with prediction column added
    */
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  /** Create a copy of this model with optional parameter overrides.
    *
    * @param extra
    *   optional parameter map to override current parameters
    * @return
    *   new model instance with copied parameters
    */
  override def copy(extra: ParamMap): GeneralizedKMeansModel = {
    val copied = new GeneralizedKMeansModel(uid, clusterCenters, kernelName)
    copyValues(copied, extra)
  }

  /** Compute the clustering cost (sum of divergences from points to their assigned centers).
    *
    * This is equivalent to the Within-Cluster Sum of Squares (WCSS) for Squared Euclidean
    * divergence. Lower cost indicates tighter, more compact clusters.
    *
    * @param dataset
    *   input dataset with features column
    * @return
    *   total cost (sum of all point-to-center distances)
    */
  def computeCost(dataset: Dataset[_]): Double = {
    val df     = dataset.toDF()
    val kernel = createKernel(kernelName, $(smoothing))

    val bcCenters = df.sparkSession.sparkContext.broadcast(clusterCenters)
    val bcKernel  = df.sparkSession.sparkContext.broadcast(kernel)

    val costUDF = udf { (features: Vector) =>
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

    val cost = df
      .select($(featuresCol))
      .agg(sum(costUDF(col($(featuresCol)))).as("cost"))
      .first()
      .getDouble(0)

    cost
  }

  /** Predict the cluster index for a single feature vector.
    *
    * Returns the index (0 to k-1) of the cluster whose center has minimum divergence to the input.
    *
    * @param features
    *   input feature vector (must have same dimension as training data)
    * @return
    *   cluster index (0-based)
    */
  def predict(features: Vector): Int = {
    val kernel  = createKernel(kernelName, $(smoothing))
    var minDist = Double.PositiveInfinity
    var minIdx  = 0
    var i       = 0
    while (i < clusterCenters.length) {
      val center = Vectors.dense(clusterCenters(i))
      val dist   = kernel.divergence(features, center)
      if (dist < minDist) {
        minDist = dist
        minIdx = i
      }
      i += 1
    }
    minIdx
  }

  /** Create Bregman kernel based on kernel name.
    */
  private def createKernel(kernelName: String, smoothing: Double): BregmanKernel = {
    kernelName match {
      case "SquaredEuclidean"                       => new SquaredEuclideanKernel()
      case name if name.startsWith("KL(")           => new KLDivergenceKernel(smoothing)
      case name if name.startsWith("ItakuraSaito(") => new ItakuraSaitoKernel(smoothing)
      case name if name.startsWith("GeneralizedI(") => new GeneralizedIDivergenceKernel(smoothing)
      case name if name.startsWith("LogisticLoss(") => new LogisticLossKernel(smoothing)
      case "L1"                                     => new L1Kernel()
      case _                                        => throw new IllegalArgumentException(s"Unknown kernel: $kernelName")
    }
  }

  override def write: MLWriter = new GeneralizedKMeansModel.GeneralizedKMeansModelWriter(this)

  override def toString: String = {
    s"GeneralizedKMeansModel: uid=$uid, k=$numClusters, features=$numFeatures, kernel=$kernelName"
  }
}

object GeneralizedKMeansModel extends MLReadable[GeneralizedKMeansModel] {

  override def read: MLReader[GeneralizedKMeansModel] = new GeneralizedKMeansModelReader

  override def load(path: String): GeneralizedKMeansModel = super.load(path)

  private[GeneralizedKMeansModel] class GeneralizedKMeansModelWriter(
      instance: GeneralizedKMeansModel
  ) extends MLWriter
      with Logging {

    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    implicit val formats: DefaultFormats.type = DefaultFormats

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession

      logInfo(s"Saving GeneralizedKMeansModel to $path")

      // Prepare centers data: (center_id, weight, vector)
      val centersData = instance.clusterCenters.indices.map { i =>
        val weight = 1.0 // GeneralizedKMeans doesn't use weighted centers currently
        val vector = Vectors.dense(instance.clusterCenters(i))
        (i, weight, vector)
      }

      // Write centers with deterministic ordering
      val centersHash = writeCenters(spark, path, centersData)

      // Collect all parameters (explicitly typed to avoid Any inference)
      val params: Map[String, Any] = Map(
        "maxIter"              -> instance.getOrDefault(instance.maxIter),
        "tol"                  -> instance.getOrDefault(instance.tol),
        "seed"                 -> instance.getOrDefault(instance.seed),
        "assignmentStrategy"   -> instance.getOrDefault(instance.assignmentStrategy),
        "smoothing"            -> instance.getOrDefault(instance.smoothing),
        "emptyClusterStrategy" -> instance.getOrDefault(instance.emptyClusterStrategy),
        "checkpointInterval"   -> instance.getOrDefault(instance.checkpointInterval),
        "initMode"             -> instance.getOrDefault(instance.initMode),
        "initSteps"            -> instance.getOrDefault(instance.initSteps),
        "featuresCol"          -> instance.getOrDefault(instance.featuresCol),
        "predictionCol"        -> instance.getOrDefault(instance.predictionCol),
        "distanceCol"          -> (if (instance.hasDistanceCol) instance.getDistanceCol else ""),
        "weightCol"            -> (if (instance.hasWeightCol) instance.getWeightCol else ""),
        "checkpointDir"        -> (if (instance.hasCheckpointDir) instance.getCheckpointDir else "")
      )

      // Create metadata object (explicitly typed to avoid Any inference)
      val metaObj: Map[String, Any] = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "GeneralizedKMeansModel",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "divergence"         -> instance.getDivergence,
        "k"                  -> instance.numClusters,
        "dim"                -> instance.numFeatures,
        "uid"                -> instance.uid,
        "kernelName"         -> instance.kernelName,
        "params"             -> params,
        "centers"            -> Map[String, Any](
          "count"    -> instance.numClusters,
          "ordering" -> "center_id ASC (0..k-1)",
          "storage"  -> "parquet"
        ),
        "checksums"          -> Map[String, String](
          "centersParquetSHA256"    -> centersHash,
          "metadataCanonicalSHA256" -> ""
        )
      )

      // Write metadata and get hash
      val json     = Serialization.write(metaObj)
      val metaHash = writeMetadata(path, json)

      // Update metadata with its own hash
      val metaObj2 = metaObj.updated(
        "checksums",
        Map(
          "centersParquetSHA256"    -> centersHash,
          "metadataCanonicalSHA256" -> metaHash
        )
      )
      val json2    = Serialization.write(metaObj2)
      writeMetadata(path, json2)

      logInfo(s"Successfully saved GeneralizedKMeansModel to $path")
    }
  }

  private class GeneralizedKMeansModelReader extends MLReader[GeneralizedKMeansModel] with Logging {

    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    implicit val formats: DefaultFormats.type = DefaultFormats

    override def load(path: String): GeneralizedKMeansModel = {
      logInfo(s"Loading GeneralizedKMeansModel from $path")

      // Read and parse metadata
      val metaStr = readMetadata(path)
      val metaJ   = JsonMethods.parse(metaStr)

      val layoutVersion = (metaJ \ "layoutVersion").extract[Int]
      val k             = (metaJ \ "k").extract[Int]
      val dim           = (metaJ \ "dim").extract[Int]
      val uid           = (metaJ \ "uid").extract[String]
      val kernelName    = (metaJ \ "kernelName").extract[String]
      val divergence    = (metaJ \ "divergence").extract[String]
      val params        = (metaJ \ "params").values.asInstanceOf[Map[String, Any]]

      // Load centers from Parquet
      val centersDF = readCenters(sparkSession, path)
      val rows      = centersDF.collect().sortBy(_.getInt(0)) // center_id

      // Validate metadata
      validateMetadata(layoutVersion, k, dim, rows.length)

      // Extract centers as arrays
      val centers = rows.map { row =>
        row.getAs[Vector]("vector").toArray
      }

      // Create model instance
      val model = new GeneralizedKMeansModel(uid, centers, kernelName)

      // Set parameters from saved metadata
      model.set(model.divergence, divergence)
      model.set(model.maxIter, params("maxIter").asInstanceOf[BigInt].toInt)
      model.set(model.tol, params("tol").asInstanceOf[Double])
      model.set(model.seed, params("seed").asInstanceOf[BigInt].toLong)
      model.set(model.assignmentStrategy, params("assignmentStrategy").asInstanceOf[String])
      model.set(model.smoothing, params("smoothing").asInstanceOf[Double])
      model.set(
        model.emptyClusterStrategy,
        params("emptyClusterStrategy").asInstanceOf[String]
      )
      model.set(
        model.checkpointInterval,
        params("checkpointInterval").asInstanceOf[BigInt].toInt
      )
      model.set(model.initMode, params("initMode").asInstanceOf[String])
      model.set(model.initSteps, params("initSteps").asInstanceOf[BigInt].toInt)
      model.set(model.featuresCol, params("featuresCol").asInstanceOf[String])
      model.set(model.predictionCol, params("predictionCol").asInstanceOf[String])

      // Set optional parameters if present
      val distanceCol = params("distanceCol").asInstanceOf[String]
      if (distanceCol.nonEmpty) {
        model.set(model.distanceCol, distanceCol)
      }

      val weightCol = params("weightCol").asInstanceOf[String]
      if (weightCol.nonEmpty) {
        model.set(model.weightCol, weightCol)
      }

      val checkpointDir = params("checkpointDir").asInstanceOf[String]
      if (checkpointDir.nonEmpty) {
        model.set(model.checkpointDir, checkpointDir)
      }

      logInfo(s"Successfully loaded GeneralizedKMeansModel from $path")

      model
    }
  }
}

/** Summary of GeneralizedKMeans training with comprehensive clustering metrics.
  *
  * Provides various quality metrics for evaluating clustering results:
  *   - Distortion-based: WCSS (within-cluster sum of squares), BCSS (between-cluster)
  *   - Silhouette coefficient: measures how similar points are to their cluster vs other clusters
  *   - Davies-Bouldin index: ratio of within-cluster to between-cluster distances (lower is better)
  *   - Dunn index: ratio of minimum separation to maximum diameter (higher is better)
  *
  * @param predictions
  *   DataFrame with predictions and features
  * @param predictionCol
  *   name of prediction column
  * @param featuresCol
  *   name of features column
  * @param clusterCenters
  *   cluster centers
  * @param kernel
  *   Bregman kernel used during training
  * @param numClusters
  *   number of clusters
  * @param numFeatures
  *   number of features
  * @param numIter
  *   number of iterations run
  * @param converged
  *   whether the algorithm converged
  * @param distortionHistory
  *   distortion at each iteration
  * @param movementHistory
  *   max center movement at each iteration
  */
class GeneralizedKMeansSummary(
    val predictions: DataFrame,
    val predictionCol: String,
    val featuresCol: String,
    val clusterCenters: Array[Array[Double]],
    val kernel: BregmanKernel,
    val numClusters: Int,
    val numFeatures: Int,
    val numIter: Int,
    val converged: Boolean,
    val distortionHistory: Array[Double],
    val movementHistory: Array[Double]
) extends Serializable
    with Logging {

  /** Number of data points.
    */
  lazy val numPoints: Long = predictions.count()

  /** Final distortion (sum of distances to assigned centers). Also known as Within-Cluster Sum of
    * Squares (WCSS).
    */
  lazy val finalDistortion: Double = distortionHistory.lastOption.getOrElse(Double.NaN)

  /** Within-Cluster Sum of Squares (WCSS). Measures compactness of clusters. Lower values indicate
    * tighter clusters.
    */
  lazy val wcss: Double = finalDistortion

  /** Cluster sizes (number of points per cluster).
    */
  lazy val clusterSizes: Array[Long] = {
    val sizesMap = predictions
      .groupBy(predictionCol)
      .count()
      .collect()
      .map(row => (row.getInt(0), row.getLong(1)))
      .toMap

    (0 until numClusters).map(i => sizesMap.getOrElse(i, 0L)).toArray
  }

  /** Between-Cluster Sum of Squares (BCSS). Measures separation between clusters. Higher values
    * indicate better separation.
    */
  lazy val bcss: Double = {
    val spark    = predictions.sparkSession
    val bcKernel = spark.sparkContext.broadcast(kernel)

    // Compute overall centroid
    val totalWeight = clusterSizes.sum.toDouble
    if (totalWeight == 0.0) {
      0.0
    } else {
      val overallCentroid = clusterCenters.zipWithIndex.map { case (center, idx) =>
        center.map(_ * clusterSizes(idx))
      }.reduce { (a, b) =>
        a.zip(b).map { case (x, y) => x + y }
      }.map(_ / totalWeight)

      // BCSS = Σ n_i * D(μ_i, μ_overall)
      val bcssValue = clusterCenters.zipWithIndex.map { case (center, idx) =>
        val size = clusterSizes(idx)
        if (size > 0) {
          val dist = bcKernel.value.divergence(
            Vectors.dense(center),
            Vectors.dense(overallCentroid)
          )
          size * dist
        } else {
          0.0
        }
      }.sum

      bcssValue
    }
  }

  /** Calinski-Harabasz Index (Variance Ratio Criterion). Ratio of between-cluster to within-cluster
    * variance. Higher values indicate better-defined clusters.
    *
    * CH = (BCSS / (k-1)) / (WCSS / (n-k))
    */
  lazy val calinskiHarabaszIndex: Double = {
    if (numClusters <= 1 || numPoints <= numClusters) {
      0.0
    } else {
      val numerator   = bcss / (numClusters - 1)
      val denominator = wcss / (numPoints - numClusters)

      if (denominator == 0.0) 0.0 else numerator / denominator
    }
  }

  /** Davies-Bouldin Index. Measures average similarity between each cluster and its most similar
    * cluster. Lower values indicate better clustering (0 is best).
    *
    * Computed as: DB = (1/k) * Σ_i max_j(R_ij) where R_ij = (s_i + s_j) / d_ij s_i = average
    * distance within cluster i d_ij = distance between cluster centers i and j
    */
  lazy val daviesBouldinIndex: Double = {
    if (numClusters <= 1) {
      0.0
    } else {
      val spark     = predictions.sparkSession
      val bcCenters = spark.sparkContext.broadcast(clusterCenters)
      val bcKernel  = spark.sparkContext.broadcast(kernel)

      // Compute average within-cluster distances
      val avgDistancesUDF = udf { (features: Vector, clusterId: Int) =>
        val center = Vectors.dense(bcCenters.value(clusterId))
        bcKernel.value.divergence(features, center)
      }

      val avgDistances = predictions
        .withColumn("dist", avgDistancesUDF(col(featuresCol), col(predictionCol)))
        .groupBy(predictionCol)
        .agg(avg("dist").as("avgDist"))
        .collect()
        .map(row => (row.getInt(0), row.getDouble(1)))
        .toMap

      // Compute pairwise center distances
      val centerDistances = Array.ofDim[Double](numClusters, numClusters)
      for (i <- 0 until numClusters; j <- i + 1 until numClusters) {
        val dist = kernel.divergence(
          Vectors.dense(clusterCenters(i)),
          Vectors.dense(clusterCenters(j))
        )
        centerDistances(i)(j) = dist
        centerDistances(j)(i) = dist
      }

      // Compute DB index
      val dbValues = (0 until numClusters).map { i =>
        if (clusterSizes(i) == 0) {
          0.0
        } else {
          val s_i      = avgDistances.getOrElse(i, 0.0)
          val maxRatio = (0 until numClusters)
            .filter(_ != i)
            .map { j =>
              if (clusterSizes(j) == 0 || centerDistances(i)(j) == 0.0) {
                0.0
              } else {
                val s_j = avgDistances.getOrElse(j, 0.0)
                (s_i + s_j) / centerDistances(i)(j)
              }
            }
            .max
          maxRatio
        }
      }

      dbValues.sum / numClusters
    }
  }

  /** Dunn Index. Ratio of minimum inter-cluster distance to maximum intra-cluster distance. Higher
    * values indicate better clustering (more compact and well-separated).
    *
    * Dunn = min_ij(d(C_i, C_j)) / max_k(diam(C_k))
    */
  lazy val dunnIndex: Double = {
    if (numClusters <= 1) {
      0.0
    } else {
      val spark     = predictions.sparkSession
      val bcCenters = spark.sparkContext.broadcast(clusterCenters)
      val bcKernel  = spark.sparkContext.broadcast(kernel)

      // Compute maximum intra-cluster distance (diameter) for each cluster
      val diameterUDF = udf { (features: Vector, clusterId: Int) =>
        val center = Vectors.dense(bcCenters.value(clusterId))
        bcKernel.value.divergence(features, center)
      }

      val maxDiameters = predictions
        .withColumn("dist", diameterUDF(col(featuresCol), col(predictionCol)))
        .groupBy(predictionCol)
        .agg(max("dist").as("maxDist"))
        .collect()
        .map(_.getDouble(1))

      val maxDiameter = if (maxDiameters.nonEmpty) maxDiameters.max else 0.0
      if (maxDiameter == 0.0) {
        0.0
      } else {
        // Compute minimum inter-cluster distance
        var minInterClusterDist = Double.MaxValue
        for (i <- 0 until numClusters; j <- i + 1 until numClusters) {
          val dist = kernel.divergence(
            Vectors.dense(clusterCenters(i)),
            Vectors.dense(clusterCenters(j))
          )
          if (dist < minInterClusterDist) {
            minInterClusterDist = dist
          }
        }

        if (minInterClusterDist == Double.MaxValue) 0.0
        else minInterClusterDist / maxDiameter
      }
    }
  }

  /** Mean Silhouette Coefficient (sampled for efficiency). Measures how similar points are to their
    * own cluster vs other clusters. Values range from -1 to 1:
    *   - Close to 1: point is well-matched to its cluster
    *   - Close to 0: point is on the border between clusters
    *   - Close to -1: point may be assigned to wrong cluster
    *
    * For large datasets, this uses sampling to make computation tractable.
    *
    * @param sampleFraction
    *   fraction of data to sample (default 0.1 = 10%)
    * @return
    *   mean silhouette coefficient
    */
  def silhouette(sampleFraction: Double = 0.1): Double = {
    if (numClusters <= 1 || numPoints <= 1) return 0.0

    require(
      sampleFraction > 0.0 && sampleFraction <= 1.0,
      s"Sample fraction must be in (0, 1], got $sampleFraction"
    )

    logInfo(s"Computing silhouette with sample fraction $sampleFraction")

    val spark     = predictions.sparkSession
    val bcCenters = spark.sparkContext.broadcast(clusterCenters)
    val bcKernel  = spark.sparkContext.broadcast(kernel)

    // Sample data for efficiency
    val sampledData = if (sampleFraction < 1.0) {
      predictions.sample(withReplacement = false, sampleFraction, seed = 42)
    } else {
      predictions
    }

    // For each point, compute:
    // a(i) = average distance to points in same cluster
    // b(i) = minimum average distance to points in other clusters
    // s(i) = (b(i) - a(i)) / max(a(i), b(i))

    val silhouetteUDF = udf { (features: Vector, clusterId: Int) =>
      val centers = bcCenters.value
      val kern    = bcKernel.value

      // Compute distance to own cluster center (proxy for a(i))
      val ownCenterDist = kern.divergence(features, Vectors.dense(centers(clusterId)))

      // Compute minimum distance to other cluster centers (proxy for b(i))
      var minOtherDist = Double.MaxValue
      var i            = 0
      while (i < centers.length) {
        if (i != clusterId) {
          val dist = kern.divergence(features, Vectors.dense(centers(i)))
          if (dist < minOtherDist) {
            minOtherDist = dist
          }
        }
        i += 1
      }

      // Compute silhouette
      val a       = ownCenterDist
      val b       = if (minOtherDist == Double.MaxValue) 0.0 else minOtherDist
      val maxDist = math.max(a, b)

      if (maxDist == 0.0) 0.0 else (b - a) / maxDist
    }

    val meanSilhouette = sampledData
      .withColumn("silhouette", silhouetteUDF(col(featuresCol), col(predictionCol)))
      .agg(avg("silhouette").as("avgSilhouette"))
      .first()
      .getDouble(0)

    meanSilhouette
  }

  override def toString: String = {
    s"""GeneralizedKMeansSummary:
       |  numClusters: $numClusters
       |  numFeatures: $numFeatures
       |  numPoints: $numPoints
       |  numIter: $numIter
       |  converged: $converged
       |  finalDistortion (WCSS): ${f"$finalDistortion%.4f"}
       |  clusterSizes: ${clusterSizes.mkString("[", ", ", "]")}
       |
       |Quality Metrics:
       |  WCSS (within-cluster sum of squares): ${f"$wcss%.4f"}
       |  BCSS (between-cluster sum of squares): ${f"$bcss%.4f"}
       |  Calinski-Harabasz Index: ${f"$calinskiHarabaszIndex%.4f"}
       |  Davies-Bouldin Index: ${f"$daviesBouldinIndex%.4f"} (lower is better)
       |  Dunn Index: ${f"$dunnIndex%.4f"} (higher is better)
       |
       |Note: Call silhouette() to compute silhouette coefficient (expensive for large datasets)
       |""".stripMargin
  }
}
