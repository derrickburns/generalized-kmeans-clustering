/*
 * Licensed to the Massive Data Science and Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Massive Data Science and Derrick R. Burns licenses this file to You under the
 * Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{ Estimator, Model }
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Parameters for Robust K-Means clustering with outlier handling.
  */
trait RobustKMeansParams extends GeneralizedKMeansParams {

  /** Fraction of points to consider as potential outliers. For trim mode: fraction to exclude from
    * center updates. For noise_cluster mode: fraction threshold for outlier detection. Default:
    * 0.05 (5% outliers)
    */
  final val outlierFraction: DoubleParam = new DoubleParam(
    this,
    "outlierFraction",
    "Fraction of points to treat as outliers",
    ParamValidators.inRange(0.0, 0.5, lowerInclusive = true, upperInclusive = true)
  )
  def getOutlierFraction: Double         = $(outlierFraction)

  /** Outlier handling mode.
    *   - "trim": Exclude outliers from center updates (trimmed k-means)
    *   - "noise_cluster": Assign outliers to special cluster -1
    *   - "m_estimator": Use M-estimator robust statistics
    * Default: "trim"
    */
  final val outlierMode: Param[String] = new Param[String](
    this,
    "outlierMode",
    "Outlier handling mode: trim, noise_cluster, m_estimator",
    ParamValidators.inArray(Array("trim", "noise_cluster", "m_estimator"))
  )
  def getOutlierMode: String           = $(outlierMode)

  /** Column name for outlier scores (distance-based). Default: "outlier_score"
    */
  final val outlierScoreCol: Param[String] = new Param[String](
    this,
    "outlierScoreCol",
    "Column name for outlier scores"
  )
  def getOutlierScoreCol: String           = $(outlierScoreCol)

  /** Column name for outlier flag. Default: "is_outlier"
    */
  final val isOutlierCol: Param[String] = new Param[String](
    this,
    "isOutlierCol",
    "Column name for outlier flag"
  )
  def getIsOutlierCol: String           = $(isOutlierCol)

  /** Distance threshold multiplier for outlier detection. Points with score > threshold *
    * median_score are outliers. Default: 3.0 (3 standard deviations)
    */
  final val outlierThreshold: DoubleParam = new DoubleParam(
    this,
    "outlierThreshold",
    "Distance threshold multiplier for outliers",
    ParamValidators.gt(0.0)
  )
  def getOutlierThreshold: Double         = $(outlierThreshold)

  setDefault(
    outlierFraction  -> 0.05,
    outlierMode      -> "trim",
    outlierScoreCol  -> "outlier_score",
    isOutlierCol     -> "is_outlier",
    outlierThreshold -> 3.0
  )
}

/** Robust K-Means clustering with outlier detection and handling.
  *
  * This estimator extends standard K-Means to handle outliers that can distort cluster centers and
  * degrade clustering quality.
  *
  * ==Outlier Modes==
  *
  * '''Trim mode''' (default): Excludes outliers from center updates
  * {{{
  * val rkm = new RobustKMeans()
  *   .setK(5)
  *   .setOutlierMode("trim")
  *   .setOutlierFraction(0.1)  // Trim top 10%
  * }}}
  *
  * '''Noise cluster mode''': Assigns outliers to cluster -1
  * {{{
  * val rkm = new RobustKMeans()
  *   .setK(5)
  *   .setOutlierMode("noise_cluster")
  *   .setOutlierThreshold(3.0)
  * }}}
  *
  * '''M-estimator mode''': Uses robust statistics to downweight outliers
  * {{{
  * val rkm = new RobustKMeans()
  *   .setK(5)
  *   .setOutlierMode("m_estimator")
  * }}}
  *
  * ==Algorithm==
  *
  *   1. Initialize centers using k-means++ or random 2. Assignment: Assign points to nearest center
  *      3. Outlier detection: Identify outliers based on distance 4. Robust update: Recompute
  *      centers excluding/downweighting outliers 5. Repeat until convergence
  *
  * ==Output Columns==
  *
  * The model adds:
  *   - `prediction`: cluster assignment (0 to k-1, or -1 for outliers in noise mode)
  *   - `outlier_score`: normalized distance to nearest center
  *   - `is_outlier`: boolean flag for identified outliers
  *
  * @see
  *   [[OutlierDetector]] for outlier detection strategies
  * @see
  *   [[GeneralizedKMeans]] for standard k-means without outlier handling
  */
class RobustKMeans(override val uid: String)
    extends Estimator[RobustKMeansModel]
    with RobustKMeansParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("robustkmeans"))

  // Parameter setters
  def setK(value: Int): this.type                   = set(k, value)
  def setDivergence(value: String): this.type       = set(divergence, value)
  def setSmoothing(value: Double): this.type        = set(smoothing, value)
  def setFeaturesCol(value: String): this.type      = set(featuresCol, value)
  def setPredictionCol(value: String): this.type    = set(predictionCol, value)
  def setWeightCol(value: String): this.type        = set(weightCol, value)
  def setMaxIter(value: Int): this.type             = set(maxIter, value)
  def setTol(value: Double): this.type              = set(tol, value)
  def setSeed(value: Long): this.type               = set(seed, value)
  def setOutlierFraction(value: Double): this.type  = set(outlierFraction, value)
  def setOutlierMode(value: String): this.type      = set(outlierMode, value)
  def setOutlierScoreCol(value: String): this.type  = set(outlierScoreCol, value)
  def setIsOutlierCol(value: String): this.type     = set(isOutlierCol, value)
  def setOutlierThreshold(value: Double): this.type = set(outlierThreshold, value)

  override def fit(dataset: Dataset[_]): RobustKMeansModel = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()

    val mode = OutlierMode.fromString($(outlierMode))
    logInfo(
      s"Training RobustKMeans: k=${$(k)}, mode=$mode, " +
        s"outlierFraction=${$(outlierFraction)}, divergence=${$(divergence)}"
    )

    df.cache()
    try {
      val kernel    = createKernel()
      val startTime = System.currentTimeMillis()

      // Initialize centers
      val initialCenters = initializeCenters(df, kernel)
      logInfo(s"Initialized ${initialCenters.length} centers")

      // Run robust Lloyd's algorithm
      val result = runRobustLloyds(df, initialCenters, kernel, mode)

      val elapsed = System.currentTimeMillis() - startTime
      logInfo(
        s"RobustKMeans completed: ${result.iterations} iterations, " +
          s"converged=${result.converged}, outliers=${result.numOutliers}"
      )

      // Create model
      val model = new RobustKMeansModel(
        uid,
        result.centers.map(Vectors.dense),
        kernel.name,
        mode.name
      )
      copyValues(model.setParent(this))

      // Training summary
      model.trainingSummary = Some(
        TrainingSummary(
          algorithm = "RobustKMeans",
          k = $(k),
          effectiveK = result.centers.length,
          dim = result.centers.headOption.map(_.length).getOrElse(0),
          numPoints = df.count(),
          iterations = result.iterations,
          converged = result.converged,
          distortionHistory = result.distortionHistory,
          movementHistory = result.movementHistory,
          assignmentStrategy = $(outlierMode),
          divergence = $(divergence),
          elapsedMillis = elapsed
        )
      )

      model
    } finally {
      df.unpersist()
    }
  }

  /** Run robust Lloyd's algorithm with outlier handling. */
  private def runRobustLloyds(
      df: DataFrame,
      initialCenters: Array[Array[Double]],
      kernel: BregmanKernel,
      mode: OutlierMode
  ): RobustResult = {
    var centers           = initialCenters
    var iteration         = 0
    var converged         = false
    val distortionHistory = scala.collection.mutable.ArrayBuffer[Double]()
    val movementHistory   = scala.collection.mutable.ArrayBuffer[Double]()
    var totalOutliers     = 0L

    // Create outlier detector based on mode
    val outlierParam = mode match {
      case OutlierMode.Trim         => $(outlierFraction)
      case OutlierMode.NoiseCluster => $(outlierThreshold)
      case OutlierMode.MEstimator   => $(outlierThreshold)
    }
    val detector     = OutlierDetector.create(mode, kernel, outlierParam)
    val updater      = RobustCenterUpdate.create(mode)

    while (iteration < $(maxIter) && !converged) {
      iteration += 1

      val centersVec = centers.map(Vectors.dense)

      // Step 1: Assign points to nearest center
      val bcKernel  = df.sparkSession.sparkContext.broadcast(kernel)
      val bcCenters = df.sparkSession.sparkContext.broadcast(centersVec)

      val assignUDF = udf { (features: Vector) =>
        val k           = bcKernel.value
        val ctrs        = bcCenters.value
        var bestCluster = 0
        var bestDist    = Double.MaxValue
        var i           = 0
        while (i < ctrs.length) {
          val dist = k.divergence(features, ctrs(i))
          if (dist < bestDist) {
            bestDist = dist
            bestCluster = i
          }
          i += 1
        }
        bestCluster
      }

      val assigned = df.withColumn("_cluster", assignUDF(col($(featuresCol))))

      // Step 2: Detect outliers
      val withOutliers = detector.detectOutliers(
        assigned,
        centersVec,
        $(featuresCol)
      )

      // Count outliers
      val numOutliers = withOutliers.filter(col(detector.isOutlierCol) === true).count()
      totalOutliers = numOutliers

      // Step 3: Robust center update
      val newCenters = updater.updateCenters(
        withOutliers,
        kernel,
        $(k),
        $(featuresCol),
        "_cluster",
        detector.isOutlierCol,
        getWeightColOpt
      )

      // Step 4: Check convergence
      val movement = computeMovement(centers, newCenters.map(_.toArray))
      movementHistory += movement

      val distortion = computeDistortion(df, newCenters, kernel)
      distortionHistory += distortion

      converged = movement < $(tol)
      centers = newCenters.map(_.toArray)

      logDebug(
        f"Iteration $iteration: distortion=$distortion%.4f, " +
          f"movement=$movement%.6f, outliers=$numOutliers"
      )
    }

    RobustResult(
      centers = centers,
      iterations = iteration,
      converged = converged,
      distortionHistory = distortionHistory.toArray,
      movementHistory = movementHistory.toArray,
      numOutliers = totalOutliers
    )
  }

  private def createKernel(): BregmanKernel = {
    ClusteringOps.createKernel($(divergence), $(smoothing)).asInstanceOf[BregmanKernel]
  }

  private def initializeCenters(df: DataFrame, kernel: BregmanKernel): Array[Array[Double]] = {
    val fraction = math.min(1.0, ($(k) * 10.0) / df.count().toDouble)
    df.select($(featuresCol))
      .sample(withReplacement = false, fraction, $(seed))
      .limit($(k))
      .collect()
      .map(_.getAs[Vector](0).toArray)
  }

  private def computeMovement(
      oldCenters: Array[Array[Double]],
      newCenters: Array[Array[Double]]
  ): Double = {
    oldCenters
      .zip(newCenters)
      .map { case (old, newC) =>
        math.sqrt(old.zip(newC).map { case (a, b) => val d = a - b; d * d }.sum)
      }
      .max
  }

  private def computeDistortion(
      df: DataFrame,
      centers: Array[Vector],
      kernel: BregmanKernel
  ): Double = {
    val bcKernel  = df.sparkSession.sparkContext.broadcast(kernel)
    val bcCenters = df.sparkSession.sparkContext.broadcast(centers)

    val distUDF = udf { (features: Vector) =>
      val k       = bcKernel.value
      val ctrs    = bcCenters.value
      var minDist = Double.MaxValue
      var i       = 0
      while (i < ctrs.length) {
        val dist = k.divergence(features, ctrs(i))
        if (dist < minDist) minDist = dist
        i += 1
      }
      minDist
    }

    df.select(distUDF(col($(featuresCol)))).agg(sum("UDF(features)")).first().getDouble(0)
  }

  private def getWeightColOpt: Option[String] =
    if (isDefined(weightCol) && $(weightCol).nonEmpty) Some($(weightCol)) else None

  override def copy(extra: ParamMap): RobustKMeans = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}

object RobustKMeans extends DefaultParamsReadable[RobustKMeans] {
  override def load(path: String): RobustKMeans = super.load(path)
}

/** Model from Robust K-Means clustering.
  *
  * Includes cluster centers and outlier detection parameters. Transform adds prediction, outlier
  * score, and outlier flag columns.
  */
class RobustKMeansModel(
    override val uid: String,
    val clusterCenters: Array[Vector],
    val divergenceName: String,
    val outlierModeName: String
) extends Model[RobustKMeansModel]
    with RobustKMeansParams
    with MLWritable
    with HasTrainingSummary
    with Logging {

  /** Number of clusters. */
  def numClusters: Int = clusterCenters.length

  /** Cluster centers as vectors. */
  def clusterCentersAsVectors: Array[Vector] = clusterCenters

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df     = dataset.toDF()
    val kernel =
      ClusteringOps.createKernel(divergenceName, $(smoothing)).asInstanceOf[BregmanKernel]
    val mode   = OutlierMode.fromString(outlierModeName)

    val bcKernel  = df.sparkSession.sparkContext.broadcast(kernel)
    val bcCenters = df.sparkSession.sparkContext.broadcast(clusterCenters)

    // Assign to nearest center
    val assignUDF = udf { (features: Vector) =>
      val k           = bcKernel.value
      val ctrs        = bcCenters.value
      var bestCluster = 0
      var bestDist    = Double.MaxValue
      var i           = 0
      while (i < ctrs.length) {
        val dist = k.divergence(features, ctrs(i))
        if (dist < bestDist) {
          bestDist = dist
          bestCluster = i
        }
        i += 1
      }
      bestCluster
    }

    // Compute distance to assigned center (for outlier score)
    val distUDF = udf { (features: Vector) =>
      val k       = bcKernel.value
      val ctrs    = bcCenters.value
      var minDist = Double.MaxValue
      var i       = 0
      while (i < ctrs.length) {
        val dist = k.divergence(features, ctrs(i))
        if (dist < minDist) minDist = dist
        i += 1
      }
      minDist
    }

    val withPred = df
      .withColumn($(predictionCol), assignUDF(col($(featuresCol))))
      .withColumn("_dist", distUDF(col($(featuresCol))))

    // Compute median for normalization
    val medianDist =
      withPred.stat.approxQuantile("_dist", Array(0.5), 0.01).headOption.getOrElse(1.0)

    val normalizedMedian = if (medianDist > 1e-10) medianDist else 1.0

    // Add outlier columns
    val threshold    = $(outlierThreshold)
    val withOutliers = withPred
      .withColumn($(outlierScoreCol), col("_dist") / lit(normalizedMedian))
      .withColumn($(isOutlierCol), col($(outlierScoreCol)) > lit(threshold))
      .drop("_dist")

    // For noise_cluster mode, reassign outliers to -1
    if (mode == OutlierMode.NoiseCluster) {
      withOutliers.withColumn(
        $(predictionCol),
        when(col($(isOutlierCol)), lit(-1)).otherwise(col($(predictionCol)))
      )
    } else {
      withOutliers
    }
  }

  override def copy(extra: ParamMap): RobustKMeansModel = {
    val copied = new RobustKMeansModel(uid, clusterCenters, divergenceName, outlierModeName)
    copyValues(copied, extra).setParent(parent)
    copied.trainingSummary = this.trainingSummary
    copied
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def write: MLWriter = new RobustKMeansModel.RobustKMeansModelWriter(this)
}

object RobustKMeansModel extends MLReadable[RobustKMeansModel] {

  override def read: MLReader[RobustKMeansModel] = new RobustKMeansModelReader

  private class RobustKMeansModelWriter(instance: RobustKMeansModel) extends MLWriter with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession
      logInfo(s"Saving RobustKMeansModel to $path")

      val centersData = instance.clusterCenters.indices.map { i =>
        (i, 1.0, instance.clusterCenters(i))
      }
      val centersHash = writeCenters(spark, path, centersData)

      val k   = instance.numClusters
      val dim = instance.clusterCenters.headOption.map(_.size).getOrElse(0)

      val params: Map[String, Any] = Map(
        "k"                -> k,
        "featuresCol"      -> instance.getOrDefault(instance.featuresCol),
        "predictionCol"    -> instance.getOrDefault(instance.predictionCol),
        "divergence"       -> instance.divergenceName,
        "outlierMode"      -> instance.outlierModeName,
        "outlierFraction"  -> instance.getOrDefault(instance.outlierFraction),
        "outlierThreshold" -> instance.getOrDefault(instance.outlierThreshold),
        "outlierScoreCol"  -> instance.getOrDefault(instance.outlierScoreCol),
        "isOutlierCol"     -> instance.getOrDefault(instance.isOutlierCol)
      )

      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaObj: Map[String, Any]             = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "RobustKMeansModel",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "divergence"         -> instance.divergenceName,
        "outlierMode"        -> instance.outlierModeName,
        "k"                  -> k,
        "dim"                -> dim,
        "uid"                -> instance.uid,
        "params"             -> params,
        "centers"            -> Map[String, Any](
          "count"    -> k,
          "ordering" -> "center_id ASC (0..k-1)",
          "storage"  -> "parquet"
        ),
        "checksums"          -> Map[String, String](
          "centersParquetSHA256" -> centersHash
        )
      )

      val json = Serialization.write(metaObj)
      writeMetadata(path, json)
      logInfo(s"RobustKMeansModel saved to $path")
    }
  }

  private class RobustKMeansModelReader extends MLReader[RobustKMeansModel] with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): RobustKMeansModel = {
      val spark = sparkSession
      logInfo(s"Loading RobustKMeansModel from $path")

      val metaStr                               = readMetadata(path)
      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaJ                                 = JsonMethods.parse(metaStr)

      val layoutVersion = (metaJ \ "layoutVersion").extract[Int]
      val k             = (metaJ \ "k").extract[Int]
      val dim           = (metaJ \ "dim").extract[Int]
      val uid           = (metaJ \ "uid").extract[String]

      val centersDF = readCenters(spark, path)
      val rows      = centersDF.collect()
      validateMetadata(layoutVersion, k, dim, rows.length)

      val centers = rows.sortBy(_.getInt(0)).map(_.getAs[Vector]("vector"))

      val paramsJ     = metaJ \ "params"
      val divergence  = (paramsJ \ "divergence").extract[String]
      val outlierMode = (paramsJ \ "outlierMode").extract[String]

      val model = new RobustKMeansModel(uid, centers, divergence, outlierMode)

      model.set(model.featuresCol, (paramsJ \ "featuresCol").extract[String])
      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])
      model.set(model.outlierFraction, (paramsJ \ "outlierFraction").extract[Double])
      model.set(model.outlierThreshold, (paramsJ \ "outlierThreshold").extract[Double])
      model.set(model.outlierScoreCol, (paramsJ \ "outlierScoreCol").extract[String])
      model.set(model.isOutlierCol, (paramsJ \ "isOutlierCol").extract[String])

      logInfo(s"RobustKMeansModel loaded from $path")
      model
    }
  }
}

/** Internal result class for robust Lloyd's algorithm. */
private[ml] case class RobustResult(
    centers: Array[Array[Double]],
    iterations: Int,
    converged: Boolean,
    distortionHistory: Array[Double],
    movementHistory: Array[Double],
    numOutliers: Long
)
