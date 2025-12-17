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

import com.massivedatascience.clusterer.ml.df.kernels.{ DTWKernel, SequenceKernel }
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{ Estimator, Model }
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{
  DefaultParamsReadable,
  DefaultParamsWritable,
  Identifiable,
  MLReadable,
  MLReader,
  MLWritable,
  MLWriter
}
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Parameters for Time Series K-Means clustering.
  */
trait TimeSeriesKMeansParams
    extends Params
    with HasFeaturesCol
    with HasPredictionCol
    with HasMaxIter
    with HasTol
    with HasSeed {

  /** Number of clusters (k). */
  final val k: IntParam = new IntParam(this, "k", "Number of clusters", ParamValidators.gt(1))
  def getK: Int         = $(k)

  /** Sequence kernel type: "dtw", "softdtw", "gak", "derivative". */
  final val kernelType: Param[String] = new Param[String](
    this,
    "kernelType",
    "Sequence kernel type for time series alignment",
    ParamValidators.inArray(Array("dtw", "softdtw", "gak", "derivative"))
  )
  def getKernelType: String           = $(kernelType)

  /** Sakoe-Chiba band width for DTW constraint. If 0, use unconstrained DTW. Larger values allow
    * more warping but increase computation.
    */
  final val window: IntParam = new IntParam(
    this,
    "window",
    "Sakoe-Chiba band width (0 for unconstrained)",
    ParamValidators.gtEq(0)
  )
  def getWindow: Int         = $(window)

  /** Smoothing parameter for soft-DTW (gamma). Smaller values approximate hard DTW; larger values
    * give smoother gradients.
    */
  final val gamma: DoubleParam = new DoubleParam(
    this,
    "gamma",
    "Soft-DTW smoothing parameter (gamma > 0)",
    ParamValidators.gt(0.0)
  )
  def getGamma: Double         = $(gamma)

  /** Bandwidth parameter for GAK kernel (sigma). */
  final val sigma: DoubleParam = new DoubleParam(
    this,
    "sigma",
    "GAK kernel bandwidth (sigma > 0)",
    ParamValidators.gt(0.0)
  )
  def getSigma: Double         = $(sigma)

  /** Maximum iterations for DBA barycenter computation within each k-means iteration. */
  final val dbaIter: IntParam = new IntParam(
    this,
    "dbaIter",
    "Maximum iterations for DBA barycenter",
    ParamValidators.gt(0)
  )
  def getDbaIter: Int         = $(dbaIter)

  /** Column name for distance to assigned cluster center. */
  final val distanceCol: Param[String] = new Param[String](
    this,
    "distanceCol",
    "Column name for distance to cluster center"
  )
  def getDistanceCol: String           = $(distanceCol)

  setDefault(
    k             -> 2,
    kernelType    -> "dtw",
    window        -> 0,
    gamma         -> 1.0,
    sigma         -> 1.0,
    dbaIter       -> 10,
    maxIter       -> 20,
    tol           -> 1e-4,
    featuresCol   -> "features",
    predictionCol -> "prediction",
    distanceCol   -> "distance"
  )
}

/** Time Series K-Means clustering with DTW-based distance measures.
  *
  * Clusters time series using elastic alignment distances (DTW family) instead of Euclidean
  * distance. This handles temporal shifts, warping, and different-length sequences.
  *
  * ==Algorithm==
  *
  * Uses the k-means framework with DTW distance and DBA (DTW Barycenter Averaging) for centroids:
  *
  *   1. Initialize k centroids (random selection from data) 2. '''Assignment:''' Assign each series
  *      to nearest centroid using DTW distance 3. '''Update:''' Recompute centroids using DBA
  *      (iterative DTW-based averaging) 4. Repeat until convergence
  *
  * ==Kernel Options==
  *
  *   - '''dtw''' - Standard Dynamic Time Warping, elastic alignment
  *   - '''softdtw''' - Differentiable DTW, smoother optimization landscape
  *   - '''gak''' - Global Alignment Kernel, positive-definite
  *   - '''derivative''' - Shape-based DTW, invariant to offset/scale
  *
  * ==Example Usage==
  *
  * {{{
  * import com.massivedatascience.clusterer.ml.TimeSeriesKMeans
  * import org.apache.spark.ml.linalg.Vectors
  *
  * // Time series as feature vectors (each element is a time point)
  * val df = Seq(
  *   Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0),
  *   Vectors.dense(1.1, 2.1, 3.1, 4.1, 5.1),
  *   Vectors.dense(5.0, 4.0, 3.0, 2.0, 1.0),
  *   Vectors.dense(5.1, 4.1, 3.1, 2.1, 1.1)
  * ).map(Tuple1(_)).toDF("features")
  *
  * val tsKMeans = new TimeSeriesKMeans()
  *   .setK(2)
  *   .setKernelType("dtw")
  *   .setWindow(5)        // Sakoe-Chiba band
  *   .setMaxIter(20)
  *   .setDbaIter(10)      // DBA iterations for centroids
  *
  * val model = tsKMeans.fit(df)
  * val predictions = model.transform(df)
  *
  * // Access cluster centroids
  * model.clusterCenters.foreach(c => println(c.toArray.mkString(", ")))
  * }}}
  *
  * ==Scalability Note==
  *
  * DTW distance is O(nm) for sequences of length n and m. With Sakoe-Chiba window w, complexity
  * reduces to O(n*w). For large datasets:
  *
  *   - Use window constraint to limit warping and speed up
  *   - Consider sampling for initialization
  *   - Use soft-DTW for smoother convergence
  *
  * @see
  *   [[SequenceKernel]] for kernel implementations
  * @see
  *   [[DTWBarycenter]] for DBA algorithm
  */
class TimeSeriesKMeans(override val uid: String)
    extends Estimator[TimeSeriesKMeansModel]
    with TimeSeriesKMeansParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("tskmeans"))

  // Parameter setters
  def setK(value: Int): this.type                = set(k, value)
  def setKernelType(value: String): this.type    = set(kernelType, value)
  def setWindow(value: Int): this.type           = set(window, value)
  def setGamma(value: Double): this.type         = set(gamma, value)
  def setSigma(value: Double): this.type         = set(sigma, value)
  def setDbaIter(value: Int): this.type          = set(dbaIter, value)
  def setMaxIter(value: Int): this.type          = set(maxIter, value)
  def setTol(value: Double): this.type           = set(tol, value)
  def setSeed(value: Long): this.type            = set(seed, value)
  def setFeaturesCol(value: String): this.type   = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setDistanceCol(value: String): this.type   = set(distanceCol, value)

  override def fit(dataset: Dataset[_]): TimeSeriesKMeansModel = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()

    logInfo(
      s"Starting Time Series K-Means: k=${$(k)}, kernel=${$(kernelType)}, " +
        s"window=${$(window)}, maxIter=${$(maxIter)}"
    )

    df.cache()
    try {
      val startTime = System.currentTimeMillis()

      // Collect all time series
      val series = df.select($(featuresCol)).collect().map(_.getAs[Vector](0))
      val n      = series.length

      if (n < $(k)) {
        throw new IllegalArgumentException(s"Need at least k=${$(k)} points, but got $n")
      }

      // Create sequence kernel
      val kernel = createKernel()
      logInfo(s"Using kernel: ${kernel.name}")

      // Initialize centroids (random selection)
      val rng             = new scala.util.Random($(seed))
      val initIndices     = rng.shuffle((0 until n).toList).take($(k))
      var centers         = initIndices.map(i => series(i)).toArray
      val assignments     = new Array[Int](n)
      var prevAssignments = Array.fill(n)(-1)

      var iteration         = 0
      var converged         = false
      val distortionHistory = scala.collection.mutable.ArrayBuffer[Double]()

      while (iteration < $(maxIter) && !converged) {
        iteration += 1

        // Assignment step: assign each series to nearest center
        var totalDistortion = 0.0
        for (i <- 0 until n) {
          var bestCluster = 0
          var bestDist    = Double.MaxValue

          for (c <- 0 until $(k)) {
            val dist = kernel match {
              case dtw: DTWKernel => dtw.distance(series(i), centers(c))
              case _              => kernel.squaredDistance(series(i), centers(c))
            }
            if (dist < bestDist) {
              bestDist = dist
              bestCluster = c
            }
          }

          assignments(i) = bestCluster
          totalDistortion += bestDist
        }

        distortionHistory += totalDistortion

        // Check convergence (no assignment changes)
        val numChanged = assignments.zip(prevAssignments).count { case (a, b) => a != b }
        converged = numChanged == 0 || numChanged.toDouble / n < $(tol)

        logDebug(f"Iteration $iteration: distortion=$totalDistortion%.4f, changed=$numChanged")

        if (!converged) {
          // Update step: recompute centers using DBA
          centers = computeCentroids(series, assignments, kernel)
        }

        prevAssignments = assignments.clone()
      }

      val elapsed = System.currentTimeMillis() - startTime
      logInfo(s"Time Series K-Means completed: $iteration iterations, converged=$converged")

      // Create model
      val model = new TimeSeriesKMeansModel(uid, centers)
      copyValues(model.setParent(this))

      // Training summary
      model.trainingSummary = Some(
        TrainingSummary(
          algorithm = "TimeSeriesKMeans",
          k = $(k),
          effectiveK = centers.length,
          dim = series.headOption.map(_.size).getOrElse(0),
          numPoints = n,
          iterations = iteration,
          converged = converged,
          distortionHistory = distortionHistory.toArray,
          movementHistory = Array.empty,
          assignmentStrategy = $(kernelType),
          divergence = $(kernelType),
          elapsedMillis = elapsed
        )
      )

      model
    } finally {
      df.unpersist()
    }
  }

  /** Compute centroids using DBA for each cluster. */
  private def computeCentroids(
      series: Array[Vector],
      assignments: Array[Int],
      kernel: SequenceKernel
  ): Array[Vector] = {
    val kVal    = $(k)
    val centers = new Array[Vector](kVal)

    for (c <- 0 until kVal) {
      val clusterSeries = series.indices.filter(assignments(_) == c).map(series(_)).toArray

      if (clusterSeries.isEmpty) {
        // Handle empty cluster: reinitialize with random point
        centers(c) = series(scala.util.Random.nextInt(series.length))
        logWarning(s"Empty cluster $c, reinitializing with random series")
      } else if (clusterSeries.length == 1) {
        centers(c) = clusterSeries(0)
      } else {
        // Use DBA to compute barycenter
        centers(c) = kernel.barycenter(clusterSeries, None, $(dbaIter))
      }
    }

    centers
  }

  /** Create sequence kernel based on parameters. */
  private def createKernel(): SequenceKernel = {
    val windowOpt = if ($(window) > 0) Some($(window)) else None
    SequenceKernel.create($(kernelType), windowOpt, $(gamma), $(sigma))
  }

  override def copy(extra: ParamMap): TimeSeriesKMeans = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    require(
      schema($(featuresCol)).dataType.typeName == "vector",
      s"Features column must be Vector type"
    )
    schema
  }
}

object TimeSeriesKMeans extends DefaultParamsReadable[TimeSeriesKMeans] {
  override def load(path: String): TimeSeriesKMeans = super.load(path)
}

/** Model from Time Series K-Means fitting.
  *
  * Contains cluster centroids (as time series) and provides transform for prediction.
  */
class TimeSeriesKMeansModel(
    override val uid: String,
    val clusterCenters: Array[Vector]
) extends Model[TimeSeriesKMeansModel]
    with TimeSeriesKMeansParams
    with MLWritable
    with Logging
    with HasTrainingSummary {

  /** Optional access to training summary. */
  def summaryOption: Option[TrainingSummary] = trainingSummary

  /** Number of clusters. */
  def numClusters: Int = clusterCenters.length

  /** Average length of cluster centroids. */
  def avgCentroidLength: Double =
    if (clusterCenters.isEmpty) 0.0
    else clusterCenters.map(_.size).sum.toDouble / clusterCenters.length

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()

    val bcCenters   = df.sparkSession.sparkContext.broadcast(clusterCenters)
    val kernelType  = $(this.kernelType)
    val windowParam = $(window)
    val gammaParam  = $(gamma)
    val sigmaParam  = $(sigma)

    // UDF for prediction
    val predictUDF = udf { (features: Vector) =>
      val windowOpt = if (windowParam > 0) Some(windowParam) else None
      val kernel    = SequenceKernel.create(kernelType, windowOpt, gammaParam, sigmaParam)

      var bestCluster = 0
      var bestDist    = Double.MaxValue

      for (c <- bcCenters.value.indices) {
        val dist = kernel match {
          case dtw: DTWKernel => dtw.distance(features, bcCenters.value(c))
          case _              => kernel.squaredDistance(features, bcCenters.value(c))
        }
        if (dist < bestDist) {
          bestDist = dist
          bestCluster = c
        }
      }

      bestCluster
    }

    // UDF for distance to assigned center
    val distanceUDF = udf { (features: Vector, prediction: Int) =>
      val windowOpt = if (windowParam > 0) Some(windowParam) else None
      val kernel    = SequenceKernel.create(kernelType, windowOpt, gammaParam, sigmaParam)

      kernel match {
        case dtw: DTWKernel => dtw.distance(features, bcCenters.value(prediction))
        case _              => kernel.squaredDistance(features, bcCenters.value(prediction))
      }
    }

    val withPrediction = df.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
    withPrediction.withColumn(
      $(distanceCol),
      distanceUDF(col($(featuresCol)), col($(predictionCol)))
    )
  }

  override def copy(extra: ParamMap): TimeSeriesKMeansModel = {
    val copied = new TimeSeriesKMeansModel(uid, clusterCenters)
    copyValues(copied, extra).setParent(parent)
    copied.trainingSummary = this.trainingSummary
    copied
  }

  override def transformSchema(schema: StructType): StructType = schema

  override def write: MLWriter = new TimeSeriesKMeansModel.TimeSeriesKMeansModelWriter(this)
}

object TimeSeriesKMeansModel extends MLReadable[TimeSeriesKMeansModel] {

  override def read: MLReader[TimeSeriesKMeansModel] = new TimeSeriesKMeansModelReader

  private class TimeSeriesKMeansModelWriter(instance: TimeSeriesKMeansModel)
      extends MLWriter
      with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession
      logInfo(s"Saving TimeSeriesKMeansModel to $path")

      // Store centers with index as weight (for ordering)
      val centersData = instance.clusterCenters.zipWithIndex.map { case (center, idx) =>
        (idx, 1.0, center)
      }.toIndexedSeq

      val centersHash = writeCenters(spark, path, centersData)

      val params: Map[String, Any] = Map(
        "k"             -> instance.getOrDefault(instance.k),
        "featuresCol"   -> instance.getOrDefault(instance.featuresCol),
        "predictionCol" -> instance.getOrDefault(instance.predictionCol),
        "distanceCol"   -> instance.getOrDefault(instance.distanceCol),
        "kernelType"    -> instance.getOrDefault(instance.kernelType),
        "window"        -> instance.getOrDefault(instance.window),
        "gamma"         -> instance.getOrDefault(instance.gamma),
        "sigma"         -> instance.getOrDefault(instance.sigma),
        "dbaIter"       -> instance.getOrDefault(instance.dbaIter)
      )

      val k   = instance.numClusters
      val dim = instance.clusterCenters.headOption.map(_.size).getOrElse(0)

      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaObj: Map[String, Any]             = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "TimeSeriesKMeansModel",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "k"                  -> k,
        "dim"                -> dim,
        "uid"                -> instance.uid,
        "params"             -> params,
        "centers"            -> Map[String, Any](
          "count"    -> k,
          "ordering" -> "center_id ASC",
          "storage"  -> "parquet"
        ),
        "checksums"          -> Map[String, String](
          "centersParquetSHA256" -> centersHash
        )
      )

      val json = Serialization.write(metaObj)
      writeMetadata(path, json)
      logInfo(s"TimeSeriesKMeansModel saved to $path")
    }
  }

  private class TimeSeriesKMeansModelReader extends MLReader[TimeSeriesKMeansModel] with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): TimeSeriesKMeansModel = {
      val spark = sparkSession
      logInfo(s"Loading TimeSeriesKMeansModel from $path")

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

      val sortedRows = rows.sortBy(_.getInt(0))
      val centers    = sortedRows.map(_.getAs[Vector]("vector"))

      val paramsJ = metaJ \ "params"

      val model = new TimeSeriesKMeansModel(uid, centers)
      model.set(model.k, k)
      model.set(model.featuresCol, (paramsJ \ "featuresCol").extract[String])
      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])
      model.set(model.distanceCol, (paramsJ \ "distanceCol").extract[String])
      model.set(model.kernelType, (paramsJ \ "kernelType").extract[String])
      model.set(model.window, (paramsJ \ "window").extract[Int])
      model.set(model.gamma, (paramsJ \ "gamma").extract[Double])
      model.set(model.sigma, (paramsJ \ "sigma").extract[Double])
      model.set(model.dbaIter, (paramsJ \ "dbaIter").extract[Int])

      logInfo(s"TimeSeriesKMeansModel loaded from $path")
      model
    }
  }
}
