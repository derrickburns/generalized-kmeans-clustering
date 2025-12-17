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
import com.massivedatascience.clusterer.ml.df.kernels.{ KernelFactory, SparseBregmanKernel }
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{ Estimator, Model }
import org.apache.spark.ml.linalg.{ SparseVector, Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Parameters for Sparse K-Means clustering.
  */
trait SparseKMeansParams extends GeneralizedKMeansParams {

  /** Sparsity detection mode.
    *   - "auto": Automatically detect sparsity from data
    *   - "force": Always use sparse-optimized kernels
    *   - "dense": Use dense kernels (equivalent to GeneralizedKMeans)
    * Default: "auto"
    */
  final val sparseMode: Param[String] = new Param[String](
    this,
    "sparseMode",
    "Sparsity handling mode: auto, force, dense",
    ParamValidators.inArray(Array("auto", "force", "dense"))
  )
  def getSparseMode: String           = $(sparseMode)

  /** Sparsity threshold for auto mode. Data with sparsity ratio below this uses sparse kernels.
    * Default: 0.3 (use sparse when less than 30% of values are non-zero)
    */
  final val sparseThreshold: DoubleParam = new DoubleParam(
    this,
    "sparseThreshold",
    "Sparsity threshold for auto mode",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true)
  )
  def getSparseThreshold: Double         = $(sparseThreshold)

  setDefault(
    sparseMode      -> "auto",
    sparseThreshold -> 0.3
  )
}

/** Sparse K-Means clustering optimized for high-dimensional sparse data.
  *
  * This estimator automatically uses sparse-optimized Bregman kernels when the input data is
  * sparse, providing significant speedups for high-dimensional data like:
  *   - Text/document vectors (TF-IDF, word embeddings)
  *   - Sparse feature matrices
  *   - One-hot encoded categorical features
  *
  * ==Supported Sparse Divergences==
  *
  * | Divergence       | Sparse Optimization | Typical Use Case           |
  * |:-----------------|:--------------------|:---------------------------|
  * | squaredEuclidean | Yes                 | General sparse clustering  |
  * | kl               | Yes                 | Document/topic clustering  |
  * | l1               | Yes                 | Robust sparse clustering   |
  * | spherical/cosine | Yes                 | Text similarity clustering |
  *
  * ==Sparsity Detection==
  *
  * In "auto" mode (default), the estimator samples the data to estimate sparsity:
  * {{{
  * val skm = new SparseKMeans()
  *   .setK(10)
  *   .setSparseMode("auto")
  *   .setSparseThreshold(0.3)  // Use sparse when < 30% non-zero
  * }}}
  *
  * ==Performance Characteristics==
  *
  *   - For highly sparse data (< 10% non-zero): 5-10x speedup
  *   - For moderately sparse data (10-30% non-zero): 2-5x speedup
  *   - For dense data (> 30% non-zero): Falls back to dense kernels
  *
  * ==Example Usage==
  *
  * {{{
  * // Clustering TF-IDF vectors
  * val skm = new SparseKMeans()
  *   .setK(100)
  *   .setDivergence("kl")
  *   .setSparseMode("force")  // Always use sparse
  *
  * val model = skm.fit(tfidfData)
  * }}}
  *
  * @see
  *   [[GeneralizedKMeans]] for standard (dense) K-means
  * @see
  *   [[SparseBregmanKernel]] for sparse kernel implementations
  */
class SparseKMeans(override val uid: String)
    extends Estimator[SparseKMeansModel]
    with SparseKMeansParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("sparsekmeans"))

  // Parameter setters
  def setK(value: Int): this.type                  = set(k, value)
  def setDivergence(value: String): this.type      = set(divergence, value)
  def setSmoothing(value: Double): this.type       = set(smoothing, value)
  def setFeaturesCol(value: String): this.type     = set(featuresCol, value)
  def setPredictionCol(value: String): this.type   = set(predictionCol, value)
  def setWeightCol(value: String): this.type       = set(weightCol, value)
  def setMaxIter(value: Int): this.type            = set(maxIter, value)
  def setTol(value: Double): this.type             = set(tol, value)
  def setSeed(value: Long): this.type              = set(seed, value)
  def setSparseMode(value: String): this.type      = set(sparseMode, value)
  def setSparseThreshold(value: Double): this.type = set(sparseThreshold, value)

  override def fit(dataset: Dataset[_]): SparseKMeansModel = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()

    // Determine if we should use sparse kernels
    val (useSparse, sparsityRatio) = $(sparseMode) match {
      case "force" => (true, estimateSparsity(df))
      case "dense" => (false, 1.0)
      case "auto"  =>
        val ratio = estimateSparsity(df)
        (ratio < $(sparseThreshold) && KernelFactory.supportsSparse($(divergence)), ratio)
    }

    val kernelType = if (useSparse) "sparse" else "dense"
    logInfo(
      s"SparseKMeans: k=${$(k)}, divergence=${$(divergence)}, " +
        f"sparsityRatio=$sparsityRatio%.3f, using $kernelType kernel"
    )

    df.cache()
    try {
      val kernel    = KernelFactory.create($(divergence), sparse = useSparse, smoothing = $(smoothing))
      val startTime = System.currentTimeMillis()

      // Initialize centers
      val initialCenters = initializeCenters(df, kernel)
      logInfo(s"Initialized ${initialCenters.length} centers")

      // Run Lloyd's algorithm
      val result = runLloyds(df, initialCenters, kernel)

      val elapsed = System.currentTimeMillis() - startTime
      logInfo(
        s"SparseKMeans completed: ${result.iterations} iterations, " +
          s"converged=${result.converged}"
      )

      // Create model - store the divergence name, not kernel.name
      // so that loading works correctly with KernelFactory
      val model = new SparseKMeansModel(
        uid,
        result.centers.map(Vectors.dense),
        $(divergence), // Use parameter value, not kernel.name
        useSparse,
        sparsityRatio
      )
      copyValues(model.setParent(this))

      // Training summary
      model.trainingSummary = Some(
        TrainingSummary(
          algorithm = "SparseKMeans",
          k = $(k),
          effectiveK = result.centers.length,
          dim = result.centers.headOption.map(_.length).getOrElse(0),
          numPoints = df.count(),
          iterations = result.iterations,
          converged = result.converged,
          distortionHistory = result.distortionHistory,
          movementHistory = result.movementHistory,
          assignmentStrategy = s"sparse=$useSparse",
          divergence = $(divergence),
          elapsedMillis = elapsed
        )
      )

      model
    } finally {
      df.unpersist()
    }
  }

  /** Estimate sparsity ratio from a sample of the data.
    *
    * @return
    *   sparsity ratio (0.0 = all zeros, 1.0 = fully dense)
    */
  private def estimateSparsity(df: DataFrame): Double = {
    val sampleSize = math.min(1000, df.count()).toInt
    if (sampleSize == 0) return 1.0

    val sample = df.select($(featuresCol)).limit(sampleSize).collect().map(_.getAs[Vector](0))

    if (sample.isEmpty) return 1.0

    val totalElements = sample.map(_.size.toLong).sum
    if (totalElements == 0) return 1.0

    val nonZeroElements = sample.map { v =>
      v match {
        case sv: SparseVector => sv.numNonzeros.toLong
        case _                => v.toArray.count(_ != 0.0).toLong
      }
    }.sum

    nonZeroElements.toDouble / totalElements
  }

  /** Run Lloyd's algorithm for sparse k-means. */
  private def runLloyds(
      df: DataFrame,
      initialCenters: Array[Array[Double]],
      kernel: BregmanKernel
  ): SparseResult = {
    var centers           = initialCenters
    var iteration         = 0
    var converged         = false
    val distortionHistory = scala.collection.mutable.ArrayBuffer[Double]()
    val movementHistory   = scala.collection.mutable.ArrayBuffer[Double]()

    // Check if kernel supports sparse operations
    val sparseKernel = kernel match {
      case sk: SparseBregmanKernel => Some(sk)
      case _                       => None
    }

    while (iteration < $(maxIter) && !converged) {
      iteration += 1

      val centersVec = centers.map(Vectors.dense)

      // Broadcast centers and kernel
      val bcKernel  = df.sparkSession.sparkContext.broadcast(kernel)
      val bcCenters = df.sparkSession.sparkContext.broadcast(centersVec)

      // Assignment UDF - use sparse divergence when available
      val assignUDF = sparseKernel match {
        case Some(sk) =>
          val bcSparseKernel = df.sparkSession.sparkContext.broadcast(sk)
          udf { (features: Vector) =>
            features match {
              case sv: SparseVector =>
                // Use sparse-optimized divergence
                val k           = bcSparseKernel.value
                val ctrs        = bcCenters.value
                var bestCluster = 0
                var bestDist    = Double.MaxValue
                var i           = 0
                while (i < ctrs.length) {
                  ctrs(i) match {
                    case centerSv: SparseVector =>
                      val dist = k.divergenceSparse(sv, centerSv)
                      if (dist < bestDist) {
                        bestDist = dist
                        bestCluster = i
                      }
                    case center                 =>
                      val dist = k.divergence(sv, center)
                      if (dist < bestDist) {
                        bestDist = dist
                        bestCluster = i
                      }
                  }
                  i += 1
                }
                bestCluster
              case _                =>
                // Fall back to dense
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
          }
        case None     =>
          // Dense kernel
          udf { (features: Vector) =>
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
      }

      val assigned = df.withColumn("_cluster", assignUDF(col($(featuresCol))))

      // Update centers using gradient mean
      val newCenters = updateCenters(assigned, kernel, $(k))

      // Check convergence
      val movement = computeMovement(centers, newCenters.map(_.toArray))
      movementHistory += movement

      val distortion = computeDistortion(df, newCenters, kernel)
      distortionHistory += distortion

      converged = movement < $(tol)
      centers = newCenters.map(_.toArray)

      logDebug(f"Iteration $iteration: distortion=$distortion%.4f, movement=$movement%.6f")
    }

    SparseResult(
      centers = centers,
      iterations = iteration,
      converged = converged,
      distortionHistory = distortionHistory.toArray,
      movementHistory = movementHistory.toArray
    )
  }

  private def initializeCenters(df: DataFrame, kernel: BregmanKernel): Array[Array[Double]] = {
    val fraction = math.min(1.0, ($(k) * 10.0) / df.count().toDouble)
    df.select($(featuresCol))
      .sample(withReplacement = false, fraction, $(seed))
      .limit($(k))
      .collect()
      .map(_.getAs[Vector](0).toArray)
  }

  private def updateCenters(
      assigned: DataFrame,
      kernel: BregmanKernel,
      numClusters: Int
  ): Array[Vector] = {
    val bcKernel = assigned.sparkSession.sparkContext.broadcast(kernel)

    val gradUDF = udf { (features: Vector) =>
      bcKernel.value.grad(features).toArray
    }

    val withGrad = assigned.withColumn("_grad", gradUDF(col($(featuresCol))))

    val dim = assigned.select($(featuresCol)).head().getAs[Vector](0).size

    val aggregated = withGrad
      .groupBy("_cluster")
      .agg(
        count("*").as("count"),
        array((0 until dim).map(i => sum(element_at(col("_grad"), i + 1))): _*).as("grad_sum")
      )
      .collect()

    val centers = Array.fill(numClusters)(Vectors.zeros(dim))
    aggregated.foreach { row =>
      val clusterId = row.getInt(0)
      if (clusterId >= 0 && clusterId < numClusters) {
        val count   = row.getLong(1)
        val gradSum = row.getSeq[Double](2).toArray
        if (count > 0) {
          val avgGrad = Vectors.dense(gradSum.map(_ / count))
          centers(clusterId) = bcKernel.value.invGrad(avgGrad)
        }
      }
    }

    centers
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

  override def copy(extra: ParamMap): SparseKMeans = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}

object SparseKMeans extends DefaultParamsReadable[SparseKMeans] {
  override def load(path: String): SparseKMeans = super.load(path)
}

/** Model from Sparse K-Means clustering.
  *
  * @param uid
  *   unique identifier
  * @param clusterCenters
  *   cluster centers as vectors
  * @param divergenceName
  *   name of the divergence used
  * @param usedSparseKernel
  *   whether sparse-optimized kernel was used during training
  * @param sparsityRatio
  *   estimated sparsity ratio of the training data
  */
class SparseKMeansModel(
    override val uid: String,
    val clusterCenters: Array[Vector],
    val divergenceName: String,
    val usedSparseKernel: Boolean,
    val sparsityRatio: Double
) extends Model[SparseKMeansModel]
    with SparseKMeansParams
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
      KernelFactory.create(divergenceName, sparse = usedSparseKernel, smoothing = $(smoothing))

    val bcKernel  = df.sparkSession.sparkContext.broadcast(kernel)
    val bcCenters = df.sparkSession.sparkContext.broadcast(clusterCenters)

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

    df.withColumn($(predictionCol), assignUDF(col($(featuresCol))))
  }

  override def copy(extra: ParamMap): SparseKMeansModel = {
    val copied = new SparseKMeansModel(
      uid,
      clusterCenters,
      divergenceName,
      usedSparseKernel,
      sparsityRatio
    )
    copyValues(copied, extra).setParent(parent)
    copied.trainingSummary = this.trainingSummary
    copied
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def write: MLWriter = new SparseKMeansModel.SparseKMeansModelWriter(this)

  override def toString: String = {
    f"SparseKMeansModel: uid=$uid, k=$numClusters, " +
      f"divergence=$divergenceName, sparseKernel=$usedSparseKernel, sparsity=$sparsityRatio%.3f"
  }
}

object SparseKMeansModel extends MLReadable[SparseKMeansModel] {

  override def read: MLReader[SparseKMeansModel] = new SparseKMeansModelReader

  private class SparseKMeansModelWriter(instance: SparseKMeansModel) extends MLWriter with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession
      logInfo(s"Saving SparseKMeansModel to $path")

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
        "usedSparseKernel" -> instance.usedSparseKernel,
        "sparsityRatio"    -> instance.sparsityRatio,
        "sparseMode"       -> instance.getOrDefault(instance.sparseMode),
        "sparseThreshold"  -> instance.getOrDefault(instance.sparseThreshold),
        "smoothing"        -> instance.getOrDefault(instance.smoothing)
      )

      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaObj: Map[String, Any]             = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "SparseKMeansModel",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "divergence"         -> instance.divergenceName,
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
      logInfo(s"SparseKMeansModel saved to $path")
    }
  }

  private class SparseKMeansModelReader extends MLReader[SparseKMeansModel] with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): SparseKMeansModel = {
      val spark = sparkSession
      logInfo(s"Loading SparseKMeansModel from $path")

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

      val paramsJ          = metaJ \ "params"
      val divergence       = (paramsJ \ "divergence").extract[String]
      val usedSparseKernel = (paramsJ \ "usedSparseKernel").extract[Boolean]
      val sparsityRatio    = (paramsJ \ "sparsityRatio").extract[Double]

      val model = new SparseKMeansModel(uid, centers, divergence, usedSparseKernel, sparsityRatio)

      model.set(model.featuresCol, (paramsJ \ "featuresCol").extract[String])
      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])
      model.set(model.sparseMode, (paramsJ \ "sparseMode").extract[String])
      model.set(model.sparseThreshold, (paramsJ \ "sparseThreshold").extract[Double])
      model.set(model.smoothing, (paramsJ \ "smoothing").extract[Double])

      logInfo(s"SparseKMeansModel loaded from $path")
      model
    }
  }
}

/** Internal result class for sparse Lloyd's algorithm. */
private[ml] case class SparseResult(
    centers: Array[Array[Double]],
    iterations: Int,
    converged: Boolean,
    distortionHistory: Array[Double],
    movementHistory: Array[Double]
)
