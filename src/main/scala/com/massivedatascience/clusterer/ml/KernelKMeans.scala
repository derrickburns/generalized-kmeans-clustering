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

import com.massivedatascience.clusterer.ml.df.kernels.{ MercerKernel, RBFKernel }
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{ Estimator, Model }
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{ DefaultParamsReadable, DefaultParamsWritable, Identifiable, MLReadable, MLReader, MLWritable, MLWriter }
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Parameters for Kernel K-Means clustering.
  */
trait KernelKMeansParams
    extends Params
    with HasFeaturesCol
    with HasPredictionCol
    with HasMaxIter
    with HasTol
    with HasSeed {

  /** Number of clusters (k). */
  final val k: IntParam = new IntParam(this, "k", "Number of clusters", ParamValidators.gt(1))
  def getK: Int         = $(k)

  /** Mercer kernel type: "rbf", "polynomial", "linear", "laplacian".
    * Note: This is distinct from Bregman divergence (BregmanKernel).
    */
  final val kernelType: Param[String] = new Param[String](
    this,
    "kernelType",
    "Mercer kernel type",
    ParamValidators.inArray(Array("rbf", "polynomial", "linear", "laplacian", "sigmoid"))
  )
  def getKernelType: String           = $(kernelType)

  /** Kernel bandwidth/scaling parameter (gamma).
    * For RBF: k(x,y) = exp(-gamma * ||x-y||²)
    * For polynomial: k(x,y) = (gamma * <x,y> + coef0)^degree
    */
  final val gamma: DoubleParam = new DoubleParam(
    this,
    "gamma",
    "Kernel bandwidth parameter",
    ParamValidators.gt(0.0)
  )
  def getGamma: Double         = $(gamma)

  /** Polynomial degree (for polynomial kernel). */
  final val degree: IntParam = new IntParam(
    this,
    "degree",
    "Polynomial kernel degree",
    ParamValidators.gtEq(1)
  )
  def getDegree: Int         = $(degree)

  /** Constant term for polynomial/sigmoid kernels. */
  final val coef0: DoubleParam = new DoubleParam(this, "coef0", "Kernel constant term")
  def getCoef0: Double         = $(coef0)

  /** Whether to use automatic gamma selection (median heuristic). */
  final val autoGamma: BooleanParam = new BooleanParam(
    this,
    "autoGamma",
    "Use median heuristic for gamma"
  )
  def getAutoGamma: Boolean         = $(autoGamma)

  setDefault(
    k           -> 2,
    kernelType  -> "rbf",
    gamma       -> 1.0,
    degree      -> 3,
    coef0       -> 0.0,
    autoGamma   -> false,
    maxIter     -> 20,
    tol         -> 1e-4,
    featuresCol -> "features",
    predictionCol -> "prediction"
  )
}

/** Kernel K-Means clustering for non-linearly separable data.
  *
  * Uses positive-definite Mercer kernels to implicitly map data to
  * high-dimensional feature spaces where clusters become linearly separable.
  *
  * ==Algorithm==
  *
  * The key insight is that K-Means only requires distances, which can be
  * computed in kernel space without explicit feature mapping:
  *
  * {{{
  * ||φ(x) - μ_c||² = k(x,x) - 2/|C_c| Σ_{j∈C_c} k(x, x_j) + 1/|C_c|² Σ_{i,j∈C_c} k(x_i, x_j)
  * }}}
  *
  * Where φ is the implicit feature map and μ_c = 1/|C_c| Σ_{j∈C_c} φ(x_j).
  *
  * ==Available Kernels==
  *
  *   - '''RBF (Gaussian):''' k(x,y) = exp(-γ||x-y||²) - Local, universal
  *   - '''Polynomial:''' k(x,y) = (γ⟨x,y⟩ + c)^d - Global, captures interactions
  *   - '''Linear:''' k(x,y) = ⟨x,y⟩ - Equivalent to standard K-Means
  *   - '''Laplacian:''' k(x,y) = exp(-γ||x-y||₁) - Robust to outliers
  *
  * ==Example Usage==
  *
  * {{{
  * val kkm = new KernelKMeans()
  *   .setK(3)
  *   .setKernelType("rbf")
  *   .setGamma(0.1)        // or setAutoGamma(true) for median heuristic
  *   .setMaxIter(50)
  *
  * val model = kkm.fit(data)
  * val predictions = model.transform(data)
  * }}}
  *
  * ==Scalability Note==
  *
  * Kernel K-Means requires O(n²) kernel evaluations per iteration.
  * For large datasets, consider:
  *   - Using a representative subset
  *   - Nyström approximation (future enhancement)
  *   - Random Fourier Features (future enhancement)
  *
  * @see
  *   [[MercerKernel]] for kernel definitions
  */
class KernelKMeans(override val uid: String)
    extends Estimator[KernelKMeansModel]
    with KernelKMeansParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("kernelkmeans"))

  // Parameter setters
  def setK(value: Int): this.type             = set(k, value)
  def setKernelType(value: String): this.type = set(kernelType, value)
  def setGamma(value: Double): this.type      = set(gamma, value)
  def setDegree(value: Int): this.type        = set(degree, value)
  def setCoef0(value: Double): this.type      = set(coef0, value)
  def setAutoGamma(value: Boolean): this.type = set(autoGamma, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setMaxIter(value: Int): this.type       = set(maxIter, value)
  def setTol(value: Double): this.type        = set(tol, value)
  def setSeed(value: Long): this.type         = set(seed, value)

  override def fit(dataset: Dataset[_]): KernelKMeansModel = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()

    logInfo(s"Starting Kernel K-Means: k=${$(k)}, kernel=${$(kernelType)}, maxIter=${$(maxIter)}")

    df.cache()
    try {
      val startTime = System.currentTimeMillis()

      // Collect all data points (kernel k-means needs pairwise distances)
      val points = df.select($(featuresCol)).collect().map(_.getAs[Vector](0))
      val n      = points.length

      if (n > 10000) {
        logWarning(s"Large dataset ($n points). Kernel K-Means has O(n²) complexity.")
      }

      // Create kernel (possibly with auto-gamma)
      val kernel = createKernel(points)
      logInfo(s"Using kernel: ${kernel.name}")

      // Precompute kernel matrix diagonal (k(x_i, x_i))
      val diagK = points.map(p => kernel(p, p))

      // Initialize cluster assignments randomly
      val rng = new scala.util.Random($(seed))
      var assignments = Array.fill(n)(rng.nextInt($(k)))

      var iteration           = 0
      var converged           = false
      val distortionHistory   = scala.collection.mutable.ArrayBuffer[Double]()

      while (iteration < $(maxIter) && !converged) {
        iteration += 1

        // Compute cluster kernel sums: for each cluster c, compute Σ_{i,j∈c} k(x_i, x_j)
        val clusterSums  = computeClusterKernelSums(points, assignments, kernel)
        val clusterSizes = (0 until $(k)).map(c => assignments.count(_ == c)).toArray

        // Assignment step: assign each point to nearest cluster in kernel space
        val newAssignments = new Array[Int](n)
        var totalDistortion = 0.0

        for (i <- 0 until n) {
          var bestCluster = 0
          var bestDist    = Double.MaxValue

          for (c <- 0 until $(k)) {
            if (clusterSizes(c) > 0) {
              // ||φ(x_i) - μ_c||² = k(x_i,x_i) - 2/|C_c| Σ_j k(x_i,x_j) + 1/|C_c|² Σ_{j,l} k(x_j,x_l)
              val term1 = diagK(i)
              val term2 = 2.0 * computePointClusterSum(points, i, assignments, c, kernel) / clusterSizes(c)
              val term3 = clusterSums(c) / (clusterSizes(c).toDouble * clusterSizes(c))
              val dist  = term1 - term2 + term3

              if (dist < bestDist) {
                bestDist = dist
                bestCluster = c
              }
            }
          }

          newAssignments(i) = bestCluster
          totalDistortion += bestDist
        }

        distortionHistory += totalDistortion

        // Check convergence (number of changed assignments)
        val numChanged = assignments.zip(newAssignments).count { case (a, b) => a != b }
        converged = numChanged == 0 || (numChanged.toDouble / n < $(tol))

        logDebug(f"Iteration $iteration: distortion=$totalDistortion%.4f, changed=$numChanged")

        assignments = newAssignments
      }

      val elapsed = System.currentTimeMillis() - startTime
      logInfo(s"Kernel K-Means completed: $iteration iterations, converged=$converged")

      // Create model with support vectors (all points, since we need kernel evaluations)
      val model = new KernelKMeansModel(uid, points, assignments, kernel)
      copyValues(model.setParent(this))

      // Training summary
      model.trainingSummary = Some(
        TrainingSummary(
          algorithm = "KernelKMeans",
          k = $(k),
          effectiveK = $(k),
          dim = points.headOption.map(_.size).getOrElse(0),
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

  /** Compute Σ_{i,j∈c} k(x_i, x_j) for each cluster c. */
  private def computeClusterKernelSums(
      points: Array[Vector],
      assignments: Array[Int],
      kernel: MercerKernel
  ): Array[Double] = {
    val k    = $(this.k)
    val sums = Array.fill(k)(0.0)

    for (i <- points.indices; j <- points.indices) {
      if (assignments(i) == assignments(j)) {
        sums(assignments(i)) += kernel(points(i), points(j))
      }
    }

    sums
  }

  /** Compute Σ_{j∈c} k(x_i, x_j) for point i and cluster c. */
  private def computePointClusterSum(
      points: Array[Vector],
      pointIdx: Int,
      assignments: Array[Int],
      clusterId: Int,
      kernel: MercerKernel
  ): Double = {
    var sum = 0.0
    for (j <- points.indices) {
      if (assignments(j) == clusterId) {
        sum += kernel(points(pointIdx), points(j))
      }
    }
    sum
  }

  private def createKernel(points: Array[Vector]): MercerKernel = {
    val actualGamma = if ($(autoGamma)) {
      // Median heuristic
      RBFKernel.withMedianHeuristic(points, sampleSize = 1000).gamma
    } else {
      $(gamma)
    }

    MercerKernel.create($(kernelType), actualGamma, $(degree), $(coef0))
  }

  override def copy(extra: ParamMap): KernelKMeans = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    require(
      schema($(featuresCol)).dataType.typeName == "vector",
      s"Features column must be Vector type"
    )
    schema
  }
}

object KernelKMeans extends DefaultParamsReadable[KernelKMeans] {
  override def load(path: String): KernelKMeans = super.load(path)
}

/** Model from Kernel K-Means fitting.
  *
  * Stores support vectors and their cluster assignments for prediction.
  * Prediction assigns new points to nearest cluster in kernel space.
  */
class KernelKMeansModel(
    override val uid: String,
    val supportVectors: Array[Vector],
    val supportAssignments: Array[Int],
    val kernel: MercerKernel
) extends Model[KernelKMeansModel]
    with KernelKMeansParams
    with MLWritable
    with Logging
    with HasTrainingSummary {

  /** Optional access to training summary without throwing. */
  def summaryOption: Option[TrainingSummary] = trainingSummary

  /** Number of support vectors. */
  def numSupportVectors: Int = supportVectors.length

  /** Cluster sizes. */
  def clusterSizes: Array[Int] = {
    val sizes = Array.fill($(k))(0)
    supportAssignments.foreach(c => sizes(c) += 1)
    sizes
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()

    val bcSupportVectors    = df.sparkSession.sparkContext.broadcast(supportVectors)
    val bcSupportAssignments = df.sparkSession.sparkContext.broadcast(supportAssignments)
    val bcKernel            = df.sparkSession.sparkContext.broadcast(kernel)
    val numClusters         = $(k)

    // Precompute cluster kernel sums and sizes
    val clusterSizes = Array.fill(numClusters)(0)
    supportAssignments.foreach(c => clusterSizes(c) += 1)

    val clusterSums = Array.fill(numClusters)(0.0)
    for (i <- supportVectors.indices; j <- supportVectors.indices) {
      if (supportAssignments(i) == supportAssignments(j)) {
        clusterSums(supportAssignments(i)) += kernel(supportVectors(i), supportVectors(j))
      }
    }

    val bcClusterSizes = df.sparkSession.sparkContext.broadcast(clusterSizes)
    val bcClusterSums  = df.sparkSession.sparkContext.broadcast(clusterSums)

    val predictUDF = udf { (features: Vector) =>
      val k     = bcKernel.value
      val diagK = k(features, features)

      var bestCluster = 0
      var bestDist    = Double.MaxValue

      for (c <- 0 until numClusters) {
        val size = bcClusterSizes.value(c)
        if (size > 0) {
          // Compute sum of kernel values to cluster members
          var pointSum = 0.0
          for (j <- bcSupportVectors.value.indices) {
            if (bcSupportAssignments.value(j) == c) {
              pointSum += k(features, bcSupportVectors.value(j))
            }
          }

          val term1 = diagK
          val term2 = 2.0 * pointSum / size
          val term3 = bcClusterSums.value(c) / (size.toDouble * size)
          val dist  = term1 - term2 + term3

          if (dist < bestDist) {
            bestDist = dist
            bestCluster = c
          }
        }
      }

      bestCluster
    }

    df.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  override def copy(extra: ParamMap): KernelKMeansModel = {
    val copied = new KernelKMeansModel(uid, supportVectors, supportAssignments, kernel)
    copyValues(copied, extra).setParent(parent)
    copied.trainingSummary = this.trainingSummary
    copied
  }

  override def transformSchema(schema: StructType): StructType = schema

  override def write: MLWriter = new KernelKMeansModel.KernelKMeansModelWriter(this)
}

object KernelKMeansModel extends MLReadable[KernelKMeansModel] {

  override def read: MLReader[KernelKMeansModel] = new KernelKMeansModelReader

  private class KernelKMeansModelWriter(instance: KernelKMeansModel)
      extends MLWriter
      with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession
      logInfo(s"Saving KernelKMeansModel to $path")

      // Store support vectors with assignments (using weight field for assignment)
      val centersData = instance.supportVectors.indices.map { i =>
        (i, instance.supportAssignments(i).toDouble, instance.supportVectors(i))
      }

      val centersHash = writeCenters(spark, path, centersData)

      val params: Map[String, Any] = Map(
        "k"            -> instance.getOrDefault(instance.k),
        "featuresCol"  -> instance.getOrDefault(instance.featuresCol),
        "predictionCol" -> instance.getOrDefault(instance.predictionCol),
        "kernelType"   -> instance.getOrDefault(instance.kernelType),
        "gamma"        -> instance.getOrDefault(instance.gamma),
        "degree"       -> instance.getOrDefault(instance.degree),
        "coef0"        -> instance.getOrDefault(instance.coef0)
      )

      val k   = instance.getOrDefault(instance.k)
      val dim = instance.supportVectors.headOption.map(_.size).getOrElse(0)

      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaObj: Map[String, Any] = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "KernelKMeansModel",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "k"                  -> k,
        "dim"                -> dim,
        "numSupportVectors"  -> instance.supportVectors.length,
        "uid"                -> instance.uid,
        "kernelName"         -> instance.kernel.name,
        "params"             -> params,
        "supportVectors"     -> Map[String, Any](
          "count"    -> instance.supportVectors.length,
          "ordering" -> "center_id ASC",
          "storage"  -> "parquet"
        ),
        "checksums"          -> Map[String, String](
          "centersParquetSHA256" -> centersHash
        )
      )

      val json = Serialization.write(metaObj)
      writeMetadata(path, json)
      logInfo(s"KernelKMeansModel saved to $path")
    }
  }

  private class KernelKMeansModelReader extends MLReader[KernelKMeansModel] with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): KernelKMeansModel = {
      val spark = sparkSession
      logInfo(s"Loading KernelKMeansModel from $path")

      val metaStr                                = readMetadata(path)
      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaJ                                  = JsonMethods.parse(metaStr)

      val layoutVersion     = (metaJ \ "layoutVersion").extract[Int]
      val k                 = (metaJ \ "k").extract[Int]
      val dim               = (metaJ \ "dim").extract[Int]
      val numSupportVectors = (metaJ \ "numSupportVectors").extract[Int]
      val uid               = (metaJ \ "uid").extract[String]

      val centersDF = readCenters(spark, path)
      val rows      = centersDF.collect()
      validateMetadata(layoutVersion, numSupportVectors, dim, rows.length)

      val sortedRows  = rows.sortBy(_.getInt(0))
      val vectors     = sortedRows.map(_.getAs[Vector]("vector"))
      val assignments = sortedRows.map(_.getDouble(1).toInt)

      val paramsJ    = metaJ \ "params"
      val kernelType = (paramsJ \ "kernelType").extract[String]
      val gamma      = (paramsJ \ "gamma").extract[Double]
      val degree     = (paramsJ \ "degree").extract[Int]
      val coef0      = (paramsJ \ "coef0").extract[Double]

      val kernel = MercerKernel.create(kernelType, gamma, degree, coef0)

      val model = new KernelKMeansModel(uid, vectors, assignments, kernel)
      model.set(model.k, k)
      model.set(model.featuresCol, (paramsJ \ "featuresCol").extract[String])
      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])
      model.set(model.kernelType, kernelType)
      model.set(model.gamma, gamma)
      model.set(model.degree, degree)
      model.set(model.coef0, coef0)

      logInfo(s"KernelKMeansModel loaded from $path")
      model
    }
  }
}
