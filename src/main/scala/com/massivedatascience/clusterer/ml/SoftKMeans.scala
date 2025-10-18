/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
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

import com.massivedatascience.clusterer.ml.df.BregmanKernel
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.internal.Logging

/** Parameters for Soft K-Means clustering (fuzzy c-means).
  */
trait SoftKMeansParams extends GeneralizedKMeansParams {

  /** Inverse temperature parameter controlling soft assignment sharpness.
    *
    * Higher values → sharper assignments (approaches hard clustering) Lower values → softer assignments (more fuzzy
    * membership)
    *
    * Default: 1.0
    */
  final val beta = new DoubleParam(
    this,
    "beta",
    "Inverse temperature parameter for soft assignments",
    ParamValidators.gt(0.0)
  )

  def getBeta: Double = $(beta)

  /** Minimum membership probability to avoid numerical issues. Default: 1e-10
    */
  final val minMembership = new DoubleParam(
    this,
    "minMembership",
    "Minimum membership probability",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = true, upperInclusive = true)
  )

  def getMinMembership: Double = $(minMembership)

  /** Column name for membership probabilities output. Default: "probabilities"
    */
  final val probabilityCol = new Param[String](
    this,
    "probabilityCol",
    "Column name for membership probabilities"
  )

  def getProbabilityCol: String = $(probabilityCol)

  setDefault(
    beta -> 1.0,
    minMembership -> 1e-10,
    probabilityCol -> "probabilities"
  )
}

/** Soft K-Means clustering (fuzzy c-means) - DataFrame API.
  *
  * Soft K-Means extends K-means by computing probabilistic cluster memberships instead of hard assignments. Each point
  * has a membership probability for each cluster, computed using:
  *
  * p(cluster c | point x) ∝ exp(-β * D_φ(x, μ_c))
  *
  * where β is the inverse temperature parameter and D_φ is the Bregman divergence.
  *
  * Benefits:
  *   - Fuzzy clustering: points can belong to multiple clusters
  *   - Better modeling of overlapping clusters
  *   - Useful for mixture model estimation
  *   - Provides uncertainty estimates
  *
  * Algorithm:
  *   1. E-step: Compute soft assignments using Boltzmann distribution
  *   2. M-step: Update centers as weighted averages with membership weights
  *   3. Repeat until convergence
  *
  * Example usage:
  * {{{
  *   val softKMeans = new SoftKMeans()
  *     .setK(3)
  *     .setBeta(2.0)
  *     .setDivergence("squaredEuclidean")
  *     .setMaxIter(100)
  *
  *   val model = softKMeans.fit(dataset)
  *   val predictions = model.transform(dataset)
  *   // predictions contains: prediction (hard assignment), probabilities (soft memberships)
  * }}}
  *
  * @param uid
  *   Unique identifier
  */
class SoftKMeans(override val uid: String)
    extends Estimator[SoftKMeansModel]
    with SoftKMeansParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("softkmeans"))

  // Parameter setters

  def setBeta(value: Double): this.type = set(beta, value)
  def setMinMembership(value: Double): this.type = set(minMembership, value)
  def setProbabilityCol(value: String): this.type = set(probabilityCol, value)

  // Inherited parameter setters
  def setK(value: Int): this.type = set(k, value)
  def setDivergence(value: String): this.type = set(divergence, value)
  def setSmoothing(value: Double): this.type = set(smoothing, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setWeightCol(value: String): this.type = set(weightCol, value)
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setTol(value: Double): this.type = set(tol, value)
  def setSeed(value: Long): this.type = set(seed, value)
  def setDistanceCol(value: String): this.type = set(distanceCol, value)

  override def fit(dataset: Dataset[_]): SoftKMeansModel = {
    transformSchema(dataset.schema, logging = true)

    val df = dataset.toDF()

    logInfo(
      s"Starting Soft K-Means: k=${$(k)}, β=${$(beta)}, " +
        s"divergence=${$(divergence)}, maxIter=${$(maxIter)}"
    )

    // Cache data for iterative algorithm
    df.cache()

    try {
      // Create kernel for distance calculations
      val kernel = createKernel($(divergence), $(smoothing))

      // Initialize centers
      val initialCenters = initializeCenters(
        df,
        $(featuresCol),
        if (hasWeightCol) Some($(weightCol)) else None,
        kernel
      )

      logInfo(s"Initialized ${initialCenters.length} centers")

      // Convert to Vector objects
      var centers = initialCenters.map(Vectors.dense)
      var previousCost = Double.MaxValue
      var iteration = 0
      var converged = false

      while (iteration < $(maxIter) && !converged) {
        iteration += 1
        logDebug(s"Soft K-Means iteration $iteration")

        // E-step: Compute soft assignments
        val memberships = computeSoftAssignments(df, centers, kernel)

        // M-step: Update centers using soft assignments
        val newCenters = computeSoftCenters(memberships, centers.length, kernel)

        // Check convergence
        val cost = computeSoftCost(memberships, newCenters, kernel)
        val improvement = (previousCost - cost) / math.max(math.abs(previousCost), 1e-10)

        logDebug(f"Iteration $iteration: cost = $cost%.6f, improvement = $improvement%.8f")

        if (improvement < $(tol)) {
          converged = true
          logInfo(s"Soft K-Means converged after $iteration iterations")
        }

        previousCost = cost
        centers = newCenters
      }

      if (!converged) {
        logWarning(s"Soft K-Means did not converge after ${$(maxIter)} iterations")
      }

      // Create model
      val model = new SoftKMeansModel(uid, centers, $(beta), $(minMembership), kernel)
      model.modelDivergence = $(divergence)
      model.modelSmoothing = $(smoothing)
      copyValues(model.setParent(this))

    } finally {
      df.unpersist()
    }
  }

  /** Create kernel for distance calculations.
    */
  private def createKernel(divName: String, smooth: Double): BregmanKernel = {
    import com.massivedatascience.clusterer.ml.df._
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

  /** Initialize cluster centers.
    */
  private def initializeCenters(
    df: DataFrame,
    featuresCol: String,
    weightCol: Option[String],
    kernel: BregmanKernel
  ): Array[Array[Double]] = {

    // For soft k-means, use random initialization by default (simpler and often works well)
    initializeRandom(df, featuresCol, $(k), $(seed))
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

  /** Compute soft assignment probabilities for all points.
    *
    * Uses the Boltzmann distribution: p(c|x) ∝ exp(-β * D_φ(x, μ_c))
    *
    * Uses the log-sum-exp trick for numerical stability.
    */
  private def computeSoftAssignments(
    df: DataFrame,
    centers: Array[Vector],
    kernel: BregmanKernel
  ): DataFrame = {

    val betaValue = $(beta)
    val minMembershipValue = $(minMembership)

    // Create UDF for soft assignment
    val softAssignUDF = udf { (features: Vector) =>
      // Compute distances to all centers
      val distances = centers.map(center => kernel.divergence(features, center))

      // Compute unnormalized probabilities: exp(-β * distance)
      // Use min distance for numerical stability (log-sum-exp trick)
      val minDistance = distances.min
      val unnormalizedProbs = distances.map { dist =>
        math.exp(-betaValue * (dist - minDistance))
      }

      // Normalize to get probabilities
      val totalProb = unnormalizedProbs.sum
      val probabilities: Array[Double] = if (totalProb > 1e-100) {
        unnormalizedProbs.map(_ / totalProb).toArray
      } else {
        // Fallback: uniform distribution
        Array.fill(centers.length)(1.0 / centers.length)
      }

      // Apply minimum membership threshold
      val adjustedProbs = probabilities.map(p => math.max(p, minMembershipValue))
      val adjustedSum = adjustedProbs.sum
      val normalizedProbs = adjustedProbs.map(_ / adjustedSum)

      Vectors.dense(normalizedProbs)
    }

    df.withColumn("probabilities", softAssignUDF(col($(featuresCol))))
  }

  /** Compute new cluster centers using soft assignments.
    *
    * Each center is the weighted average of all points, where weights are membership probabilities.
    */
  private def computeSoftCenters(
    memberships: DataFrame,
    numClusters: Int,
    kernel: BregmanKernel
  ): Array[Vector] = {

    val featCol: String = $(featuresCol)
    val weightColOpt: Option[String] = if (hasWeightCol) Some($(weightCol)) else None

    // Collect membership data to driver (needed for weighted averaging)
    // For very large datasets, this could be memory-intensive
    val membershipData = if (weightColOpt.isDefined) {
      memberships.select(featCol, "probabilities", weightColOpt.get).collect()
    } else {
      memberships.select(featCol, "probabilities").collect()
    }

    // For each cluster, compute weighted sum of points
    val clusterCenters = (0 until numClusters).map { clusterId =>
      var weightedSum: Array[Double] = null
      var totalWeight = 0.0

      membershipData.foreach { row =>
        val features = row.getAs[Vector](0).toArray
        val probs = row.getAs[Vector](1).toArray
        val clusterProb = probs(clusterId)

        val finalWeight = weightColOpt match {
          case Some(wCol) => clusterProb * row.getAs[Double](2) // Third column is weight
          case None => clusterProb
        }

        if (weightedSum == null) {
          weightedSum = Array.fill(features.length)(0.0)
        }

        // Add weighted contribution
        var i = 0
        while (i < features.length) {
          weightedSum(i) += features(i) * finalWeight
          i += 1
        }
        totalWeight += finalWeight
      }

      if (totalWeight > 1e-10) {
        Vectors.dense(weightedSum.map(_ / totalWeight))
      } else {
        // Empty cluster - return zero vector (will be handled by caller)
        logWarning(s"Cluster $clusterId has insufficient soft membership weight")
        Vectors.dense(Array.fill(weightedSum.length)(0.0))
      }
    }.toArray

    clusterCenters
  }

  /** Compute the soft clustering cost.
    *
    * Cost = Σ_x Σ_c p(c|x) * D_φ(x, μ_c)
    */
  private def computeSoftCost(
    memberships: DataFrame,
    centers: Array[Vector],
    kernel: BregmanKernel
  ): Double = {

    val costUDF = udf { (features: Vector, probs: Vector) =>
      probs.toArray.zipWithIndex.map { case (prob, clusterId) =>
        val distance = kernel.divergence(features, centers(clusterId))
        prob * distance
      }.sum
    }

    memberships
      .select(costUDF(col($(featuresCol)), col("probabilities")).as("cost"))
      .agg(sum("cost"))
      .head()
      .getDouble(0)
  }

  override def copy(extra: ParamMap): SoftKMeans = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}

object SoftKMeans extends DefaultParamsReadable[SoftKMeans] {
  override def load(path: String): SoftKMeans = super.load(path)
}

/** Model produced by Soft K-Means clustering.
  *
  * Contains cluster centers and methods for computing soft and hard assignments.
  *
  * @param uid
  *   Unique identifier
  * @param clusterCenters
  *   Final cluster centers
  * @param beta
  *   Inverse temperature parameter
  * @param minMembership
  *   Minimum membership probability
  * @param kernel
  *   Kernel for distance calculations
  */
class SoftKMeansModel(
  override val uid: String,
  val clusterCenters: Array[Vector],
  val betaValue: Double,
  val minMembershipValue: Double,
  private val kernel: BregmanKernel
) extends org.apache.spark.ml.Model[SoftKMeansModel]
    with SoftKMeansParams
    with MLWritable
    with Logging {

  def numClusters: Int = clusterCenters.length

  /** Transform dataset by adding prediction and probability columns.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val df = dataset.toDF()

    // Add soft assignments (probabilities)
    val withProbabilities = computeSoftAssignments(df)

    // Add hard assignment (prediction)
    val hardAssignUDF = udf { (probs: Vector) =>
      probs.toArray.zipWithIndex.maxBy(_._1)._2
    }

    val result = withProbabilities
      .withColumn($(predictionCol), hardAssignUDF(col($(probabilityCol))))

    // Add distance column if requested
    if (hasDistanceCol) {
      val distanceUDF = udf { (features: Vector, prediction: Int) =>
        kernel.divergence(features, clusterCenters(prediction))
      }
      result.withColumn($(distanceCol), distanceUDF(col($(featuresCol)), col($(predictionCol))))
    } else {
      result
    }
  }

  /** Compute soft assignment probabilities.
    */
  private def computeSoftAssignments(df: DataFrame): DataFrame = {
    val softAssignUDF = udf { (features: Vector) =>
      // Compute distances to all centers
      val distances = clusterCenters.map(center => kernel.divergence(features, center))

      // Compute unnormalized probabilities: exp(-β * distance)
      val minDistance = distances.min
      val unnormalizedProbs = distances.map { dist =>
        math.exp(-betaValue * (dist - minDistance))
      }

      // Normalize
      val totalProb = unnormalizedProbs.sum
      val probabilities: Array[Double] = if (totalProb > 1e-100) {
        unnormalizedProbs.map(_ / totalProb).toArray
      } else {
        Array.fill(clusterCenters.length)(1.0 / clusterCenters.length)
      }

      // Apply minimum membership threshold
      val adjustedProbs = probabilities.map(p => math.max(p, minMembershipValue))
      val adjustedSum = adjustedProbs.sum
      val normalizedProbs = adjustedProbs.map(_ / adjustedSum)

      Vectors.dense(normalizedProbs)
    }

    df.withColumn($(probabilityCol), softAssignUDF(col($(featuresCol))))
  }

  /** Predict cluster for a single vector (hard assignment).
    */
  def predict(features: Vector): Int = {
    val distances = clusterCenters.map(center => kernel.divergence(features, center))
    distances.zipWithIndex.minBy(_._1)._2
  }

  /** Compute soft probabilities for a single vector.
    */
  def predictSoft(features: Vector): Vector = {
    val distances = clusterCenters.map(center => kernel.divergence(features, center))

    val minDistance = distances.min
    val unnormalizedProbs = distances.map { dist =>
      math.exp(-betaValue * (dist - minDistance))
    }

    val totalProb = unnormalizedProbs.sum
    val probabilities: Array[Double] = if (totalProb > 1e-100) {
      unnormalizedProbs.map(_ / totalProb).toArray
    } else {
      Array.fill(clusterCenters.length)(1.0 / clusterCenters.length)
    }

    val adjustedProbs = probabilities.map(p => math.max(p, minMembershipValue))
    val adjustedSum = adjustedProbs.sum
    val normalizedProbs = adjustedProbs.map(_ / adjustedSum)

    Vectors.dense(normalizedProbs)
  }

  /** Compute clustering cost.
    */
  def computeCost(dataset: Dataset[_]): Double = {
    val df = dataset.toDF()

    val costUDF = udf { (features: Vector) =>
      val prediction = predict(features)
      kernel.divergence(features, clusterCenters(prediction))
    }

    df.select(costUDF(col($(featuresCol))).as("cost"))
      .agg(sum("cost"))
      .head()
      .getDouble(0)
  }

  /** Compute soft clustering cost.
    */
  def computeSoftCost(dataset: Dataset[_]): Double = {
    val df = dataset.toDF()
    val withProbs = computeSoftAssignments(df)

    val costUDF = udf { (features: Vector, probs: Vector) =>
      probs.toArray.zipWithIndex.map { case (prob, clusterId) =>
        val distance = kernel.divergence(features, clusterCenters(clusterId))
        prob * distance
      }.sum
    }

    withProbs
      .select(costUDF(col($(featuresCol)), col($(probabilityCol))).as("cost"))
      .agg(sum("cost"))
      .head()
      .getDouble(0)
  }

  /** Compute effective number of clusters (entropy-based measure).
    */
  def effectiveNumberOfClusters(dataset: Dataset[_]): Double = {
    val df = dataset.toDF()
    val withProbs = computeSoftAssignments(df)

    val entropyUDF = udf { (probs: Vector) =>
      val entropy = -probs.toArray.map(p => if (p > minMembershipValue) p * math.log(p) else 0.0).sum
      math.exp(entropy) // Effective number of clusters for this point
    }

    withProbs
      .select(entropyUDF(col($(probabilityCol))).as("effectiveClusters"))
      .agg(avg("effectiveClusters"))
      .head()
      .getDouble(0)
  }

  override def copy(extra: ParamMap): SoftKMeansModel = {
    val copied = new SoftKMeansModel(uid, clusterCenters, betaValue, minMembershipValue, kernel)
    copyValues(copied, extra).setParent(parent)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
    // Note: probability column will be added dynamically during transform
  }

  override def write: MLWriter = new SoftKMeansModel.SoftKMeansModelWriter(this)

  // Get divergence from model (need to track this)
  private[ml] var modelDivergence: String = "squaredEuclidean"
  private[ml] var modelSmoothing: Double = 1e-10
}

object SoftKMeansModel extends MLReadable[SoftKMeansModel] {

  override def read: MLReader[SoftKMeansModel] = new SoftKMeansModelReader

  override def load(path: String): SoftKMeansModel = super.load(path)

  private class SoftKMeansModelWriter(instance: SoftKMeansModel) extends MLWriter with Logging {

    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession

      logInfo(s"Saving SoftKMeansModel to $path")

      // Prepare centers data: (center_id, weight, vector)
      // For Soft K-Means, all centers have uniform weight (1.0)
      val centersData = instance.clusterCenters.indices.map { i =>
        val weight = 1.0
        val vector = instance.clusterCenters(i)
        (i, weight, vector)
      }

      // Write centers with deterministic ordering
      val centersHash = writeCenters(spark, path, centersData)
      logInfo(s"Centers saved with SHA-256: $centersHash")

      // Collect all model parameters
      val params = Map(
        "k" -> instance.numClusters,
        "featuresCol" -> instance.getOrDefault(instance.featuresCol),
        "predictionCol" -> instance.getOrDefault(instance.predictionCol),
        "probabilityCol" -> instance.getOrDefault(instance.probabilityCol),
        "beta" -> instance.betaValue,
        "minMembership" -> instance.minMembershipValue,
        "divergence" -> instance.modelDivergence,
        "smoothing" -> instance.modelSmoothing
      )

      val k = instance.numClusters
      val dim = instance.clusterCenters.headOption.map(_.size).getOrElse(0)

      // Build metadata object
      implicit val formats = DefaultFormats
      val metaObj = Map(
        "layoutVersion" -> LayoutVersion,
        "algo" -> "SoftKMeansModel",
        "sparkMLVersion" -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "divergence" -> instance.modelDivergence,
        "k" -> k,
        "dim" -> dim,
        "uid" -> instance.uid,
        "params" -> params,
        "centers" -> Map(
          "count" -> k,
          "ordering" -> "center_id ASC (0..k-1)",
          "storage" -> "parquet"
        ),
        "checksums" -> Map(
          "centersParquetSHA256" -> centersHash
        )
      )

      // Serialize to JSON
      val json = Serialization.write(metaObj)(formats)

      // Write metadata
      val metadataHash = writeMetadata(path, json)
      logInfo(s"Metadata saved with SHA-256: $metadataHash")
      logInfo(s"SoftKMeansModel successfully saved to $path")
    }
  }

  private class SoftKMeansModelReader extends MLReader[SoftKMeansModel] with Logging {

    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): SoftKMeansModel = {
      val spark = sparkSession

      logInfo(s"Loading SoftKMeansModel from $path")

      // Read metadata
      val metaStr = readMetadata(path)
      implicit val formats = DefaultFormats
      val metaJ = JsonMethods.parse(metaStr)

      // Extract and validate layout version
      val layoutVersion = (metaJ \ "layoutVersion").extract[Int]
      val k = (metaJ \ "k").extract[Int]
      val dim = (metaJ \ "dim").extract[Int]
      val uid = (metaJ \ "uid").extract[String]
      val divergence = (metaJ \ "divergence").extract[String]

      logInfo(s"Model metadata: layoutVersion=$layoutVersion, k=$k, dim=$dim, divergence=$divergence")

      // Read centers
      val centersDF = readCenters(spark, path)
      val rows = centersDF.collect()

      // Validate metadata
      validateMetadata(layoutVersion, k, dim, rows.length)

      // Extract centers (sorted by center_id)
      val centers = rows.sortBy(_.getInt(0)).map { row =>
        row.getAs[Vector]("vector")
      }

      // Extract parameters
      val paramsJ = metaJ \ "params"
      val beta = (paramsJ \ "beta").extract[Double]
      val minMembership = (paramsJ \ "minMembership").extract[Double]
      val smoothing = (paramsJ \ "smoothing").extract[Double]

      // Create kernel
      import com.massivedatascience.clusterer.ml.df._
      val kernel: BregmanKernel = divergence match {
        case "squaredEuclidean" => new SquaredEuclideanKernel()
        case "kl"               => new KLDivergenceKernel(smoothing)
        case "itakuraSaito"     => new ItakuraSaitoKernel(smoothing)
        case "generalizedI"     => new GeneralizedIDivergenceKernel(smoothing)
        case "logistic"         => new LogisticLossKernel(smoothing)
        case "l1" | "manhattan" => new L1Kernel()
        case _                  => new SquaredEuclideanKernel()
      }

      // Reconstruct model
      val model = new SoftKMeansModel(uid, centers, beta, minMembership, kernel)
      model.modelDivergence = divergence
      model.modelSmoothing = smoothing

      // Set parameters
      model.set(model.featuresCol, (paramsJ \ "featuresCol").extract[String])
      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])
      model.set(model.probabilityCol, (paramsJ \ "probabilityCol").extract[String])

      logInfo(s"SoftKMeansModel successfully loaded from $path")
      model
    }
  }
}
