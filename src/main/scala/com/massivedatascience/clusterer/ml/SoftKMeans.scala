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
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Parameters for Soft K-Means clustering (fuzzy c-means).
  */
trait SoftKMeansParams extends GeneralizedKMeansParams {

  /** Inverse temperature parameter controlling soft assignment sharpness.
    *
    * Higher values → sharper assignments (approaches hard clustering) Lower values → softer
    * assignments (more fuzzy membership)
    *
    * Default: 1.0
    */
  final val beta      = new DoubleParam(
    this,
    "beta",
    "Inverse temperature parameter for soft assignments",
    ParamValidators.gt(0.0)
  )
  def getBeta: Double = $(beta)

  /** Minimum membership probability to avoid numerical issues. Default: 1e-10 */
  final val minMembership      = new DoubleParam(
    this,
    "minMembership",
    "Minimum membership probability",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = true, upperInclusive = true)
  )
  def getMinMembership: Double = $(minMembership)

  /** Column name for membership probabilities output. Default: "probabilities" */
  final val probabilityCol      =
    new Param[String](this, "probabilityCol", "Column name for membership probabilities")
  def getProbabilityCol: String = $(probabilityCol)

  setDefault(
    beta           -> 1.0,
    minMembership  -> 1e-10,
    probabilityCol -> "probabilities"
  )
}

/** Soft K-Means clustering (fuzzy c-means) - DataFrame API.
  *
  * p(cluster c | point x) ∝ exp(-β * D_φ(x, μ_c))
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
  def setBeta(value: Double): this.type           = set(beta, value)
  def setMinMembership(value: Double): this.type  = set(minMembership, value)
  def setProbabilityCol(value: String): this.type = set(probabilityCol, value)

  // Inherited parameter setters
  def setK(value: Int): this.type                = set(k, value)
  def setDivergence(value: String): this.type    = set(divergence, value)
  def setSmoothing(value: Double): this.type     = set(smoothing, value)
  def setFeaturesCol(value: String): this.type   = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setWeightCol(value: String): this.type     = set(weightCol, value)
  def setMaxIter(value: Int): this.type          = set(maxIter, value)
  def setTol(value: Double): this.type           = set(tol, value)
  def setSeed(value: Long): this.type            = set(seed, value)
  def setDistanceCol(value: String): this.type   = set(distanceCol, value)

  override def fit(dataset: Dataset[_]): SoftKMeansModel = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()
    logInfo(
      s"Starting Soft K-Means: k=${$(k)}, β=${$(beta)}, divergence=${$(divergence)}, maxIter=${$(maxIter)}"
    )

    df.cache()
    try {
      val kernel = createKernel($(divergence), $(smoothing))

      // Initialize centers
      val initialCenters = initializeCenters(
        df,
        $(featuresCol),
        if (hasWeightCol) Some($(weightCol)) else None,
        kernel
      )
      logInfo(s"Initialized ${initialCenters.length} centers")

      var centers           = initialCenters.map(Vectors.dense)
      var previousCost      = Double.MaxValue
      var iteration         = 0
      var converged         = false
      val distortionHistory = scala.collection.mutable.ArrayBuffer[Double]()
      val movementHistory   = scala.collection.mutable.ArrayBuffer[Double]()
      val startTime         = System.currentTimeMillis()

      while (iteration < $(maxIter) && !converged) {
        iteration += 1
        logDebug(s"Soft K-Means iteration $iteration")

        // E-step
        val memberships = computeSoftAssignments(df, centers, kernel)

        // M-step
        val newCenters = computeSoftCenters(memberships, centers.length, kernel)

        // Movement
        val maxMovement = centers
          .zip(newCenters)
          .map { case (old, newC) =>
            math.sqrt(old.toArray.zip(newC.toArray).map { case (a, b) => val d = a - b; d * d }.sum)
          }
          .max
        movementHistory += maxMovement

        // Cost / convergence
        val cost        = computeSoftCost(memberships, newCenters, kernel)
        distortionHistory += cost
        val improvement = (previousCost - cost) / math.max(math.abs(previousCost), 1e-10)
        logDebug(
          f"Iteration $iteration: cost = $cost%.6f, improvement = $improvement%.8f, maxMove = $maxMovement%.6f"
        )

        if (improvement < $(tol)) {
          converged = true
          logInfo(s"Soft K-Means converged after $iteration iterations")
        }

        previousCost = cost
        centers = newCenters
      }

      val elapsed = System.currentTimeMillis() - startTime
      if (!converged) logWarning(s"Soft K-Means did not converge after ${$(maxIter)} iterations")

      val model = new SoftKMeansModel(uid, centers, $(beta), $(minMembership), kernel)
      model.modelDivergence = $(divergence)
      model.modelSmoothing = $(smoothing)
      copyValues(model.setParent(this))

      val dim     = centers.headOption.map(_.size).getOrElse(0)
      val summary = TrainingSummary(
        algorithm = "SoftKMeans",
        k = $(k),
        effectiveK = centers.length,
        dim = dim,
        numPoints = df.count(),
        iterations = iteration,
        converged = converged,
        distortionHistory = distortionHistory.toArray,
        movementHistory = movementHistory.toArray,
        assignmentStrategy = "SoftEM",
        divergence = $(divergence),
        elapsedMillis = elapsed
      )
      model.trainingSummary = Some(summary)
      logInfo(s"Training summary:\n${summary.convergenceReport}")

      model
    } finally {
      df.unpersist()
    }
  }

  /** Create kernel for distance calculations. */
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

  /** Initialize cluster centers. */
  private def initializeCenters(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      kernel: BregmanKernel
  ): Array[Array[Double]] = {
    // Random initialization by default
    initializeRandom(df, featuresCol, $(k), $(seed))
  }

  /** Random initialization: sample k random points. */
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
      .map(_.getAs.toArray)
  }

  /** Compute soft assignment probabilities for all points (Boltzmann distribution). */
  private def computeSoftAssignments(
      df: DataFrame,
      centers: Array[Vector],
      kernel: BregmanKernel
  ): DataFrame = {
    val betaValue          = $(beta)
    val minMembershipValue = $(minMembership)

    val softAssignUDF = udf { (features: Vector) =>
      val distances         = centers.map(center => kernel.divergence(features, center))
      val minDistance       = distances.min
      val unnormalizedProbs = distances.map(dist => math.exp(-betaValue * (dist - minDistance)))
      val totalProb         = unnormalizedProbs.sum
      val probabilities     =
        if (totalProb > 1e-100) unnormalizedProbs.map(_ / totalProb).toArray
        else Array.fill(centers.length)(1.0 / centers.length)
      val adjusted          = probabilities.map(p => math.max(p, minMembershipValue))
      val adjustedSum       = adjusted.sum
      Vectors.dense(adjusted.map(_ / adjustedSum))
    }

    // Use the configured probabilityCol for output
    df.withColumn($(probabilityCol), softAssignUDF(col($(featuresCol))))
  }

  /** Compute new cluster centers using soft assignments. */
  private def computeSoftCenters(
      memberships: DataFrame,
      numClusters: Int,
      kernel: BregmanKernel
  ): Array[Vector] = {
    val featCol: String              = $(featuresCol)
    val weightColOpt: Option[String] = if (hasWeightCol) Some($(weightCol)) else None

    val membershipData =
      if (weightColOpt.isDefined)
        memberships.select(featCol, $(probabilityCol), weightColOpt.get).collect()
      else memberships.select(featCol, $(probabilityCol)).collect()

    val centers = (0 until numClusters).map { clusterId =>
      var weightedSum: Array[Double] = null
      var totalWeight                = 0.0

      membershipData.foreach { row =>
        val features    = row.getAs.toArray
        val probs       = row.getAs.toArray
        val clusterProb = probs(clusterId)
        val w           = weightColOpt.map(_ => row.getAs).getOrElse(1.0)
        val finalWeight = clusterProb * w

        if (weightedSum == null) weightedSum = Array.fill(features.length)(0.0)
        var i = 0
        while (i < features.length) {
          weightedSum(i) += features(i) * finalWeight
          i += 1
        }
        totalWeight += finalWeight
      }

      if (totalWeight > 1e-10) Vectors.dense(weightedSum.map(_ / totalWeight))
      else {
        // Empty cluster - return zeros (caller can decide handling)
        val dim = Option(weightedSum).map(_.length).getOrElse(0)
        if (dim == 0) Vectors.dense(Array.empty[Double]) else Vectors.dense(Array.fill(dim)(0.0))
      }
    }.toArray

    centers
  }

  /** Compute the soft clustering cost: Σ_x Σ_c p(c|x) * D_φ(x, μ_c). */
  private def computeSoftCost(
      memberships: DataFrame,
      centers: Array[Vector],
      kernel: BregmanKernel
  ): Double = {
    val costUDF = udf { (features: Vector, probs: Vector) =>
      probs.toArray.zipWithIndex.map { case (p, c) =>
        p * kernel.divergence(features, centers(c))
      }.sum
    }
    memberships
      .select(costUDF(col($(featuresCol)), col($(probabilityCol))).as("cost"))
      .agg(sum("cost"))
      .head()
      .getDouble(0)
  }

  override def copy(extra: ParamMap): SoftKMeans = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object SoftKMeans extends DefaultParamsReadable[SoftKMeans] {
  override def load(path: String): SoftKMeans = super.load(path)
}
