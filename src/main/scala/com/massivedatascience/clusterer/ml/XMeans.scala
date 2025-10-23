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

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType
import org.apache.spark.internal.Logging

/** Parameters for X-Means clustering.
  */
trait XMeansParams extends GeneralizedKMeansParams {

  /** Minimum number of clusters to try. Must be >= 2. Default: 2
    */
  final val minK = new IntParam(
    this,
    "minK",
    "Minimum number of clusters to try",
    ParamValidators.gtEq(2)
  )

  def getMinK: Int = $(minK)

  /** Maximum number of clusters to try. Must be >= minK. Default: 20
    */
  final val maxK = new IntParam(
    this,
    "maxK",
    "Maximum number of clusters to try",
    ParamValidators.gtEq(2)
  )

  def getMaxK: Int = $(maxK)

  /** Information criterion for model selection: "bic" (Bayesian) or "aic" (Akaike). Default: "bic"
    */
  final val criterion = new Param[String](
    this,
    "criterion",
    "Information criterion: bic or aic",
    (value: String) => value == "bic" || value == "aic"
  )

  def getCriterion: String = $(criterion)

  /** Minimum improvement in criterion score to continue trying larger k. Default: -1.0 (always
    * continue)
    */
  final val improvementThreshold = new DoubleParam(
    this,
    "improvementThreshold",
    "Minimum criterion improvement to continue"
  )

  def getImprovementThreshold: Double = $(improvementThreshold)

  setDefault(
    minK                 -> 2,
    maxK                 -> 20,
    criterion            -> "bic",
    improvementThreshold -> -1.0,
    k                    -> 2 // Initial k for starting, will be overridden
  )
}

/** X-Means clustering - automatically determines optimal k using information criteria.
  *
  * X-means extends k-means by trying multiple values of k and selecting the best using BIC or AIC.
  *
  * Algorithm:
  *   1. For each k from minK to maxK: a. Run GeneralizedKMeans with k clusters b. Compute BIC or
  *      AIC score c. Track best k 2. Return model with optimal k
  *
  * Information Criteria:
  *   - BIC = -2*log-likelihood + p*log(n)
  *   - AIC = -2*log-likelihood + 2*p
  *   - Lower scores are better
  *
  * Benefits:
  *   - Eliminates need to specify k in advance
  *   - Principled statistical criterion
  *   - Works with all Bregman divergences
  *
  * Example usage:
  * {{{
  *   val xmeans = new XMeans()
  *     .setMinK(2)
  *     .setMaxK(10)
  *     .setCriterion("bic")
  *     .setDivergence("squaredEuclidean")
  *
  *   val model = xmeans.fit(dataset)
  *   println(s"Optimal k = ${model.numClusters}")
  * }}}
  *
  * @param uid
  *   Unique identifier
  */
class XMeans(override val uid: String)
    extends Estimator[GeneralizedKMeansModel]
    with XMeansParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("xmeans"))

  // Parameter setters

  def setMinK(value: Int): this.type                    = set(minK, value)
  def setMaxK(value: Int): this.type                    = set(maxK, value)
  def setCriterion(value: String): this.type            = set(criterion, value)
  def setImprovementThreshold(value: Double): this.type = set(improvementThreshold, value)

  // Inherited parameter setters
  def setDivergence(value: String): this.type    = set(divergence, value)
  def setSmoothing(value: Double): this.type     = set(smoothing, value)
  def setFeaturesCol(value: String): this.type   = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setWeightCol(value: String): this.type     = set(weightCol, value)
  def setMaxIter(value: Int): this.type          = set(maxIter, value)
  def setTol(value: Double): this.type           = set(tol, value)
  def setSeed(value: Long): this.type            = set(seed, value)
  def setDistanceCol(value: String): this.type   = set(distanceCol, value)

  override def fit(dataset: Dataset[_]): GeneralizedKMeansModel = {
    transformSchema(dataset.schema, logging = true)

    val df = dataset.toDF()

    logInfo(
      s"Starting X-Means: k range [${$(minK)}, ${$(maxK)}], " +
        s"criterion=${$(criterion)}, divergence=${$(divergence)}"
    )

    // Validate input data domain requirements for the selected divergence
    com.massivedatascience.util.DivergenceDomainValidator.validateDataFrame(
      df,
      $(featuresCol),
      $(divergence),
      maxSamples = Some(1000)
    )

    // Cache data for multiple k trials
    df.cache()

    val n        = df.count()
    val firstRow = df.select($(featuresCol)).head()
    val d        = firstRow.getAs[org.apache.spark.ml.linalg.Vector](0).size

    logInfo(s"Data: n=$n points, d=$d dimensions")

    // Try each value of k
    var bestModel: GeneralizedKMeansModel = null
    var bestScore                         = Double.MaxValue // Lower is better
    var bestK                             = $(minK)

    for (currentK <- $(minK) to $(maxK)) {
      logInfo(s"Trying k=$currentK...")

      // Create k-means estimator with current k
      val kmeans = new GeneralizedKMeans(uid + s"_k${currentK}")
        .setK(currentK)
        .setDivergence($(divergence))
        .setSmoothing($(smoothing))
        .setFeaturesCol($(featuresCol))
        .setPredictionCol($(predictionCol))
        .setMaxIter($(maxIter))
        .setTol($(tol))
        .setSeed($(seed))

      if (hasWeightCol) {
        kmeans.setWeightCol($(weightCol))
      }

      if (hasDistanceCol) {
        kmeans.setDistanceCol($(distanceCol))
      }

      // Fit model
      val model = kmeans.fit(df)

      // Compute information criterion score
      val cost  = model.computeCost(df)
      val score = computeScore(cost, currentK, n, d)

      logInfo(
        f"  k=$currentK: ${$(criterion).toUpperCase}=$score%.2f, cost=$cost%.4f"
      )

      // Check if this is better
      if (score < bestScore) {
        val improvement = bestScore - score
        logInfo(f"  Improvement: $improvement%.2f")

        bestScore = score
        bestModel = model
        bestK = currentK

        // If improvement is below threshold, consider stopping early
        if (currentK > $(minK) && improvement < - $(improvementThreshold)) {
          logInfo(s"Improvement below threshold")
        }
      } else {
        logInfo(f"  No improvement (previous best: $bestScore%.2f)")
      }
    }

    df.unpersist()

    logInfo(
      f"X-Means completed: optimal k=$bestK, ${$(criterion).toUpperCase}=$bestScore%.2f"
    )

    // Copy parameters to best model
    copyValues(bestModel)
  }

  /** Compute BIC or AIC score for a clustering.
    *
    * Lower scores are better.
    *
    * BIC = -2*log-likelihood + p*log(n) AIC = -2*log-likelihood + 2*p
    *
    * where:
    *   - log-likelihood is computed from the clustering cost
    *   - p = k*d + 1 (number of parameters: k centers of d dimensions + variance)
    *   - n = number of data points
    *
    * For k-means with squared Euclidean distance: log-likelihood ≈ -cost/(2*variance) -
    * n*log(sigma) - n*log(2π)/2
    */
  private def computeScore(cost: Double, k: Int, n: Long, d: Int): Double = {
    val nDouble = n.toDouble

    // Estimate variance from average cost
    val variance = math.max(cost / nDouble, 1e-10) // Avoid division by zero
    val sigma    = math.sqrt(variance)

    // Log-likelihood (simplified Gaussian assumption)
    val logLikelihood = -cost / (2 * variance) -
      nDouble * math.log(sigma) -
      nDouble * math.log(2 * math.Pi) / 2

    // Number of parameters: k centers * d dimensions + 1 variance parameter
    val numParams = k * d + 1

    // Compute criterion
    $(criterion) match {
      case "bic" => -2 * logLikelihood + numParams * math.log(nDouble)
      case "aic" => -2 * logLikelihood + 2 * numParams
      case _     => -2 * logLikelihood + numParams * math.log(nDouble) // Default to BIC
    }
  }

  override def copy(extra: ParamMap): XMeans = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    // Delegate to GeneralizedKMeans for schema validation
    val kmeans =
      new GeneralizedKMeans().setFeaturesCol($(featuresCol)).setPredictionCol($(predictionCol))

    if (hasWeightCol) kmeans.setWeightCol($(weightCol))
    if (hasDistanceCol) kmeans.setDistanceCol($(distanceCol))

    kmeans.transformSchema(schema)
  }
}

object XMeans extends DefaultParamsReadable[XMeans] {
  override def load(path: String): XMeans = super.load(path)
}
