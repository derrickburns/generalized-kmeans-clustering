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

package com.massivedatascience.clusterer

import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory

import scala.math.{log, exp}

/** Configuration for Bregman mixture model estimation.
  *
  * @param maxIterations
  *   Maximum EM iterations
  * @param convergenceThreshold
  *   Convergence threshold for log-likelihood
  * @param minMixingWeight
  *   Minimum mixing weight to avoid degeneracy
  * @param regularization
  *   Regularization parameter for mixing weights
  * @param initializationMethod
  *   Method for initializing parameters
  */
case class BregmanMixtureConfig(
  maxIterations: Int = 100,
  convergenceThreshold: Double = 1e-6,
  minMixingWeight: Double = 1e-8,
  regularization: Double = 1e-6,
  initializationMethod: String = "kmeans"
) {

  require(maxIterations > 0, s"Max iterations must be positive, got: $maxIterations")
  require(
    convergenceThreshold > 0.0,
    s"Convergence threshold must be positive, got: $convergenceThreshold"
  )
  require(
    minMixingWeight > 0.0 && minMixingWeight < 1.0,
    s"Minimum mixing weight must be in (0,1), got: $minMixingWeight"
  )
  require(regularization >= 0.0, s"Regularization must be non-negative, got: $regularization")
}

/** A component in a Bregman mixture model.
  *
  * @param center
  *   The component center (mean parameter)
  * @param mixingWeight
  *   The mixing weight (prior probability) of this component
  * @param componentId
  *   Unique identifier for this component
  */
case class MixtureComponent(center: BregmanCenter, mixingWeight: Double, componentId: Int) {

  require(
    mixingWeight >= 0.0 && mixingWeight <= 1.0,
    s"Mixing weight must be in [0,1], got: $mixingWeight"
  )
}

/** Result of Bregman mixture model estimation.
  *
  * @param components
  *   The estimated mixture components
  * @param logLikelihood
  *   Final log-likelihood value
  * @param responsibilities
  *   RDD of posterior probabilities for each point
  * @param iterations
  *   Number of EM iterations performed
  * @param converged
  *   Whether the algorithm converged
  * @param config
  *   Configuration used
  */
case class BregmanMixtureResult(
  components: IndexedSeq[MixtureComponent],
  logLikelihood: Double,
  responsibilities: RDD[(BregmanPoint, Array[Double])],
  iterations: Int,
  converged: Boolean,
  config: BregmanMixtureConfig
) {

  /** Get the number of components in the mixture.
    */
  def numComponents: Int = components.length

  /** Get the mixing weights as an array.
    */
  def mixingWeights: Array[Double] = components.map(_.mixingWeight).toArray

  /** Get the component centers.
    */
  def centers: IndexedSeq[BregmanCenter] = components.map(_.center)

  /** Compute the effective number of components (based on mixing weights).
    */
  def effectiveNumComponents: Double = {
    val weights = mixingWeights
    val entropy = -weights.map(w => if (w > 1e-10) w * log(w) else 0.0).sum
    exp(entropy)
  }

  /** Assign points to components using maximum a posteriori (MAP) estimation.
    */
  def mapAssignments: RDD[(BregmanPoint, Int)] = {
    responsibilities.map { case (point, posteriors) =>
      val assignment = posteriors.zipWithIndex.maxBy(_._1)._2
      (point, assignment)
    }
  }

  /** Compute the Bayesian Information Criterion (BIC) for model selection.
    */
  def bic(numDataPoints: Long, dimensionality: Int): Double = {
    val numParameters =
      numComponents * (dimensionality + 1) - 1 // centers + mixing weights - constraint
    -2.0 * logLikelihood + numParameters * log(numDataPoints)
  }

  /** Compute the Akaike Information Criterion (AIC) for model selection.
    */
  def aic(dimensionality: Int): Double = {
    val numParameters = numComponents * (dimensionality + 1) - 1
    -2.0 * logLikelihood + 2.0 * numParameters
  }

  /** Get comprehensive statistics about the mixture model.
    */
  def getStats: Map[String, Double] = {
    Map(
      "logLikelihood"          -> logLikelihood,
      "numComponents"          -> numComponents.toDouble,
      "effectiveNumComponents" -> effectiveNumComponents,
      "iterations"             -> iterations.toDouble,
      "converged"              -> (if (converged) 1.0 else 0.0),
      "minMixingWeight"        -> mixingWeights.min,
      "maxMixingWeight"        -> mixingWeights.max,
      "mixingWeightEntropy"    -> -mixingWeights.map(w => if (w > 1e-10) w * log(w) else 0.0).sum
    )
  }
}

/** Bregman mixture model estimation using the EM algorithm.
  *
  * This estimates mixture models of the form: p(x) = Σ_k π_k * p_k(x)
  *
  * where π_k are mixing weights and p_k(x) are exponential family distributions corresponding to Bregman divergences.
  *
  * The likelihood for each component is: p_k(x) ∝ exp(-D_φ(x, μ_k))
  *
  * where D_φ is the Bregman divergence and μ_k is the component center.
  */
case class BregmanMixtureModel(config: BregmanMixtureConfig = BregmanMixtureModel.defaultConfig) {

  @transient private lazy val logger = LoggerFactory.getLogger(getClass.getName)

  /** Estimate a Bregman mixture model using the EM algorithm.
    *
    * @param data
    *   Input data points
    * @param numComponents
    *   Number of mixture components
    * @param pointOps
    *   Bregman point operations
    * @param initialCenters
    *   Optional initial centers (if not provided, will use initialization method)
    * @return
    *   Estimated mixture model
    */
  def fit(
    data: RDD[BregmanPoint],
    numComponents: Int,
    pointOps: BregmanPointOps,
    initialCenters: Option[IndexedSeq[BregmanCenter]] = None
  ): BregmanMixtureResult = {

    require(numComponents > 0, s"Number of components must be positive, got: $numComponents")

    logger.info(
      s"Estimating Bregman mixture model: k=$numComponents, method=${config.initializationMethod}"
    )

    // Cache data for performance
    data.cache()
    val numDataPoints = data.count()

    // Initialize parameters
    val (initialComponents, initialResponsibilities) =
      initializeParameters(data, numComponents, pointOps, initialCenters)

    logger.info(
      s"Initialized mixture model with $numComponents components on $numDataPoints data points"
    )

    // EM iterations
    var components            = initialComponents
    var responsibilities      = initialResponsibilities
    var previousLogLikelihood = Double.NegativeInfinity
    var iteration             = 0
    var converged             = false

    while (iteration < config.maxIterations && !converged) {
      iteration += 1
      logger.debug(s"EM iteration $iteration")

      // E-step: Update responsibilities (posterior probabilities)
      responsibilities = computeResponsibilities(data, components, pointOps)

      // M-step: Update parameters based on responsibilities
      components = updateParameters(responsibilities, numComponents, pointOps)

      // Check convergence
      val logLikelihood       = computeLogLikelihood(responsibilities, components, pointOps)
      val improvement         = logLikelihood - previousLogLikelihood
      val relativeImprovement = improvement / math.max(math.abs(previousLogLikelihood), 1e-10)

      logger.debug(
        f"Iteration $iteration: log-likelihood = $logLikelihood%.6f, improvement = $improvement%.8f"
      )

      if (math.abs(relativeImprovement) < config.convergenceThreshold) {
        converged = true
        logger.info(s"EM algorithm converged after $iteration iterations")
      }

      previousLogLikelihood = logLikelihood
    }

    if (!converged) {
      logger.warn(s"EM algorithm did not converge after ${config.maxIterations} iterations")
    }

    val finalLogLikelihood = computeLogLikelihood(responsibilities, components, pointOps)

    BregmanMixtureResult(
      components = components,
      logLikelihood = finalLogLikelihood,
      responsibilities = responsibilities,
      iterations = iteration,
      converged = converged,
      config = config
    )
  }

  /** Initialize mixture model parameters.
    */
  private def initializeParameters(
    data: RDD[BregmanPoint],
    numComponents: Int,
    pointOps: BregmanPointOps,
    initialCenters: Option[IndexedSeq[BregmanCenter]]
  ): (IndexedSeq[MixtureComponent], RDD[(BregmanPoint, Array[Double])]) = {

    val centers = initialCenters.getOrElse {
      config.initializationMethod match {
        case "kmeans" =>
          logger.info("Initializing with K-means clustering")
          val selector = KMeansSelector(KMeansSelector.K_MEANS_PARALLEL)
          selector.init(pointOps, data, numComponents, None, 1, 42L).head

        case "random" =>
          logger.info("Initializing with random points")
          data
            .takeSample(false, numComponents, 42L)
            .zipWithIndex
            .map { case (point, _) =>
              pointOps.toCenter(point)
            }
            .toIndexedSeq

        case _ =>
          throw new IllegalArgumentException(
            s"Unknown initialization method: ${config.initializationMethod}"
          )
      }
    }

    // Initialize with uniform mixing weights
    val uniformWeight = 1.0 / numComponents
    val components = centers.zipWithIndex.map { case (center, id) =>
      MixtureComponent(center, uniformWeight, id)
    }

    // Initialize responsibilities uniformly
    val responsibilities = data.map { point =>
      val uniformResponsibility = Array.fill(numComponents)(uniformWeight)
      (point, uniformResponsibility)
    }

    (components, responsibilities)
  }

  /** E-step: Compute responsibilities (posterior probabilities).
    *
    * r_ik = π_k * p_k(x_i) / Σ_j π_j * p_j(x_i)
    */
  private def computeResponsibilities(
    data: RDD[BregmanPoint],
    components: IndexedSeq[MixtureComponent],
    pointOps: BregmanPointOps
  ): RDD[(BregmanPoint, Array[Double])] = {

    val broadcastComponents = data.sparkContext.broadcast(components)

    data.map { point =>
      val comps = broadcastComponents.value

      // Compute log-probabilities for numerical stability
      val logProbs = comps.map { component =>
        val distance = pointOps.distance(point, component.center)
        log(component.mixingWeight) - distance // log(π_k) - D_φ(x, μ_k)
      }

      // Normalize using log-sum-exp trick
      val maxLogProb         = logProbs.max
      val normalizedLogProbs = logProbs.map(_ - maxLogProb)
      val sumExp             = normalizedLogProbs.map(exp).sum

      val responsibilities = normalizedLogProbs.map { logProb =>
        val responsibility = exp(logProb) / sumExp
        math.max(responsibility, config.minMixingWeight) // Avoid numerical issues
      }.toArray

      // Re-normalize after applying minimum threshold
      val totalResp                  = responsibilities.sum
      val normalizedResponsibilities = responsibilities.map(_ / totalResp)

      (point, normalizedResponsibilities)
    }
  }

  /** M-step: Update parameters based on responsibilities.
    */
  private def updateParameters(
    responsibilities: RDD[(BregmanPoint, Array[Double])],
    numComponents: Int,
    pointOps: BregmanPointOps
  ): IndexedSeq[MixtureComponent] = {

    // Compute effective counts and weighted sums for each component
    val componentStats = responsibilities
      .flatMap { case (point, resps) =>
        resps.zipWithIndex.map { case (resp, componentId) =>
          val weightedPoint = pointOps.scale(point, resp)
          (componentId, (weightedPoint, resp))
        }
      }
      .aggregateByKey((pointOps.make(), 0.0))(
        // Sequence operation
        { case ((accumulator, totalWeight), (weightedPoint, weight)) =>
          accumulator.add(weightedPoint)
          (accumulator, totalWeight + weight)
        },
        // Combiner operation
        { case ((acc1, weight1), (acc2, weight2)) =>
          acc1.add(acc2)
          (acc1, weight1 + weight2)
        }
      )
      .collectAsMap()

    val totalDataWeight = componentStats.values.map(_._2).sum

    // Update components
    (0 until numComponents).map { componentId =>
      componentStats.get(componentId) match {
        case Some((accumulator, effectiveCount)) if effectiveCount > pointOps.weightThreshold =>
          val center = pointOps.toCenter(accumulator.asImmutable)
          val mixingWeight = math.max(
            (effectiveCount + config.regularization) / (totalDataWeight + numComponents * config.regularization),
            config.minMixingWeight
          )
          MixtureComponent(center, mixingWeight, componentId)

        case _ =>
          // Handle degenerate component
          logger.warn(s"Component $componentId has insufficient data, using minimum mixing weight")
          val defaultCenter = pointOps.toCenter(pointOps.make().asImmutable)
          MixtureComponent(defaultCenter, config.minMixingWeight, componentId)
      }
    }
  }

  /** Compute the log-likelihood of the data given the current parameters.
    */
  private def computeLogLikelihood(
    responsibilities: RDD[(BregmanPoint, Array[Double])],
    components: IndexedSeq[MixtureComponent],
    pointOps: BregmanPointOps
  ): Double = {

    val broadcastComponents = responsibilities.sparkContext.broadcast(components)

    responsibilities
      .map { case (point, _) =>
        val comps = broadcastComponents.value

        // Compute log p(x_i) = log(Σ_k π_k * p_k(x_i))
        val logProbs = comps.map { component =>
          val distance = pointOps.distance(point, component.center)
          log(component.mixingWeight) - distance
        }

        // Use log-sum-exp for numerical stability
        val maxLogProb = logProbs.max
        val logSumExp  = maxLogProb + log(logProbs.map(lp => exp(lp - maxLogProb)).sum)
        logSumExp
      }
      .sum()
  }
}

object BregmanMixtureModel {

  /** Default configuration for mixture model estimation.
    */
  def defaultConfig: BregmanMixtureConfig = {
    BregmanMixtureConfig(
      maxIterations = 100,
      convergenceThreshold = 1e-6,
      minMixingWeight = 1e-8,
      regularization = 1e-6,
      initializationMethod = "kmeans"
    )
  }

  /** Create mixture model with specified number of iterations.
    */
  def apply(maxIterations: Int): BregmanMixtureModel = {
    val config = defaultConfig.copy(maxIterations = maxIterations)
    BregmanMixtureModel(config)
  }

  /** Create mixture model with high precision settings.
    */
  def highPrecision(): BregmanMixtureModel = {
    val config = defaultConfig.copy(
      maxIterations = 200,
      convergenceThreshold = 1e-8,
      minMixingWeight = 1e-12,
      regularization = 1e-8
    )
    BregmanMixtureModel(config)
  }

  /** Create mixture model optimized for large datasets.
    */
  def forLargeDatasets(): BregmanMixtureModel = {
    val config = defaultConfig.copy(
      maxIterations = 50,
      convergenceThreshold = 1e-5,
      regularization = 1e-4 // More regularization for stability
    )
    BregmanMixtureModel(config)
  }

  /** Quick mixture model estimation with reasonable defaults.
    */
  def quick(
    data: RDD[BregmanPoint],
    numComponents: Int,
    pointOps: BregmanPointOps
  ): BregmanMixtureResult = {

    val config = defaultConfig.copy(maxIterations = 50)
    val model  = BregmanMixtureModel(config)
    model.fit(data, numComponents, pointOps)
  }
}
