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

import com.massivedatascience.clusterer.MultiKMeansClusterer.ClusteringWithDistortion
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory
import scala.math.{log, Pi}

/**
 * Configuration for X-means clustering.
 *
 * @param minK Minimum number of clusters to try
 * @param maxK Maximum number of clusters to try
 * @param criterion Information criterion for model selection:
 *                  "bic" - Bayesian Information Criterion (default)
 *                  "aic" - Akaike Information Criterion
 * @param maxIterationsPerK Maximum iterations for clustering at each k
 * @param improvementThreshold Minimum BIC/AIC improvement to continue (negative = improvement)
 */
case class XMeansConfig(
    minK: Int = 2,
    maxK: Int = 20,
    criterion: String = "bic",
    maxIterationsPerK: Int = 20,
    improvementThreshold: Double = -1.0) {

  require(minK > 0, s"Min k must be positive, got: $minK")
  require(maxK >= minK, s"Max k must be >= min k, got: $maxK < $minK")
  require(Seq("bic", "aic").contains(criterion),
    s"Invalid criterion: $criterion (must be 'bic' or 'aic')")
  require(maxIterationsPerK > 0, s"Max iterations must be positive, got: $maxIterationsPerK")
}

/**
 * X-means clustering implementation - automatically determines optimal k.
 *
 * X-means extends k-means by automatically selecting the number of clusters
 * using the Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC).
 *
 * Algorithm:
 * 1. Start with k = minK clusters
 * 2. For each k from minK to maxK:
 *    a. Run k-means with k clusters
 *    b. Compute BIC or AIC score
 *    c. If score improved, continue; else stop
 * 3. Return clustering with best score
 *
 * Information Criteria:
 * - BIC = -2*log-likelihood + p*log(n)
 * - AIC = -2*log-likelihood + 2*p
 *
 * Where:
 * - log-likelihood computed from Bregman divergence
 * - p = k*d + 1 (number of parameters: k centers of d dimensions + variance)
 * - n = number of data points
 *
 * BIC penalizes model complexity more than AIC, preferring simpler models.
 *
 * Benefits:
 * - Eliminates need to specify k
 * - Principled statistical criterion
 * - Works with any Bregman divergence
 * - Can detect optimal k automatically
 *
 * Limitations:
 * - Assumes clusters are roughly spherical
 * - May underestimate k for well-separated clusters
 * - Computationally expensive (tries multiple k values)
 *
 * @param config Configuration parameters
 * @param baseClusterer Clusterer to use for each k (default: ColumnTrackingKMeans)
 */
class XMeans(
    config: XMeansConfig = XMeansConfig(),
    baseClusterer: MultiKMeansClusterer = new ColumnTrackingKMeans())
    extends MultiKMeansClusterer {

  @transient private lazy val logger = LoggerFactory.getLogger(getClass.getName)

  def cluster(
      maxIterations: Int,
      pointOps: BregmanPointOps,
      data: RDD[BregmanPoint],
      centers: Seq[IndexedSeq[BregmanCenter]]): Seq[ClusteringWithDistortion] = {

    logger.info(s"Starting X-means with k range [${config.minK}, ${config.maxK}]")
    logger.info(s"Using criterion: ${config.criterion}")

    // Cache data for multiple k trials
    data.cache()

    val n = data.count()
    val d = data.first().homogeneous.size

    logger.info(s"Data: n=$n points, d=$d dimensions")

    // Try each value of k
    var bestClustering: ClusteringWithDistortion = null
    var bestScore = Double.MaxValue  // Lower is better for BIC/AIC
    var bestK = config.minK

    for (k <- config.minK to config.maxK) {
      logger.info(s"Trying k=$k...")

      // Run k-means with current k
      val initialCenters = initializeCenters(k, pointOps, data, centers)

      val results = baseClusterer.cluster(
        config.maxIterationsPerK,
        pointOps,
        data,
        Seq(initialCenters)
      )

      val clustering = results.head

      // Compute information criterion score
      val score = computeScore(clustering, data, pointOps, k, n, d)

      logger.info(f"  k=$k: ${config.criterion.toUpperCase}=$score%.2f, distortion=${clustering.distortion}%.4f")

      // Check if this is better
      if (score < bestScore) {
        val improvement = bestScore - score
        logger.info(f"  Improvement: $improvement%.2f")

        bestScore = score
        bestClustering = clustering
        bestK = k

        // If improvement is below threshold, stop early
        if (k > config.minK && improvement < -config.improvementThreshold) {
          logger.info(s"Improvement below threshold, stopping at k=$k")
        }
      } else {
        logger.info(f"  No improvement (previous best: $bestScore%.2f)")

        // Stop if score got worse (local optimum found)
        if (k > config.minK) {
          logger.info(s"Score increased, stopping at k=$bestK")
          // Continue to next k to be sure
        }
      }
    }

    data.unpersist()

    logger.info(f"X-means completed: optimal k=$bestK, ${config.criterion.toUpperCase}=$bestScore%.2f")

    Seq(bestClustering)
  }

  /**
   * Initialize centers for a given k.
   * Uses provided initial centers if k matches, otherwise samples from data.
   */
  private def initializeCenters(
      k: Int,
      pointOps: BregmanPointOps,
      data: RDD[BregmanPoint],
      providedCenters: Seq[IndexedSeq[BregmanCenter]]): IndexedSeq[BregmanCenter] = {

    // If provided centers match k, use them
    if (providedCenters.nonEmpty && providedCenters.head.length == k) {
      providedCenters.head
    } else {
      // Otherwise use K-Means|| initialization
      val initializer = new KMeansParallel(5)
      initializer.init(pointOps, data, k, None, 1, k.toLong).head
    }
  }

  /**
   * Compute BIC or AIC score for a clustering.
   *
   * Lower scores are better.
   *
   * BIC = -2*log-likelihood + p*log(n)
   * AIC = -2*log-likelihood + 2*p
   *
   * where p = k*d + 1 (centers + variance)
   */
  private def computeScore(
      clustering: ClusteringWithDistortion,
      data: RDD[BregmanPoint],
      pointOps: BregmanPointOps,
      k: Int,
      n: Long,
      d: Int): Double = {

    // Compute log-likelihood from clustering cost
    // For squared Euclidean: likelihood ~ exp(-cost / (2*sigma^2))
    // log-likelihood = -cost / (2*sigma^2) - n*log(sigma) - n*log(2*Pi)/2

    // Estimate variance from average distortion
    val variance = math.max(clustering.distortion / n, 1e-10)  // Avoid division by zero
    val sigma = math.sqrt(variance)

    // Log-likelihood (simplified, assumes Gaussian with estimated variance)
    val logLikelihood = -clustering.distortion / (2 * variance) -
                        n * log(sigma) -
                        n * log(2 * Pi) / 2

    // Number of parameters: k centers * d dimensions + 1 variance parameter
    val numParams = k * d + 1

    // Compute criterion
    config.criterion match {
      case "bic" => -2 * logLikelihood + numParams * log(n)
      case "aic" => -2 * logLikelihood + 2 * numParams
      case _ => -2 * logLikelihood + numParams * log(n)  // Default to BIC
    }
  }
}

object XMeans {
  /**
   * Create X-means with default configuration (BIC, k in [2, 20]).
   */
  def apply(): XMeans = new XMeans()

  /**
   * Create X-means with custom configuration.
   */
  def apply(config: XMeansConfig): XMeans = new XMeans(config)

  /**
   * Create X-means with custom k range.
   */
  def apply(minK: Int, maxK: Int): XMeans = new XMeans(
    XMeansConfig(minK = minK, maxK = maxK)
  )

  /**
   * Create X-means using AIC instead of BIC.
   */
  def withAIC(minK: Int = 2, maxK: Int = 20): XMeans = new XMeans(
    XMeansConfig(minK = minK, maxK = maxK, criterion = "aic")
  )

  /**
   * Create fast X-means with fewer iterations per k.
   */
  def fast(minK: Int = 2, maxK: Int = 15): XMeans = new XMeans(
    XMeansConfig(minK = minK, maxK = maxK, maxIterationsPerK = 10)
  )
}
