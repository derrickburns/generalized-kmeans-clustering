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

/** Configuration for annealed (deterministic annealing) k-means clustering.
  *
  * Annealed k-means gradually transitions from soft to hard clustering by increasing the inverse temperature parameter
  * (beta) according to an annealing schedule.
  *
  * @param initialBeta
  *   Starting inverse temperature (low = soft, high = hard)
  * @param finalBeta
  *   Ending inverse temperature
  * @param annealingSchedule
  *   Strategy for increasing beta: "exponential" - β_new = β_old * annealingRate "linear" - β_new = β_old +
  *   annealingRate
  * @param annealingRate
  *   Rate at which beta increases
  * @param stepsPerTemperature
  *   Number of EM iterations at each temperature
  * @param maxTemperatures
  *   Maximum number of temperature steps
  * @param convergenceThreshold
  *   Threshold for considering convergence at each temperature
  * @param minMembership
  *   Minimum membership probability (from BregmanSoftKMeansConfig)
  */
case class AnnealedKMeansConfig(
  initialBeta: Double = 0.1,
  finalBeta: Double = 100.0,
  annealingSchedule: String = "exponential",
  annealingRate: Double = 1.5,
  stepsPerTemperature: Int = 5,
  maxTemperatures: Int = 20,
  convergenceThreshold: Double = 1e-4,
  minMembership: Double = 1e-10
) extends ConfigValidator {

  requirePositive(initialBeta, "Initial beta")
  requireGreaterThan(finalBeta, initialBeta, "Final beta")
  requireGreaterThan(annealingRate, 1.0, "Annealing rate")
  requirePositive(stepsPerTemperature, "Steps per temperature")
  requirePositive(maxTemperatures, "Max temperatures")
  requirePositive(convergenceThreshold, "Convergence threshold")
  requireOneOf(annealingSchedule, Seq("exponential", "linear"), "Annealing schedule")
}

/** Annealed (deterministic annealing) k-means clustering implementation.
  *
  * This algorithm gradually transitions from soft to hard clustering using a temperature parameter, providing several
  * benefits over standard k-means:
  *
  * Benefits:
  *   - Better escape from local minima (starts globally, refines locally)
  *   - More stable convergence (smooth transition soft→hard)
  *   - Can automatically determine k via cluster splitting
  *   - Works with any Bregman divergence
  *
  * Algorithm:
  *   1. Start with low beta (high temperature) = very soft clustering 2. Run soft k-means (BregmanSoftKMeans) for a few
  *      iterations 3. Increase beta (decrease temperature) = make clustering sharper 4. Repeat until beta is high (low
  *      temperature) = hard clustering 5. Final result approaches standard k-means
  *
  * The annealing schedule controls how quickly we transition from soft to hard:
  *   - Exponential: β_t+1 = rate * β_t (faster)
  *   - Linear: β_t+1 = β_t + rate (slower, more stable)
  *
  * Properties:
  *   - Quality: Often better than standard k-means (fewer local minima)
  *   - Speed: Slower than standard k-means (multiple temperature steps)
  *   - Robustness: More robust to initialization
  *
  * @param config
  *   Configuration parameters
  */
class AnnealedKMeans(config: AnnealedKMeansConfig = AnnealedKMeansConfig()) extends MultiKMeansClusterer with Logging {

  def cluster(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centers: Seq[IndexedSeq[BregmanCenter]]
  ): Seq[ClusteringWithDistortion] = {

    logger.info(s"Starting annealed k-means with ${centers.size} initial center sets")
    logger.info(s"Annealing schedule: ${config.annealingSchedule}, rate: ${config.annealingRate}")
    logger.info(s"Temperature range: β = ${config.initialBeta} → ${config.finalBeta}")

    // Process each initial center set independently
    centers.map { initialCenters =>
      trainAnnealed(pointOps, data, initialCenters)
    }
  }

  /** Train annealed k-means on a single initial center set.
    */
  private def trainAnnealed(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    initialCenters: IndexedSeq[BregmanCenter]
  ): ClusteringWithDistortion = {

    val k = initialCenters.length
    logger.info(s"Training annealed k-means with k=$k")

    // Cache data for multiple passes
    data.cache()

    var currentCenters  = initialCenters
    var currentBeta     = config.initialBeta
    var temperatureStep = 0
    var totalIterations = 0

    // Annealing loop: gradually increase beta (decrease temperature)
    while (currentBeta < config.finalBeta && temperatureStep < config.maxTemperatures) {
      logger.info(f"Temperature step $temperatureStep: β = $currentBeta%.4f")

      // Run soft k-means at this temperature
      val softConfig = BregmanSoftKMeansConfig(
        beta = currentBeta,
        minMembership = config.minMembership,
        maxIterations = config.stepsPerTemperature,
        convergenceThreshold = config.convergenceThreshold,
        computeObjective = true
      )

      val softKMeans = new BregmanSoftKMeans(softConfig)
      val softResult =
        softKMeans.clusterSoft(config.stepsPerTemperature, pointOps, data, currentCenters)

      currentCenters = softResult.centers
      totalIterations += softResult.iterations

      logger.info(
        f"  Completed ${softResult.iterations} iterations, " +
          f"objective: ${softResult.objective}%.4f, " +
          f"converged: ${softResult.converged}"
      )

      // Update temperature for next step
      currentBeta = nextBeta(currentBeta)
      temperatureStep += 1
    }

    // Final hard clustering step at high temperature
    logger.info(f"Final hard clustering step at β = $currentBeta%.4f")
    val finalConfig = BregmanSoftKMeansConfig(
      beta = config.finalBeta,
      minMembership = config.minMembership,
      maxIterations = config.stepsPerTemperature,
      convergenceThreshold = config.convergenceThreshold,
      computeObjective = true
    )

    val finalSoftKMeans = new BregmanSoftKMeans(finalConfig)
    val finalResult =
      finalSoftKMeans.clusterSoft(config.stepsPerTemperature, pointOps, data, currentCenters)

    totalIterations += finalResult.iterations

    // Compute final distortion
    val distortion = pointOps.distortion(data, finalResult.centers)

    logger.info(f"Annealed k-means completed in $totalIterations total iterations")
    logger.info(f"Final distortion: $distortion%.4f")
    logger.info(f"Effective number of clusters: ${finalResult.effectiveNumberOfClusters}%.2f")

    data.unpersist()

    ClusteringWithDistortion(distortion, finalResult.centers)
  }

  /** Compute next beta value according to annealing schedule.
    */
  private def nextBeta(currentBeta: Double): Double = {
    val nextBeta = config.annealingSchedule match {
      case "exponential" => currentBeta * config.annealingRate
      case "linear"      => currentBeta + config.annealingRate
      case _             => currentBeta * config.annealingRate // fallback
    }

    // Don't exceed final beta during intermediate steps
    math.min(nextBeta, config.finalBeta)
  }
}

object AnnealedKMeans {

  /** Create an annealed k-means clusterer with default configuration.
    */
  def apply(): AnnealedKMeans = new AnnealedKMeans()

  /** Create an annealed k-means clusterer with custom configuration.
    */
  def apply(config: AnnealedKMeansConfig): AnnealedKMeans = new AnnealedKMeans(config)

  /** Create a fast annealed k-means with aggressive annealing.
    */
  def fast(): AnnealedKMeans = new AnnealedKMeans(
    AnnealedKMeansConfig(
      initialBeta = 0.5,
      finalBeta = 50.0,
      annealingRate = 2.0,
      stepsPerTemperature = 3,
      maxTemperatures = 10
    )
  )

  /** Create a high-quality annealed k-means with gradual annealing.
    */
  def highQuality(): AnnealedKMeans = new AnnealedKMeans(
    AnnealedKMeansConfig(
      initialBeta = 0.05,
      finalBeta = 200.0,
      annealingRate = 1.3,
      stepsPerTemperature = 10,
      maxTemperatures = 30
    )
  )

  /** Create an annealed k-means optimized for escape from local minima.
    */
  def robustInit(): AnnealedKMeans = new AnnealedKMeans(
    AnnealedKMeansConfig(
      initialBeta = 0.01, // Very soft start
      finalBeta = 100.0,
      annealingRate = 1.5,
      stepsPerTemperature = 5,
      maxTemperatures = 25
    )
  )
}
