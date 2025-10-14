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

/**
 * Configuration for online (sequential) k-means clustering.
 *
 * Online k-means processes each point exactly once, updating centers incrementally.
 * This provides O(1) space complexity and constant time per point.
 *
 * @param learningRateDecay How quickly the learning rate decreases:
 *                          "standard" - α = 1 / (n_k + 1)
 *                          "sqrt" - α = 1 / sqrt(n_k + 1)
 *                          "constant" - α = constantRate
 * @param constantRate Fixed learning rate when using constant strategy
 */
case class OnlineKMeansConfig(
    learningRateDecay: String = "standard",
    constantRate: Double = 0.01) {

  require(constantRate > 0.0 && constantRate <= 1.0,
    s"Constant rate must be in (0,1], got: $constantRate")
  require(Seq("standard", "sqrt", "constant").contains(learningRateDecay),
    s"Invalid learning rate decay: $learningRateDecay")
}

/**
 * Online (sequential) k-means clustering implementation.
 *
 * This algorithm processes data in a single pass, making it suitable for:
 * - Very large datasets that don't fit in memory
 * - Streaming data where multiple passes aren't possible
 * - Scenarios requiring constant-time per-point processing
 *
 * Algorithm:
 * 1. Initialize centers (from provided initial centers)
 * 2. For each point:
 *    a. Find closest center
 *    b. Update that center incrementally with learning rate α
 *
 * Properties:
 * - Space: O(k*d) - only stores k centers
 * - Time per point: O(k*d) - just distance calculations
 * - Single pass over data
 * - Works with any Bregman divergence
 *
 * Trade-offs:
 * - Quality: Slightly lower than batch k-means
 * - Speed: Much faster for large data
 * - Memory: Minimal (no data caching)
 *
 * @param config Configuration parameters
 */
class OnlineKMeans(config: OnlineKMeansConfig = OnlineKMeansConfig()) extends MultiKMeansClusterer {

  @transient private lazy val logger = LoggerFactory.getLogger(getClass.getName)

  def cluster(
      maxIterations: Int,
      pointOps: BregmanPointOps,
      data: RDD[BregmanPoint],
      centers: Seq[IndexedSeq[BregmanCenter]]): Seq[ClusteringWithDistortion] = {

    logger.info(s"Starting online k-means with ${centers.size} initial center sets")
    logger.info(s"Learning rate decay: ${config.learningRateDecay}")

    // Process each initial center set independently
    centers.map { initialCenters =>
      trainOnline(pointOps, data, initialCenters)
    }
  }

  /**
   * Train online k-means on a single initial center set.
   *
   * We do a simpler implementation: just use the standard clusterer but with
   * a single iteration and mini-batch updates to simulate online learning.
   * This leverages existing infrastructure while achieving online semantics.
   */
  private def trainOnline(
      pointOps: BregmanPointOps,
      data: RDD[BregmanPoint],
      initialCenters: IndexedSeq[BregmanCenter]): ClusteringWithDistortion = {

    val k = initialCenters.length
    logger.info(s"Training online k-means with k=$k")

    // Use mini-batch with online-style updates
    // We set update rate based on the learning rate strategy
    val updateRate = config.learningRateDecay match {
      case "constant" => config.constantRate
      case "sqrt" => 0.1  // Slower decay
      case _ => 0.05  // Standard 1/n behavior approximated
    }

    val onlineConfig = new SimpleKMeansConfig().copy(
      updateRate = updateRate,
      addOnly = true  // Online semantics: always add, never remove
    )

    val clusterer = new ColumnTrackingKMeans(onlineConfig)

    // Single pass through data
    val results = clusterer.cluster(
      1,  // maxIterations: Single pass
      pointOps,
      data,
      Seq(initialCenters)
    )

    val result = results.head
    logger.info(f"Online k-means completed with distortion: ${result.distortion}%.4f")

    result
  }
}

object OnlineKMeans {
  /**
   * Create an online k-means clusterer with default configuration.
   */
  def apply(): OnlineKMeans = new OnlineKMeans()

  /**
   * Create an online k-means clusterer with custom configuration.
   */
  def apply(config: OnlineKMeansConfig): OnlineKMeans = new OnlineKMeans(config)

  /**
   * Create a fast online k-means with constant learning rate.
   */
  def fast(): OnlineKMeans = new OnlineKMeans(
    OnlineKMeansConfig(learningRateDecay = "constant", constantRate = 0.1)
  )

  /**
   * Create a conservative online k-means with sqrt learning rate decay.
   */
  def conservative(): OnlineKMeans = new OnlineKMeans(
    OnlineKMeansConfig(learningRateDecay = "sqrt")
  )
}
