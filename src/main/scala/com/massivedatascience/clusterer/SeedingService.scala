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

/** Service for initializing cluster centers.
  *
  * This trait centralizes all seeding/initialization strategies for k-means clustering. It provides:
  * - Consistent API across all initialization methods
  * - Deterministic seeding with seed control
  * - Support for weighted and unweighted data
  * - Easy to add new initialization strategies
  *
  * Design principles:
  * - All methods are deterministic given same seed
  * - Strategies handle edge cases (k > n, duplicate points, etc.)
  * - Performance characteristics are documented
  * - Support both RDD and DataFrame workflows
  *
  * Example usage:
  * {{{
  *   val seeding = SeedingService.kMeansPlusPlus(k = 10, seed = 42)
  *   val centers = seeding.selectInitialCenters(data, ops)
  * }}}
  */
trait SeedingService extends Serializable {

  /** Human-readable name of this seeding strategy */
  def name: String

  /** Select initial cluster centers from data.
    *
    * @param data
    *   RDD of data points to cluster
    * @param ops
    *   Bregman point operations for distance calculations
    * @return
    *   initial cluster centers
    */
  def selectInitialCenters(
    data: RDD[BregmanPoint],
    ops: BregmanPointOps
  ): IndexedSeq[BregmanCenter]

  /** Number of centers to select */
  def k: Int

  /** Random seed for deterministic initialization */
  def seed: Long

  /** Whether this strategy requires multiple passes over data */
  def requiresMultiplePasses: Boolean
}

/** Random seeding - select k random points as initial centers.
  *
  * This is the simplest and fastest initialization strategy. It samples k random points from the data uniformly.
  *
  * Characteristics:
  * - Time: O(k) - single sample operation
  * - Quality: Low - may select poor initial centers
  * - Best for: Quick prototyping, when k << n
  *
  * @param k
  *   number of centers to select
  * @param seed
  *   random seed
  */
case class RandomSeeding(k: Int, seed: Long) extends SeedingService {
  require(k > 0, s"k must be positive, got $k")

  override def name: String = s"random(k=$k)"

  override def selectInitialCenters(
    data: RDD[BregmanPoint],
    ops: BregmanPointOps
  ): IndexedSeq[BregmanCenter] = {
    val n = data.count()
    require(n > 0, "Data cannot be empty")

    val actualK = math.min(k, n.toInt)
    val fraction = math.min(1.0, actualK.toDouble / n * 2.0) // Oversample for safety

    val sample = data
      .sample(withReplacement = false, fraction, seed)
      .take(actualK)
      .map(ops.toCenter)
      .toIndexedSeq

    require(sample.nonEmpty, "Failed to sample initial centers")
    sample
  }

  override def requiresMultiplePasses: Boolean = false
}

/** K-means++ seeding - iteratively select centers far from existing ones.
  *
  * This strategy selects the first center randomly, then iteratively selects remaining centers with probability
  * proportional to squared distance from nearest existing center. Provides better initial centers than random
  * selection.
  *
  * Characteristics:
  * - Time: O(k * n) - k passes over data
  * - Quality: High - provably O(log k) approximation
  * - Best for: General purpose, balanced speed/quality
  *
  * @param k
  *   number of centers to select
  * @param seed
  *   random seed
  * @param oversamplingFactor
  *   multiplier for parallel k-means++ (select multiple per round)
  */
case class KMeansPlusPlusSeeding(
  k: Int,
  seed: Long,
  oversamplingFactor: Int = 2
) extends SeedingService {

  require(k > 0, s"k must be positive, got $k")
  require(oversamplingFactor > 0, s"Oversampling factor must be positive, got $oversamplingFactor")

  override def name: String = s"kMeans++(k=$k, oversampling=$oversamplingFactor)"

  override def selectInitialCenters(
    data: RDD[BregmanPoint],
    ops: BregmanPointOps
  ): IndexedSeq[BregmanCenter] = {
    val n = data.count()
    require(n > 0, "Data cannot be empty")

    val actualK = math.min(k, n.toInt)

    // Use KMeansPlusPlus implementation
    val kpp = new KMeansPlusPlus(ops)

    // Get candidate centers (all data points as candidates)
    val candidates = data.map(ops.toCenter).collect().toIndexedSeq
    val weights    = IndexedSeq.fill(candidates.length)(1.0)

    kpp.goodCenters(
      seed = seed,
      candidateCenters = candidates,
      weights = weights,
      totalRequested = actualK,
      perRound = oversamplingFactor,
      numPreselected = 0
    )
  }

  override def requiresMultiplePasses: Boolean = true
}

/** K-means|| (parallel k-means++) seeding.
  *
  * This is the parallel version of k-means++. Instead of selecting one center per round, it selects multiple centers
  * in parallel, reducing the number of passes over data.
  *
  * Characteristics:
  * - Time: O(log(k) * n) - fewer passes than k-means++
  * - Quality: High - similar to k-means++
  * - Best for: Large datasets where multiple passes are expensive
  *
  * @param k
  *   number of centers to select
  * @param seed
  *   random seed
  * @param rounds
  *   number of sampling rounds (fewer = faster, more = better quality)
  */
case class KMeansParallelSeeding(
  k: Int,
  seed: Long,
  rounds: Int = 5
) extends SeedingService {

  require(k > 0, s"k must be positive, got $k")
  require(rounds > 0, s"Rounds must be positive, got $rounds")

  override def name: String = s"kMeans||(k=$k, rounds=$rounds)"

  override def selectInitialCenters(
    data: RDD[BregmanPoint],
    ops: BregmanPointOps
  ): IndexedSeq[BregmanCenter] = {
    val n = data.count()
    require(n > 0, "Data cannot be empty")

    val actualK   = math.min(k, n.toInt)
    val perRound  = math.max(1, actualK / rounds)

    // Use KMeansPlusPlus with higher oversampling for parallel behavior
    val kpp = new KMeansPlusPlus(ops)

    val candidates = data.map(ops.toCenter).collect().toIndexedSeq
    val weights    = IndexedSeq.fill(candidates.length)(1.0)

    kpp.goodCenters(
      seed = seed,
      candidateCenters = candidates,
      weights = weights,
      totalRequested = actualK,
      perRound = perRound,
      numPreselected = 0
    )
  }

  override def requiresMultiplePasses: Boolean = true
}

/** Grid-based seeding - partition space into grid and select one center per cell.
  *
  * This strategy partitions the feature space into a grid and selects one representative point per grid cell. Useful
  * for uniformly distributed data.
  *
  * Characteristics:
  * - Time: O(n) - single pass
  * - Quality: Medium - depends on data distribution
  * - Best for: Uniformly distributed data, deterministic initialization
  *
  * @param k
  *   number of centers to select
  * @param seed
  *   random seed
  */
case class GridSeeding(k: Int, seed: Long) extends SeedingService {
  require(k > 0, s"k must be positive, got $k")

  override def name: String = s"grid(k=$k)"

  override def selectInitialCenters(
    data: RDD[BregmanPoint],
    ops: BregmanPointOps
  ): IndexedSeq[BregmanCenter] = {
    val n = data.count()
    require(n > 0, "Data cannot be empty")

    val actualK = math.min(k, n.toInt)

    // Simple grid-based sampling: partition data into k buckets and take one from each
    val buckets = data
      .zipWithIndex()
      .groupBy { case (_, idx) => (idx % actualK).toInt }
      .mapValues(_.map(_._1).take(1))
      .collectAsMap()

    buckets.values.flatten.map(ops.toCenter).toIndexedSeq
  }

  override def requiresMultiplePasses: Boolean = false
}

/** Factory methods for creating seeding services */
object SeedingService {

  /** Random seeding
    *
    * @param k
    *   number of centers
    * @param seed
    *   random seed
    */
  def random(k: Int, seed: Long = 42): SeedingService = {
    RandomSeeding(k, seed)
  }

  /** K-means++ seeding
    *
    * @param k
    *   number of centers
    * @param seed
    *   random seed
    * @param oversamplingFactor
    *   oversampling for parallel selection
    */
  def kMeansPlusPlus(k: Int, seed: Long = 42, oversamplingFactor: Int = 2): SeedingService = {
    KMeansPlusPlusSeeding(k, seed, oversamplingFactor)
  }

  /** K-means|| (parallel k-means++) seeding
    *
    * @param k
    *   number of centers
    * @param seed
    *   random seed
    * @param rounds
    *   number of sampling rounds
    */
  def kMeansParallel(k: Int, seed: Long = 42, rounds: Int = 5): SeedingService = {
    KMeansParallelSeeding(k, seed, rounds)
  }

  /** Grid-based seeding
    *
    * @param k
    *   number of centers
    * @param seed
    *   random seed
    */
  def grid(k: Int, seed: Long = 42): SeedingService = {
    GridSeeding(k, seed)
  }

  /** Default seeding strategy (k-means++) */
  def default(k: Int, seed: Long = 42): SeedingService = {
    kMeansPlusPlus(k, seed)
  }

  /** Parse seeding strategy from string name
    *
    * @param name
    *   strategy name
    * @param k
    *   number of centers
    * @param seed
    *   random seed
    * @return
    *   seeding service
    */
  def fromString(name: String, k: Int, seed: Long = 42): SeedingService = {
    val normalized = name.toLowerCase.replaceAll("[\\s-_]", "")
    normalized match {
      case "random"                     => RandomSeeding(k, seed)
      case "kmeans++" | "kmeanspp"      => KMeansPlusPlusSeeding(k, seed)
      case "kmeans||" | "kmeansparallel" => KMeansParallelSeeding(k, seed)
      case "grid"                       => GridSeeding(k, seed)
      case _ =>
        throw new IllegalArgumentException(
          s"Unknown seeding strategy: $name. Supported: random, kmeans++, kmeans||, grid"
        )
    }
  }
}
