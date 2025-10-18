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
import scala.util.Random

/** Strategy for handling empty clusters during k-means iterations.
  *
  * Empty clusters can occur when all points are reassigned away from a center. This trait defines
  * how to reseed (reinitialize) empty clusters to maintain the target number of clusters.
  *
  * Design principles:
  *   - Each policy defines a complete reseeding strategy
  *   - Policies are stateless and deterministic (given same seed)
  *   - Cost/performance tradeoffs are documented
  *   - Support both RDD and DataFrame operations
  *
  * Example usage:
  * {{{
  *   val policy = ReseedPolicy.farthest()
  *   val newCenters = policy.reseedEmptyClusters(
  *     data = dataRDD,
  *     centers = currentCenters,
  *     emptyClusters = Set(2, 5),
  *     ops = pointOps,
  *     seed = 42
  *   )
  * }}}
  */
trait ReseedPolicy extends Serializable {

  /** Human-readable name of this policy */
  def name: String

  /** Reseed empty clusters by selecting replacement centers.
    *
    * @param data
    *   RDD of data points
    * @param centers
    *   current cluster centers
    * @param emptyClusters
    *   indices of empty clusters to reseed
    * @param ops
    *   Bregman point operations
    * @param seed
    *   random seed for deterministic reseeding
    * @return
    *   new centers with empty clusters reseeded
    */
  def reseedEmptyClusters(
      data: RDD[BregmanPoint],
      centers: IndexedSeq[BregmanCenter],
      emptyClusters: Set[Int],
      ops: BregmanPointOps,
      seed: Long
  ): IndexedSeq[BregmanCenter]

  /** Whether this policy requires distance computation to all points.
    *
    * If true, expect O(n*k) distance computations per reseed operation. If false, expect O(1) or
    * O(k) operations.
    */
  def requiresFullScan: Boolean

  /** Expected computational cost category */
  def costCategory: ReseedCostCategory
}

/** Cost categories for reseed policies */
sealed trait ReseedCostCategory

object ReseedCostCategory {

  /** O(1) - Constant time, no data scan */
  case object Constant extends ReseedCostCategory

  /** O(k) - Linear in number of clusters */
  case object Linear extends ReseedCostCategory

  /** O(n) - Linear in dataset size, single pass */
  case object SinglePass extends ReseedCostCategory

  /** O(n*k) - Full distance computation */
  case object FullScan extends ReseedCostCategory
}

/** No reseeding - leave empty clusters as-is.
  *
  * This policy does not reseed empty clusters, effectively reducing k. Useful when the target
  * number of clusters is a maximum rather than exact requirement.
  *
  * Cost: O(1)
  */
case object NoReseedPolicy extends ReseedPolicy {
  override def name: String = "none"

  override def reseedEmptyClusters(
      data: RDD[BregmanPoint],
      centers: IndexedSeq[BregmanCenter],
      emptyClusters: Set[Int],
      ops: BregmanPointOps,
      seed: Long
  ): IndexedSeq[BregmanCenter] = {
    // Return centers unchanged
    centers
  }

  override def requiresFullScan: Boolean = false

  override def costCategory: ReseedCostCategory = ReseedCostCategory.Constant
}

/** Random reseeding - select random points from data as new centers.
  *
  * This policy samples random points from the dataset to replace empty cluster centers. Fast but
  * may select suboptimal centers (e.g., points close to existing centers).
  *
  * Cost: O(k) for k empty clusters - requires small sample from data Tradeoff: Fast but potentially
  * suboptimal center placement
  */
case class RandomReseedPolicy(sampleSize: Int = 100) extends ReseedPolicy {
  require(sampleSize > 0, s"Sample size must be positive, got $sampleSize")

  override def name: String = s"random(sample=$sampleSize)"

  override def reseedEmptyClusters(
      data: RDD[BregmanPoint],
      centers: IndexedSeq[BregmanCenter],
      emptyClusters: Set[Int],
      ops: BregmanPointOps,
      seed: Long
  ): IndexedSeq[BregmanCenter] = {
    if (emptyClusters.isEmpty) {
      return centers
    }

    // Sample points from data (small sample for efficiency)
    val sampleFraction = math.min(1.0, sampleSize.toDouble / math.max(1, data.count()))
    val sample         = data.sample(withReplacement = false, sampleFraction, seed).collect()

    if (sample.isEmpty) {
      return centers // Can't reseed without data
    }

    // Use seeded random to select replacement centers
    val rng = new Random(seed)

    emptyClusters.foldLeft(centers) { (currentCenters, emptyIdx) =>
      if (emptyIdx >= 0 && emptyIdx < currentCenters.length && sample.nonEmpty) {
        // Pick a random sample point
        val replacement = sample(rng.nextInt(sample.length))
        val newCenter   = ops.toCenter(replacement)
        currentCenters.updated(emptyIdx, newCenter)
      } else {
        currentCenters
      }
    }
  }

  override def requiresFullScan: Boolean = false

  override def costCategory: ReseedCostCategory = ReseedCostCategory.Linear
}

/** Farthest-point reseeding - select points with maximum distance to nearest center.
  *
  * This policy finds points that are farthest from their nearest cluster center and uses them to
  * reseed empty clusters. Provides better center placement than random selection but requires
  * distance computation.
  *
  * Cost: O(n*k) - requires full pass over data with distance computations Tradeoff: Higher quality
  * centers but more expensive
  */
case class FarthestPointReseedPolicy(numCandidates: Int = 100) extends ReseedPolicy {
  require(numCandidates > 0, s"Number of candidates must be positive, got $numCandidates")

  override def name: String = s"farthest(candidates=$numCandidates)"

  override def reseedEmptyClusters(
      data: RDD[BregmanPoint],
      centers: IndexedSeq[BregmanCenter],
      emptyClusters: Set[Int],
      ops: BregmanPointOps,
      seed: Long
  ): IndexedSeq[BregmanCenter] = {
    if (emptyClusters.isEmpty) {
      return centers
    }

    // Find points with largest distance to their nearest center
    val farthestPoints = data.map { point =>
      val distance = ops.pointCost(centers, point)
      (distance, point)
    }.top(numCandidates)(Ordering.by(_._1)).map(_._2)

    if (farthestPoints.isEmpty) {
      return centers // Can't reseed without data
    }

    // Use seeded random to select from farthest points
    val rng = new Random(seed)

    emptyClusters.foldLeft(centers) { (currentCenters, emptyIdx) =>
      if (emptyIdx >= 0 && emptyIdx < currentCenters.length && farthestPoints.nonEmpty) {
        val replacement = farthestPoints(rng.nextInt(farthestPoints.length))
        val newCenter   = ops.toCenter(replacement)
        currentCenters.updated(emptyIdx, newCenter)
      } else {
        currentCenters
      }
    }
  }

  override def requiresFullScan: Boolean = true

  override def costCategory: ReseedCostCategory = ReseedCostCategory.FullScan
}

/** Split largest cluster - divide the largest cluster into two.
  *
  * This policy identifies the cluster with the most points and splits it by selecting two points
  * from that cluster as new centers. Good for balancing cluster sizes but requires cluster
  * membership tracking.
  *
  * Cost: O(n) - single pass to count cluster memberships Tradeoff: Balanced clusters but requires
  * extra bookkeeping
  */
case class SplitLargestReseedPolicy(perturbation: Double = 0.1) extends ReseedPolicy {
  require(perturbation > 0.0, s"Perturbation must be positive, got $perturbation")

  override def name: String = s"splitLargest(perturb=$perturbation)"

  override def reseedEmptyClusters(
      data: RDD[BregmanPoint],
      centers: IndexedSeq[BregmanCenter],
      emptyClusters: Set[Int],
      ops: BregmanPointOps,
      seed: Long
  ): IndexedSeq[BregmanCenter] = {
    if (emptyClusters.isEmpty) {
      return centers
    }

    // Count cluster memberships
    val clusterSizes =
      data.map(point => ops.findClosestCluster(centers, point)).countByValue().toMap

    val rng = new Random(seed)

    emptyClusters.foldLeft(centers) { (currentCenters, emptyIdx) =>
      if (emptyIdx >= 0 && emptyIdx < currentCenters.length) {
        // Find largest non-empty cluster
        val largestCluster = clusterSizes.toSeq.sortBy(-_._2).headOption.map(_._1)

        largestCluster match {
          case Some(clusterId) if clusterId < currentCenters.length =>
            // Perturb the largest cluster center to create new center
            val originalCenter  = currentCenters(clusterId)
            val perturbedArray  = ops.toPoint(originalCenter).homogeneous.toArray.map { x =>
              x * (1.0 + perturbation * (rng.nextDouble() - 0.5))
            }
            val perturbedVector = org.apache.spark.ml.linalg.Vectors.dense(perturbedArray)
            val perturbedCenter = ops.toCenter(
              com.massivedatascience.linalg.WeightedVector
                .fromInhomogeneousWeighted(perturbedVector, 1.0)
            )
            currentCenters.updated(emptyIdx, perturbedCenter)

          case _ =>
            currentCenters // Can't split if no valid cluster found
        }
      } else {
        currentCenters
      }
    }
  }

  override def requiresFullScan: Boolean = false

  override def costCategory: ReseedCostCategory = ReseedCostCategory.SinglePass
}

/** Factory methods for creating reseed policies */
object ReseedPolicy {

  /** No reseeding - accept fewer than k clusters */
  def none: ReseedPolicy = NoReseedPolicy

  /** Random reseeding from sampled points
    *
    * @param sampleSize
    *   number of points to sample
    */
  def random(sampleSize: Int = 100): ReseedPolicy = RandomReseedPolicy(sampleSize)

  /** Farthest-point reseeding - select outliers
    *
    * @param numCandidates
    *   number of farthest points to consider
    */
  def farthest(numCandidates: Int = 100): ReseedPolicy = FarthestPointReseedPolicy(numCandidates)

  /** Split largest cluster to reseed
    *
    * @param perturbation
    *   amount to perturb cluster center (as fraction)
    */
  def splitLargest(perturbation: Double = 0.1): ReseedPolicy = SplitLargestReseedPolicy(
    perturbation
  )

  /** Default reseed policy (random with moderate sample) */
  def default: ReseedPolicy = RandomReseedPolicy(100)

  /** Parse reseed policy from string name
    *
    * @param name
    *   policy name (e.g., "none", "random", "farthest", "splitLargest")
    * @return
    *   reseed policy
    */
  def fromString(name: String): ReseedPolicy = {
    val normalized = name.toLowerCase.replaceAll("[\\s-_]", "")
    normalized match {
      case "none" | "noreseed"      => NoReseedPolicy
      case "random"                 => RandomReseedPolicy()
      case "farthest" | "outlier"   => FarthestPointReseedPolicy()
      case "splitlargest" | "split" => SplitLargestReseedPolicy()
      case _                        =>
        throw new IllegalArgumentException(
          s"Unknown reseed policy: $name. Supported: none, random, farthest, splitLargest"
        )
    }
  }
}
