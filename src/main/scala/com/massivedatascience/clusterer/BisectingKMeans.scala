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

/** Configuration for bisecting k-means clustering.
  *
  * @param maxIterationsPerSplit
  *   Maximum iterations for each bisection
  * @param minClusterSize
  *   Minimum size for a cluster to be split
  * @param splitCriterion
  *   Criterion for selecting cluster to split: "largest" - split cluster with most points "highest_cost" - split
  *   cluster with highest cost
  */
case class BisectingKMeansConfig(
  maxIterationsPerSplit: Int = 20,
  minClusterSize: Int = 2,
  splitCriterion: String = "largest"
) extends ConfigValidator {

  requirePositive(maxIterationsPerSplit, "Max iterations per split")
  requireAtLeast(minClusterSize, 2, "Min cluster size")
  requireOneOf(splitCriterion, Seq("largest", "highest_cost"), "Split criterion")
}

/** Bisecting k-means clustering implementation.
  *
  * This is a hierarchical divisive clustering algorithm that:
  *   1. Starts with all points in one cluster 2. Repeatedly selects a cluster and splits it into two using k-means
  *      (k=2) 3. Continues until reaching target k clusters
  *
  * Benefits over standard k-means:
  *   - More deterministic (less sensitive to initialization)
  *   - Better handling of imbalanced cluster sizes
  *   - Produces a clustering hierarchy (dendrogram) as byproduct
  *   - Often faster for large k (only splits locally)
  *   - Generally higher quality than random initialization
  *
  * Algorithm:
  *   1. Start: all data in one cluster 2. While num_clusters < k:
  *      a. Select largest cluster (or highest cost cluster) b. Split it into 2 using k-means with k=2 c. Add both
  *         sub-clusters to active clusters 3. Return final k clusters
  *
  * Works with any Bregman divergence.
  *
  * @param config
  *   Configuration parameters
  * @param baseClusterer
  *   Clusterer to use for each split (default: ColumnTrackingKMeans)
  */
class BisectingKMeans(
  config: BisectingKMeansConfig = BisectingKMeansConfig(),
  baseClusterer: MultiKMeansClusterer = new ColumnTrackingKMeans()
) extends MultiKMeansClusterer
    with Logging {

  def cluster(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centers: Seq[IndexedSeq[BregmanCenter]]
  ): Seq[ClusteringWithDistortion] = {

    logger.info(s"Starting bisecting k-means with ${centers.size} initial center sets")

    // For bisecting k-means, we only use the first center set to determine target k
    val targetK = centers.head.length
    logger.info(s"Target k = $targetK")

    // Process each run independently (though typically runs=1 for bisecting)
    centers.map { initialCenters =>
      bisectingCluster(targetK, pointOps, data, initialCenters)
    }
  }

  /** Perform bisecting clustering to reach target k.
    */
  private def bisectingCluster(
    targetK: Int,
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    initialCenters: IndexedSeq[BregmanCenter]
  ): ClusteringWithDistortion = {

    logger.info(s"Bisecting clustering to k=$targetK")

    // Cache data for multiple splits
    data.cache()

    // Start with all data in one cluster
    // Use first initial center as starting point
    var clusters = List(
      ClusterNode(
        data = data,
        center = initialCenters.head,
        size = data.count(),
        cost = pointOps.distortion(data, IndexedSeq(initialCenters.head))
      )
    )

    var iteration = 0

    // Bisect until we reach target k
    while (clusters.length < targetK && iteration < targetK * 2) {
      iteration += 1

      // Select cluster to split
      val (toSplit, remaining) = selectClusterToSplit(clusters)

      if (toSplit.size < config.minClusterSize) {
        logger.warn(s"Cannot split cluster of size ${toSplit.size} (< ${config.minClusterSize})")
        // Can't split further, return what we have
        return buildFinalClustering(clusters, pointOps, data)
      }

      logger.info(
        f"Iteration $iteration: Splitting cluster with ${toSplit.size} points, " +
          f"cost ${toSplit.cost}%.4f"
      )

      // Split the selected cluster into 2
      val (left, right) = splitCluster(toSplit, pointOps)

      // Add both sub-clusters to active clusters
      clusters = left :: right :: remaining

      logger.info(f"  Split into clusters of size ${left.size} and ${right.size}")
    }

    val result = buildFinalClustering(clusters, pointOps, data)

    data.unpersist()

    logger.info(
      f"Bisecting k-means completed with k=${clusters.length}, " +
        f"distortion: ${result.distortion}%.4f"
    )

    result
  }

  /** Select which cluster to split next.
    */
  private def selectClusterToSplit(
    clusters: List[ClusterNode]
  ): (ClusterNode, List[ClusterNode]) = {
    val splittable = clusters.filter(_.size >= config.minClusterSize)

    if (splittable.isEmpty) {
      // No cluster can be split
      (clusters.head, clusters.tail)
    } else {
      val selected = config.splitCriterion match {
        case "largest"      => splittable.maxBy(_.size)
        case "highest_cost" => splittable.maxBy(_.cost)
        case _              => splittable.maxBy(_.size)
      }

      val remaining = clusters.filterNot(_ == selected)
      (selected, remaining)
    }
  }

  /** Split a cluster into two using k-means with k=2.
    */
  private def splitCluster(
    node: ClusterNode,
    pointOps: BregmanPointOps
  ): (ClusterNode, ClusterNode) = {

    // Initialize with k=2 using K-Means++ style initialization
    val initializer = new KMeansParallel(2)
    val initialCenters = initializer
      .init(
        pointOps,
        node.data,
        2,
        None,
        1,
        System.nanoTime()
      )
      .head

    // Cluster into 2
    val results = baseClusterer.cluster(
      config.maxIterationsPerSplit,
      pointOps,
      node.data,
      Seq(initialCenters)
    )

    val twoWayClustering = results.head

    // Assign points to the two clusters
    val assignments = node.data
      .map { point =>
        val cluster = pointOps.findClosestCluster(twoWayClustering.centers, point)
        (cluster, point)
      }
      .cache()

    // Build left and right clusters
    val leftData  = assignments.filter(_._1 == 0).map(_._2).cache()
    val rightData = assignments.filter(_._1 == 1).map(_._2).cache()

    val leftSize  = leftData.count()
    val rightSize = rightData.count()

    val leftCenter  = twoWayClustering.centers(0)
    val rightCenter = twoWayClustering.centers(1)

    val leftCost = if (leftSize > 0) {
      pointOps.distortion(leftData, IndexedSeq(leftCenter))
    } else 0.0

    val rightCost = if (rightSize > 0) {
      pointOps.distortion(rightData, IndexedSeq(rightCenter))
    } else 0.0

    assignments.unpersist()

    val left  = ClusterNode(leftData, leftCenter, leftSize, leftCost)
    val right = ClusterNode(rightData, rightCenter, rightSize, rightCost)

    (left, right)
  }

  /** Build final clustering from list of cluster nodes.
    */
  private def buildFinalClustering(
    clusters: List[ClusterNode],
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint]
  ): ClusteringWithDistortion = {

    val centers   = clusters.map(_.center).toIndexedSeq
    val totalCost = pointOps.distortion(data, centers)

    ClusteringWithDistortion(totalCost, centers)
  }

  /** Internal representation of a cluster during bisection.
    */
  private case class ClusterNode(
    data: RDD[BregmanPoint],
    center: BregmanCenter,
    size: Long,
    cost: Double
  )
}

object BisectingKMeans {

  /** Create bisecting k-means with default configuration.
    */
  def apply(): BisectingKMeans = new BisectingKMeans()

  /** Create bisecting k-means with custom configuration.
    */
  def apply(config: BisectingKMeansConfig): BisectingKMeans = {
    new BisectingKMeans(config)
  }

  /** Create bisecting k-means that splits by highest cost.
    */
  def byCost(): BisectingKMeans = new BisectingKMeans(
    BisectingKMeansConfig(splitCriterion = "highest_cost")
  )

  /** Create fast bisecting k-means with fewer iterations per split.
    */
  def fast(): BisectingKMeans = new BisectingKMeans(
    BisectingKMeansConfig(maxIterationsPerSplit = 10)
  )
}
