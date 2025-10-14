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
import com.massivedatascience.clusterer.coreset.{BregmanCoreset, BregmanSensitivity, CoresetConfig, CoresetResult, WeightedPoint}
import org.apache.spark.rdd.RDD

/**
 * Configuration for core-set based K-means clustering.
 * 
 * @param coresetConfig Configuration for core-set construction
 * @param maxIterations Maximum iterations for clustering on core-set
 * @param refinementIterations Number of refinement iterations on full data
 * @param convergenceThreshold Convergence threshold for core-set clustering
 * @param enableRefinement Whether to refine centers on full data after core-set clustering
 */
case class CoresetKMeansConfig(
    coresetConfig: CoresetConfig,
    maxIterations: Int = 50,
    refinementIterations: Int = 3,
    convergenceThreshold: Double = 1e-6,
    enableRefinement: Boolean = true) extends ConfigValidator {

  requirePositive(maxIterations, "Max iterations")
  requireNonNegative(refinementIterations, "Refinement iterations")
  requirePositive(convergenceThreshold, "Convergence threshold")
}

/**
 * Result of core-set based clustering with detailed metrics.
 * 
 * @param centers Final cluster centers
 * @param distortion Final clustering distortion
 * @param coresetResult Result from core-set construction
 * @param coresetIterations Number of iterations used on core-set
 * @param refinementIterations Number of refinement iterations used
 * @param totalTime Total clustering time in milliseconds
 * @param config Configuration used
 */
case class CoresetKMeansResult(
    centers: IndexedSeq[BregmanCenter],
    distortion: Double,
    coresetResult: CoresetResult,
    coresetIterations: Int,
    refinementIterations: Int,
    totalTime: Long,
    config: CoresetKMeansConfig) {
  
  /**
   * Get comprehensive statistics about the clustering result.
   */
  def getStats: Map[String, Double] = {
    val coresetStats = coresetResult.getStats
    val clusteringStats = Map(
      "numCenters" -> centers.length.toDouble,
      "finalDistortion" -> distortion,
      "coresetIterations" -> coresetIterations.toDouble,
      "refinementIterations" -> refinementIterations.toDouble,
      "totalTimeMs" -> totalTime.toDouble,
      "speedupEstimate" -> (coresetResult.originalSize.toDouble / coresetResult.coreset.length)
    )
    
    coresetStats ++ clusteringStats
  }
}

/**
 * Core-set based K-means clustering for Bregman divergences.
 * 
 * This implementation first constructs a small representative core-set,
 * performs exact clustering on the core-set, then optionally refines
 * the centers using the full dataset.
 */
case class CoresetKMeans(config: CoresetKMeansConfig = CoresetKMeans.defaultConfig)
    extends MultiKMeansClusterer with Logging {
  
  /**
   * Cluster the given points using core-set approximation.
   */
  def cluster(
      maxIterations: Int,
      pointOps: BregmanPointOps,
      data: RDD[BregmanPoint],
      initialCenters: Seq[IndexedSeq[BregmanCenter]]): Seq[ClusteringWithDistortion] = {
    
    require(initialCenters.nonEmpty, "At least one set of initial centers must be provided")
    
    val startTime = System.currentTimeMillis()
    
    // Use the number of centers from the first run to determine k
    val k = initialCenters.head.length
    logger.info(s"Starting core-set K-means with k=$k, ${initialCenters.length} runs")
    
    // Step 1: Build core-set
    logger.info("Building core-set...")
    val coresetBuilder = new BregmanCoreset(config.coresetConfig)
    val coresetResult = coresetBuilder.buildCoreset(data, k, pointOps)
    
    logger.info(s"Core-set built: ${coresetResult.coreset.length} points " +
      f"(${coresetResult.compressionRatio * 100}%.2f%% of original)")
    
    // Step 2: Cluster on core-set
    logger.info("Clustering on core-set...")
    val coresetClustering = clusterCoreset(coresetResult.coreset, initialCenters, pointOps)
    
    // Step 3: Refine on full data if enabled
    val finalResults = if (config.enableRefinement && config.refinementIterations > 0) {
      logger.info("Refining centers on full dataset...")
      coresetClustering.map { case ClusteringWithDistortion(_, centers) =>
        refineOnFullData(centers, data, pointOps)
      }
    } else {
      coresetClustering
    }
    
    val totalTime = System.currentTimeMillis() - startTime
    logger.info(f"Core-set clustering completed in ${totalTime}ms")
    
    finalResults
  }
  
  /**
   * Perform exact clustering on the core-set using in-memory Lloyd's algorithm.
   */
  private def clusterCoreset(
      coreset: Seq[WeightedPoint],
      initialCenters: Seq[IndexedSeq[BregmanCenter]],
      pointOps: BregmanPointOps): Seq[ClusteringWithDistortion] = {
    
    val coresetPoints = coreset.map(_.point)
    
    // Run Lloyd's algorithm for each set of initial centers
    initialCenters.map { centers =>
      val finalCenters = lloydIterationsOnCoreset(coresetPoints, centers, pointOps)
      val distortion = coresetPoints.map(p => pointOps.findClosestDistance(finalCenters, p)).sum
      ClusteringWithDistortion(distortion, finalCenters)
    }
  }
  
  /**
   * Run Lloyd's algorithm iterations on the core-set.
   */
  private def lloydIterationsOnCoreset(
      points: Seq[BregmanPoint],
      initialCenters: IndexedSeq[BregmanCenter],
      pointOps: BregmanPointOps): IndexedSeq[BregmanCenter] = {
    
    var centers = initialCenters
    var previousDistortion = Double.MaxValue
    
    for (iteration <- 1 to config.maxIterations) {
      // Assign points to closest centers
      val assignments = points.map { point =>
        val (closestIndex, distance) = pointOps.findClosest(centers, point)
        (closestIndex, point, distance)
      }
      
      // Compute new centers
      val newCenters = computeNewCentersFromAssignments(assignments, centers.length, pointOps)
      
      // Check convergence
      val distortion = assignments.map(_._3).sum
      val improvement = math.abs(previousDistortion - distortion) / math.max(previousDistortion, 1e-10)
      
      if (improvement < config.convergenceThreshold) {
        logger.debug(s"Core-set clustering converged after $iteration iterations")
        return newCenters
      }
      
      centers = newCenters
      previousDistortion = distortion
    }
    
    centers
  }
  
  /**
   * Compute new centers from point assignments.
   */
  private def computeNewCentersFromAssignments(
      assignments: Seq[(Int, BregmanPoint, Double)],
      numClusters: Int,
      pointOps: BregmanPointOps): IndexedSeq[BregmanCenter] = {
    
    // Group points by cluster
    val clusterGroups = assignments
      .filter(_._1 >= 0)
      .groupBy(_._1)
      .mapValues(_.map(_._2))
    
    // Compute center for each cluster
    (0 until numClusters).map { i =>
      clusterGroups.get(i) match {
        case Some(clusterPoints) if clusterPoints.nonEmpty =>
          val accumulator = pointOps.make()
          clusterPoints.foreach(accumulator.add)
          if (accumulator.weight > pointOps.weightThreshold) {
            pointOps.toCenter(accumulator.asImmutable)
          } else {
            // Empty cluster - keep previous center or create default
            pointOps.toCenter(pointOps.make().asImmutable)
          }
        case _ =>
          // Empty cluster
          logger.warn(s"Cluster $i is empty in core-set clustering")
          pointOps.toCenter(pointOps.make().asImmutable)
      }
    }
  }
  
  /**
   * Refine cluster centers using the full dataset.
   */
  private def refineOnFullData(
      initialCenters: IndexedSeq[BregmanCenter],
      fullData: RDD[BregmanPoint],
      pointOps: BregmanPointOps): ClusteringWithDistortion = {
    
    var centers = initialCenters
    var previousDistortion = Double.MaxValue
    
    for (iteration <- 1 to config.refinementIterations) {
      logger.debug(s"Refinement iteration $iteration")
      
      // Assign points to nearest centers
      val assignments = fullData.map { point =>
        val (closestIndex, distance) = pointOps.findClosest(centers, point)
        (closestIndex, point, distance)
      }
      
      // Compute new centers
      val newCenters = computeNewCenters(assignments, centers.length, pointOps)
      
      // Compute distortion
      val distortion = assignments.map(_._3).sum()
      
      logger.debug(f"Refinement iteration $iteration: distortion = $distortion%.6f")
      
      // Check convergence
      val improvement = (previousDistortion - distortion) / previousDistortion
      if (improvement < config.convergenceThreshold) {
        logger.debug(f"Refinement converged after $iteration iterations")
        return ClusteringWithDistortion(distortion, newCenters)
      }
      
      centers = newCenters
      previousDistortion = distortion
    }
    
    val finalDistortion = fullData.map(point => pointOps.findClosestDistance(centers, point)).sum()
    ClusteringWithDistortion(finalDistortion, centers)
  }
  
  /**
   * Compute new cluster centers from point assignments.
   */
  private def computeNewCenters(
      assignments: RDD[(Int, BregmanPoint, Double)],
      numClusters: Int,
      pointOps: BregmanPointOps): IndexedSeq[BregmanCenter] = {
    
    // Aggregate points by cluster
    val clusterSums = assignments
      .filter(_._1 >= 0) // Filter out unassigned points
      .map { case (cluster, point, _) => (cluster, point) }
      .aggregateByKey(pointOps.make())(
        (accumulator, point) => accumulator.add(point),
        (acc1, acc2) => acc1.add(acc2)
      )
      .collectAsMap()
    
    // Convert to centers, handling empty clusters
    (0 until numClusters).map { i =>
      clusterSums.get(i) match {
        case Some(sum) if sum.weight > pointOps.weightThreshold =>
          pointOps.toCenter(sum.asImmutable)
        case _ =>
          // Handle empty cluster - use a random point or previous center
          logger.warn(s"Cluster $i is empty during refinement")
          // For now, create a zero-weight center
          pointOps.toCenter(pointOps.make().asImmutable)
      }
    }
  }
}

object CoresetKMeans {
  
  /**
   * Default configuration for core-set K-means.
   */
  def defaultConfig: CoresetKMeansConfig = {
    CoresetKMeansConfig(
      coresetConfig = CoresetConfig(
        coresetSize = 1000,
        epsilon = 0.1,
        sensitivity = BregmanSensitivity.hybrid()
      ),
      maxIterations = 50,
      refinementIterations = 3,
      enableRefinement = true
    )
  }
  
  /**
   * Create CoresetKMeans with specified core-set size.
   */
  def apply(coresetSize: Int): CoresetKMeans = {
    val config = defaultConfig.copy(
      coresetConfig = defaultConfig.coresetConfig.copy(coresetSize = coresetSize)
    )
    CoresetKMeans(config)
  }
  
  /**
   * Create CoresetKMeans with specified compression ratio.
   */
  def withCompressionRatio(compressionRatio: Double): CoresetKMeans = {
    require(compressionRatio > 0.0 && compressionRatio <= 1.0, 
      s"Compression ratio must be in (0,1], got: $compressionRatio")
    
    // Core-set size will be determined at runtime based on data size
    val config = defaultConfig.copy(
      coresetConfig = defaultConfig.coresetConfig.copy(
        coresetSize = math.max(100, (10000 * compressionRatio).toInt) // Rough estimate
      )
    )
    CoresetKMeans(config)
  }
  
  /**
   * Create high-quality CoresetKMeans with conservative parameters.
   */
  def highQuality(epsilon: Double = 0.05): CoresetKMeans = {
    val config = CoresetKMeansConfig(
      coresetConfig = CoresetConfig(
        coresetSize = 5000, // Larger core-set for better quality
        epsilon = epsilon,
        sensitivity = BregmanSensitivity.hybrid(distanceWeight = 0.7, densityWeight = 0.3),
        minSamplingProb = 1e-5
      ),
      maxIterations = 100,
      refinementIterations = 5,
      convergenceThreshold = 1e-8,
      enableRefinement = true
    )
    CoresetKMeans(config)
  }
  
  /**
   * Create fast CoresetKMeans with aggressive compression.
   */
  def fast(coresetSize: Int = 100): CoresetKMeans = {
    val config = CoresetKMeansConfig(
      coresetConfig = CoresetConfig(
        coresetSize = coresetSize,
        epsilon = 0.2, // More aggressive approximation
        sensitivity = BregmanSensitivity.uniform(), // Faster uniform sampling
        minSamplingProb = 1e-4
      ),
      maxIterations = 20,
      refinementIterations = 1,
      enableRefinement = true
    )
    CoresetKMeans(config)
  }
  
  /**
   * Quick clustering with automatic parameter selection.
   */
  def quick(
      points: RDD[BregmanPoint],
      k: Int,
      pointOps: BregmanPointOps,
      compressionRatio: Double = 0.01): CoresetKMeansResult = {
    
    val startTime = System.currentTimeMillis()
    
    // Build core-set
    val coresetResult = BregmanCoreset.quick(points, k, pointOps, compressionRatio)
    
    // Initialize centers using k-means++
    val selector = KMeansSelector(KMeansSelector.K_MEANS_PARALLEL)
    val initialCenters = selector.init(pointOps, points, k, None, 1, 42L).head
    
    // Cluster using core-set
    val clusterer = CoresetKMeans.withCompressionRatio(compressionRatio)
    val clusteringResults = clusterer.cluster(50, pointOps, points, Seq(initialCenters))
    
    val totalTime = System.currentTimeMillis() - startTime
    val bestResult = clusteringResults.minBy(_.distortion)
    
    CoresetKMeansResult(
      centers = bestResult.centers,
      distortion = bestResult.distortion,
      coresetResult = coresetResult,
      coresetIterations = 50, // Approximate
      refinementIterations = clusterer.config.refinementIterations,
      totalTime = totalTime,
      config = clusterer.config
    )
  }
}