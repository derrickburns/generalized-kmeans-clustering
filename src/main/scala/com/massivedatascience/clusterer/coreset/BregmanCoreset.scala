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

package com.massivedatascience.clusterer.coreset

import com.massivedatascience.clusterer.{BregmanPoint, BregmanPointOps}
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory

import scala.util.Random

/**
 * Configuration for Bregman core-set construction.
 * 
 * @param coresetSize Target size of the core-set
 * @param epsilon Approximation parameter (smaller = better quality, larger core-set)
 * @param sensitivity Sensitivity computation strategy
 * @param seed Random seed for reproducibility
 * @param minSamplingProb Minimum sampling probability to avoid extreme weights
 * @param maxWeight Maximum weight allowed for core-set points
 */
case class CoresetConfig(
    coresetSize: Int,
    epsilon: Double = 0.1,
    sensitivity: BregmanSensitivity = BregmanSensitivity.hybrid(),
    seed: Long = 42L,
    minSamplingProb: Double = 1e-6,
    maxWeight: Double = 1e6) {
  
  require(coresetSize > 0, s"Core-set size must be positive, got: $coresetSize")
  require(epsilon > 0.0 && epsilon < 1.0, s"Epsilon must be in (0,1), got: $epsilon")
  require(minSamplingProb > 0.0 && minSamplingProb <= 1.0, 
    s"Minimum sampling probability must be in (0,1], got: $minSamplingProb")
  require(maxWeight > 1.0, s"Maximum weight must be > 1.0, got: $maxWeight")
}

/**
 * Result of core-set construction with quality metrics.
 * 
 * @param coreset The constructed core-set
 * @param originalSize Size of the original dataset
 * @param compressionRatio Ratio of core-set size to original size
 * @param totalSensitivity Sum of all sensitivity scores
 * @param avgSamplingProb Average sampling probability used
 * @param config Configuration used for construction
 */
case class CoresetResult(
    coreset: Seq[WeightedPoint],
    originalSize: Long,
    compressionRatio: Double,
    totalSensitivity: Double,
    avgSamplingProb: Double,
    config: CoresetConfig) {
  
  /**
   * Get the effective size of the core-set (sum of importance weights).
   */
  def effectiveSize: Double = coreset.map(_.importance).sum
  
  /**
   * Get statistics about the core-set.
   */
  def getStats: Map[String, Double] = {
    val weights = coreset.map(_.importance)
    val sensitivities = coreset.map(_.sensitivity)
    
    Map(
      "coresetSize" -> coreset.length.toDouble,
      "originalSize" -> originalSize.toDouble,
      "compressionRatio" -> compressionRatio,
      "effectiveSize" -> effectiveSize,
      "totalSensitivity" -> totalSensitivity,
      "avgSamplingProb" -> avgSamplingProb,
      "minWeight" -> (if (weights.nonEmpty) weights.min else 0.0),
      "maxWeight" -> (if (weights.nonEmpty) weights.max else 0.0),
      "avgWeight" -> (if (weights.nonEmpty) weights.sum / weights.length else 0.0),
      "minSensitivity" -> (if (sensitivities.nonEmpty) sensitivities.min else 0.0),
      "maxSensitivity" -> (if (sensitivities.nonEmpty) sensitivities.max else 0.0),
      "avgSensitivity" -> (if (sensitivities.nonEmpty) sensitivities.sum / sensitivities.length else 0.0)
    )
  }
}

/**
 * Bregman core-set construction for efficient approximate clustering.
 * 
 * This class implements sensitivity-based sampling to create a small weighted
 * subset of points that preserves the clustering structure for Bregman divergences.
 */
class BregmanCoreset(config: CoresetConfig) extends Serializable {
  
  @transient private lazy val logger = LoggerFactory.getLogger(getClass.getName)
  
  /**
   * Construct a core-set from the given points.
   * 
   * @param points RDD of points to create core-set from
   * @param k Number of clusters (affects sensitivity computation)
   * @param pointOps Bregman point operations
   * @return CoresetResult containing the core-set and metadata
   */
  def buildCoreset(
      points: RDD[BregmanPoint],
      k: Int,
      pointOps: BregmanPointOps): CoresetResult = {
    
    require(k > 0, s"Number of clusters must be positive, got: $k")
    
    val originalSize = points.count()
    logger.info(s"Building core-set from $originalSize points, target size: ${config.coresetSize}")
    
    if (originalSize <= config.coresetSize) {
      logger.info("Dataset smaller than target core-set size, returning all points")
      val allPoints = points.collect().map(WeightedPoint(_))
      return CoresetResult(
        coreset = allPoints,
        originalSize = originalSize,
        compressionRatio = 1.0,
        totalSensitivity = allPoints.length.toDouble,
        avgSamplingProb = 1.0,
        config = config
      )
    }
    
    // Step 1: Compute sensitivity scores
    logger.info("Computing sensitivity scores...")
    val sensitivities = config.sensitivity.computeBatchSensitivity(points, k, pointOps)
    val totalSensitivity = sensitivities.map(_._2).sum()
    
    logger.info(f"Total sensitivity: $totalSensitivity%.4f")
    
    // Step 2: Compute sampling probabilities
    logger.info("Computing sampling probabilities...")
    val samplingProbs = sensitivities.map { case (point, sensitivity) =>
      val prob = math.min(1.0, math.max(config.minSamplingProb, 
        config.coresetSize * sensitivity / totalSensitivity))
      (point, sensitivity, prob)
    }
    
    val avgSamplingProb = samplingProbs.map(_._3).mean()
    logger.info(f"Average sampling probability: $avgSamplingProb%.6f")
    
    // Step 3: Perform stratified sampling
    logger.info("Performing stratified sampling...")
    val coresetRDD = samplingProbs.mapPartitionsWithIndex { (partitionId, iter) =>
      val random = new Random(config.seed + partitionId)
      
      iter.flatMap { case (point, sensitivity, prob) =>
        if (random.nextDouble() < prob) {
          val importance = math.min(config.maxWeight, 1.0 / prob)
          Some(WeightedPoint(point, importance, sensitivity))
        } else {
          None
        }
      }
    }
    
    // Step 4: Collect core-set
    val coreset = coresetRDD.collect()
    val actualCoresetSize = coreset.length
    val compressionRatio = actualCoresetSize.toDouble / originalSize
    
    logger.info(s"Core-set construction complete:")
    logger.info(s"  Original size: $originalSize")
    logger.info(s"  Core-set size: $actualCoresetSize")
    logger.info(f"  Compression ratio: ${compressionRatio * 100}%.2f%%")
    logger.info(f"  Effective size: ${coreset.map(_.importance).sum}%.1f")
    
    CoresetResult(
      coreset = coreset,
      originalSize = originalSize,
      compressionRatio = compressionRatio,
      totalSensitivity = totalSensitivity,
      avgSamplingProb = avgSamplingProb,
      config = config
    )
  }
  
  /**
   * Build a core-set with automatic size estimation based on epsilon.
   * 
   * Uses the theoretical bound: coreset_size = O(k * log(k) / epsilon^2)
   */
  def buildAdaptiveCoreset(
      points: RDD[BregmanPoint],
      k: Int,
      pointOps: BregmanPointOps): CoresetResult = {
    
    val theoreticalSize = math.ceil(k * math.log(k) / (config.epsilon * config.epsilon)).toInt
    val adaptiveSize = math.max(theoreticalSize, config.coresetSize)
    
    logger.info(s"Using adaptive core-set size: $adaptiveSize (theoretical: $theoreticalSize)")
    
    val adaptiveConfig = config.copy(coresetSize = adaptiveSize)
    val adaptiveCoreset = new BregmanCoreset(adaptiveConfig)
    
    adaptiveCoreset.buildCoreset(points, k, pointOps)
  }
  
  /**
   * Build multiple core-sets with different parameters for ensemble methods.
   */
  def buildEnsembleCoresets(
      points: RDD[BregmanPoint],
      k: Int,
      pointOps: BregmanPointOps,
      numCoresets: Int = 3): Seq[CoresetResult] = {
    
    require(numCoresets > 0, s"Number of core-sets must be positive, got: $numCoresets")
    
    logger.info(s"Building ensemble of $numCoresets core-sets")
    
    (0 until numCoresets).map { i =>
      // Vary the random seed and slightly vary core-set size
      val seedVariation = config.seed + i * 12345
      val sizeVariation = (config.coresetSize * (0.8 + 0.4 * i / numCoresets)).toInt
      
      val variedConfig = config.copy(
        seed = seedVariation,
        coresetSize = math.max(k, sizeVariation)
      )
      
      val variedCoreset = new BregmanCoreset(variedConfig)
      variedCoreset.buildCoreset(points, k, pointOps)
    }
  }
}

object BregmanCoreset {
  
  /**
   * Create a BregmanCoreset with default configuration.
   */
  def apply(coresetSize: Int): BregmanCoreset = {
    new BregmanCoreset(CoresetConfig(coresetSize = coresetSize))
  }
  
  /**
   * Create a BregmanCoreset with specified epsilon (approximation quality).
   */
  def withEpsilon(coresetSize: Int, epsilon: Double): BregmanCoreset = {
    new BregmanCoreset(CoresetConfig(coresetSize = coresetSize, epsilon = epsilon))
  }
  
  /**
   * Create a BregmanCoreset with custom sensitivity computation.
   */
  def withSensitivity(coresetSize: Int, sensitivity: BregmanSensitivity): BregmanCoreset = {
    new BregmanCoreset(CoresetConfig(coresetSize = coresetSize, sensitivity = sensitivity))
  }
  
  /**
   * Quick core-set construction with reasonable defaults.
   * 
   * @param points Points to create core-set from
   * @param k Number of clusters
   * @param pointOps Bregman operations
   * @param compressionRatio Target compression ratio (0.01 = 1% of original size)
   * @return CoresetResult
   */
  def quick(
      points: RDD[BregmanPoint],
      k: Int,
      pointOps: BregmanPointOps,
      compressionRatio: Double = 0.01): CoresetResult = {
    
    val originalSize = points.count()
    val targetSize = math.max(k, (originalSize * compressionRatio).toInt)
    
    val coreset = BregmanCoreset(targetSize)
    coreset.buildCoreset(points, k, pointOps)
  }
  
  /**
   * Create a high-quality core-set with conservative parameters.
   */
  def highQuality(
      points: RDD[BregmanPoint],
      k: Int,
      pointOps: BregmanPointOps,
      epsilon: Double = 0.05): CoresetResult = {
    
    val theoreticalSize = math.ceil(k * math.log(k) / (epsilon * epsilon)).toInt
    val conservativeSize = math.max(theoreticalSize, k * 10) // At least 10 points per cluster
    
    val config = CoresetConfig(
      coresetSize = conservativeSize,
      epsilon = epsilon,
      sensitivity = BregmanSensitivity.hybrid(distanceWeight = 0.7, densityWeight = 0.3),
      minSamplingProb = 1e-5
    )
    
    val coreset = new BregmanCoreset(config)
    coreset.buildCoreset(points, k, pointOps)
  }
}