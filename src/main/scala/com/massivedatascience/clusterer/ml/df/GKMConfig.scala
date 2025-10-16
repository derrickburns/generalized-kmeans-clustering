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

package com.massivedatascience.clusterer.ml.df

/** Configuration for generalized k-means clustering.
  *
  * This provides a more convenient, immutable configuration object that can be easily constructed and modified. It
  * reduces ceremony compared to working directly with Spark ML Params.
  *
  * Example usage:
  * {{{
  *   // Simple configuration
  *   val config = GKMConfig(k = 5, maxIter = 100)
  *
  *   // With method chaining
  *   val config = GKMConfig.default
  *     .withK(10)
  *     .withKernel("kl")
  *     .withTolerance(1e-6)
  *
  *   // Preset configurations
  *   val config = GKMConfig.euclidean(k = 5)
  *   val config = GKMConfig.kl(k = 10, maxIter = 50)
  * }}}
  */
case class GKMConfig(
  k: Int = 5,
  maxIter: Int = 20,
  tolerance: Double = 1e-4,
  seed: Long = 42L,
  kernel: String = "squaredEuclidean",
  initMode: String = "kmeans++",
  featuresCol: String = "features",
  predictionCol: String = "prediction",
  weightCol: Option[String] = None,
  distanceCol: Option[String] = None,
  // Advanced options
  miniBatchFraction: Double = 1.0,
  reseedPolicy: String = "random",
  validateData: Boolean = true,
  checkpointInterval: Int = 10,
  enableTelemetry: Boolean = false
) {

  /** Set number of clusters */
  def withK(k: Int): GKMConfig = copy(k = k)

  /** Set maximum iterations */
  def withMaxIter(maxIter: Int): GKMConfig = copy(maxIter = maxIter)

  /** Set convergence tolerance */
  def withTolerance(tolerance: Double): GKMConfig = copy(tolerance = tolerance)

  /** Set random seed */
  def withSeed(seed: Long): GKMConfig = copy(seed = seed)

  /** Set distance kernel/divergence */
  def withKernel(kernel: String): GKMConfig = copy(kernel = kernel)

  /** Set initialization mode */
  def withInitMode(initMode: String): GKMConfig = copy(initMode = initMode)

  /** Set features column name */
  def withFeaturesCol(col: String): GKMConfig = copy(featuresCol = col)

  /** Set prediction column name */
  def withPredictionCol(col: String): GKMConfig = copy(predictionCol = col)

  /** Set weight column */
  def withWeightCol(col: String): GKMConfig = copy(weightCol = Some(col))

  /** Remove weight column */
  def withoutWeightCol(): GKMConfig = copy(weightCol = None)

  /** Set distance column */
  def withDistanceCol(col: String): GKMConfig = copy(distanceCol = Some(col))

  /** Remove distance column */
  def withoutDistanceCol(): GKMConfig = copy(distanceCol = None)

  /** Set mini-batch fraction (1.0 = full batch) */
  def withMiniBatchFraction(fraction: Double): GKMConfig = copy(miniBatchFraction = fraction)

  /** Set reseed policy for empty clusters */
  def withReseedPolicy(policy: String): GKMConfig = copy(reseedPolicy = policy)

  /** Enable/disable data validation */
  def withValidation(validate: Boolean): GKMConfig = copy(validateData = validate)

  /** Set checkpoint interval */
  def withCheckpointInterval(interval: Int): GKMConfig = copy(checkpointInterval = interval)

  /** Enable/disable telemetry */
  def withTelemetry(enabled: Boolean): GKMConfig = copy(enableTelemetry = enabled)

  /** Validate configuration and return errors if any */
  def validate(): GKMResult[GKMConfig] = {
    if (k <= 0) {
      GKMResult.failure(InvalidK(k, -1)) // n unknown at config time
    } else if (maxIter < 1) {
      GKMResult.failure(InvalidMaxIterations(maxIter))
    } else if (tolerance < 0.0) {
      GKMResult.failure(InvalidTolerance(tolerance))
    } else if (seed < 0) {
      GKMResult.failure(InvalidSeed(seed))
    } else if (miniBatchFraction <= 0.0 || miniBatchFraction > 1.0) {
      GKMResult.failure(InvalidState(s"miniBatchFraction must be in (0, 1], got $miniBatchFraction"))
    } else if (checkpointInterval < 1) {
      GKMResult.failure(InvalidState(s"checkpointInterval must be >= 1, got $checkpointInterval"))
    } else {
      GKMResult.success(this)
    }
  }

  /** Convert to human-readable summary */
  def summary: String = {
    val sb = new StringBuilder
    sb.append("K-Means Configuration:\n")
    sb.append(s"  k: $k\n")
    sb.append(s"  maxIter: $maxIter\n")
    sb.append(s"  tolerance: $tolerance\n")
    sb.append(s"  kernel: $kernel\n")
    sb.append(s"  initMode: $initMode\n")
    sb.append(s"  seed: $seed\n")
    sb.append(s"  featuresCol: $featuresCol\n")
    sb.append(s"  predictionCol: $predictionCol\n")
    weightCol.foreach(col => sb.append(s"  weightCol: $col\n"))
    distanceCol.foreach(col => sb.append(s"  distanceCol: $col\n"))
    if (miniBatchFraction < 1.0) {
      sb.append(s"  miniBatchFraction: $miniBatchFraction\n")
    }
    sb.append(s"  reseedPolicy: $reseedPolicy\n")
    sb.append(s"  validateData: $validateData\n")
    sb.append(s"  checkpointInterval: $checkpointInterval\n")
    sb.append(s"  enableTelemetry: $enableTelemetry\n")
    sb.toString()
  }
}

object GKMConfig {

  /** Default configuration */
  def default: GKMConfig = GKMConfig()

  /** Configuration for Euclidean distance (squared) */
  def euclidean(k: Int, maxIter: Int = 20, tolerance: Double = 1e-4): GKMConfig = {
    GKMConfig(k = k, maxIter = maxIter, tolerance = tolerance, kernel = "squaredEuclidean")
  }

  /** Configuration for KL divergence */
  def kl(k: Int, maxIter: Int = 20, tolerance: Double = 1e-4): GKMConfig = {
    GKMConfig(k = k, maxIter = maxIter, tolerance = tolerance, kernel = "kl")
      .withValidation(true) // KL requires positive features
  }

  /** Configuration for Manhattan distance (L1) */
  def manhattan(k: Int, maxIter: Int = 20, tolerance: Double = 1e-4): GKMConfig = {
    GKMConfig(k = k, maxIter = maxIter, tolerance = tolerance, kernel = "manhattan")
  }

  /** Configuration for Itakura-Saito divergence */
  def itakuraSaito(k: Int, maxIter: Int = 20, tolerance: Double = 1e-4): GKMConfig = {
    GKMConfig(k = k, maxIter = maxIter, tolerance = tolerance, kernel = "itakuraSaito")
      .withValidation(true) // Itakura-Saito requires positive features
  }

  /** Configuration for cosine similarity */
  def cosine(k: Int, maxIter: Int = 20, tolerance: Double = 1e-4): GKMConfig = {
    GKMConfig(k = k, maxIter = maxIter, tolerance = tolerance, kernel = "cosine")
  }

  /** Mini-batch configuration (10% sampling) */
  def miniBatch(k: Int, fraction: Double = 0.1, maxIter: Int = 50): GKMConfig = {
    GKMConfig(k = k, maxIter = maxIter, miniBatchFraction = fraction)
  }

  /** Fast configuration (fewer iterations, looser tolerance) */
  def fast(k: Int): GKMConfig = {
    GKMConfig(k = k, maxIter = 10, tolerance = 1e-3)
  }

  /** High-quality configuration (more iterations, tighter tolerance) */
  def highQuality(k: Int): GKMConfig = {
    GKMConfig(k = k, maxIter = 100, tolerance = 1e-6)
  }

  /** Debugging configuration (telemetry enabled, validation on) */
  def debug(k: Int): GKMConfig = {
    GKMConfig(k = k, enableTelemetry = true, validateData = true)
  }

  /** Production configuration (validation on, telemetry off) */
  def production(k: Int, maxIter: Int = 20): GKMConfig = {
    GKMConfig(k = k, maxIter = maxIter, validateData = true, enableTelemetry = false)
  }

  /** Weighted clustering configuration */
  def weighted(k: Int, weightCol: String, maxIter: Int = 20): GKMConfig = {
    GKMConfig(k = k, maxIter = maxIter, weightCol = Some(weightCol))
  }
}

/** Builder for constructing GKMConfig with fluent API.
  *
  * This provides an alternative to the case class copy methods for those who prefer builder pattern.
  *
  * Example:
  * {{{
  *   val config = GKMConfigBuilder()
  *     .setK(10)
  *     .setKernel("kl")
  *     .setMaxIter(50)
  *     .build()
  * }}}
  */
class GKMConfigBuilder private (private var config: GKMConfig) {

  def setK(k: Int): GKMConfigBuilder = { config = config.withK(k); this }
  def setMaxIter(maxIter: Int): GKMConfigBuilder = { config = config.withMaxIter(maxIter); this }
  def setTolerance(tolerance: Double): GKMConfigBuilder = { config = config.withTolerance(tolerance); this }
  def setSeed(seed: Long): GKMConfigBuilder = { config = config.withSeed(seed); this }
  def setKernel(kernel: String): GKMConfigBuilder = { config = config.withKernel(kernel); this }
  def setInitMode(initMode: String): GKMConfigBuilder = { config = config.withInitMode(initMode); this }
  def setFeaturesCol(col: String): GKMConfigBuilder = { config = config.withFeaturesCol(col); this }
  def setPredictionCol(col: String): GKMConfigBuilder = { config = config.withPredictionCol(col); this }
  def setWeightCol(col: String): GKMConfigBuilder = { config = config.withWeightCol(col); this }
  def setDistanceCol(col: String): GKMConfigBuilder = { config = config.withDistanceCol(col); this }
  def setMiniBatchFraction(fraction: Double): GKMConfigBuilder = { config = config.withMiniBatchFraction(fraction); this }
  def setReseedPolicy(policy: String): GKMConfigBuilder = { config = config.withReseedPolicy(policy); this }
  def setValidation(validate: Boolean): GKMConfigBuilder = { config = config.withValidation(validate); this }
  def setCheckpointInterval(interval: Int): GKMConfigBuilder = { config = config.withCheckpointInterval(interval); this }
  def setTelemetry(enabled: Boolean): GKMConfigBuilder = { config = config.withTelemetry(enabled); this }

  /** Build the configuration */
  def build(): GKMConfig = config

  /** Build and validate the configuration */
  def buildValidated(): GKMResult[GKMConfig] = config.validate()
}

object GKMConfigBuilder {

  /** Create a new builder with default configuration */
  def apply(): GKMConfigBuilder = new GKMConfigBuilder(GKMConfig.default)

  /** Create a new builder from existing configuration */
  def apply(config: GKMConfig): GKMConfigBuilder = new GKMConfigBuilder(config)
}

/** Preset configurations for common use cases */
object GKMPresets {

  /** Text clustering (cosine similarity, higher k) */
  def textClustering(k: Int = 20): GKMConfig = {
    GKMConfig.cosine(k, maxIter = 30)
  }

  /** Image clustering (Euclidean, moderate k) */
  def imageClustering(k: Int = 10): GKMConfig = {
    GKMConfig.euclidean(k, maxIter = 50)
  }

  /** Topic modeling (KL divergence, many clusters) */
  def topicModeling(k: Int = 50): GKMConfig = {
    GKMConfig.kl(k, maxIter = 100, tolerance = 1e-5)
  }

  /** Anomaly detection (small k, tight tolerance) */
  def anomalyDetection(k: Int = 3): GKMConfig = {
    GKMConfig.euclidean(k, maxIter = 100, tolerance = 1e-6)
  }

  /** Large dataset (mini-batch, many iterations) */
  def largeDataset(k: Int = 10): GKMConfig = {
    GKMConfig.miniBatch(k, fraction = 0.1, maxIter = 100)
  }

  /** Streaming/online (very small batch, many iterations) */
  def streaming(k: Int = 10): GKMConfig = {
    GKMConfig.miniBatch(k, fraction = 0.01, maxIter = 1000)
      .withCheckpointInterval(100)
  }

  /** Robust clustering (Manhattan, high quality) */
  def robust(k: Int = 5): GKMConfig = {
    GKMConfig.manhattan(k, maxIter = 100, tolerance = 1e-6)
  }
}
