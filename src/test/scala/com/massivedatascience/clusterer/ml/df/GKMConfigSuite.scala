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

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class GKMConfigSuite extends AnyFunSuite with Matchers {

  test("GKMConfig.default should have sensible defaults") {
    val config = GKMConfig.default

    assert(config.k == 5)
    assert(config.maxIter == 20)
    assert(config.tolerance == 1e-4)
    assert(config.seed == 42L)
    assert(config.kernel == "squaredEuclidean")
    assert(config.initMode == "kmeans++")
    assert(config.featuresCol == "features")
    assert(config.predictionCol == "prediction")
    assert(config.weightCol.isEmpty)
    assert(config.miniBatchFraction == 1.0)
  }

  test("GKMConfig should support method chaining") {
    val config = GKMConfig.default
      .withK(10)
      .withKernel("kl")
      .withMaxIter(50)
      .withTolerance(1e-6)

    assert(config.k == 10)
    assert(config.kernel == "kl")
    assert(config.maxIter == 50)
    assert(config.tolerance == 1e-6)
  }

  test("GKMConfig should support column configuration") {
    val config = GKMConfig.default
      .withFeaturesCol("myFeatures")
      .withPredictionCol("myPrediction")
      .withWeightCol("myWeight")
      .withDistanceCol("myDistance")

    assert(config.featuresCol == "myFeatures")
    assert(config.predictionCol == "myPrediction")
    assert(config.weightCol.contains("myWeight"))
    assert(config.distanceCol.contains("myDistance"))
  }

  test("GKMConfig should support removing optional columns") {
    val config = GKMConfig.default
      .withWeightCol("weight")
      .withDistanceCol("distance")
      .withoutWeightCol()
      .withoutDistanceCol()

    assert(config.weightCol.isEmpty)
    assert(config.distanceCol.isEmpty)
  }

  test("GKMConfig.validate should accept valid configuration") {
    val config = GKMConfig.default
    val result = config.validate()

    assert(result.isSuccess)
    assert(result.get == config)
  }

  test("GKMConfig.validate should reject invalid k") {
    val config = GKMConfig.default.withK(0)
    val result = config.validate()

    assert(result.isFailure)
    assert(result.error.exists(_.isInstanceOf[InvalidK]))
  }

  test("GKMConfig.validate should reject invalid maxIter") {
    val config = GKMConfig.default.withMaxIter(0)
    val result = config.validate()

    assert(result.isFailure)
    assert(result.error.exists(_.isInstanceOf[InvalidMaxIterations]))
  }

  test("GKMConfig.validate should reject invalid tolerance") {
    val config = GKMConfig.default.withTolerance(-0.1)
    val result = config.validate()

    assert(result.isFailure)
    assert(result.error.exists(_.isInstanceOf[InvalidTolerance]))
  }

  test("GKMConfig.validate should reject invalid seed") {
    val config = GKMConfig.default.withSeed(-1)
    val result = config.validate()

    assert(result.isFailure)
    assert(result.error.exists(_.isInstanceOf[InvalidSeed]))
  }

  test("GKMConfig.validate should reject invalid miniBatchFraction") {
    val config1 = GKMConfig.default.withMiniBatchFraction(0.0)
    assert(config1.validate().isFailure)

    val config2 = GKMConfig.default.withMiniBatchFraction(1.5)
    assert(config2.validate().isFailure)

    val config3 = GKMConfig.default.withMiniBatchFraction(0.5)
    assert(config3.validate().isSuccess)
  }

  test("GKMConfig.summary should generate readable output") {
    val config = GKMConfig.default.withK(10).withKernel("kl")
    val summary = config.summary

    assert(summary.contains("k: 10"))
    assert(summary.contains("kernel: kl"))
    assert(summary.contains("K-Means Configuration"))
  }

  test("GKMConfig.euclidean should create Euclidean config") {
    val config = GKMConfig.euclidean(k = 10, maxIter = 50)

    assert(config.k == 10)
    assert(config.maxIter == 50)
    assert(config.kernel == "squaredEuclidean")
  }

  test("GKMConfig.kl should create KL config with validation") {
    val config = GKMConfig.kl(k = 10)

    assert(config.k == 10)
    assert(config.kernel == "kl")
    assert(config.validateData)
  }

  test("GKMConfig.manhattan should create Manhattan config") {
    val config = GKMConfig.manhattan(k = 5)

    assert(config.k == 5)
    assert(config.kernel == "manhattan")
  }

  test("GKMConfig.itakuraSaito should create Itakura-Saito config") {
    val config = GKMConfig.itakuraSaito(k = 8)

    assert(config.k == 8)
    assert(config.kernel == "itakuraSaito")
    assert(config.validateData)
  }

  test("GKMConfig.cosine should create cosine config") {
    val config = GKMConfig.cosine(k = 15)

    assert(config.k == 15)
    assert(config.kernel == "cosine")
  }

  test("GKMConfig.miniBatch should create mini-batch config") {
    val config = GKMConfig.miniBatch(k = 10, fraction = 0.2, maxIter = 100)

    assert(config.k == 10)
    assert(config.miniBatchFraction == 0.2)
    assert(config.maxIter == 100)
  }

  test("GKMConfig.fast should create fast config") {
    val config = GKMConfig.fast(k = 5)

    assert(config.k == 5)
    assert(config.maxIter == 10)
    assert(config.tolerance == 1e-3)
  }

  test("GKMConfig.highQuality should create high-quality config") {
    val config = GKMConfig.highQuality(k = 5)

    assert(config.k == 5)
    assert(config.maxIter == 100)
    assert(config.tolerance == 1e-6)
  }

  test("GKMConfig.debug should enable telemetry and validation") {
    val config = GKMConfig.debug(k = 5)

    assert(config.k == 5)
    assert(config.enableTelemetry)
    assert(config.validateData)
  }

  test("GKMConfig.production should enable validation, disable telemetry") {
    val config = GKMConfig.production(k = 5)

    assert(config.k == 5)
    assert(config.validateData)
    assert(!config.enableTelemetry)
  }

  test("GKMConfig.weighted should set weight column") {
    val config = GKMConfig.weighted(k = 5, weightCol = "myWeight")

    assert(config.k == 5)
    assert(config.weightCol.contains("myWeight"))
  }

  test("GKMConfigBuilder should build configuration") {
    val config = GKMConfigBuilder()
      .setK(10)
      .setKernel("kl")
      .setMaxIter(50)
      .setTolerance(1e-6)
      .build()

    assert(config.k == 10)
    assert(config.kernel == "kl")
    assert(config.maxIter == 50)
    assert(config.tolerance == 1e-6)
  }

  test("GKMConfigBuilder should support all setters") {
    val config = GKMConfigBuilder()
      .setK(10)
      .setMaxIter(50)
      .setTolerance(1e-6)
      .setSeed(123)
      .setKernel("kl")
      .setInitMode("random")
      .setFeaturesCol("f")
      .setPredictionCol("p")
      .setWeightCol("w")
      .setDistanceCol("d")
      .setMiniBatchFraction(0.5)
      .setReseedPolicy("farthest")
      .setValidation(true)
      .setCheckpointInterval(20)
      .setTelemetry(true)
      .build()

    assert(config.k == 10)
    assert(config.maxIter == 50)
    assert(config.tolerance == 1e-6)
    assert(config.seed == 123)
    assert(config.kernel == "kl")
    assert(config.initMode == "random")
    assert(config.featuresCol == "f")
    assert(config.predictionCol == "p")
    assert(config.weightCol.contains("w"))
    assert(config.distanceCol.contains("d"))
    assert(config.miniBatchFraction == 0.5)
    assert(config.reseedPolicy == "farthest")
    assert(config.validateData)
    assert(config.checkpointInterval == 20)
    assert(config.enableTelemetry)
  }

  test("GKMConfigBuilder should build from existing config") {
    val original = GKMConfig.default.withK(10)
    val modified = GKMConfigBuilder(original)
      .setMaxIter(100)
      .build()

    assert(modified.k == 10) // Preserved from original
    assert(modified.maxIter == 100) // Modified
  }

  test("GKMConfigBuilder.buildValidated should validate") {
    val validResult = GKMConfigBuilder()
      .setK(10)
      .buildValidated()

    assert(validResult.isSuccess)

    val invalidResult = GKMConfigBuilder()
      .setK(-1)
      .buildValidated()

    assert(invalidResult.isFailure)
  }

  test("GKMPresets.textClustering should use cosine") {
    val config = GKMPresets.textClustering(k = 20)

    assert(config.k == 20)
    assert(config.kernel == "cosine")
    assert(config.maxIter == 30)
  }

  test("GKMPresets.imageClustering should use Euclidean") {
    val config = GKMPresets.imageClustering(k = 10)

    assert(config.k == 10)
    assert(config.kernel == "squaredEuclidean")
  }

  test("GKMPresets.topicModeling should use KL") {
    val config = GKMPresets.topicModeling(k = 50)

    assert(config.k == 50)
    assert(config.kernel == "kl")
    assert(config.maxIter == 100)
  }

  test("GKMPresets.anomalyDetection should use tight tolerance") {
    val config = GKMPresets.anomalyDetection(k = 3)

    assert(config.k == 3)
    assert(config.tolerance == 1e-6)
  }

  test("GKMPresets.largeDataset should use mini-batch") {
    val config = GKMPresets.largeDataset(k = 10)

    assert(config.k == 10)
    assert(config.miniBatchFraction == 0.1)
    assert(config.maxIter == 100)
  }

  test("GKMPresets.streaming should use very small batch") {
    val config = GKMPresets.streaming(k = 10)

    assert(config.k == 10)
    assert(config.miniBatchFraction == 0.01)
    assert(config.maxIter == 1000)
    assert(config.checkpointInterval == 100)
  }

  test("GKMPresets.robust should use Manhattan") {
    val config = GKMPresets.robust(k = 5)

    assert(config.k == 5)
    assert(config.kernel == "manhattan")
    assert(config.maxIter == 100)
  }

  test("GKMConfig should be immutable") {
    val original = GKMConfig.default
    val modified = original.withK(10)

    assert(original.k == 5) // Original unchanged
    assert(modified.k == 10) // New instance
  }

  test("GKMConfig should be serializable") {
    val config = GKMConfig.default

    val stream = new java.io.ByteArrayOutputStream()
    val oos = new java.io.ObjectOutputStream(stream)
    oos.writeObject(config)
    oos.close()

    val bytes = stream.toByteArray
    assert(bytes.nonEmpty)
  }
}
