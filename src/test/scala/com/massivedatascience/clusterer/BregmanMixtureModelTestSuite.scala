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

import com.massivedatascience.clusterer.TestingUtils._
import com.massivedatascience.linalg.WeightedVector
import org.apache.spark.ml.linalg.Vectors
import org.scalatest._
import funsuite._

class BregmanMixtureModelTestSuite extends AnyFunSuite with LocalClusterSparkContext {

  val pointOps = BregmanPointOps(BregmanPointOps.EUCLIDEAN)

  test("BregmanMixtureModel should estimate basic mixture parameters") {
    val points = sc.parallelize(Seq(
      // Component 1: around (1, 1)
      BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
      BregmanPoint(WeightedVector(Vectors.dense(1.1, 0.9)), 2.02),
      BregmanPoint(WeightedVector(Vectors.dense(0.9, 1.1)), 2.02),
      // Component 2: around (3, 3)
      BregmanPoint(WeightedVector(Vectors.dense(3.0, 3.0)), 18.0),
      BregmanPoint(WeightedVector(Vectors.dense(3.1, 2.9)), 18.58),
      BregmanPoint(WeightedVector(Vectors.dense(2.9, 3.1)), 18.02)
    ))

    points.cache()

    val mixtureModel = BregmanMixtureModel.defaultConfig
    val model = BregmanMixtureModel(mixtureModel)
    val result = model.fit(points, 2, pointOps)

    // Verify basic properties
    assert(result.numComponents == 2)
    assert(result.iterations > 0)
    assert(result.logLikelihood < 0.0) // Log-likelihood should be negative

    // Verify mixing weights
    val mixingWeights = result.mixingWeights
    assert(mixingWeights.length == 2)
    assert(mixingWeights.sum ~= 1.0 absTol 1E-6) // Should sum to 1
    mixingWeights.foreach(w => assert(w > 0.0 && w < 1.0)) // Should be valid probabilities

    // Verify responsibilities
    val responsibilities = result.responsibilities.collect()
    assert(responsibilities.length == 6)
    
    responsibilities.foreach { case (_, posteriors) =>
      assert(posteriors.length == 2)
      assert(posteriors.sum ~= 1.0 absTol 1E-8)
      posteriors.foreach(p => assert(p > 0.0))
    }
  }

  test("BregmanMixtureModel should handle single component case") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
      BregmanPoint(WeightedVector(Vectors.dense(1.1, 1.1)), 2.42),
      BregmanPoint(WeightedVector(Vectors.dense(0.9, 0.9)), 1.62)
    ))

    points.cache()

    val model = BregmanMixtureModel()
    val result = model.fit(points, 1, pointOps)

    assert(result.numComponents == 1)
    assert(result.mixingWeights(0) ~= 1.0 absTol 1E-8)

    val responsibilities = result.responsibilities.collect()
    responsibilities.foreach { case (_, posteriors) =>
      assert(posteriors.length == 1)
      assert(posteriors(0) ~= 1.0 absTol 1E-8)
    }
  }

  test("BregmanMixtureModel should compute effective number of components") {
    val points = sc.parallelize(Seq(
      // Strongly separated clusters
      BregmanPoint(WeightedVector(Vectors.dense(0.0, 0.0)), 0.0),
      BregmanPoint(WeightedVector(Vectors.dense(0.1, 0.1)), 0.02),
      BregmanPoint(WeightedVector(Vectors.dense(10.0, 10.0)), 200.0),
      BregmanPoint(WeightedVector(Vectors.dense(10.1, 10.1)), 204.02)
    ))

    points.cache()

    val model = BregmanMixtureModel()
    val result = model.fit(points, 2, pointOps)

    val effectiveNumComponents = result.effectiveNumComponents

    // Should be close to 2 since we have 2 well-separated clusters
    assert(effectiveNumComponents >= 1.5 && effectiveNumComponents <= 2.0)
    assert(effectiveNumComponents > 1.8, s"Expected effective components > 1.8, got: $effectiveNumComponents")
  }

  test("BregmanMixtureModel should produce valid MAP assignments") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(0.0, 0.0)), 0.0),
      BregmanPoint(WeightedVector(Vectors.dense(0.1, 0.1)), 0.02),
      BregmanPoint(WeightedVector(Vectors.dense(5.0, 5.0)), 50.0),
      BregmanPoint(WeightedVector(Vectors.dense(5.1, 5.1)), 52.02)
    ))

    points.cache()

    val model = BregmanMixtureModel()
    val result = model.fit(points, 2, pointOps)

    val mapAssignments = result.mapAssignments.collect()
    assert(mapAssignments.length == 4)

    mapAssignments.foreach { case (_, assignment) =>
      assert(assignment >= 0 && assignment < 2)
    }

    // Points that are close should likely have the same assignment
    val assignmentMap = mapAssignments.toMap
    val expectedSameCluster = Set(
      (BregmanPoint(WeightedVector(Vectors.dense(0.0, 0.0)), 0.0),
       BregmanPoint(WeightedVector(Vectors.dense(0.1, 0.1)), 0.02)),
      (BregmanPoint(WeightedVector(Vectors.dense(5.0, 5.0)), 50.0),
       BregmanPoint(WeightedVector(Vectors.dense(5.1, 5.1)), 52.02))
    )
  }

  test("BregmanMixtureModel should compute BIC and AIC correctly") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
      BregmanPoint(WeightedVector(Vectors.dense(1.1, 0.9)), 2.02),
      BregmanPoint(WeightedVector(Vectors.dense(3.0, 3.0)), 18.0),
      BregmanPoint(WeightedVector(Vectors.dense(3.1, 2.9)), 18.58)
    ))

    points.cache()
    val numDataPoints = points.count()

    val model = BregmanMixtureModel()
    val result = model.fit(points, 2, pointOps)

    val dimensionality = 2 // 2D points
    val bic = result.bic(numDataPoints, dimensionality)
    val aic = result.aic(dimensionality)

    // Both should be finite and reasonable
    assert(!bic.isInfinity && !bic.isNaN && !aic.isInfinity && !aic.isNaN)
    assert(bic > 0.0 && aic > 0.0) // Typically positive for small datasets
  }

  test("BregmanMixtureModel should handle initialization methods") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
      BregmanPoint(WeightedVector(Vectors.dense(1.1, 0.9)), 2.02),
      BregmanPoint(WeightedVector(Vectors.dense(3.0, 3.0)), 18.0),
      BregmanPoint(WeightedVector(Vectors.dense(3.1, 2.9)), 18.58)
    ))

    points.cache()

    // Test with K-means initialization
    val kmeansConfig = BregmanMixtureConfig(initializationMethod = "kmeans")
    val kmeansModel = BregmanMixtureModel(kmeansConfig)
    val kmeansResult = kmeansModel.fit(points, 2, pointOps)

    // Test with random initialization
    val randomConfig = BregmanMixtureConfig(initializationMethod = "random")
    val randomModel = BregmanMixtureModel(randomConfig)
    val randomResult = randomModel.fit(points, 2, pointOps)

    // Both should produce valid results
    assert(kmeansResult.numComponents == 2)
    assert(randomResult.numComponents == 2)
    assert(kmeansResult.mixingWeights.sum ~= 1.0 absTol 1E-6)
    assert(randomResult.mixingWeights.sum ~= 1.0 absTol 1E-6)
  }

  test("BregmanMixtureModel should handle high precision settings") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
      BregmanPoint(WeightedVector(Vectors.dense(1.01, 0.99)), 2.0002),
      BregmanPoint(WeightedVector(Vectors.dense(3.0, 3.0)), 18.0),
      BregmanPoint(WeightedVector(Vectors.dense(3.01, 2.99)), 18.0002)
    ))

    points.cache()

    val model = BregmanMixtureModel.highPrecision()
    val result = model.fit(points, 2, pointOps)

    assert(result.numComponents == 2)
    assert(result.config.convergenceThreshold == 1e-8)
    assert(result.config.maxIterations == 200)

    val stats = result.getStats
    assert(stats.contains("logLikelihood"))
    assert(stats.contains("effectiveNumComponents"))
  }

  test("BregmanMixtureModel should work with factory methods") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
      BregmanPoint(WeightedVector(Vectors.dense(2.0, 2.0)), 8.0),
      BregmanPoint(WeightedVector(Vectors.dense(3.0, 3.0)), 18.0)
    ))

    points.cache()

    // Test quick method
    val quickResult = BregmanMixtureModel.quick(points, 2, pointOps)
    assert(quickResult.numComponents == 2)
    assert(quickResult.config.maxIterations == 50)

    // Test large datasets method
    val largeDatasetModel = BregmanMixtureModel.forLargeDatasets()
    val largeResult = largeDatasetModel.fit(points, 2, pointOps)
    assert(largeResult.numComponents == 2)
    assert(largeResult.config.maxIterations == 50)
    assert(largeResult.config.regularization == 1e-4)
  }

  test("BregmanMixtureModel should handle convergence properly") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(0.0, 0.0)), 0.0),
      BregmanPoint(WeightedVector(Vectors.dense(10.0, 10.0)), 200.0)
    ))

    points.cache()

    // Test with very few iterations
    val limitedConfig = BregmanMixtureConfig(maxIterations = 3, convergenceThreshold = 1e-12)
    val limitedModel = BregmanMixtureModel(limitedConfig)
    val result = limitedModel.fit(points, 2, pointOps)

    assert(result.iterations <= 3)
    assert(result.iterations > 0)
    // Should either converge or hit max iterations
  }

  test("BregmanMixtureModel should provide comprehensive statistics") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
      BregmanPoint(WeightedVector(Vectors.dense(1.1, 0.9)), 2.02),
      BregmanPoint(WeightedVector(Vectors.dense(3.0, 3.0)), 18.0),
      BregmanPoint(WeightedVector(Vectors.dense(3.1, 2.9)), 18.58)
    ))

    points.cache()

    val model = BregmanMixtureModel()
    val result = model.fit(points, 2, pointOps)

    val stats = result.getStats
    val expectedKeys = Set("logLikelihood", "numComponents", "effectiveNumComponents", 
                          "iterations", "converged", "minMixingWeight", "maxMixingWeight", 
                          "mixingWeightEntropy")

    expectedKeys.foreach { key =>
      assert(stats.contains(key), s"Missing key: $key")
      assert(!stats(key).isInfinity && !stats(key).isNaN, s"Non-finite value for key: $key")
    }

    assert(stats("numComponents") == 2.0)
    assert(stats("converged") == 1.0 || stats("converged") == 0.0)
    assert(stats("minMixingWeight") > 0.0)
    assert(stats("maxMixingWeight") <= 1.0)
  }
}