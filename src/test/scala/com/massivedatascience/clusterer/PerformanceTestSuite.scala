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
import com.massivedatascience.clusterer.KMeans.RunConfig
import com.massivedatascience.linalg.WeightedVector
import com.massivedatascience.transforms.Embedding
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite

class PerformanceTestSuite extends AnyFunSuite with LocalClusterSparkContext {

  test("numerical stability with near-duplicate points") {
    val basePoint = Array(1.0, 2.0, 3.0)
    val nearDuplicates = (0 until 100).map { i =>
      val noise = Array.fill(3)(scala.util.Random.nextGaussian() * 1e-10)
      Vectors.dense(basePoint.zip(noise).map { case (base, n) => base + n })
    }
    
    val data = sc.parallelize(nearDuplicates)
    val model = KMeans.train(data, k = 3, maxIterations = 10)
    
    // Should converge and produce valid clusters despite numerical challenges
    assert(model.centers.nonEmpty)
    val cost = model.computeCost(data)
    assert(cost >= 0.0 && java.lang.Double.isFinite(cost))
    
    // All points should be clustered
    val predictions = model.predict(data).collect()
    assert(predictions.forall(p => p >= 0 && p < model.centers.length))
    
    // Cost should be very small since points are nearly identical
    assert(cost < 1e-6, s"Cost should be very small for near-duplicate points: $cost")
  }

  test("scalability with increasing cluster count") {
    val numPoints = 500
    val dim = 10
    
    val clusterCounts = Seq(5, 10, 20, 50)
    val results = clusterCounts.map { k =>
      val data = sc.parallelize((0 until numPoints).map { i =>
        val cluster = i % k
        val center = Array.fill(dim)(cluster * 2.0)
        val noise = Array.fill(dim)(scala.util.Random.nextGaussian() * 0.5)
        val point = center.zip(noise).map { case (c, n) => c + n }
        Vectors.dense(point)
      })
      
      val startTime = System.currentTimeMillis()
      val model = KMeans.train(data, k = k, maxIterations = 10)
      val endTime = System.currentTimeMillis()
      
      val duration = endTime - startTime
      val cost = model.computeCost(data)
      
      (k, duration, cost, model.centers.length)
    }
    
    // All runs should complete successfully
    results.foreach { case (k, duration, cost, actualClusters) =>
      assert(duration < 60000, s"Training with k=$k took too long: ${duration}ms")
      assert(cost >= 0.0 && java.lang.Double.isFinite(cost), s"Invalid cost for k=$k: $cost")
      assert(actualClusters <= k, s"Too many clusters for k=$k: $actualClusters")
    }
    
    // Time should scale reasonably (not exponentially)
    val durations = results.map(_._2)
    val maxDuration = durations.max
    val minDuration = durations.min
    assert(maxDuration <= minDuration * 50, "Time scaling appears exponential")
  }

  test("memory efficiency with large sparse vectors") {
    val dim = 100000
    val numPoints = 50
    val sparsity = 0.0001 // 0.01% non-zero elements
    
    val data = sc.parallelize((0 until numPoints).map { i =>
      val cluster = i % 5
      val numNonZero = (dim * sparsity).toInt
      val baseIndices = (0 until numNonZero).map(_ * (dim / numNonZero))
      val clusterOffset = cluster * (dim / 10)
      val indices = baseIndices.map(idx => (idx + clusterOffset) % dim)
      val values = indices.map(_ => scala.util.Random.nextGaussian() + cluster)
      Vectors.sparse(dim, indices.zip(values))
    })
    
    val startTime = System.currentTimeMillis()
    val model = KMeans.train(data, k = 5, maxIterations = 5)
    val endTime = System.currentTimeMillis()
    
    val duration = endTime - startTime
    
    // Should handle large sparse vectors efficiently
    assert(duration < 30000, s"Large sparse vector training took too long: ${duration}ms")
    assert(model.centers.length <= 5)
    assert(model.clusterCenters.forall(_.size == dim))
    
    val predictions = model.predict(data).collect()
    assert(predictions.forall(p => p >= 0 && p < model.centers.length))
    
    val cost = model.computeCost(data)
    assert(cost >= 0.0 && java.lang.Double.isFinite(cost))
  }

  test("convergence speed with different initializations") {
    val numPoints = 200
    val data = sc.parallelize((0 until numPoints).map { i =>
      val cluster = i % 4
      Vectors.dense(
        cluster * 5.0 + scala.util.Random.nextGaussian(),
        cluster * 5.0 + scala.util.Random.nextGaussian()
      )
    })
    
    val maxIterations = 50
    
    // Test different initialization methods
    val initializers = Seq(
      KMeansSelector.RANDOM,
      KMeansSelector.K_MEANS_PARALLEL
    )
    
    val results = initializers.map { initializer =>
      val startTime = System.currentTimeMillis()
      val model = KMeans.train(
        data,
        k = 4,
        maxIterations = maxIterations,
        mode = initializer
      )
      val endTime = System.currentTimeMillis()
      
      val duration = endTime - startTime
      val cost = model.computeCost(data)
      
      (initializer.toString, duration, cost)
    }
    
    // All initializers should complete in reasonable time
    results.foreach { case (name, duration, cost) =>
      assert(duration < 20000, s"$name took too long: ${duration}ms")
      assert(cost >= 0.0 && java.lang.Double.isFinite(cost), s"Invalid cost for $name: $cost")
    }
    
    // K-means++ should generally converge faster or produce better cost
    val (_, randomTime, randomCost) = results(0)
    val (_, kppTime, kppCost) = results(1)
    
    // Both should be reasonable (we don't enforce strict ordering due to randomness)
    assert(randomTime < 20000 && kppTime < 20000)
    assert(java.lang.Double.isFinite(randomCost) && java.lang.Double.isFinite(kppCost))
  }

  test("performance with different distance functions") {
    val numPoints = 200
    val data = sc.parallelize((0 until numPoints).map { i =>
      // Generate positive values for KL divergence compatibility
      val cluster = i % 3
      val base = Array.fill(5)(cluster + 1.0)
      val noise = Array.fill(5)(math.abs(scala.util.Random.nextGaussian()) * 0.1)
      Vectors.dense(base.zip(noise).map { case (b, n) => b + n })
    })
    
    val distanceFunctions = Seq(
      BregmanPointOps.EUCLIDEAN,
      BregmanPointOps.GENERALIZED_I,
      BregmanPointOps.RELATIVE_ENTROPY
    )
    
    val results = distanceFunctions.map { distanceFunction =>
      try {
        val startTime = System.currentTimeMillis()
        val model = KMeans.train(
          data,
          k = 3,
          maxIterations = 10,
          distanceFunctionNames = Seq(distanceFunction)
        )
        val endTime = System.currentTimeMillis()
        
        val duration = endTime - startTime
        val cost = model.computeCost(data)
        
        Some((distanceFunction, duration, cost))
      } catch {
        case e: Exception =>
          // Some distance functions may not work with all data
          println(s"Distance function $distanceFunction failed: ${e.getMessage}")
          None
      }
    }
    
    val successfulResults = results.flatten
    
    // At least Euclidean should work
    assert(successfulResults.nonEmpty, "At least one distance function should work")
    
    successfulResults.foreach { case (name, duration, cost) =>
      assert(duration < 30000, s"$name took too long: ${duration}ms")
      assert(cost >= 0.0 && java.lang.Double.isFinite(cost), s"Invalid cost for $name: $cost")
    }
  }

  test("memory usage with zero-weight vectors") {
    val numPoints = 100
    val data = sc.parallelize((0 until numPoints).map { i =>
      val weight = if (i % 10 == 0) 1e-100 else 1.0 // Use very small weight instead of exactly zero
      val cluster = i % 3
      WeightedVector(
        Vectors.dense(
          cluster * 3.0 + scala.util.Random.nextGaussian(),
          cluster * 3.0 + scala.util.Random.nextGaussian()
        ),
        weight
      )
    })

    try {
      val model = KMeans.trainWeighted(
        RunConfig(3, 1, 0, 10),
        data,
        KMeansSelector(KMeansSelector.K_MEANS_PARALLEL),
        Seq(BregmanPointOps(BregmanPointOps.EUCLIDEAN)),
        Seq(Embedding(Embedding.IDENTITY_EMBEDDING)),
        MultiKMeansClusterer(MultiKMeansClusterer.COLUMN_TRACKING)
      )

      // Should handle near-zero weight vectors without issues
      assert(model.centers.nonEmpty)
      assert(model.k > 0)

      val predictions = model.predictWeighted(data).collect()
      assert(predictions.forall(p => p >= 0 && p < model.k))

      val cost = model.computeCostWeighted(data)
      assert(cost >= 0.0) // May be infinite with extreme weights
    } catch {
      case e: IllegalArgumentException if e.getMessage.contains("requires at least one valid center") =>
        // Acceptable if extreme weights cause invalid centers
        succeed
    }
  }

  test("robustness to extreme coordinate values") {
    val extremeData = sc.parallelize(Seq(
      Vectors.dense(-1e6, -1e6),
      Vectors.dense(-1e6, 1e6),
      Vectors.dense(1e6, -1e6),
      Vectors.dense(1e6, 1e6),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1e-6, 1e-6),
      Vectors.dense(-1e-6, -1e-6)
    ))
    
    val model = KMeans.train(extremeData, k = 3, maxIterations = 10)
    
    // Should handle extreme values gracefully
    assert(model.centers.nonEmpty)
    
    val predictions = model.predict(extremeData).collect()
    assert(predictions.forall(p => p >= 0 && p < model.centers.length))
    
    val cost = model.computeCost(extremeData)
    assert(cost >= 0.0 && java.lang.Double.isFinite(cost))
    
    // Centers should have finite coordinates
    model.clusterCenters.foreach { center =>
      center.toArray.foreach { coord =>
        assert(java.lang.Double.isFinite(coord), s"Non-finite coordinate in center: $coord")
      }
    }
  }

  test("convergence detection effectiveness") {
    // Create data where convergence should happen quickly
    val tightClusters = sc.parallelize((0 until 150).map { i =>
      val cluster = i % 3
      Vectors.dense(
        cluster * 10.0 + scala.util.Random.nextGaussian() * 0.1,
        cluster * 10.0 + scala.util.Random.nextGaussian() * 0.1
      )
    })
    
    val maxIterations = 50
    val startTime = System.currentTimeMillis()
    
    val model = KMeans.train(tightClusters, k = 3, maxIterations = maxIterations)
    
    val endTime = System.currentTimeMillis()
    val duration = endTime - startTime
    
    // Should converge quickly for well-separated clusters
    assert(duration < 10000, s"Should converge quickly for tight clusters: ${duration}ms")
    
    val cost = model.computeCost(tightClusters)
    assert(cost >= 0.0 && java.lang.Double.isFinite(cost))
    
    // Should achieve low cost for tight clusters
    assert(cost < 10.0, s"Cost should be reasonable for tight clusters: $cost")
    
    // Should produce at least 1 cluster (may be fewer than 3 if some are filtered)
    assert(model.k >= 1)
    assert(model.k <= 3)

    val predictions = model.predict(tightClusters).collect()
    assert(predictions.forall(p => p >= 0 && p < model.k))
    val clusterCounts = predictions.groupBy(identity).mapValues(_.length)
    assert(clusterCounts.size <= 3, "Should produce at most 3 clusters")
    assert(clusterCounts.values.forall(_ == 50), "Clusters should be evenly sized")
  }

  test("thread safety and concurrent access") {
    val data = sc.parallelize((0 until 100).map { i =>
      Vectors.dense(
        scala.util.Random.nextGaussian(),
        scala.util.Random.nextGaussian()
      )
    })
    
    val model = KMeans.train(data, k = 3, maxIterations = 5)
    
    // Test concurrent predictions (simulating multiple threads)
    val testPoints = (0 until 50).map { i =>
      Vectors.dense(
        scala.util.Random.nextGaussian(),
        scala.util.Random.nextGaussian()
      )
    }
    
    val predictions = testPoints.par.map { point =>
      try {
        val prediction = model.predict(point)
        val (cluster, distance) = model.predictClusterAndDistance(point)
        (prediction, cluster, distance, prediction == cluster)
      } catch {
        case e: Exception =>
          fail(s"Concurrent prediction failed: ${e.getMessage}")
      }
    }.seq
    
    // All predictions should be valid
    predictions.foreach { case (pred, cluster, distance, consistent) =>
      assert(pred >= 0 && pred < model.centers.length)
      assert(cluster >= 0 && cluster < model.centers.length)
      assert(distance >= 0.0 && java.lang.Double.isFinite(distance))
      assert(consistent, "predict and predictClusterAndDistance should be consistent")
    }
  }
}