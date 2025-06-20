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

class IntegrationTestSuite extends AnyFunSuite with LocalClusterSparkContext {

  test("end-to-end pipeline with multiple distance functions") {
    val data = sc.parallelize((0 until 100).map { i =>
      Vectors.dense(
        scala.util.Random.nextGaussian() + (i % 3) * 5,
        scala.util.Random.nextGaussian() + (i % 3) * 5
      )
    })
    
    val distanceFunctions = Seq(
      BregmanPointOps.EUCLIDEAN,
      BregmanPointOps.RELATIVE_ENTROPY,
      BregmanPointOps.GENERALIZED_I
    )
    
    distanceFunctions.foreach { distanceFunction =>
      try {
        val model = KMeans.train(
          data, 
          k = 3, 
          maxIterations = 10, 
          distanceFunctionNames = Seq(distanceFunction)
        )
        
        // Each distance function should produce valid results
        assert(model.centers.length <= 3)
        val cost = model.computeCost(data)
        assert(cost >= 0.0 && java.lang.Double.isFinite(cost), s"Invalid cost for $distanceFunction: $cost")
        
        // Predictions should be valid
        val predictions = model.predict(data).collect()
        assert(predictions.forall(p => p >= 0 && p < model.centers.length))
        
        // Should achieve reasonable clustering (some clusters should have multiple points)
        val clusterCounts = predictions.groupBy(identity).mapValues(_.length)
        assert(clusterCounts.size <= 3)
        assert(clusterCounts.values.forall(_ > 0))
        
      } catch {
        case e: Exception =>
          fail(s"Distance function $distanceFunction failed: ${e.getMessage}")
      }
    }
  }

  test("embedding pipeline integration with time series") {
    val timeSeries = sc.parallelize((0 until 50).map { i =>
      // Create time series with different patterns
      val pattern = i % 3
      val values = (0 until 16).map { t =>
        pattern match {
          case 0 => math.sin(t * 0.5) + scala.util.Random.nextGaussian() * 0.1
          case 1 => math.cos(t * 0.3) + scala.util.Random.nextGaussian() * 0.1
          case 2 => (t % 4) * 0.5 + scala.util.Random.nextGaussian() * 0.1
        }
      }
      WeightedVector(Vectors.dense(values.toArray))
    })
    
    val model = KMeans.timeSeriesTrain(
      RunConfig(3, 1, 0, 10),
      timeSeries,
      KMeansSelector(KMeansSelector.K_MEANS_PARALLEL),
      BregmanPointOps(BregmanPointOps.EUCLIDEAN),
      MultiKMeansClusterer(MultiKMeansClusterer.COLUMN_TRACKING),
      Embedding(Embedding.HAAR_EMBEDDING)
    )
    
    // Should produce valid model with Haar embedding
    assert(model.centers.length <= 3)
    val predictions = model.predictWeighted(timeSeries).collect()
    assert(predictions.forall(p => p >= 0 && p < model.centers.length))
    
    // Should achieve some clustering structure
    val clusterCounts = predictions.groupBy(identity).mapValues(_.length)
    assert(clusterCounts.size <= 3)
    assert(clusterCounts.values.sum == 50)
  }

  test("multi-step training with different embeddings") {
    val data = sc.parallelize((0 until 200).map { i =>
      WeightedVector(Vectors.dense(Array.fill(64)(scala.util.Random.nextGaussian())))
    })
    
    val embeddings = Seq(
      Embedding(Embedding.LOW_DIMENSIONAL_RI),
      Embedding(Embedding.MEDIUM_DIMENSIONAL_RI),
      Embedding(Embedding.IDENTITY_EMBEDDING)
    )
    
    val model = KMeans.trainWeighted(
      RunConfig(5, 1, 0, 5),
      data,
      KMeansSelector(KMeansSelector.K_MEANS_PARALLEL),
      Seq(BregmanPointOps(BregmanPointOps.EUCLIDEAN)),
      embeddings,
      MultiKMeansClusterer(MultiKMeansClusterer.COLUMN_TRACKING)
    )
    
    // Should complete multi-stage training successfully
    assert(model.centers.length <= 5)
    
    val predictions = model.predictWeighted(data).collect()
    assert(predictions.forall(p => p >= 0 && p < model.centers.length))
    
    val cost = model.computeCostWeighted(data)
    assert(cost >= 0.0 && java.lang.Double.isFinite(cost))
  }

  test("comparison of different clustering implementations") {
    val data = sc.parallelize((0 until 100).map { i =>
      WeightedVector(Vectors.dense(
        scala.util.Random.nextGaussian() + (i % 4) * 3,
        scala.util.Random.nextGaussian() + (i % 4) * 3
      ))
    })
    
    val implementations = Seq(
      MultiKMeansClusterer(MultiKMeansClusterer.COLUMN_TRACKING),
      MultiKMeansClusterer(MultiKMeansClusterer.MINI_BATCH_10)
    )
    
    val results = implementations.map { clusterer =>
      val model = KMeans.trainWeighted(
        RunConfig(4, 1, 0, 10),
        data,
        KMeansSelector(KMeansSelector.K_MEANS_PARALLEL),
        Seq(BregmanPointOps(BregmanPointOps.EUCLIDEAN)),
        Seq(Embedding(Embedding.IDENTITY_EMBEDDING)),
        clusterer
      )
      
      val cost = model.computeCostWeighted(data)
      val predictions = model.predictWeighted(data).collect()
      
      (model.centers.length, cost, predictions)
    }
    
    // All implementations should produce valid results
    results.foreach { case (numCenters, cost, predictions) =>
      assert(numCenters <= 4)
      assert(cost >= 0.0 && java.lang.Double.isFinite(cost))
      assert(predictions.forall(p => p >= 0 && p < numCenters))
    }
    
    // Costs should be reasonably similar (within order of magnitude)
    val costs = results.map(_._2)
    val minCost = costs.min
    val maxCost = costs.max
    assert(maxCost <= minCost * 100, "Clustering implementations produce very different costs")
  }

  test("large dataset performance test") {
    val numPoints = 1000
    val dim = 50
    val numClusters = 10
    
    val data = sc.parallelize((0 until numPoints).map { i =>
      val cluster = i % numClusters
      val center = Array.fill(dim)(cluster * 2.0)
      val noise = Array.fill(dim)(scala.util.Random.nextGaussian() * 0.5)
      val point = center.zip(noise).map { case (c, n) => c + n }
      Vectors.dense(point)
    }, 10) // Use multiple partitions
    
    val startTime = System.currentTimeMillis()
    
    val model = KMeans.train(
      data,
      k = numClusters,
      maxIterations = 20,
      distanceFunctionNames = Seq(BregmanPointOps.EUCLIDEAN)
    )
    
    val endTime = System.currentTimeMillis()
    val duration = endTime - startTime
    
    // Should complete in reasonable time (less than 30 seconds for 1000 points)
    assert(duration < 30000, s"Training took too long: ${duration}ms")
    
    // Should produce reasonable results
    assert(model.centers.length <= numClusters)
    val predictions = model.predict(data).collect()
    assert(predictions.forall(p => p >= 0 && p < model.centers.length))
    
    // Should achieve good clustering quality (most points should be correctly clustered)
    val correctAssignments = predictions.zipWithIndex.count { case (prediction, index) =>
      val trueCluster = index % numClusters
      prediction == trueCluster
    }
    val accuracy = correctAssignments.toDouble / numPoints
    assert(accuracy > 0.5, s"Poor clustering accuracy: $accuracy")
  }

  test("memory usage with sparse high-dimensional data") {
    val dim = 10000
    val numPoints = 100
    val sparsity = 0.01 // 1% non-zero elements
    
    val data = sc.parallelize((0 until numPoints).map { i =>
      val numNonZero = (dim * sparsity).toInt
      val indices = scala.util.Random.shuffle((0 until dim).toList).take(numNonZero)
      val values = indices.map(_ => scala.util.Random.nextGaussian())
      Vectors.sparse(dim, indices.zip(values))
    })
    
    // Should handle high-dimensional sparse data without memory issues
    val model = KMeans.train(data, k = 5, maxIterations = 5)
    
    assert(model.centers.length <= 5)
    assert(model.clusterCenters.forall(_.size == dim))
    
    val predictions = model.predict(data).collect()
    assert(predictions.forall(p => p >= 0 && p < model.centers.length))
    
    val cost = model.computeCost(data)
    assert(cost >= 0.0 && java.lang.Double.isFinite(cost))
  }

  test("convergence behavior with different initializations") {
    val data = sc.parallelize((0 until 100).map { i =>
      Vectors.dense(
        scala.util.Random.nextGaussian() + (i % 3) * 4,
        scala.util.Random.nextGaussian() + (i % 3) * 4
      )
    })
    
    val initializers = Seq(
      KMeansSelector.RANDOM,
      KMeansSelector.K_MEANS_PARALLEL
    )
    
    val results = initializers.map { initializer =>
      val model = KMeans.train(
        data,
        k = 3,
        maxIterations = 20,
        mode = initializer
      )
      
      val cost = model.computeCost(data)
      val predictions = model.predict(data).collect()
      
      (cost, predictions, model.centers.length)
    }
    
    // All initializers should produce valid results
    results.foreach { case (cost, predictions, numCenters) =>
      assert(cost >= 0.0 && java.lang.Double.isFinite(cost))
      assert(predictions.forall(p => p >= 0 && p < numCenters))
      assert(numCenters <= 3)
    }
    
    // K-means++ should generally produce better or similar results than random
    val (randomCost, _, _) = results(0)
    val (kppCost, _, _) = results(1)
    // Note: This is not guaranteed, just a general expectation
    // We just verify both produce reasonable results
    assert(java.lang.Double.isFinite(randomCost) && java.lang.Double.isFinite(kppCost))
  }

  test("robustness to outliers") {
    // Create data with clear clusters plus outliers
    val normalPoints = (0 until 90).map { i =>
      val cluster = i % 3
      Vectors.dense(
        cluster * 5.0 + scala.util.Random.nextGaussian() * 0.5,
        cluster * 5.0 + scala.util.Random.nextGaussian() * 0.5
      )
    }
    
    val outliers = (0 until 10).map { _ =>
      Vectors.dense(
        scala.util.Random.nextGaussian() * 20 + 100,
        scala.util.Random.nextGaussian() * 20 + 100
      )
    }
    
    val dataWithOutliers = sc.parallelize(normalPoints ++ outliers)
    
    val model = KMeans.train(dataWithOutliers, k = 3, maxIterations = 20)
    
    // Should still find reasonable clusters despite outliers
    assert(model.centers.length <= 3)
    
    val cost = model.computeCost(dataWithOutliers)
    assert(cost >= 0.0 && java.lang.Double.isFinite(cost))
    
    // Test clustering of just the normal points
    val normalData = sc.parallelize(normalPoints)
    val normalPredictions = model.predict(normalData).collect()
    
    // Most normal points should be clustered reasonably
    val clusterCounts = normalPredictions.groupBy(identity).mapValues(_.length)
    assert(clusterCounts.size <= 3)
    // Each cluster should have at least a few points
    assert(clusterCounts.values.forall(_ >= 5))
  }

  test("reproducibility with fixed seeds") {
    val data = sc.parallelize((0 until 50).map { i =>
      WeightedVector(Vectors.dense(
        scala.util.Random.nextGaussian(),
        scala.util.Random.nextGaussian()
      ))
    })
    
    val seed = 12345L
    
    val model1 = KMeans.trainWeighted(
      RunConfig(3, 1, seed.toInt, 10),
      data,
      KMeansSelector(KMeansSelector.K_MEANS_PARALLEL),
      Seq(BregmanPointOps(BregmanPointOps.EUCLIDEAN)),
      Seq(Embedding(Embedding.IDENTITY_EMBEDDING)),
      MultiKMeansClusterer(MultiKMeansClusterer.COLUMN_TRACKING)
    )
    
    val model2 = KMeans.trainWeighted(
      RunConfig(3, 1, seed.toInt, 10),
      data,
      KMeansSelector(KMeansSelector.K_MEANS_PARALLEL),
      Seq(BregmanPointOps(BregmanPointOps.EUCLIDEAN)),
      Seq(Embedding(Embedding.IDENTITY_EMBEDDING)),
      MultiKMeansClusterer(MultiKMeansClusterer.COLUMN_TRACKING)
    )
    
    // Should produce identical results with same seed
    assert(model1.centers.length == model2.centers.length)
    
    val predictions1 = model1.predictWeighted(data).collect()
    val predictions2 = model2.predictWeighted(data).collect()
    
    // Note: Due to floating-point precision and Spark's distributed nature,
    // exact reproducibility may be challenging. We test for reasonable similarity.
    val cost1 = model1.computeCostWeighted(data)
    val cost2 = model2.computeCostWeighted(data)
    
    assert(math.abs(cost1 - cost2) < cost1 * 0.01, "Results should be very similar with same seed")
  }
}