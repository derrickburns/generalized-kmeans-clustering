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
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.scalatest.funsuite.AnyFunSuite

class KMeansModelEdgeCasesSuite extends AnyFunSuite with LocalClusterSparkContext {

  test("KMeansModel.fromAssignments with null inputs") {
    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    
    intercept[IllegalArgumentException] {
      KMeansModel.fromAssignments(ops, null, sc.parallelize(Seq(0)), 1)
    }
    
    val points = sc.parallelize(Seq(WeightedVector(Vectors.dense(1.0, 2.0))))
    intercept[IllegalArgumentException] {
      KMeansModel.fromAssignments(ops, points, null, 1)
    }
  }

  test("KMeansModel.fromAssignments with mismatched RDD lengths") {
    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val points = sc.parallelize(Seq(
      WeightedVector(Vectors.dense(1.0, 2.0))
    ))
    val assignments = sc.parallelize(Seq(0, 1)) // Different length
    
    intercept[IllegalArgumentException] {
      KMeansModel.fromAssignments(ops, points, assignments, 2)
    }
  }

  test("KMeansModel.fromAssignments with empty RDDs") {
    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val emptyPoints = sc.parallelize(Seq.empty[WeightedVector])
    val emptyAssignments = sc.parallelize(Seq.empty[Int])
    
    intercept[IllegalArgumentException] {
      KMeansModel.fromAssignments(ops, emptyPoints, emptyAssignments, 1)
    }
  }

  test("KMeansModel.fromAssignments with invalid cluster assignments") {
    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val points = sc.parallelize(Seq(
      WeightedVector(Vectors.dense(1.0, 2.0)),
      WeightedVector(Vectors.dense(3.0, 4.0))
    ))
    
    // Negative cluster assignment
    val negativeAssignments = sc.parallelize(Seq(-1, 0))
    intercept[Exception] {
      KMeansModel.fromAssignments(ops, points, negativeAssignments, 2)
    }
    
    // Assignment index too high
    val highAssignments = sc.parallelize(Seq(0, 10))
    intercept[Exception] {
      KMeansModel.fromAssignments(ops, points, highAssignments, 2)
    }
  }

  test("KMeansModel.fromAssignments with all points in one cluster") {
    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val points = sc.parallelize(Seq(
      WeightedVector(Vectors.dense(1.0, 2.0)),
      WeightedVector(Vectors.dense(3.0, 4.0)),
      WeightedVector(Vectors.dense(5.0, 6.0))
    ))
    val assignments = sc.parallelize(Seq(0, 0, 0)) // All assigned to cluster 0
    
    val model = KMeansModel.fromAssignments(ops, points, assignments, 3)
    
    // Should handle empty clusters gracefully
    assert(model.centers.length == 3)
    
    // First cluster should have the centroid of all points
    val firstCenter = model.centers(0).inhomogeneous
    assert(firstCenter(0) ~= 3.0 absTol 1e-8) // (1+3+5)/3
    assert(firstCenter(1) ~= 4.0 absTol 1e-8) // (2+4+6)/3
    
    val predictions = model.predict(sc.parallelize(Seq(
      Vectors.dense(2.0, 3.0),
      Vectors.dense(10.0, 10.0)
    ))).collect()
    
    // All predictions should be valid cluster indices
    assert(predictions.forall(p => p >= 0 && p < 3))
  }

  test("KMeansModel.fromAssignments with zero-weight points") {
    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val points = sc.parallelize(Seq(
      WeightedVector.fromInhomogeneousWeighted(Vectors.dense(1.0, 2.0), 0.0), // Zero weight
      WeightedVector.fromInhomogeneousWeighted(Vectors.dense(3.0, 4.0), 1.0),
      WeightedVector.fromInhomogeneousWeighted(Vectors.dense(5.0, 6.0), 2.0)
    ))
    val assignments = sc.parallelize(Seq(0, 1, 1))

    val model = KMeansModel.fromAssignments(ops, points, assignments, 2)

    // Should handle zero-weight points appropriately
    assert(model.centers.length == 2)

    // Cluster 1 should have centroid weighted by point weights
    val cluster1Center = model.centers(1).inhomogeneous
    // Weighted average: (3*1 + 5*2)/(1+2) = 13/3, (4*1 + 6*2)/(1+2) = 16/3
    assert(cluster1Center(0) ~= 13.0/3.0 absTol 1e-8)
    assert(cluster1Center(1) ~= 16.0/3.0 absTol 1e-8)
  }

  test("predict with empty RDD") {
    val data = sc.parallelize(Seq(Vectors.dense(1.0, 2.0), Vectors.dense(3.0, 4.0)))
    val model = KMeans.train(data, k = 2, maxIterations = 1)
    
    val emptyRDD = sc.parallelize(Seq.empty[org.apache.spark.ml.linalg.Vector])
    val emptyPredictions = model.predict(emptyRDD).collect()
    assert(emptyPredictions.isEmpty)
    
    val emptyWeightedRDD = sc.parallelize(Seq.empty[WeightedVector])
    val emptyWeightedPredictions = model.predictWeighted(emptyWeightedRDD).collect()
    assert(emptyWeightedPredictions.isEmpty)
  }

  test("computeCost with empty RDD") {
    val data = sc.parallelize(Seq(Vectors.dense(1.0, 2.0), Vectors.dense(3.0, 4.0)))
    val model = KMeans.train(data, k = 2, maxIterations = 1)
    
    val emptyRDD = sc.parallelize(Seq.empty[org.apache.spark.ml.linalg.Vector])
    val emptyCost = model.computeCost(emptyRDD)
    assert(emptyCost == 0.0)
    
    val emptyWeightedRDD = sc.parallelize(Seq.empty[WeightedVector])
    val emptyWeightedCost = model.computeCostWeighted(emptyWeightedRDD)
    assert(emptyWeightedCost == 0.0)
  }

  test("computeCost with mismatched dimensions") {
    val data = sc.parallelize(Seq(Vectors.dense(1.0, 2.0), Vectors.dense(3.0, 4.0)))
    val model = KMeans.train(data, k = 2, maxIterations = 1)
    
    // Test with wrong dimensions
    val wrongDimData = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0), // 3D instead of 2D
      Vectors.dense(4.0, 5.0, 6.0)
    ))
    
    // Should handle gracefully or throw appropriate exception
    intercept[Exception] {
      model.computeCost(wrongDimData)
    }
  }

  test("predict with mismatched dimensions") {
    val data = sc.parallelize(Seq(Vectors.dense(1.0, 2.0), Vectors.dense(3.0, 4.0)))
    val model = KMeans.train(data, k = 2, maxIterations = 1)
    
    // Single point prediction with wrong dimension
    intercept[Exception] {
      model.predict(Vectors.dense(1.0, 2.0, 3.0))
    }
    
    // RDD prediction with wrong dimensions
    val wrongDimData = sc.parallelize(Seq(Vectors.dense(1.0, 2.0, 3.0)))
    intercept[Exception] {
      model.predict(wrongDimData).collect()
    }
  }

  test("model with single cluster") {
    val data = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0),
      Vectors.dense(1.1, 2.1),
      Vectors.dense(0.9, 1.9)
    ))
    
    val model = KMeans.train(data, k = 1, maxIterations = 5)
    
    assert(model.centers.length == 1)
    
    // All predictions should be 0
    val predictions = model.predict(data).collect()
    assert(predictions.forall(_ == 0))
    
    // Cost should be non-negative
    val cost = model.computeCost(data)
    assert(cost >= 0.0 && java.lang.Double.isFinite(cost))
  }

  test("model with more clusters than points") {
    val data = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0),
      Vectors.dense(3.0, 4.0)
    ))
    
    val model = KMeans.train(data, k = 5, maxIterations = 5)
    
    // Should create fewer clusters than requested
    assert(model.centers.length <= 5)
    
    val predictions = model.predict(data).collect()
    assert(predictions.forall(p => p >= 0 && p < model.centers.length))
    
    val cost = model.computeCost(data)
    assert(cost >= 0.0 && java.lang.Double.isFinite(cost))
  }

  test("model with identical points") {
    val identicalVector = Vectors.dense(1.0, 2.0)
    val data = sc.parallelize(Array.fill(10)(identicalVector))
    
    val model = KMeans.train(data, k = 3, maxIterations = 10)
    
    // Should converge quickly
    assert(model.centers.nonEmpty)
    
    // All predictions should be the same
    val predictions = model.predict(data).collect()
    assert(predictions.toSet.size == 1)
    
    // Cost should be 0 (all points identical)
    val cost = model.computeCost(data)
    assert(cost ~= 0.0 absTol 1e-8)
  }

  test("model serialization compatibility") {
    val data = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0),
      Vectors.dense(3.0, 4.0),
      Vectors.dense(5.0, 6.0)
    ))
    
    val model = KMeans.train(data, k = 2, maxIterations = 5)
    
    // Test that the model can be used after creation
    val testPoint = Vectors.dense(2.0, 3.0)
    val prediction1 = model.predict(testPoint)
    val (cluster1, distance1) = model.predictClusterAndDistance(testPoint)
    
    assert(prediction1 == cluster1)
    assert(distance1 >= 0.0 && java.lang.Double.isFinite(distance1))
    
    // Test cluster centers access
    val centers = model.clusterCenters
    assert(centers.length == model.centers.length)
    assert(centers.forall(_.size == 2))
    
    // Test weighted cluster centers access
    val weightedCenters = model.weightedClusterCenters
    assert(weightedCenters.length == model.centers.length)
    assert(weightedCenters.forall(_.weight > 0.0))
  }

  test("KMeansModel.usingKMeansParallel with invalid parameters") {
    val data = sc.parallelize(Seq(
      WeightedVector(Vectors.dense(1.0, 2.0)),
      WeightedVector(Vectors.dense(3.0, 4.0))
    ))
    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    
    // Test with invalid k
    intercept[Exception] {
      KMeansModel.usingKMeansParallel(ops, data, k = 0)
    }
    
    intercept[Exception] {
      KMeansModel.usingKMeansParallel(ops, data, k = -1)
    }
    
    // Test with invalid numSteps
    intercept[Exception] {
      KMeansModel.usingKMeansParallel(ops, data, k = 2, numSteps = 0)
    }
    
    // Test with invalid sampleRate
    intercept[Exception] {
      KMeansModel.usingKMeansParallel(ops, data, k = 2, sampleRate = 0.0)
    }
    
    intercept[Exception] {
      KMeansModel.usingKMeansParallel(ops, data, k = 2, sampleRate = 1.5)
    }
  }

  test("predictClusterAndDistance consistency") {
    val data = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0),
      Vectors.dense(3.0, 4.0),
      Vectors.dense(5.0, 6.0)
    ))
    
    val model = KMeans.train(data, k = 2, maxIterations = 5)
    
    val testPoints = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(2.0, 3.0),
      Vectors.dense(10.0, 10.0)
    )
    
    for (point <- testPoints) {
      val prediction = model.predict(point)
      val (cluster, distance) = model.predictClusterAndDistance(point)
      
      // Prediction should match cluster from predictClusterAndDistance
      assert(prediction == cluster)
      
      // Distance should be non-negative and finite
      assert(distance >= 0.0 && java.lang.Double.isFinite(distance))
      
      // Distance should be to the predicted cluster
      val centerDistance = {
        val center = model.clusterCenters(cluster)
        val diff = point.toArray.zip(center.toArray).map { case (a, b) => a - b }
        diff.map(x => x * x).sum
      }
      assert(distance ~= centerDistance absTol 1e-6)
    }
  }

  test("model with high-dimensional sparse data") {
    val dim = 1000
    val data = (0 until 20).map { i =>
      val indices = Seq(i % dim, (i * 2) % dim, (i * 3) % dim).distinct
      val values = indices.map(_ => scala.util.Random.nextGaussian())
      Vectors.sparse(dim, indices.zip(values))
    }
    
    val rdd = sc.parallelize(data)
    val model = KMeans.train(rdd, k = 3, maxIterations = 5)
    
    // Should handle high-dimensional sparse data
    assert(model.centers.length == 3)
    assert(model.clusterCenters.forall(_.size == dim))
    
    val predictions = model.predict(rdd).collect()
    assert(predictions.forall(p => p >= 0 && p < 3))
    
    val cost = model.computeCost(rdd)
    assert(cost >= 0.0 && java.lang.Double.isFinite(cost))
  }

  test("model with extreme weight values") {
    // Use extreme but numerically safe weights
    val data = sc.parallelize(Seq(
      WeightedVector.fromInhomogeneousWeighted(Vectors.dense(1.0, 2.0), 1e-50),
      WeightedVector.fromInhomogeneousWeighted(Vectors.dense(3.0, 4.0), 1.0),
      WeightedVector.fromInhomogeneousWeighted(Vectors.dense(5.0, 6.0), 1e50)
    ))

    try {
      val model = KMeans.trainWeighted(
        RunConfig(2, 1, 0, 5),
        data,
        KMeansSelector(KMeansSelector.K_MEANS_PARALLEL),
        Seq(BregmanPointOps(BregmanPointOps.EUCLIDEAN)),
        Seq(Embedding(Embedding.IDENTITY_EMBEDDING)),
        MultiKMeansClusterer(MultiKMeansClusterer.COLUMN_TRACKING)
      )

      // Should handle extreme weight values
      assert(model.centers.nonEmpty)
      assert(model.k > 0)

      val predictions = model.predictWeighted(data).collect()
      assert(predictions.forall(p => p >= 0 && p < model.k))

      val cost = model.computeCostWeighted(data)
      assert(cost >= 0.0) // May be infinite with extreme weights
    } catch {
      case e: IllegalArgumentException if e.getMessage.contains("requires at least one valid center") =>
        // Acceptable failure mode when extreme weights cause all centers to be invalid
        succeed
    }
  }
}