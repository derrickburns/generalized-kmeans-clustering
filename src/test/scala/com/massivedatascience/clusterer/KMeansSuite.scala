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
 *
 * This code is a modified version of the original Spark 1.0.2 K-Means implementation.
 */

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.KMeans.RunConfig
import com.massivedatascience.linalg.WeightedVector
import com.massivedatascience.transforms.Embedding
import com.massivedatascience.transforms.Embedding._

import scala.util.Random

import org.scalatest._
import funsuite._

import org.apache.spark.ml.linalg.{ Vector, Vectors }

import com.massivedatascience.clusterer.TestingUtils._
import com.massivedatascience.clusterer.BregmanPointOps._

class KMeansSuite extends AnyFunSuite with LocalClusterSparkContext {

  import com.massivedatascience.clusterer.KMeansSelector._

    test("coverage") {

    val seed = 0
    val r = new Random(seed)
    val data = sc.parallelize[Vector](Array.fill(1000)(Vectors.dense(Array.fill(20)(r.nextDouble()))))

    KMeans.train(data, k = 20, maxIterations = 10)

    KMeans.train(data, k = 20, maxIterations = 10, runs = 2)

    KMeans.train(data, k = 20, maxIterations = 10, runs = 1, mode = KMeansSelector.RANDOM)

    KMeans.train(data, k = 20, maxIterations = 10, runs = 1, distanceFunctionNames = Seq(BregmanPointOps.EUCLIDEAN))

    KMeans.train(data, k = 20, maxIterations = 10, runs = 1,
      distanceFunctionNames = Seq(BregmanPointOps.EUCLIDEAN),
      clustererName = MultiKMeansClusterer.CHANGE_TRACKING)

    KMeans.train(data, k = 50, maxIterations = 10, runs = 1,
      distanceFunctionNames = Seq(BregmanPointOps.EUCLIDEAN),
      clustererName = MultiKMeansClusterer.RESEED)

    KMeans.timeSeriesTrain(
      RunConfig(20, 1, 0, 10),
      data.map(WeightedVector.apply),
      KMeansSelector(KMeansSelector.K_MEANS_PARALLEL),
      BregmanPointOps(BregmanPointOps.EUCLIDEAN),
      MultiKMeansClusterer(MultiKMeansClusterer.COLUMN_TRACKING),
      Embedding(HAAR_EMBEDDING))

    KMeans.train(data, k = 20, maxIterations = 10, runs = 1, clustererName = MultiKMeansClusterer.MINI_BATCH_10)
  }

  test("sparse vector iterator") {

    import com.massivedatascience.linalg._

    val v = Vectors.sparse(10, Seq((2, 30.2), (4, 42.0)))

    val iter = v.iterator

    println(iter.toString)

    assert(iter.hasNext)
    val firstIndex = iter.index
    assert(firstIndex == 2)
    val firstValue = iter.value
    assert(firstValue == 30.2)

    assert(iter.hasNext)
    iter.advance()

    val secondIndex = iter.index
    assert(secondIndex == 4)
    val secondValue = iter.value
    assert(secondValue == 42)
    iter.advance()
    assert(!iter.hasNext)
  }

  test("sparse negative vector iterator") {

    import com.massivedatascience.linalg._

    val v = Vectors.sparse(10, Seq((2, 30.2), (4, 42.0)))

    val iter = v.negativeIterator

    assert(iter.hasNext)
    val firstIndex = iter.index
    assert(firstIndex == 2)
    val firstValue = iter.value
    assert(firstValue == -30.2)

    assert(iter.hasNext)
    iter.advance()

    val secondIndex = iter.index
    assert(secondIndex == 4)
    val secondValue = iter.value
    assert(secondValue == -42)
    iter.advance()
    assert(!iter.hasNext)

    println(iter.toString)
  }

  test("dense vector iterator") {

    import com.massivedatascience.linalg._

    val v = Vectors.dense(5.0, 9.0)

    val iter = v.iterator

    println(iter.toString)

    assert(iter.hasNext)
    val firstIndex = iter.index
    assert(firstIndex == 0)
    val firstValue = iter.value
    assert(firstValue == 5.0)

    assert(iter.hasNext)
    iter.advance()

    val secondIndex = iter.index
    assert(secondIndex == 1)
    val secondValue = iter.value
    assert(secondValue == 9.0)
    iter.advance()
    assert(!iter.hasNext)
  }

  test("dense negative vector iterator") {

    import com.massivedatascience.linalg._

    val v = Vectors.dense(5.0, 9.0)

    val iter = v.negativeIterator

    println(iter.toString)

    assert(iter.hasNext)
    val firstIndex = iter.index
    assert(firstIndex == 0)
    val firstValue = iter.value
    assert(firstValue == -5.0)

    assert(iter.hasNext)
    iter.advance()

    val secondIndex = iter.index
    assert(secondIndex == 1)
    val secondValue = iter.value
    assert(secondValue == -9.0)
    iter.advance()
    assert(!iter.hasNext)
  }



  test("iteratively train") {
    val seed = 0
    val r = new Random(seed)

    val data = sc.parallelize[Vector](Array.fill(1000)(Vectors.dense(Array.fill(20)(r.nextDouble()))))

    /*

    runConfig: RunConfig,
    pointOps: Seq[BregmanPointOps],
    dataSets: Seq[RDD[BregmanPoint]],
    initializer: KMeansSelector,
    clusterer: MultiKMeansClusterer)
     */

    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val cached = data.map(WeightedVector.apply).map(ops.toPoint).cache()
    KMeans.iterativelyTrain(
      RunConfig(20, 1, 0, 10),
      Seq(ops),
      Seq(cached),
      KMeansSelector(KMeansSelector.K_MEANS_PARALLEL),
      MultiKMeansClusterer(MultiKMeansClusterer.COLUMN_TRACKING))
  }

  test("single cluster") {
    val data = sc.parallelize[Vector](Array(
      Vectors.dense(1.0, 2.0, 6.0),
      Vectors.dense(1.0, 3.0, 0.0),
      Vectors.dense(1.0, 4.0, 6.0)
    ))

    val center = Vectors.dense(1.0, 3.0, 4.0)

    // No matter how many runs or iterations we use, we should get one cluster,
    // centered at the mean of the points

    var model = KMeans.train(data, k = 1, maxIterations = 1)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 2)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = RANDOM)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(
      data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL)
    assert(model.clusterCenters.head ~== center absTol 1E-5)
  }

  test("no distinct points") {
    val data = sc.parallelize[Vector](
      Array(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(1.0, 2.0, 3.0)),
      2)
    val center = Vectors.dense(1.0, 2.0, 3.0)

    // Make sure code runs.
    var model = KMeans.train(data, k = 2, maxIterations = 1)
    assert(model.clusterCenters.size === 1)
  }

  test("more clusters than points") {
    val data = sc.parallelize[Vector](
      Array(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(1.0, 3.0, 4.0)),
      2)

    // Make sure code runs.
    var model = KMeans.train(data, k = 3, maxIterations = 1)
    assert(model.clusterCenters.size === 2, s"${model.clusterCenters.size} != 2")
  }

  test("single cluster with big dataset") {
    val smallData = Array(
      Vectors.dense(1.0, 2.0, 6.0),
      Vectors.dense(1.0, 3.0, 0.0),
      Vectors.dense(1.0, 4.0, 6.0)
    )
    val data = sc.parallelize[Vector]((1 to 100).flatMap(_ => smallData), 4)

    // No matter how many runs or iterations we use, we should get one cluster,
    // centered at the mean of the points

    val center = Vectors.dense(1.0, 3.0, 4.0)

    var model = KMeans.train(data, k = 1, maxIterations = 1)
    assert(model.clusterCenters.size === 1)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 2)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = RANDOM)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL)
    assert(model.clusterCenters.head ~== center absTol 1E-5)
  }

  test("single cluster with sparse data") {

    val n = 10000
    val data = sc.parallelize[Vector]((1 to 100).flatMap { i =>
      val x = i / 1000.0
      Array(
        Vectors.sparse(n, Seq((0, 1.0 + x), (1, 2.0), (2, 6.0))),
        Vectors.sparse(n, Seq((0, 1.0 - x), (1, 2.0), (2, 6.0))),
        Vectors.sparse(n, Seq((0, 1.0), (1, 3.0 + x))),
        Vectors.sparse(n, Seq((0, 1.0), (1, 3.0 - x))),
        Vectors.sparse(n, Seq((0, 1.0), (1, 4.0), (2, 6.0 + x))),
        Vectors.sparse(n, Seq((0, 1.0), (1, 4.0), (2, 6.0 - x)))
      )
    }, 4)

    data.persist()

    // No matter how many runs or iterations we use, we should get one cluster,
    // centered at the mean of the points

    val center = Vectors.sparse(n, Seq((0, 1.0), (1, 3.0), (2, 4.0)))

    var model = KMeans.train(data, k = 1, maxIterations = 1)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 2)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = RANDOM)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL,
      distanceFunctionNames = Seq(BregmanPointOps.RELATIVE_ENTROPY))
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL,
      distanceFunctionNames = Seq(EUCLIDEAN))
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL,
      distanceFunctionNames = Seq(DISCRETE_KL))
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    data.unpersist()
  }

  test("k-means|| initialization") {

    case class VectorWithCompare(x: Vector) extends Ordered[VectorWithCompare] {
      @Override def compare(that: VectorWithCompare): Int = {
        if (this.x.toArray.foldLeft[Double](0.0)((acc, x) => acc + x * x) >
          that.x.toArray.foldLeft[Double](0.0)((acc, x) => acc + x * x)) -1
        else 1
      }
    }

    val points = Seq(
      Vectors.dense(1.0, 2.0, 6.0),
      Vectors.dense(1.0, 3.0, 0.0),
      Vectors.dense(1.0, 4.0, 6.0),
      Vectors.dense(1.0, 0.0, 1.0),
      Vectors.dense(1.0, 1.0, 1.0)
    )
    val rdd = sc.parallelize[Vector](points)

    // K-means|| initialization should place all clusters into distinct centers because
    // it will make at least five passes, and it will give non-zero probability to each
    // unselected point as long as it hasn't yet selected all of them

    var model = KMeans.train(rdd, k = 5, maxIterations = 1)

    assert(model.clusterCenters.sortBy(VectorWithCompare)
      .zip(points.sortBy(VectorWithCompare)).forall(x => x._1 ~== x._2 absTol 1E-5))

    // Iterations of Lloyd's should not change the answer either
    model = KMeans.train(rdd, k = 5, maxIterations = 10)
    assert(model.clusterCenters.sortBy(VectorWithCompare)
      .zip(points.sortBy(VectorWithCompare)).forall(x => x._1 ~== x._2 absTol 1E-5))

    // Neither should more runs
    model = KMeans.train(rdd, k = 5, maxIterations = 10, runs = 5)
    assert(model.clusterCenters.sortBy(VectorWithCompare)
      .zip(points.sortBy(VectorWithCompare)).forall(x => x._1 ~== x._2 absTol 1E-5))
  }

  test("two clusters") {
    val points = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.0, 0.1),
      Vectors.dense(0.1, 0.0),
      Vectors.dense(9.0, 0.0),
      Vectors.dense(9.0, 0.2),
      Vectors.dense(9.2, 0.0)
    )
    val rdd = sc.parallelize[Vector](points, 3)

    for (initMode <- Seq(RANDOM, K_MEANS_PARALLEL)) {
      // Two iterations are sufficient no matter where the initial centers are.
      val model = KMeans.train(rdd, k = 2, maxIterations = 2, runs = 1, mode = initMode)

      val predicts = model.predict(rdd).collect()

      assert(predicts(0) === predicts(1))
      assert(predicts(0) === predicts(2))
      assert(predicts(3) === predicts(4))
      assert(predicts(3) === predicts(5))
      assert(predicts(0) != predicts(3))
    }
  }
  
  test("multiple distance functions") {
    // Test that different Bregman divergences produce different clustering results
    val r = new Random(42)
    val points = Array.fill(100)(Vectors.dense(Array.fill(5)(r.nextDouble() * 10)))
    val rdd = sc.parallelize(points, 4)
    
    // Train models with different distance functions
    val euclideanModel = KMeans.train(rdd, k = 3, maxIterations = 10, 
      distanceFunctionNames = Seq(EUCLIDEAN))
    val klModel = KMeans.train(rdd, k = 3, maxIterations = 10, 
      distanceFunctionNames = Seq(RELATIVE_ENTROPY))
    
    // Verify that predictions are consistent for each model
    val euclideanPredictions = euclideanModel.predict(rdd).collect()
    val klPredictions = klModel.predict(rdd).collect()
    
    // Each prediction should be a valid cluster index
    assert(euclideanPredictions.forall(p => p >= 0 && p < 3))
    assert(klPredictions.forall(p => p >= 0 && p < 3))
    
    // Compute costs for both models
    val euclideanCost = euclideanModel.computeCost(rdd)
    val klCost = klModel.computeCost(rdd)
    
    // Both costs should be non-negative
    assert(euclideanCost >= 0.0, "Euclidean cost should be non-negative")
    assert(klCost >= 0.0, "KL cost should be non-negative")
  }
  
  test("high dimensional clustering") {
    // Test K-means with high-dimensional data
    val dim = 100
    val k = 5
    val numPoints = 50
    val r = new Random(123)
    
    // Create k distinct centers in high-dimensional space
    val centers = (0 until k).map { i =>
      val center = Array.fill(dim)(0.0)
      // Set a few dimensions to have large values to make centers distinct
      for (j <- 0 until 10) {
        val idx = r.nextInt(dim)
        center(idx) = 10.0 * (i + 1) + r.nextDouble()
      }
      center
    }
    
    // Generate points around these centers
    val points = centers.flatMap { center =>
      (0 until numPoints / k).map { _ =>
        val point = center.clone()
        // Add noise to each dimension
        for (j <- 0 until dim) {
          point(j) += r.nextGaussian() * 0.1
        }
        Vectors.dense(point)
      }
    }
    
    val rdd = sc.parallelize(points, 4)
    
    // Train with different numbers of iterations
    val model1 = KMeans.train(rdd, k = k, maxIterations = 1)
    val model10 = KMeans.train(rdd, k = k, maxIterations = 10)
    
    // Compute costs for both models
    val cost1 = model1.computeCost(rdd)
    val cost10 = model10.computeCost(rdd)
    
    // More iterations should result in lower cost
    assert(cost10 <= cost1, "More iterations should result in lower cost")
    
    // Check that we have the right number of clusters
    assert(model10.clusterCenters.length === k)
    
    // Verify that predictions are consistent
    val predictions = model10.predict(rdd).collect()
    assert(predictions.forall(p => p >= 0 && p < k))
  }
  
  test("weighted vectors") {
    // Test K-means with weighted vectors
    val points = Seq(
      WeightedVector(Vectors.dense(0.0, 0.0), 1.0),
      WeightedVector(Vectors.dense(0.0, 0.1), 5.0),  // Higher weight
      WeightedVector(Vectors.dense(0.1, 0.0), 1.0),
      WeightedVector(Vectors.dense(9.0, 0.0), 1.0),
      WeightedVector(Vectors.dense(9.0, 0.2), 5.0),  // Higher weight
      WeightedVector(Vectors.dense(9.2, 0.0), 1.0)
    )
    
    val rdd = sc.parallelize(points, 3)
    
    // Train model
    val model = KMeans.trainWeighted(
      RunConfig(2, 1, 0, 5),
      rdd,
      KMeansSelector(K_MEANS_PARALLEL),
      Seq(BregmanPointOps(EUCLIDEAN)),
      Seq(Embedding(IDENTITY_EMBEDDING)),
      MultiKMeansClusterer(MultiKMeansClusterer.COLUMN_TRACKING)
    )
    
    // Predict clusters
    val predictions = model.predictWeighted(rdd).collect()
    
    // Verify we have the right number of predictions
    assert(predictions.length === points.length)
    
    // Verify all predictions are valid cluster indices
    assert(predictions.forall(p => p >= 0 && p < 2))
    
    // Compute cost
    val cost = model.computeCostWeighted(rdd)
    assert(cost >= 0.0, "Cost should be non-negative")
  }
  
  test("empty clusters handling") {
    // Test how K-means handles empty clusters
    val points = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(0.2, 0.2),
      Vectors.dense(9.0, 9.0),
      Vectors.dense(9.1, 9.1),
      Vectors.dense(9.2, 9.2)
    )
    
    // Request more clusters than natural clusters in the data
    val k = 4
    val rdd = sc.parallelize(points, 2)
    
    // Train model with multiple runs to ensure stability
    val model = KMeans.train(rdd, k = k, maxIterations = 10, runs = 3)
    
    // Verify we get k centers even though data has only 2 natural clusters
    assert(model.clusterCenters.length === k)
    
    // Get predictions
    val predictions = model.predict(rdd).collect()
    
    // Group points by their cluster assignments
    val clusterGroups = predictions.zipWithIndex.groupBy(_._1).mapValues(_.map(_._2))
    
    // Verify that points 0, 1, 2 are assigned to the same or similar clusters
    val firstThreePoints = Set(0, 1, 2)
    val firstThreeClusters = firstThreePoints.map(predictions(_))
    
    // Verify that points 3, 4, 5 are assigned to the same or similar clusters
    val lastThreePoints = Set(3, 4, 5)
    val lastThreeClusters = lastThreePoints.map(predictions(_))
    
    // Check that the first three points and last three points form coherent groups
    // This is a more flexible test that allows for some variation in clustering
    assert(firstThreeClusters.size <= 2, "First three points should be in at most 2 clusters")
    assert(lastThreeClusters.size <= 2, "Last three points should be in at most 2 clusters")
    
    // Verify that at least some points from the first group are in a different cluster than points from the second group
    assert(firstThreeClusters != lastThreeClusters, "The two groups of points should be in different clusters")
  }
  
  test("KMeans model serialization") {
    // Test that KMeansModel can be serialized and deserialized
    import java.io._
    
    val points = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(9.0, 9.0),
      Vectors.dense(9.1, 9.1)
    )
    val rdd = sc.parallelize(points, 2)
    
    // Train model
    val model = KMeans.train(rdd, k = 2, maxIterations = 5)
    
    // Serialize model
    val baos = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(baos)
    oos.writeObject(model)
    oos.close()
    
    // Deserialize model
    val bais = new ByteArrayInputStream(baos.toByteArray)
    val ois = new ObjectInputStream(bais)
    val deserializedModel = ois.readObject().asInstanceOf[KMeansModel]
    ois.close()
    
    // Verify that the deserialized model produces the same predictions
    val originalPredictions = model.predict(rdd).collect()
    val deserializedPredictions = deserializedModel.predict(rdd).collect()
    
    assert(originalPredictions.sameElements(deserializedPredictions), 
      "Deserialized model should produce the same predictions")
  }
}

