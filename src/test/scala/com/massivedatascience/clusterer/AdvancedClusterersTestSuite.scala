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

import com.massivedatascience.linalg.WeightedVector
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite

class AdvancedClusterersTestSuite extends AnyFunSuite with LocalClusterSparkContext {

  // === BisectingKMeans Tests ===

  test("BisectingKMeans should cluster basic 2D data") {
    val data = sc.parallelize(Seq(
      WeightedVector(Vectors.dense(1.0, 1.0)),
      WeightedVector(Vectors.dense(1.5, 1.2)),
      WeightedVector(Vectors.dense(1.2, 1.5)),
      WeightedVector(Vectors.dense(10.0, 10.0)),
      WeightedVector(Vectors.dense(10.5, 10.2)),
      WeightedVector(Vectors.dense(10.2, 10.5))
    )).cache()

    val model = KMeans.train(
      data.map(_.homogeneous),
      k = 2,
      maxIterations = 20,
      clustererName = MultiKMeansClusterer.BISECTING
    )

    assert(model.centers.length == 2)
    val cost = model.computeCost(data.map(_.homogeneous))
    assert(cost >= 0.0)
    assert(java.lang.Double.isFinite(cost))

    // Check that clusters are separated
    val predictions = model.predict(data.map(_.homogeneous)).collect()
    val cluster0Count = predictions.count(_ == 0)
    val cluster1Count = predictions.count(_ == 1)
    assert(cluster0Count == 3 || cluster1Count == 3, "Should separate into 3+3 clusters")
  }

  test("BisectingKMeans with different split criteria") {
    val data = sc.parallelize((0 until 60).map { i =>
      val cluster = i % 3
      val x = cluster * 10.0 + scala.util.Random.nextGaussian()
      val y = cluster * 10.0 + scala.util.Random.nextGaussian()
      WeightedVector(Vectors.dense(x, y))
    }).cache()

    val criteria = Seq("largest", "highest_cost")

    criteria.foreach { criterion =>
      val config = BisectingKMeansConfig(splitCriterion = criterion)
      val clusterer = BisectingKMeans(config)

      val pointOps = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
      val bregmanData = data.map(pointOps.toPoint).cache()

      val initialCenters = KMeansRandom.init(
        pointOps, bregmanData, 3, None, 1, 0L
      )

      val result = clusterer.cluster(20, pointOps, bregmanData, initialCenters).head

      assert(result.centers.length == 3, s"Criterion $criterion should produce 3 centers")
      assert(result.distortion >= 0.0, s"Criterion $criterion distortion should be non-negative")
      assert(java.lang.Double.isFinite(result.distortion), s"Criterion $criterion distortion should be finite")

      bregmanData.unpersist()
    }
  }

  test("BisectingKMeans fast variant") {
    val data = sc.parallelize((0 until 40).map { _ =>
      WeightedVector(Vectors.dense(
        scala.util.Random.nextGaussian(),
        scala.util.Random.nextGaussian()
      ))
    })

    val model = KMeans.train(
      data.map(_.homogeneous),
      k = 3,
      maxIterations = 10,
      clustererName = MultiKMeansClusterer.BISECTING_FAST
    )

    assert(model.centers.length == 3)
    assert(model.computeCost(data.map(_.homogeneous)) >= 0.0)
  }

  // === XMeans Tests ===

  test("XMeans should find optimal k automatically") {
    // Create data with 3 clear clusters
    val data = sc.parallelize((0 until 60).map { i =>
      val cluster = i / 20
      val x = cluster * 15.0 + scala.util.Random.nextGaussian()
      val y = cluster * 15.0 + scala.util.Random.nextGaussian()
      WeightedVector(Vectors.dense(x, y))
    }).cache()

    val config = XMeansConfig(minK = 2, maxK = 5)
    val clusterer = XMeans(config)

    val pointOps = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val bregmanData = data.map(pointOps.toPoint).cache()

    val initialCenters = KMeansRandom.init(
      pointOps, bregmanData, 2, None, 1, 0L
    )

    val result = clusterer.cluster(20, pointOps, bregmanData, initialCenters).head

    // X-means should find k=3 or close to it
    assert(result.centers.length >= 2 && result.centers.length <= 5)
    assert(result.distortion >= 0.0)
    assert(java.lang.Double.isFinite(result.distortion))

    bregmanData.unpersist()
  }

  test("XMeans with BIC vs AIC") {
    val data = sc.parallelize((0 until 40).map { i =>
      val cluster = if (i < 20) 0.0 else 10.0
      WeightedVector(Vectors.dense(
        cluster + scala.util.Random.nextGaussian(),
        cluster + scala.util.Random.nextGaussian()
      ))
    }).cache()

    val pointOps = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val bregmanData = data.map(pointOps.toPoint).cache()

    // Test BIC
    val bicClusterer = XMeans(XMeansConfig(minK = 1, maxK = 4, criterion = "bic"))
    val bicInitial = KMeansRandom.init(pointOps, bregmanData, 1, None, 1, 0L)
    val bicResult = bicClusterer.cluster(20, pointOps, bregmanData, bicInitial).head

    // Test AIC
    val aicClusterer = XMeans(XMeansConfig(minK = 1, maxK = 4, criterion = "aic"))
    val aicInitial = KMeansRandom.init(pointOps, bregmanData, 1, None, 1, 0L)
    val aicResult = aicClusterer.cluster(20, pointOps, bregmanData, aicInitial).head

    // Both should find valid clusters
    assert(bicResult.centers.length >= 1 && bicResult.centers.length <= 4)
    assert(aicResult.centers.length >= 1 && aicResult.centers.length <= 4)
    assert(java.lang.Double.isFinite(bicResult.distortion))
    assert(java.lang.Double.isFinite(aicResult.distortion))

    bregmanData.unpersist()
  }

  test("XMeans fast variant") {
    val data = sc.parallelize((0 until 30).map { _ =>
      WeightedVector(Vectors.dense(
        scala.util.Random.nextGaussian() * 2,
        scala.util.Random.nextGaussian() * 2
      ))
    })

    val model = KMeans.train(
      data.map(_.homogeneous),
      k = 2,  // Will be ignored, XMeans finds optimal k
      maxIterations = 10,
      clustererName = MultiKMeansClusterer.XMEANS_FAST
    )

    // XMeans should find some k in the range
    assert(model.centers.length >= 2 && model.centers.length <= 15)
    assert(model.computeCost(data.map(_.homogeneous)) >= 0.0)
  }

  // === ConstrainedKMeans Tests ===

  test("ConstrainedKMeans with must-link constraints") {
    val data = sc.parallelize(Seq(
      WeightedVector(Vectors.dense(1.0, 1.0)),    // 0
      WeightedVector(Vectors.dense(1.5, 1.2)),    // 1
      WeightedVector(Vectors.dense(1.2, 1.5)),    // 2
      WeightedVector(Vectors.dense(10.0, 10.0)),  // 3
      WeightedVector(Vectors.dense(10.5, 10.2)),  // 4
      WeightedVector(Vectors.dense(10.2, 10.5))   // 5
    )).cache()

    // Points 0 and 1 must be in same cluster
    val constraints = Constraints(mustLink = Set((0L, 1L)))
    val clusterer = ConstrainedKMeans(constraints)

    val pointOps = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val bregmanData = data.map(pointOps.toPoint).cache()

    val initialCenters = KMeansRandom.init(
      pointOps, bregmanData, 2, None, 1, 0L
    )

    val result = clusterer.cluster(20, pointOps, bregmanData, initialCenters).head

    assert(result.centers.length == 2)
    assert(result.distortion >= 0.0)

    bregmanData.unpersist()
  }

  test("ConstrainedKMeans with cannot-link constraints") {
    val data = sc.parallelize(Seq(
      WeightedVector(Vectors.dense(1.0, 1.0)),
      WeightedVector(Vectors.dense(1.5, 1.2)),
      WeightedVector(Vectors.dense(10.0, 10.0)),
      WeightedVector(Vectors.dense(10.5, 10.2))
    )).cache()

    // Points 0 and 2 cannot be in same cluster
    val constraints = Constraints(cannotLink = Set((0L, 2L)))
    val clusterer = ConstrainedKMeans(constraints)

    val pointOps = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val bregmanData = data.map(pointOps.toPoint).cache()

    val initialCenters = KMeansRandom.init(
      pointOps, bregmanData, 2, None, 1, 0L
    )

    val result = clusterer.cluster(20, pointOps, bregmanData, initialCenters).head

    assert(result.centers.length == 2)
    assert(result.distortion >= 0.0)

    bregmanData.unpersist()
  }

  test("ConstrainedKMeans with no constraints behaves like standard k-means") {
    val data = sc.parallelize((0 until 30).map { _ =>
      WeightedVector(Vectors.dense(
        scala.util.Random.nextGaussian(),
        scala.util.Random.nextGaussian()
      ))
    })

    // No constraints
    val constraints = Constraints()
    val clusterer = ConstrainedKMeans(constraints)

    val pointOps = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val bregmanData = data.map(pointOps.toPoint).cache()

    val initialCenters = KMeansRandom.init(
      pointOps, bregmanData, 3, None, 1, 0L
    )

    val result = clusterer.cluster(10, pointOps, bregmanData, initialCenters).head

    assert(result.centers.length == 3)
    assert(result.distortion >= 0.0)

    bregmanData.unpersist()
  }

  test("ConstrainedKMeans soft constraints") {
    val data = sc.parallelize((0 until 20).map { i =>
      val x = if (i < 10) 0.0 else 10.0
      WeightedVector(Vectors.dense(
        x + scala.util.Random.nextGaussian(),
        x + scala.util.Random.nextGaussian()
      ))
    }).cache()

    // Soft constraints: points 0 and 15 cannot link (different natural clusters)
    val constraints = Constraints(cannotLink = Set((0L, 15L)))
    val clusterer = ConstrainedKMeans.withSoftConstraints(constraints, penalty = 10.0)

    val pointOps = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val bregmanData = data.map(pointOps.toPoint).cache()

    val initialCenters = KMeansRandom.init(
      pointOps, bregmanData, 2, None, 1, 0L
    )

    val result = clusterer.cluster(20, pointOps, bregmanData, initialCenters).head

    assert(result.centers.length == 2)
    assert(result.distortion >= 0.0)

    bregmanData.unpersist()
  }

  // === Integration Tests ===

  test("All three advanced clusterers should handle same dataset") {
    val data = sc.parallelize((0 until 60).map { i =>
      val cluster = i / 20
      val x = cluster * 10.0 + scala.util.Random.nextGaussian()
      val y = cluster * 10.0 + scala.util.Random.nextGaussian()
      WeightedVector(Vectors.dense(x, y))
    }).cache()

    val clusterers = Seq(
      MultiKMeansClusterer.BISECTING,
      MultiKMeansClusterer.XMEANS,
      MultiKMeansClusterer.BISECTING_FAST
    )

    clusterers.foreach { name =>
      val model = KMeans.train(
        data.map(_.homogeneous),
        k = 3,
        maxIterations = 20,
        clustererName = name
      )

      assert(model.centers.nonEmpty, s"$name should produce clusters")
      val cost = model.computeCost(data.map(_.homogeneous))
      assert(cost >= 0.0, s"$name should have non-negative cost")
      assert(java.lang.Double.isFinite(cost), s"$name should have finite cost")
    }
  }
}
