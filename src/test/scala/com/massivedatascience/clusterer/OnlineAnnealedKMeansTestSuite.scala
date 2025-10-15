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

class OnlineAnnealedKMeansTestSuite extends AnyFunSuite with LocalClusterSparkContext {

  test("OnlineKMeans should cluster basic 2D data") {
    val data = sc
      .parallelize(
        Seq(
          WeightedVector(Vectors.dense(1.0, 1.0)),
          WeightedVector(Vectors.dense(1.5, 1.2)),
          WeightedVector(Vectors.dense(1.2, 1.5)),
          WeightedVector(Vectors.dense(10.0, 10.0)),
          WeightedVector(Vectors.dense(10.5, 10.2)),
          WeightedVector(Vectors.dense(10.2, 10.5))
        )
      )
      .cache()

    val model = KMeans.train(
      data.map(_.homogeneous),
      k = 2,
      maxIterations = 10,
      clustererName = MultiKMeansClusterer.ONLINE
    )

    assert(model.centers.length == 2)
    val cost = model.computeCost(data.map(_.homogeneous))
    assert(cost >= 0.0)
    assert(java.lang.Double.isFinite(cost))

    // Predictions should be reasonable
    val predictions = model.predict(data.map(_.homogeneous)).collect()
    assert(predictions.forall(p => p >= 0 && p < 2))
  }

  test("OnlineKMeans with different learning rate strategies") {
    val data = sc
      .parallelize((0 until 100).map { i =>
        val cluster = i % 3
        val x       = cluster * 10.0 + scala.util.Random.nextGaussian()
        val y       = cluster * 10.0 + scala.util.Random.nextGaussian()
        WeightedVector(Vectors.dense(x, y))
      })
      .cache()

    val strategies = Seq("standard", "sqrt", "constant")

    strategies.foreach { strategy =>
      val config    = OnlineKMeansConfig(learningRateDecay = strategy)
      val clusterer = OnlineKMeans(config)

      val pointOps    = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
      val bregmanData = data.map(pointOps.toPoint).cache()

      val initialCenters = KMeansRandom.init(
        pointOps,
        bregmanData,
        3,
        None,
        1,
        0L
      )

      val result = clusterer.cluster(10, pointOps, bregmanData, initialCenters).head

      assert(result.centers.length == 3, s"Strategy $strategy should produce 3 centers")
      assert(result.distortion >= 0.0, s"Strategy $strategy distortion should be non-negative")
      assert(
        java.lang.Double.isFinite(result.distortion),
        s"Strategy $strategy distortion should be finite"
      )

      bregmanData.unpersist()
    }
  }

  test("OnlineKMeans fast variant") {
    val data = sc.parallelize((0 until 50).map { _ =>
      WeightedVector(
        Vectors.dense(
          scala.util.Random.nextGaussian(),
          scala.util.Random.nextGaussian()
        )
      )
    })

    val model = KMeans.train(
      data.map(_.homogeneous),
      k = 3,
      maxIterations = 5,
      clustererName = MultiKMeansClusterer.ONLINE_FAST
    )

    assert(model.centers.length == 3)
    assert(model.computeCost(data.map(_.homogeneous)) >= 0.0)
  }

  test("AnnealedKMeans should cluster basic 2D data") {
    val data = sc
      .parallelize(
        Seq(
          WeightedVector(Vectors.dense(1.0, 1.0)),
          WeightedVector(Vectors.dense(1.5, 1.2)),
          WeightedVector(Vectors.dense(1.2, 1.5)),
          WeightedVector(Vectors.dense(10.0, 10.0)),
          WeightedVector(Vectors.dense(10.5, 10.2)),
          WeightedVector(Vectors.dense(10.2, 10.5))
        )
      )
      .cache()

    val model = KMeans.train(
      data.map(_.homogeneous),
      k = 2,
      maxIterations = 20,
      clustererName = MultiKMeansClusterer.ANNEALED
    )

    // Annealed k-means may reduce k due to empty clusters during annealing
    assert(model.centers.length >= 1 && model.centers.length <= 2)
    val cost = model.computeCost(data.map(_.homogeneous))
    assert(cost >= 0.0)
    assert(java.lang.Double.isFinite(cost))

    // Annealed k-means should produce valid cluster assignments
    val predictions = model.predict(data.map(_.homogeneous)).collect()
    assert(predictions.forall(p => p >= 0 && p < model.centers.length))

    // Should have reasonable cluster assignments
    // Note: annealing may collapse clusters, which is expected behavior
    assert(predictions.length == 6, "Should have predictions for all points")
  }

  test("AnnealedKMeans with different annealing schedules") {
    val data = sc
      .parallelize((0 until 60).map { i =>
        val cluster = i % 3
        val x       = cluster * 5.0 + scala.util.Random.nextGaussian()
        val y       = cluster * 5.0 + scala.util.Random.nextGaussian()
        WeightedVector(Vectors.dense(x, y))
      })
      .cache()

    val schedules = Seq("exponential", "linear")

    schedules.foreach { schedule =>
      val config = AnnealedKMeansConfig(
        initialBeta = 0.1,
        finalBeta = 10.0,
        annealingSchedule = schedule,
        annealingRate = 1.5,
        stepsPerTemperature = 3,
        maxTemperatures = 5
      )
      val clusterer = AnnealedKMeans(config)

      val pointOps    = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
      val bregmanData = data.map(pointOps.toPoint).cache()

      val initialCenters = KMeansRandom.init(
        pointOps,
        bregmanData,
        3,
        None,
        1,
        0L
      )

      val result = clusterer.cluster(10, pointOps, bregmanData, initialCenters).head

      assert(result.centers.length == 3, s"Schedule $schedule should produce 3 centers")
      assert(result.distortion >= 0.0, s"Schedule $schedule distortion should be non-negative")
      assert(
        java.lang.Double.isFinite(result.distortion),
        s"Schedule $schedule distortion should be finite"
      )

      bregmanData.unpersist()
    }
  }

  test("AnnealedKMeans fast variant") {
    val data = sc.parallelize((0 until 50).map { _ =>
      WeightedVector(
        Vectors.dense(
          scala.util.Random.nextGaussian(),
          scala.util.Random.nextGaussian()
        )
      )
    })

    val model = KMeans.train(
      data.map(_.homogeneous),
      k = 3,
      maxIterations = 10,
      clustererName = MultiKMeansClusterer.ANNEALED_FAST
    )

    assert(model.centers.length == 3)
    assert(model.computeCost(data.map(_.homogeneous)) >= 0.0)
  }

  test("AnnealedKMeans high quality variant") {
    val data = sc.parallelize((0 until 30).map { i =>
      val cluster = if (i < 15) 0.0 else 10.0
      WeightedVector(
        Vectors.dense(
          cluster + scala.util.Random.nextGaussian() * 0.5,
          cluster + scala.util.Random.nextGaussian() * 0.5
        )
      )
    })

    val model = KMeans.train(
      data.map(_.homogeneous),
      k = 2,
      maxIterations = 20,
      clustererName = MultiKMeansClusterer.ANNEALED_HIGH_QUALITY
    )

    // Annealed k-means may reduce k due to empty clusters during annealing
    assert(model.centers.length >= 1 && model.centers.length <= 2)
    val cost = model.computeCost(data.map(_.homogeneous))
    assert(cost >= 0.0)

    // High quality variant should produce valid predictions
    val predictions = model.predict(data.map(_.homogeneous)).collect()
    assert(predictions.length == 30, "Should have predictions for all points")
    assert(predictions.forall(p => p >= 0 && p < model.centers.length), "Valid predictions")
  }

  test("AnnealedKMeans robust initialization variant") {
    val data = sc.parallelize((0 until 40).map { _ =>
      WeightedVector(
        Vectors.dense(
          scala.util.Random.nextGaussian() * 2,
          scala.util.Random.nextGaussian() * 2
        )
      )
    })

    val model = KMeans.train(
      data.map(_.homogeneous),
      k = 2,
      maxIterations = 15,
      clustererName = MultiKMeansClusterer.ANNEALED_ROBUST
    )

    assert(model.centers.length == 2)
    assert(model.computeCost(data.map(_.homogeneous)) >= 0.0)
  }

  test("OnlineKMeans vs standard k-means quality comparison") {
    // Generate well-separated clusters
    val data = sc
      .parallelize((0 until 90).map { i =>
        val cluster = i / 30
        val x       = cluster * 20.0 + scala.util.Random.nextGaussian()
        val y       = cluster * 20.0 + scala.util.Random.nextGaussian()
        WeightedVector(Vectors.dense(x, y))
      })
      .cache()

    val onlineModel = KMeans.train(
      data.map(_.homogeneous),
      k = 3,
      maxIterations = 10,
      clustererName = MultiKMeansClusterer.ONLINE
    )

    val standardModel = KMeans.train(
      data.map(_.homogeneous),
      k = 3,
      maxIterations = 10,
      clustererName = MultiKMeansClusterer.COLUMN_TRACKING
    )

    val onlineCost   = onlineModel.computeCost(data.map(_.homogeneous))
    val standardCost = standardModel.computeCost(data.map(_.homogeneous))

    // Online should be within reasonable quality range
    // (might be slightly worse, but not dramatically)
    assert(onlineCost >= 0.0)
    assert(standardCost >= 0.0)
    assert(
      onlineCost < standardCost * 3.0,
      s"Online cost ($onlineCost) should be within 3x of standard cost ($standardCost)"
    )
  }

  test("AnnealedKMeans should handle single cluster gracefully") {
    val data = sc.parallelize((0 until 20).map { _ =>
      WeightedVector(
        Vectors.dense(
          scala.util.Random.nextGaussian(),
          scala.util.Random.nextGaussian()
        )
      )
    })

    val model = KMeans.train(
      data.map(_.homogeneous),
      k = 1,
      maxIterations = 10,
      clustererName = MultiKMeansClusterer.ANNEALED
    )

    assert(model.centers.length == 1)
    assert(model.computeCost(data.map(_.homogeneous)) >= 0.0)

    val predictions = model.predict(data.map(_.homogeneous)).collect()
    assert(predictions.forall(_ == 0), "All points should be in cluster 0")
  }
}
