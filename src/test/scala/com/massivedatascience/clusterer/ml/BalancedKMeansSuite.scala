/*
 * Licensed to the Massive Data Science and Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Massive Data Science and Derrick R. Burns licenses this file to You under the
 * Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
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

package com.massivedatascience.clusterer.ml

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class BalancedKMeansSuite extends AnyFunSuite with DataFrameSuiteBase with Matchers {

  test("BalancedKMeans basic clustering with soft mode") {
    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(0.2, 0.0)),
      Tuple1(Vectors.dense(9.0, 9.0)),
      Tuple1(Vectors.dense(9.1, 9.1)),
      Tuple1(Vectors.dense(9.2, 9.0)),
      Tuple1(Vectors.dense(0.0, 9.0)),
      Tuple1(Vectors.dense(0.1, 9.1)),
      Tuple1(Vectors.dense(9.0, 0.0)),
      Tuple1(Vectors.dense(9.1, 0.1))
    )

    val df = spark.createDataFrame(data).toDF("features")

    val balancedKmeans =
      new BalancedKMeans().setK(2).setBalanceMode("soft").setMaxIter(30).setSeed(42)

    val model = balancedKmeans.fit(df)

    assert(model.clusterCenters.length == 2)

    val predictions = model.transform(df)
    assert(predictions.select("prediction").distinct().count() == 2)
  }

  test("BalancedKMeans with hard mode produces balanced clusters") {
    // Create 100 points, expect 4 clusters of ~25 each
    val data = (0 until 100).map { i =>
      val cluster = i / 25
      val x       = cluster * 10.0 + (i % 25) * 0.1
      val y       = cluster * 10.0 + (i % 25) * 0.1
      Tuple1(Vectors.dense(x, y))
    }

    val df = spark.createDataFrame(data).toDF("features")

    val balancedKmeans = new BalancedKMeans()
      .setK(4)
      .setBalanceMode("hard")
      .setMinClusterSize(20)
      .setMaxClusterSize(30)
      .setMaxIter(50)
      .setSeed(42)

    val model       = balancedKmeans.fit(df)
    val predictions = model.transform(df)

    // Check that each cluster has between 20 and 30 points
    val clusterCounts =
      predictions.groupBy("prediction").count().collect().map(r => (r.getInt(0), r.getLong(1)))

    clusterCounts.foreach { case (cluster, count) =>
      assert(count >= 20 && count <= 30, s"Cluster $cluster has $count points, expected 20-30")
    }
  }

  test("BalancedKMeans with equal-sized clusters") {
    // 50 points, 5 clusters = 10 points each
    val data = (0 until 50).map { i =>
      Tuple1(Vectors.dense(i.toDouble, i.toDouble % 10))
    }

    val df = spark.createDataFrame(data).toDF("features")

    val balancedKmeans = new BalancedKMeans()
      .setK(5)
      .setBalanceMode("hard")
      .setMinClusterSize(10)
      .setMaxClusterSize(10)
      .setMaxIter(100)
      .setSeed(42)

    val model       = balancedKmeans.fit(df)
    val predictions = model.transform(df)

    val clusterCounts = predictions.groupBy("prediction").count().collect().map(_.getLong(1))

    // Each cluster should have exactly 10 points
    clusterCounts.foreach { count =>
      assert(count == 10, s"Expected 10 points per cluster, got $count")
    }
  }

  test("BalancedKMeans soft mode with penalty") {
    val data = (0 until 60).map { i =>
      // Create natural imbalance: 30 points near origin, 30 spread out
      if (i < 30) {
        Tuple1(Vectors.dense(i * 0.01, i * 0.01))
      } else {
        Tuple1(Vectors.dense(10.0 + (i - 30) * 0.5, 10.0 + (i - 30) * 0.5))
      }
    }

    val df = spark.createDataFrame(data).toDF("features")

    val balancedKmeans = new BalancedKMeans()
      .setK(3)
      .setBalanceMode("soft")
      .setBalancePenalty(1.0)
      .setMaxIter(50)
      .setSeed(42)

    val model       = balancedKmeans.fit(df)
    val predictions = model.transform(df)

    // All clusters should have at least some points
    val clusterCounts = predictions.groupBy("prediction").count().collect()

    assert(clusterCounts.length == 3, "Expected 3 clusters")
    clusterCounts.foreach { row =>
      assert(row.getLong(1) > 0, "Each cluster should have at least one point")
    }
  }

  test("BalancedKMeans with KL divergence") {
    // Generate positive data for KL divergence
    val data = Seq(
      Tuple1(Vectors.dense(0.1, 0.9)),
      Tuple1(Vectors.dense(0.2, 0.8)),
      Tuple1(Vectors.dense(0.15, 0.85)),
      Tuple1(Vectors.dense(0.9, 0.1)),
      Tuple1(Vectors.dense(0.8, 0.2)),
      Tuple1(Vectors.dense(0.85, 0.15))
    )

    val df = spark.createDataFrame(data).toDF("features")

    val balancedKmeans = new BalancedKMeans()
      .setK(2)
      .setDivergence("kl")
      .setSmoothing(1e-6)
      .setBalanceMode("soft")
      .setMaxIter(30)
      .setSeed(42)

    val model = balancedKmeans.fit(df)
    assert(model.clusterCenters.length == 2)
  }

  test("BalancedKMeans parameter validation") {
    val balancedKmeans = new BalancedKMeans()

    // Check defaults
    assert(balancedKmeans.getK == 2)
    assert(balancedKmeans.getMinClusterSize == 1)
    assert(balancedKmeans.getMaxClusterSize == 0) // Auto
    assert(balancedKmeans.getBalanceMode == "soft")
    assert(balancedKmeans.getBalancePenalty == 0.5)
    assert(balancedKmeans.getDivergence == "squaredEuclidean")
    assert(balancedKmeans.getMaxIter == 50)

    // Set and verify
    balancedKmeans
      .setK(10)
      .setMinClusterSize(5)
      .setMaxClusterSize(100)
      .setBalanceMode("hard")
      .setBalancePenalty(0.8)
      .setDivergence("kl")
      .setMaxIter(200)

    assert(balancedKmeans.getK == 10)
    assert(balancedKmeans.getMinClusterSize == 5)
    assert(balancedKmeans.getMaxClusterSize == 100)
    assert(balancedKmeans.getBalanceMode == "hard")
    assert(balancedKmeans.getBalancePenalty == 0.8)
    assert(balancedKmeans.getDivergence == "kl")
    assert(balancedKmeans.getMaxIter == 200)
  }

  test("BalancedKMeans deterministic with same seed") {
    val data = (0 until 50).map(i => Tuple1(Vectors.dense(i.toDouble, i.toDouble % 10)))
    val df   = spark.createDataFrame(data).toDF("features")

    val km1 = new BalancedKMeans().setK(3).setBalanceMode("soft").setMaxIter(30).setSeed(12345)

    val km2 = new BalancedKMeans().setK(3).setBalanceMode("soft").setMaxIter(30).setSeed(12345)

    val model1 = km1.fit(df)
    val model2 = km2.fit(df)

    val centers1 = model1.clusterCenters.sortBy(_.toArray.sum)
    val centers2 = model2.clusterCenters.sortBy(_.toArray.sum)

    for (i <- centers1.indices) {
      for (j <- 0 until centers1(i).size) {
        assert(
          math.abs(centers1(i)(j) - centers2(i)(j)) < 0.01,
          s"Centers differ: ${centers1(i)} vs ${centers2(i)}"
        )
      }
    }
  }

  test("BalancedKMeans training summary") {
    val data = (0 until 30).map(i => Tuple1(Vectors.dense(i.toDouble, i.toDouble)))
    val df   = spark.createDataFrame(data).toDF("features")

    val balancedKmeans =
      new BalancedKMeans().setK(3).setBalanceMode("soft").setMaxIter(20).setSeed(42)

    val model = balancedKmeans.fit(df)

    assert(model.trainingSummary.isDefined)
    val summary = model.trainingSummary.get

    assert(summary.algorithm == "BalancedKMeans")
    assert(summary.k == 3)
    assert(summary.dim == 2)
    assert(summary.iterations > 0)
    assert(summary.distortionHistory.nonEmpty)
  }

  test("BalancedKMeans model transform") {
    val trainData = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(10.1, 10.1))
    )
    val trainDf   = spark.createDataFrame(trainData).toDF("features")

    val balancedKmeans = new BalancedKMeans()
      .setK(2)
      .setBalanceMode("hard")
      .setMinClusterSize(2)
      .setMaxClusterSize(2)
      .setMaxIter(20)
      .setSeed(42)

    val model = balancedKmeans.fit(trainDf)

    // Transform new data
    val testData = Seq(
      Tuple1(Vectors.dense(0.5, 0.5)),
      Tuple1(Vectors.dense(9.5, 9.5)),
      Tuple1(Vectors.dense(5.0, 5.0))
    )
    val testDf   = spark.createDataFrame(testData).toDF("features")

    val predictions = model.transform(testDf)
    assert(predictions.count() == 3)
    assert(predictions.columns.contains("prediction"))
  }

  test("BalancedKMeans with L1 divergence") {
    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(11.0, 11.0))
    )
    val df   = spark.createDataFrame(data).toDF("features")

    val balancedKmeans = new BalancedKMeans()
      .setK(2)
      .setDivergence("l1")
      .setBalanceMode("hard")
      .setMinClusterSize(2)
      .setMaxClusterSize(2)
      .setMaxIter(30)
      .setSeed(42)

    val model = balancedKmeans.fit(df)
    assert(model.clusterCenters.length == 2)

    val predictions   = model.transform(df)
    val clusterCounts = predictions.groupBy("prediction").count().collect()
    clusterCounts.foreach { row =>
      assert(row.getLong(1) == 2, "Each cluster should have exactly 2 points")
    }
  }

  test("BalancedKMeans with spherical divergence") {
    val data = Seq(
      Tuple1(Vectors.dense(1.0, 0.0)),
      Tuple1(Vectors.dense(0.9, 0.1)),
      Tuple1(Vectors.dense(0.8, 0.2)),
      Tuple1(Vectors.dense(0.0, 1.0)),
      Tuple1(Vectors.dense(0.1, 0.9)),
      Tuple1(Vectors.dense(0.2, 0.8))
    )
    val df   = spark.createDataFrame(data).toDF("features")

    val balancedKmeans = new BalancedKMeans()
      .setK(2)
      .setDivergence("spherical")
      .setBalanceMode("soft")
      .setMaxIter(30)
      .setSeed(42)

    val model = balancedKmeans.fit(df)
    assert(model.clusterCenters.length == 2)
  }

  test("BalancedKMeans handles minimum viable dataset") {
    // Just enough points for 2 clusters with minSize=1
    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(10.0, 10.0))
    )
    val df   = spark.createDataFrame(data).toDF("features")

    val balancedKmeans = new BalancedKMeans()
      .setK(2)
      .setBalanceMode("hard")
      .setMinClusterSize(1)
      .setMaxClusterSize(1)
      .setMaxIter(20)
      .setSeed(42)

    val model = balancedKmeans.fit(df)
    assert(model.clusterCenters.length == 2)

    val predictions   = model.transform(df)
    val clusterCounts = predictions.groupBy("prediction").count().collect()
    assert(clusterCounts.length == 2)
    clusterCounts.foreach { row =>
      assert(row.getLong(1) == 1, "Each cluster should have exactly 1 point")
    }
  }

  test("BalancedKMeans with auto maxClusterSize") {
    val data = (0 until 100).map(i => Tuple1(Vectors.dense(i.toDouble, i.toDouble % 10)))
    val df   = spark.createDataFrame(data).toDF("features")

    val balancedKmeans = new BalancedKMeans()
      .setK(5)
      .setBalanceMode("soft")
      .setMaxClusterSize(0) // Auto
      .setMaxIter(30)
      .setSeed(42)

    // Should not throw, auto max size is ~22 (100/5 * 1.1)
    val model = balancedKmeans.fit(df)
    assert(model.clusterCenters.length == 5)
  }

  test("BalancedKMeans with random initialization") {
    val data = (0 until 50).map(i => Tuple1(Vectors.dense(i.toDouble, i.toDouble % 10)))
    val df   = spark.createDataFrame(data).toDF("features")

    val balancedKmeans = new BalancedKMeans()
      .setK(3)
      .setInitMode("random")
      .setBalanceMode("soft")
      .setMaxIter(30)
      .setSeed(42)

    val model = balancedKmeans.fit(df)
    assert(model.clusterCenters.length == 3)
  }
}
