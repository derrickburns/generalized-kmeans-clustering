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

class MiniBatchKMeansSuite extends AnyFunSuite with DataFrameSuiteBase with Matchers {

  test("MiniBatchKMeans basic clustering with Squared Euclidean") {
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

    val mbKmeans = new MiniBatchKMeans()
      .setK(2)
      .setBatchSize(5)
      .setMaxIter(50)
      .setSeed(42)

    val model = mbKmeans.fit(df)

    assert(model.clusterCenters.length == 2)

    val predictions = model.transform(df)
    assert(predictions.select("prediction").distinct().count() == 2)
  }

  test("MiniBatchKMeans finds correct clusters on well-separated data") {
    // Generate well-separated clusters
    val cluster1 = (1 to 50).map(i => Tuple1(Vectors.dense(0.0 + i * 0.01, 0.0 + i * 0.01)))
    val cluster2 = (1 to 50).map(i => Tuple1(Vectors.dense(10.0 + i * 0.01, 10.0 + i * 0.01)))

    val df = spark.createDataFrame(cluster1 ++ cluster2).toDF("features")

    val mbKmeans = new MiniBatchKMeans()
      .setK(2)
      .setBatchSize(20)
      .setMaxIter(100)
      .setSeed(42)

    val model = mbKmeans.fit(df)
    val predictions = model.transform(df)

    // All points should be assigned to one of two clusters
    val clusterCounts = predictions
      .groupBy("prediction")
      .count()
      .collect()
      .map(r => r.getLong(1))

    assert(clusterCounts.length == 2)
    // Each cluster should have roughly 50 points (some variance expected)
    clusterCounts.foreach { count =>
      assert(count >= 40 && count <= 60, s"Cluster count $count out of expected range")
    }
  }

  test("MiniBatchKMeans with KL divergence") {
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

    val mbKmeans = new MiniBatchKMeans()
      .setK(2)
      .setDivergence("kl")
      .setSmoothing(1e-6)
      .setBatchSize(3)
      .setMaxIter(50)
      .setSeed(42)

    val model = mbKmeans.fit(df)
    assert(model.clusterCenters.length == 2)
  }

  test("MiniBatchKMeans with spherical divergence") {
    val data = Seq(
      Tuple1(Vectors.dense(1.0, 0.0)),
      Tuple1(Vectors.dense(0.9, 0.1)),
      Tuple1(Vectors.dense(0.0, 1.0)),
      Tuple1(Vectors.dense(0.1, 0.9))
    )

    val df = spark.createDataFrame(data).toDF("features")

    val mbKmeans = new MiniBatchKMeans()
      .setK(2)
      .setDivergence("spherical")
      .setBatchSize(2)
      .setMaxIter(30)
      .setSeed(42)

    val model = mbKmeans.fit(df)
    assert(model.clusterCenters.length == 2)
  }

  test("MiniBatchKMeans early stopping") {
    // Create data that converges quickly
    val cluster1 = (1 to 100).map(_ => Tuple1(Vectors.dense(0.0, 0.0)))
    val cluster2 = (1 to 100).map(_ => Tuple1(Vectors.dense(10.0, 10.0)))

    val df = spark.createDataFrame(cluster1 ++ cluster2).toDF("features")

    val mbKmeans = new MiniBatchKMeans()
      .setK(2)
      .setBatchSize(50)
      .setMaxIter(1000)
      .setMaxNoImprovement(5)
      .setSeed(42)

    val model = mbKmeans.fit(df)

    // Should converge before maxIter due to early stopping
    assert(model.trainingSummary.isDefined)
    val summary = model.trainingSummary.get
    assert(summary.iterations < 1000, s"Expected early stopping, got ${summary.iterations} iterations")
  }

  test("MiniBatchKMeans respects batch size") {
    val data = (1 to 1000).map(i => Tuple1(Vectors.dense(i.toDouble, i.toDouble)))
    val df   = spark.createDataFrame(data).toDF("features")

    val mbKmeans = new MiniBatchKMeans()
      .setK(5)
      .setBatchSize(100)
      .setMaxIter(10)
      .setSeed(42)

    // Should complete without error
    val model = mbKmeans.fit(df)
    assert(model.clusterCenters.length == 5)
  }

  test("MiniBatchKMeans with random initialization") {
    val data = (1 to 100).map(i => Tuple1(Vectors.dense(i.toDouble % 10, i.toDouble / 10)))
    val df   = spark.createDataFrame(data).toDF("features")

    val mbKmeans = new MiniBatchKMeans()
      .setK(3)
      .setBatchSize(20)
      .setMaxIter(50)
      .setInitMode("random")
      .setSeed(42)

    val model = mbKmeans.fit(df)
    assert(model.clusterCenters.length == 3)
  }

  test("MiniBatchKMeans parameter validation") {
    val mbKmeans = new MiniBatchKMeans()

    // Check defaults
    assert(mbKmeans.getK == 2)
    assert(mbKmeans.getBatchSize == 1024)
    assert(mbKmeans.getDivergence == "squaredEuclidean")
    assert(mbKmeans.getMaxIter == 100)
    assert(mbKmeans.getMaxNoImprovement == 10)
    assert(mbKmeans.getReassignmentRatio == 0.01)

    // Set and verify
    mbKmeans
      .setK(10)
      .setBatchSize(512)
      .setDivergence("kl")
      .setMaxIter(200)
      .setMaxNoImprovement(20)

    assert(mbKmeans.getK == 10)
    assert(mbKmeans.getBatchSize == 512)
    assert(mbKmeans.getDivergence == "kl")
    assert(mbKmeans.getMaxIter == 200)
    assert(mbKmeans.getMaxNoImprovement == 20)
  }

  test("MiniBatchKMeans produces deterministic results with same seed") {
    val data = (1 to 100).map(i => Tuple1(Vectors.dense(i.toDouble % 5, i.toDouble / 20)))
    val df   = spark.createDataFrame(data).toDF("features")

    val mbKmeans1 = new MiniBatchKMeans()
      .setK(3)
      .setBatchSize(20)
      .setMaxIter(30)
      .setSeed(12345)

    val mbKmeans2 = new MiniBatchKMeans()
      .setK(3)
      .setBatchSize(20)
      .setMaxIter(30)
      .setSeed(12345)

    val model1 = mbKmeans1.fit(df)
    val model2 = mbKmeans2.fit(df)

    // Centers should be identical (or very close) with same seed
    val centers1 = model1.clusterCenters.sortBy(_.toArray.sum)
    val centers2 = model2.clusterCenters.sortBy(_.toArray.sum)

    for (i <- centers1.indices) {
      for (j <- 0 until centers1(i).size) {
        assert(
          math.abs(centers1(i)(j) - centers2(i)(j)) < 0.1,
          s"Centers differ: ${centers1(i)} vs ${centers2(i)}"
        )
      }
    }
  }

  test("MiniBatchKMeans handles single cluster") {
    val data = Seq(
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(1.1, 1.1)),
      Tuple1(Vectors.dense(0.9, 0.9))
    )
    val df = spark.createDataFrame(data).toDF("features")

    // k=2 is minimum, but all points are similar
    val mbKmeans = new MiniBatchKMeans()
      .setK(2)
      .setBatchSize(3)
      .setMaxIter(20)
      .setSeed(42)

    val model = mbKmeans.fit(df)
    assert(model.clusterCenters.length == 2)
  }

  test("MiniBatchKMeans with L1 divergence") {
    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(11.0, 11.0))
    )
    val df = spark.createDataFrame(data).toDF("features")

    val mbKmeans = new MiniBatchKMeans()
      .setK(2)
      .setDivergence("l1")
      .setBatchSize(2)
      .setMaxIter(30)
      .setSeed(42)

    val model = mbKmeans.fit(df)
    assert(model.clusterCenters.length == 2)
  }

  test("MiniBatchKMeans training summary contains expected fields") {
    val data = (1 to 50).map(i => Tuple1(Vectors.dense(i.toDouble, i.toDouble)))
    val df   = spark.createDataFrame(data).toDF("features")

    val mbKmeans = new MiniBatchKMeans()
      .setK(3)
      .setBatchSize(10)
      .setMaxIter(20)
      .setSeed(42)

    val model = mbKmeans.fit(df)

    assert(model.trainingSummary.isDefined)
    val summary = model.trainingSummary.get

    assert(summary.algorithm == "MiniBatchKMeans")
    assert(summary.k == 3)
    assert(summary.dim == 2)
    assert(summary.iterations > 0)
    assert(summary.distortionHistory.nonEmpty)
  }

  test("MiniBatchKMeans model can transform new data") {
    val trainData = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(10.0, 10.0))
    )
    val trainDf = spark.createDataFrame(trainData).toDF("features")

    val mbKmeans = new MiniBatchKMeans()
      .setK(2)
      .setBatchSize(2)
      .setMaxIter(20)
      .setSeed(42)

    val model = mbKmeans.fit(trainDf)

    // Transform new data
    val testData = Seq(
      Tuple1(Vectors.dense(0.5, 0.5)),
      Tuple1(Vectors.dense(9.5, 9.5)),
      Tuple1(Vectors.dense(5.0, 5.0))
    )
    val testDf = spark.createDataFrame(testData).toDF("features")

    val predictions = model.transform(testDf)
    assert(predictions.count() == 3)
    assert(predictions.columns.contains("prediction"))
  }
}
