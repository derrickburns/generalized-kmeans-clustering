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

class ClusteringMetricsSuite extends AnyFunSuite with DataFrameSuiteBase with Matchers {

  test("ClusteringMetrics - computes cluster sizes correctly") {
    val spark = this.spark
    import spark.implicits._

    val predictions = Seq(
      (Vectors.dense(0.0, 0.0), 0),
      (Vectors.dense(1.0, 0.0), 0),
      (Vectors.dense(0.0, 1.0), 0),
      (Vectors.dense(5.0, 5.0), 1),
      (Vectors.dense(6.0, 5.0), 1)
    ).toDF("features", "prediction")

    val metrics = ClusteringMetrics(predictions)

    metrics.clusterSizes shouldBe Map(0 -> 3L, 1 -> 2L)
    metrics.k shouldBe 2
    metrics.numPoints shouldBe 5
  }

  test("ClusteringMetrics - computes centroids correctly") {
    val spark = this.spark
    import spark.implicits._

    val predictions = Seq(
      (Vectors.dense(0.0, 0.0), 0),
      (Vectors.dense(2.0, 0.0), 0),
      (Vectors.dense(10.0, 10.0), 1),
      (Vectors.dense(10.0, 12.0), 1)
    ).toDF("features", "prediction")

    val metrics = ClusteringMetrics(predictions)

    // Cluster 0 centroid: (1.0, 0.0)
    val c0 = metrics.centroids(0).toArray
    c0(0) shouldBe (1.0 +- 0.01)
    c0(1) shouldBe (0.0 +- 0.01)

    // Cluster 1 centroid: (10.0, 11.0)
    val c1 = metrics.centroids(1).toArray
    c1(0) shouldBe (10.0 +- 0.01)
    c1(1) shouldBe (11.0 +- 0.01)
  }

  test("ClusteringMetrics - computes inertia (WCSS) correctly") {
    val spark = this.spark
    import spark.implicits._

    // Two clusters with known distances from centroids
    val predictions = Seq(
      (Vectors.dense(0.0, 0.0), 0), // dist to centroid (1,0) = 1
      (Vectors.dense(2.0, 0.0), 0), // dist to centroid (1,0) = 1
      (Vectors.dense(10.0, 0.0), 1) // dist to centroid (10,0) = 0
    ).toDF("features", "prediction")

    val metrics = ClusteringMetrics(predictions)

    // Inertia = sum of squared distances to centroids
    // Cluster 0: (0,0) -> (1,0): 1, (2,0) -> (1,0): 1, total = 2
    // Cluster 1: single point, inertia = 0
    metrics.inertia shouldBe (2.0 +- 0.01)
  }

  test("ClusteringMetrics - exact silhouette for perfect clusters") {
    val spark = this.spark
    import spark.implicits._

    // Two well-separated clusters
    val predictions = Seq(
      (Vectors.dense(0.0, 0.0), 0),
      (Vectors.dense(0.1, 0.1), 0),
      (Vectors.dense(0.2, 0.0), 0),
      (Vectors.dense(10.0, 10.0), 1),
      (Vectors.dense(10.1, 9.9), 1),
      (Vectors.dense(9.9, 10.1), 1)
    ).toDF("features", "prediction")

    val metrics    = ClusteringMetrics(predictions)
    val silhouette = metrics.silhouetteScore

    // Well-separated clusters should have high silhouette (close to 1)
    silhouette should be > 0.8
  }

  test("ClusteringMetrics - approximate silhouette for perfect clusters") {
    val spark = this.spark
    import spark.implicits._

    val predictions = Seq(
      (Vectors.dense(0.0, 0.0), 0),
      (Vectors.dense(0.1, 0.1), 0),
      (Vectors.dense(10.0, 10.0), 1),
      (Vectors.dense(10.1, 9.9), 1)
    ).toDF("features", "prediction")

    val metrics          = ClusteringMetrics(predictions)
    val approxSilhouette = metrics.approximateSilhouetteScore

    // Should be positive for well-separated clusters
    approxSilhouette should be > 0.5
  }

  test("ClusteringMetrics - silhouette for overlapping clusters") {
    val spark = this.spark
    import spark.implicits._

    // Overlapping clusters
    val predictions = Seq(
      (Vectors.dense(0.0, 0.0), 0),
      (Vectors.dense(1.0, 1.0), 0),
      (Vectors.dense(1.5, 1.5), 1), // Close to cluster 0
      (Vectors.dense(2.0, 2.0), 1)
    ).toDF("features", "prediction")

    val metrics    = ClusteringMetrics(predictions)
    val silhouette = metrics.silhouetteScore

    // Overlapping clusters should have lower silhouette
    silhouette should be < 0.8
  }

  test("ClusteringMetrics - balance ratio for equal clusters") {
    val spark = this.spark
    import spark.implicits._

    val predictions = Seq(
      (Vectors.dense(0.0, 0.0), 0),
      (Vectors.dense(1.0, 0.0), 0),
      (Vectors.dense(5.0, 0.0), 1),
      (Vectors.dense(6.0, 0.0), 1)
    ).toDF("features", "prediction")

    val metrics = ClusteringMetrics(predictions)
    metrics.balanceRatio shouldBe 1.0
  }

  test("ClusteringMetrics - balance ratio for unequal clusters") {
    val spark = this.spark
    import spark.implicits._

    val predictions = Seq(
      (Vectors.dense(0.0, 0.0), 0),
      (Vectors.dense(1.0, 0.0), 0),
      (Vectors.dense(2.0, 0.0), 0),
      (Vectors.dense(3.0, 0.0), 0),
      (Vectors.dense(10.0, 0.0), 1)
    ).toDF("features", "prediction")

    val metrics = ClusteringMetrics(predictions)
    metrics.balanceRatio shouldBe (0.25 +- 0.01) // 1/4
  }

  test("ClusteringMetrics - size standard deviation") {
    val spark = this.spark
    import spark.implicits._

    val predictions = Seq(
      (Vectors.dense(0.0, 0.0), 0),
      (Vectors.dense(1.0, 0.0), 0),
      (Vectors.dense(5.0, 0.0), 1),
      (Vectors.dense(6.0, 0.0), 1)
    ).toDF("features", "prediction")

    val metrics = ClusteringMetrics(predictions)
    metrics.sizeStdDev shouldBe (0.0 +- 0.01) // Equal sizes
  }

  test("ClusteringMetrics - single cluster returns 0 silhouette") {
    val spark = this.spark
    import spark.implicits._

    val predictions = Seq(
      (Vectors.dense(0.0, 0.0), 0),
      (Vectors.dense(1.0, 0.0), 0),
      (Vectors.dense(2.0, 0.0), 0)
    ).toDF("features", "prediction")

    val metrics = ClusteringMetrics(predictions)
    metrics.silhouetteScore shouldBe 0.0
  }

  test("ClusteringMetrics - toString produces readable output") {
    val spark = this.spark
    import spark.implicits._

    val predictions = Seq(
      (Vectors.dense(0.0, 0.0), 0),
      (Vectors.dense(1.0, 0.0), 0),
      (Vectors.dense(5.0, 0.0), 1)
    ).toDF("features", "prediction")

    val metrics = ClusteringMetrics(predictions)
    val str     = metrics.toString

    str should include("Clusters: 2")
    str should include("Points: 3")
    str should include("Inertia")
    str should include("Balance ratio")
  }

  test("ClusteringMetrics - works with model predictions") {
    val spark = this.spark
    import spark.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.5, 0.5)),
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(10.5, 10.5))
    ).toDF("features")

    val model = new GeneralizedKMeans().setK(2).setSeed(42L).fit(data)

    val predictions = model.transform(data)
    val metrics     = ClusteringMetrics(predictions)

    metrics.k shouldBe 2
    metrics.numPoints shouldBe 4
    metrics.silhouetteScore should be > 0.5
  }
}
