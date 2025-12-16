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

package com.massivedatascience.clusterer.ml.df

import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests for outlier detection framework.
  *
  * Validates:
  *   - Distance-based outlier detection
  *   - Trimmed outlier detection
  *   - OutlierMode enum parsing
  *   - Robust center update strategies
  */
class OutlierDetectionSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  private val spark: SparkSession = SparkSession
    .builder()
    .master("local[2]")
    .appName("OutlierDetectionSuite")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  import spark.implicits._

  override def afterAll(): Unit = {
    try {
      spark.stop()
    } finally {
      super.afterAll()
    }
  }

  // Test data: 4 normal points near origin, 1 outlier far away
  private def testDataWithOutlier() = {
    Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 0.0)),
      Tuple1(Vectors.dense(0.0, 1.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(100.0, 100.0)) // Outlier
    ).toDF("features")
  }

  // Centers at origin
  private def singleCenterAtOrigin(): Array[Vector] = {
    Array(Vectors.dense(0.5, 0.5))
  }

  // ========== DistanceBasedOutlierDetector Tests ==========

  test("DistanceBasedOutlierDetector detects outliers beyond threshold") {
    val kernel   = new SquaredEuclideanKernel()
    val detector = new DistanceBasedOutlierDetector(kernel, threshold = 3.0)

    val df      = testDataWithOutlier()
    val centers = singleCenterAtOrigin()

    val result = detector.detectOutliers(df, centers, "features")

    result.columns should contain("outlier_score")
    result.columns should contain("is_outlier")

    // Collect results
    val outliers = result.filter($"is_outlier" === true).count()
    outliers shouldBe 1 // The point at (100, 100)
  }

  test("DistanceBasedOutlierDetector adds outlier_score column") {
    val kernel   = new SquaredEuclideanKernel()
    val detector = new DistanceBasedOutlierDetector(kernel, threshold = 5.0)

    val df      = testDataWithOutlier()
    val centers = singleCenterAtOrigin()

    val result  = detector.detectOutliers(df, centers, "features")
    val scores  = result.select("outlier_score").collect().map(_.getDouble(0))

    // All scores should be positive
    scores.foreach(_ should be >= 0.0)

    // Outlier score should be highest
    val maxScore = scores.max
    maxScore should be > 10.0 // Much larger than normal points
  }

  test("DistanceBasedOutlierDetector threshold validation") {
    val kernel = new SquaredEuclideanKernel()

    an[IllegalArgumentException] should be thrownBy {
      new DistanceBasedOutlierDetector(kernel, threshold = -1.0)
    }

    an[IllegalArgumentException] should be thrownBy {
      new DistanceBasedOutlierDetector(kernel, threshold = 0.0)
    }
  }

  test("DistanceBasedOutlierDetector with KL kernel") {
    val kernel   = new KLDivergenceKernel(smoothing = 1e-10)
    val detector = new DistanceBasedOutlierDetector(kernel, threshold = 3.0)

    // Positive data for KL
    val df = Seq(
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(0.2, 0.2)),
      Tuple1(Vectors.dense(0.15, 0.15)),
      Tuple1(Vectors.dense(10.0, 10.0)) // Outlier
    ).toDF("features")

    val centers = Array(Vectors.dense(0.15, 0.15))
    val result  = detector.detectOutliers(df, centers, "features")

    val outliers = result.filter($"is_outlier" === true).count()
    outliers should be >= 1L
  }

  // ========== TrimmedOutlierDetector Tests ==========

  test("TrimmedOutlierDetector trims specified fraction") {
    val kernel   = new SquaredEuclideanKernel()
    val detector = new TrimmedOutlierDetector(kernel, trimFraction = 0.2)

    val df      = testDataWithOutlier()
    val centers = singleCenterAtOrigin()

    val result = detector.detectOutliers(df, centers, "features")

    // 20% of 5 points = 1 point trimmed
    val outliers = result.filter($"is_outlier" === true).count()
    outliers shouldBe 1
  }

  test("TrimmedOutlierDetector with 0 trim fraction marks no outliers") {
    val kernel   = new SquaredEuclideanKernel()
    val detector = new TrimmedOutlierDetector(kernel, trimFraction = 0.0)

    val df      = testDataWithOutlier()
    val centers = singleCenterAtOrigin()

    val result   = detector.detectOutliers(df, centers, "features")
    val outliers = result.filter($"is_outlier" === true).count()

    // No trimming, no outliers (threshold is MaxValue)
    outliers shouldBe 0
  }

  test("TrimmedOutlierDetector trim fraction validation") {
    val kernel = new SquaredEuclideanKernel()

    an[IllegalArgumentException] should be thrownBy {
      new TrimmedOutlierDetector(kernel, trimFraction = -0.1)
    }

    an[IllegalArgumentException] should be thrownBy {
      new TrimmedOutlierDetector(kernel, trimFraction = 0.6)
    }

    // Valid edge cases
    noException should be thrownBy {
      new TrimmedOutlierDetector(kernel, trimFraction = 0.0)
    }
    noException should be thrownBy {
      new TrimmedOutlierDetector(kernel, trimFraction = 0.5)
    }
  }

  // ========== OutlierMode Tests ==========

  test("OutlierMode.fromString parses valid modes") {
    OutlierMode.fromString("trim") shouldBe OutlierMode.Trim
    OutlierMode.fromString("noise_cluster") shouldBe OutlierMode.NoiseCluster
    OutlierMode.fromString("m_estimator") shouldBe OutlierMode.MEstimator
  }

  test("OutlierMode.fromString is case-insensitive") {
    OutlierMode.fromString("TRIM") shouldBe OutlierMode.Trim
    OutlierMode.fromString("Noise_Cluster") shouldBe OutlierMode.NoiseCluster
  }

  test("OutlierMode.fromString rejects invalid modes") {
    an[IllegalArgumentException] should be thrownBy {
      OutlierMode.fromString("invalid")
    }
  }

  test("OutlierMode.name returns correct string") {
    OutlierMode.Trim.name shouldBe "trim"
    OutlierMode.NoiseCluster.name shouldBe "noise_cluster"
    OutlierMode.MEstimator.name shouldBe "m_estimator"
  }

  // ========== OutlierDetector Factory Tests ==========

  test("OutlierDetector.create creates correct detector type") {
    val kernel = new SquaredEuclideanKernel()

    val trimDetector = OutlierDetector.create(OutlierMode.Trim, kernel, 0.1)
    trimDetector shouldBe a[TrimmedOutlierDetector]

    val noiseDetector = OutlierDetector.create(OutlierMode.NoiseCluster, kernel, 3.0)
    noiseDetector shouldBe a[DistanceBasedOutlierDetector]

    val mEstDetector = OutlierDetector.create(OutlierMode.MEstimator, kernel, 3.0)
    mEstDetector shouldBe a[DistanceBasedOutlierDetector]
  }

  // ========== RobustCenterUpdate Tests ==========

  test("RobustCenterUpdate.create creates correct strategy") {
    val trimUpdate = RobustCenterUpdate.create(OutlierMode.Trim)
    trimUpdate shouldBe a[TrimmedCenterUpdate]

    val noiseUpdate = RobustCenterUpdate.create(OutlierMode.NoiseCluster)
    noiseUpdate shouldBe a[TrimmedCenterUpdate]

    val mEstUpdate = RobustCenterUpdate.create(OutlierMode.MEstimator, 1.5)
    mEstUpdate shouldBe a[MEstimatorCenterUpdate]
  }

  test("MEstimatorCenterUpdate huber delta validation") {
    an[IllegalArgumentException] should be thrownBy {
      new MEstimatorCenterUpdate(huberDelta = 0.0)
    }

    an[IllegalArgumentException] should be thrownBy {
      new MEstimatorCenterUpdate(huberDelta = -1.0)
    }
  }

  // ========== Integration Tests ==========

  test("Outlier detection preserves original columns") {
    val kernel   = new SquaredEuclideanKernel()
    val detector = new DistanceBasedOutlierDetector(kernel, threshold = 3.0)

    val df = Seq(
      (1, Vectors.dense(0.0, 0.0)),
      (2, Vectors.dense(1.0, 1.0)),
      (3, Vectors.dense(100.0, 100.0))
    ).toDF("id", "features")

    val centers = Array(Vectors.dense(0.5, 0.5))
    val result  = detector.detectOutliers(df, centers, "features")

    result.columns should contain("id")
    result.columns should contain("features")
    result.count() shouldBe 3
  }

  test("Multiple centers - distance to nearest") {
    val kernel   = new SquaredEuclideanKernel()
    val detector = new DistanceBasedOutlierDetector(kernel, threshold = 5.0)

    val df = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),   // Near center 1
      Tuple1(Vectors.dense(10.0, 10.0)), // Near center 2
      Tuple1(Vectors.dense(50.0, 50.0))  // Far from both (outlier)
    ).toDF("features")

    val centers = Array(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(10.0, 10.0)
    )

    val result   = detector.detectOutliers(df, centers, "features")
    val outliers = result.filter($"is_outlier" === true).count()

    outliers shouldBe 1
  }
}
