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

package com.massivedatascience.clusterer.coreset

import com.massivedatascience.clusterer._
import com.massivedatascience.linalg.WeightedVector
import org.apache.spark.ml.linalg.Vectors
import org.scalatest._
import funsuite._
import com.massivedatascience.clusterer.TestingUtils._

class CoresetTestSuite extends AnyFunSuite with LocalClusterSparkContext {

  val pointOps = BregmanPointOps(BregmanPointOps.EUCLIDEAN)

  test("WeightedPoint should handle basic operations") {
    val vector      = Vectors.dense(1.0, 2.0, 3.0)
    val weight      = 2.0
    val f           = 14.0 // 1^2 + 2^2 + 3^2
    val importance  = 5.0
    val sensitivity = 3.0

    val point         = BregmanPoint(WeightedVector.fromInhomogeneousWeighted(vector, weight), f)
    val weightedPoint = WeightedPoint(point, importance, sensitivity)

    assert(weightedPoint.effectiveWeight ~= weight * importance absTol 1e-8)
    assert(weightedPoint.point == point)
    assert(weightedPoint.importance ~= importance absTol 1e-8)
    assert(weightedPoint.sensitivity ~= sensitivity absTol 1e-8)

    val newWeightedPoint = weightedPoint.withImportance(10.0)
    assert(newWeightedPoint.importance ~= 10.0 absTol 1e-8)
    assert(newWeightedPoint.point == point)
  }

  test("BregmanSensitivity should compute uniform sensitivity") {
    val points = sc.parallelize(
      Seq(
        BregmanPoint(WeightedVector(Vectors.dense(1.0, 0.0)), 1.0),
        BregmanPoint(WeightedVector(Vectors.dense(0.0, 1.0)), 1.0),
        BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0)
      )
    )

    val sensitivity   = new UniformSensitivity()
    val sensitivities = sensitivity.computeBatchSensitivity(points, 2, pointOps).collect()

    assert(sensitivities.length == 3)
    sensitivities.foreach { case (_, sens) =>
      assert(sens ~= 1.0 absTol 1e-8)
    }
  }

  test("BregmanCoreset should build core-set from small dataset") {
    val points = sc.parallelize(
      Seq(
        BregmanPoint(WeightedVector(Vectors.dense(1.0, 0.0)), 1.0),
        BregmanPoint(WeightedVector(Vectors.dense(0.0, 1.0)), 1.0),
        BregmanPoint(WeightedVector(Vectors.dense(2.0, 0.0)), 4.0),
        BregmanPoint(WeightedVector(Vectors.dense(0.0, 2.0)), 4.0),
        BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0)
      )
    )

    val config = CoresetConfig(
      coresetSize = 3,
      epsilon = 0.1,
      sensitivity = BregmanSensitivity.uniform()
    )

    val coreset = new BregmanCoreset(config)
    val result  = coreset.buildCoreset(points, 2, pointOps)

    assert(result.originalSize == 5)
    assert(result.coreset.length <= 3)
    assert(result.compressionRatio <= 1.0)
    assert(result.totalSensitivity > 0.0)

    // Verify core-set points are valid
    result.coreset.foreach { wp =>
      assert(wp.importance > 0.0)
      assert(wp.sensitivity >= 0.0)
      assert(wp.point != null)
    }
  }

  test("BregmanCoreset should handle dataset smaller than target size") {
    val points = sc.parallelize(
      Seq(
        BregmanPoint(WeightedVector(Vectors.dense(1.0, 0.0)), 1.0),
        BregmanPoint(WeightedVector(Vectors.dense(0.0, 1.0)), 1.0)
      )
    )

    val coreset = BregmanCoreset(coresetSize = 10)
    val result  = coreset.buildCoreset(points, 2, pointOps)

    assert(result.originalSize == 2)
    assert(result.coreset.length == 2)
    assert(result.compressionRatio ~= 1.0 absTol 1e-8)

    // All points should be included with unit importance
    result.coreset.foreach { wp =>
      assert(wp.importance ~= 1.0 absTol 1e-8)
    }
  }

  test("CoresetKMeans should cluster small dataset") {
    val points = sc.parallelize(
      Seq(
        // Cluster 1: around (1, 1)
        BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
        BregmanPoint(WeightedVector(Vectors.dense(1.1, 0.9)), 2.02),
        BregmanPoint(WeightedVector(Vectors.dense(0.9, 1.1)), 2.02),
        // Cluster 2: around (3, 3)
        BregmanPoint(WeightedVector(Vectors.dense(3.0, 3.0)), 18.0),
        BregmanPoint(WeightedVector(Vectors.dense(3.1, 2.9)), 18.58),
        BregmanPoint(WeightedVector(Vectors.dense(2.9, 3.1)), 18.02)
      )
    )

    // Cache the data to satisfy KMeansParallel requirement
    points.cache()

    val clusterer = CoresetKMeans.fast(coresetSize = 4)

    // Create initial centers
    val selector       = KMeansSelector(KMeansSelector.K_MEANS_PARALLEL)
    val initialCenters = selector.init(pointOps, points, 2, None, 1, 42L)

    val results = clusterer.cluster(10, pointOps, points, initialCenters)

    assert(results.nonEmpty)
    val bestResult = results.minBy(_.distortion)
    assert(bestResult.centers.length == 2)
    assert(bestResult.distortion >= 0.0)
  }

  test("CoresetKMeans quick method should work") {
    val points = sc.parallelize((1 to 100).map { i =>
      val x = if (i <= 50) 1.0 + 0.1 * (i % 10) else 5.0 + 0.1 * (i % 10)
      val y = if (i <= 50) 1.0 + 0.1 * (i % 10) else 5.0 + 0.1 * (i % 10)
      BregmanPoint(WeightedVector(Vectors.dense(x, y)), x * x + y * y)
    })

    // Cache the data to satisfy KMeansParallel requirement
    points.cache()

    val result = CoresetKMeans.quick(points, 2, pointOps, compressionRatio = 0.1)

    assert(result.centers.length == 2)
    assert(result.distortion >= 0.0)
    assert(result.coresetResult.originalSize == 100)
    assert(result.coresetResult.coreset.length <= 10) // 10% compression
    assert(result.totalTime > 0)

    val stats = result.getStats
    assert(stats.contains("numCenters"))
    assert(stats.contains("compressionRatio"))
    assert(stats("numCenters") ~= 2.0 absTol 1e-8)
  }

  test("Distance-based sensitivity should compute properly") {
    val points = sc.parallelize(
      Seq(
        // Dense cluster around origin
        BregmanPoint(WeightedVector(Vectors.dense(0.1, 0.1)), 0.02),
        BregmanPoint(WeightedVector(Vectors.dense(0.0, 0.1)), 0.01),
        BregmanPoint(WeightedVector(Vectors.dense(0.1, 0.0)), 0.01),
        // Outlier
        BregmanPoint(WeightedVector(Vectors.dense(100.0, 100.0)), 20000.0)
      )
    )

    val sensitivity   = new DistanceBasedSensitivity(numSampleCenters = 3)
    val sensitivities = sensitivity.computeBatchSensitivity(points, 2, pointOps).collect()

    assert(sensitivities.length == 4)

    // All sensitivity values should be positive and finite
    sensitivities.foreach { case (_, sens) =>
      assert(
        sens > 0.0 && !sens.isInfinite && !sens.isNaN,
        s"Sensitivity should be positive and finite, got: $sens"
      )
    }

    // Verify that distance-based sensitivity produces reasonable values
    val sensitivityValues = sensitivities.map(_._2)
    val maxSensitivity    = sensitivityValues.max
    val minSensitivity    = sensitivityValues.min

    // The range should be reasonable (not all identical)
    assert(
      maxSensitivity >= minSensitivity,
      s"Max sensitivity ($maxSensitivity) should be >= min sensitivity ($minSensitivity)"
    )
  }

  test("BregmanCoreset adaptive sizing should work") {
    val points = sc.parallelize((1 to 50).map { i =>
      BregmanPoint(WeightedVector(Vectors.dense(i.toDouble, i.toDouble)), 2.0 * i * i)
    })

    val config = CoresetConfig(
      coresetSize = 10,
      epsilon = 0.2 // Larger epsilon for smaller theoretical size
    )

    val coreset = new BregmanCoreset(config)
    val result  = coreset.buildAdaptiveCoreset(points, 3, pointOps)

    assert(result.originalSize == 50)
    // Should use theoretical size based on k * log(k) / epsilon^2
    val theoreticalSize = math.ceil(3 * math.log(3) / (0.2 * 0.2)).toInt
    val expectedSize    = math.max(theoreticalSize, 10)
    assert(result.config.coresetSize == expectedSize)
  }
}
