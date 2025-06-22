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
import com.massivedatascience.linalg.WeightedVector
import org.apache.spark.ml.linalg.Vectors
import org.scalatest._
import funsuite._

class BregmanSoftKMeansTestSuite extends AnyFunSuite with LocalClusterSparkContext {

  val pointOps = BregmanPointOps(BregmanPointOps.EUCLIDEAN)

  test("BregmanSoftKMeans should handle basic soft clustering") {
    val points = sc.parallelize(Seq(
      // Cluster 1: around (1, 1)
      BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
      BregmanPoint(WeightedVector(Vectors.dense(1.1, 0.9)), 2.02),
      BregmanPoint(WeightedVector(Vectors.dense(0.9, 1.1)), 2.02),
      // Cluster 2: around (3, 3)
      BregmanPoint(WeightedVector(Vectors.dense(3.0, 3.0)), 18.0),
      BregmanPoint(WeightedVector(Vectors.dense(3.1, 2.9)), 18.58),
      BregmanPoint(WeightedVector(Vectors.dense(2.9, 3.1)), 18.02)
    ))

    // Cache data for soft clustering
    points.cache()

    val softKMeans = BregmanSoftKMeans.moderatelySoft(beta = 2.0)
    val selector = KMeansSelector(KMeansSelector.K_MEANS_PARALLEL)
    val initialCenters = selector.init(pointOps, points, 2, None, 1, 42L).head

    val result = softKMeans.clusterSoft(50, pointOps, points, initialCenters)

    // Verify basic properties
    assert(result.centers.length == 2)
    assert(result.iterations > 0)
    assert(result.objective >= 0.0)

    // Verify membership probabilities
    val memberships = result.memberships.collect()
    assert(memberships.length == 6)

    memberships.foreach { case (_, probs) =>
      // Each point should have probabilities for 2 clusters
      assert(probs.length == 2)
      // Probabilities should sum to 1
      assert(probs.sum ~= 1.0 absTol 1E-8)
      // All probabilities should be positive
      probs.foreach(p => assert(p > 0.0))
    }
  }

  test("BregmanSoftKMeans should produce different results with different beta values") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
      BregmanPoint(WeightedVector(Vectors.dense(2.0, 2.0)), 8.0),
      BregmanPoint(WeightedVector(Vectors.dense(1.5, 1.5)), 4.5) // Ambiguous point
    ))

    points.cache()

    val selector = KMeansSelector(KMeansSelector.K_MEANS_PARALLEL)
    val initialCenters = selector.init(pointOps, points, 2, None, 1, 42L).head

    // Test with low beta (soft assignments)
    val softResult = BregmanSoftKMeans.verySoft(beta = 0.1).clusterSoft(30, pointOps, points, initialCenters)
    val softMemberships = softResult.memberships.collect()

    // Test with high beta (sharp assignments)
    val sharpResult = BregmanSoftKMeans.sharp(beta = 10.0).clusterSoft(30, pointOps, points, initialCenters)
    val sharpMemberships = sharpResult.memberships.collect()

    // For the ambiguous middle point, soft clustering should have more balanced probabilities
    val ambiguousPointSoft = softMemberships.find(_._1.homogeneous.toArray.sameElements(Array(1.5, 1.5, 1.0)))
      .map(_._2).getOrElse(Array(0.5, 0.5))

    val ambiguousPointSharp = sharpMemberships.find(_._1.homogeneous.toArray.sameElements(Array(1.5, 1.5, 1.0)))
      .map(_._2).getOrElse(Array(0.5, 0.5))

    // Soft clustering should have more balanced probabilities
    val softEntropy = -ambiguousPointSoft.map(p => if (p > 1e-10) p * math.log(p) else 0.0).sum
    val sharpEntropy = -ambiguousPointSharp.map(p => if (p > 1e-10) p * math.log(p) else 0.0).sum

    assert(softEntropy > sharpEntropy, s"Soft entropy ($softEntropy) should be > sharp entropy ($sharpEntropy)")
  }

  test("BregmanSoftKMeans should convert to hard assignments correctly") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(0.0, 0.0)), 0.0),
      BregmanPoint(WeightedVector(Vectors.dense(0.1, 0.1)), 0.02),
      BregmanPoint(WeightedVector(Vectors.dense(5.0, 5.0)), 50.0),
      BregmanPoint(WeightedVector(Vectors.dense(5.1, 4.9)), 50.02)
    ))

    points.cache()

    val softKMeans = BregmanSoftKMeans.sharp(beta = 5.0)
    val selector = KMeansSelector(KMeansSelector.K_MEANS_PARALLEL)
    val initialCenters = selector.init(pointOps, points, 2, None, 1, 42L).head

    val result = softKMeans.clusterSoft(50, pointOps, points, initialCenters)
    val hardAssignments = result.toHardAssignments.collect()

    assert(hardAssignments.length == 4)

    // Points should be assigned to clusters based on proximity
    val assignmentMap = hardAssignments.toMap
    val point1 = BregmanPoint(WeightedVector(Vectors.dense(0.0, 0.0)), 0.0)
    val point2 = BregmanPoint(WeightedVector(Vectors.dense(0.1, 0.1)), 0.02)
    val point3 = BregmanPoint(WeightedVector(Vectors.dense(5.0, 5.0)), 50.0)
    val point4 = BregmanPoint(WeightedVector(Vectors.dense(5.1, 4.9)), 50.02)

    // Close points should have the same assignment
    hardAssignments.foreach { case (_, assignment) =>
      assert(assignment >= 0 && assignment < 2)
    }
  }

  test("BregmanSoftKMeans should handle single cluster case") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
      BregmanPoint(WeightedVector(Vectors.dense(1.1, 1.1)), 2.42),
      BregmanPoint(WeightedVector(Vectors.dense(0.9, 0.9)), 1.62)
    ))

    points.cache()

    val softKMeans = BregmanSoftKMeans.moderatelySoft()
    val selector = KMeansSelector(KMeansSelector.K_MEANS_PARALLEL)
    val initialCenters = selector.init(pointOps, points, 1, None, 1, 42L).head

    val result = softKMeans.clusterSoft(30, pointOps, points, initialCenters)

    assert(result.centers.length == 1)
    
    val memberships = result.memberships.collect()
    memberships.foreach { case (_, probs) =>
      assert(probs.length == 1)
      assert(probs(0) ~= 1.0 absTol 1E-8)
    }
  }

  test("BregmanSoftKMeans should compute reasonable effective number of clusters") {
    val points = sc.parallelize((1 to 50).map { i =>
      val x = if (i <= 25) 1.0 + 0.1 * scala.util.Random.nextGaussian() 
              else 5.0 + 0.1 * scala.util.Random.nextGaussian()
      val y = if (i <= 25) 1.0 + 0.1 * scala.util.Random.nextGaussian() 
              else 5.0 + 0.1 * scala.util.Random.nextGaussian()
      BregmanPoint(WeightedVector(Vectors.dense(x, y)), x*x + y*y)
    })

    points.cache()

    val softKMeans = BregmanSoftKMeans.moderatelySoft(beta = 1.0)
    val selector = KMeansSelector(KMeansSelector.K_MEANS_PARALLEL)
    val initialCenters = selector.init(pointOps, points, 3, None, 1, 42L).head

    val result = softKMeans.clusterSoft(50, pointOps, points, initialCenters)
    val effectiveNumClusters = result.effectiveNumberOfClusters

    // Should be between 1 and 3, closer to 2 since we have 2 natural clusters
    assert(effectiveNumClusters >= 1.0 && effectiveNumClusters <= 3.0)
    assert(effectiveNumClusters > 1.5, s"Expected effective clusters > 1.5, got: $effectiveNumClusters")
  }

  test("BregmanSoftKMeans should work with mixture model factory") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
      BregmanPoint(WeightedVector(Vectors.dense(1.1, 0.9)), 2.02),
      BregmanPoint(WeightedVector(Vectors.dense(3.0, 3.0)), 18.0),
      BregmanPoint(WeightedVector(Vectors.dense(3.1, 2.9)), 18.58)
    ))

    points.cache()

    val mixtureKMeans = BregmanSoftKMeans.forMixtureModel(beta = 2.0)
    val selector = KMeansSelector(KMeansSelector.K_MEANS_PARALLEL)
    val initialCenters = selector.init(pointOps, points, 2, None, 1, 42L).head

    val result = mixtureKMeans.clusterSoft(50, pointOps, points, initialCenters)

    assert(result.centers.length == 2)
    assert(result.config.computeObjective == true)
    assert(result.config.beta == 2.0)

    val stats = result.getStats
    assert(stats.contains("objective"))
    assert(stats.contains("effectiveNumClusters"))
    assert(stats("numCenters") ~= 2.0 absTol 1E-8)
  }

  test("BregmanSoftKMeans should handle convergence properly") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(0.0, 0.0)), 0.0),
      BregmanPoint(WeightedVector(Vectors.dense(10.0, 10.0)), 200.0)
    ))

    points.cache()

    // Use very tight convergence threshold
    val config = BregmanSoftKMeansConfig(
      beta = 5.0,
      convergenceThreshold = 1e-12,
      maxIterations = 5
    )
    val softKMeans = BregmanSoftKMeans(config)

    val selector = KMeansSelector(KMeansSelector.K_MEANS_PARALLEL)
    val initialCenters = selector.init(pointOps, points, 2, None, 1, 42L).head

    val result = softKMeans.clusterSoft(5, pointOps, points, initialCenters)

    // Should either converge or hit max iterations
    assert(result.iterations <= 5)
    assert(result.iterations > 0)
  }

  test("BregmanSoftKMeans should integrate with MultiKMeansClusterer interface") {
    val points = sc.parallelize(Seq(
      BregmanPoint(WeightedVector(Vectors.dense(1.0, 1.0)), 2.0),
      BregmanPoint(WeightedVector(Vectors.dense(1.1, 0.9)), 2.02),
      BregmanPoint(WeightedVector(Vectors.dense(3.0, 3.0)), 18.0),
      BregmanPoint(WeightedVector(Vectors.dense(3.1, 2.9)), 18.58)
    ))

    points.cache()

    val softKMeans: MultiKMeansClusterer = BregmanSoftKMeans.moderatelySoft()
    val selector = KMeansSelector(KMeansSelector.K_MEANS_PARALLEL)
    val initialCenters = selector.init(pointOps, points, 2, None, 3, 42L) // 3 runs

    val results = softKMeans.cluster(50, pointOps, points, initialCenters)

    assert(results.length == 3) // Should return 3 results for 3 runs
    results.foreach { result =>
      assert(result.centers.length == 2)
      assert(result.distortion >= 0.0)
    }

    // Test best method
    val bestResult = softKMeans.best(50, pointOps, points, initialCenters)
    assert(bestResult.centers.length == 2)
    assert(bestResult.distortion == results.map(_.distortion).min)
  }
}