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

package com.rincaro.clusterer

import com.rincaro.clusterer.base.KMeans
import com.rincaro.clusterer.metrics.FastEuOps
import org.scalatest.FunSuite


class KMeansSuite extends FunSuite with LocalSparkContext {

  val EPSILON = 1e-4


  def prettyPrint(point: Array[Double]): String = point.mkString("(", ", ", ")")

  def prettyPrint(points: Array[Array[Double]]): String = {
    points.map(prettyPrint).mkString("(", "; ", ")")
  }

  // L1 distance between two points
  def distance1(v1: Array[Double], v2: Array[Double]): Double = {
    v1.zip(v2).map {
      case (a, b) => math.abs(a - b)
    }.max
  }

  // Assert that two vectors are equal within tolerance EPSILON
  def assertEqual(v1: Array[Double], v2: Array[Double]) {
    def errorMessage = prettyPrint(v1) + " did not equal " + prettyPrint(v2)
    assert(v1.length == v2.length, errorMessage)
    assert(distance1(v1, v2) <= EPSILON, errorMessage)
  }

  // Assert that two sets of points are equal, within EPSILON tolerance
  def assertSetsEqual(set1: Array[Array[Double]], set2: Array[Array[Double]]) {
    def errorMessage = prettyPrint(set1) + " did not equal " + prettyPrint(set2)
    assert(set1.length == set2.length, errorMessage)
    for (v <- set1) {
      val closestDistance = set2.map(w => distance1(v, w)).min
      if (closestDistance > EPSILON) {
        fail(errorMessage)
      }
    }
    for (v <- set2) {
      val closestDistance = set1.map(w => distance1(v, w)).min
      if (closestDistance > EPSILON) {
        fail(errorMessage)
      }
    }
  }

  test("no distinct points") {
    val data = sc.parallelize(
      Array[(Option[String],Array[Double])](
        (Some("first"), Array(1.0, 2.0, 3.0)),
        (Some("second"), Array(1.0, 2.0, 3.0)),
        (Some("third"), Array(1.0, 2.0, 3.0))))

    // Make sure code runs.
    val model = KMeans.train(new FastEuOps[String])(data, k=2, maxIterations=1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assert(model.clusterCenters.size === 1)
  }

  test("more clusters than points") {
    val data = sc.parallelize(
      Array[(Option[String],Array[Double])](
        (Some("first"), Array(1.0, 2.0, 3.0)),
        (Some("second"), Array(1.0, 3.0, 4.0))))

    // Make sure code runs.
    val model = KMeans.train(new FastEuOps[String])(data, k=3, maxIterations=1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assert(model.clusterCenters.size === 2)
  }

  test("single cluster") {
    val data = sc.parallelize(
      Array[(Option[String],Array[Double])](
        (Some("first"), Array(1.0, 2.0, 6.0)),
      (Some("second"), Array(1.0, 3.0, 0.0)),
      (Some("third"), Array(1.0, 4.0, 6.0))
    ))

    // No matter how many runs or iterations we use, we should get one cluster,
    // centered at the mean of the points

    var model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 1, runs = 1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))

    model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 2, runs = 1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))

    model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 5, runs = 1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))

    model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 1, runs = 5, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))

    model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 1, runs = 5, initializationMode = KMeans.RANDOM)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))

    model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 1, runs = 1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))
  }

  test("single cluster with big dataset") {
    val smallData = Array[(Option[String],Array[Double])](
      (Some("first"), Array(1.0, 2.0, 6.0)),
      (Some("second"), Array(1.0, 3.0, 0.0)),
      (Some("third"), Array(1.0, 4.0, 6.0)))

    val data = sc.parallelize((1 to 100).flatMap(_ => smallData), 4)

    // No matter how many runs or iterations we use, we should get one cluster,
    // centered at the mean of the points

    var model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 1, runs = 1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))

    model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 1, runs = 2, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))

    model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 5, runs = 1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))

    model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 1, runs = 5, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))

    model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 1, runs = 5, initializationMode = KMeans.RANDOM)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))

    model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 1, runs = 1, initializationMode = KMeans.RANDOM)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))

    model = KMeans.train(new FastEuOps[String])(data, k = 1, maxIterations = 1, runs = 1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, Array(Array(1.0, 3.0, 4.0)))
  }


  test("k-means|| initialization on FastEuOps") {
    val points = Array[(Option[String],Array[Double])](
      (Some("first"), Array(1.0, 2.0, 6.0)),
      (Some("second"), Array(1.0, 3.0, 0.0)),
      (Some("third"), Array(1.0, 4.0, 6.0)),
      (Some("fourth"), Array(1.0, 2.0, 1.0)),
      (Some("fifth"), Array(1.0, 1.0, 1.0))
    )
    val rdd = sc.parallelize(points)

    // K-means|| initialization should place all clusters into distinct centers because
    // it will make at least five passes, and it will give non-zero probability to each
    // unselected point as long as it hasn't yet selected all of them

    var model = KMeans.train(new FastEuOps[String])(rdd, k = 5, maxIterations = 1, runs = 1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, points.map(_._2))

    // Iterations of Lloyd's should not change the answer either
    model = KMeans.train(new FastEuOps[String])(rdd, k = 5, maxIterations = 10, runs = 1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, points.map(_._2))

    // Neither should more runs
    model = KMeans.train(new FastEuOps[String])(rdd, k = 5, maxIterations = 10, runs = 5, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.inh}, points.map(_._2))
  }


  test("k-means|| initialization on FastEuOps one run, one iteration") {
    val points = Array[(Option[String],Array[Double])](
      (Some("first"), Array(1.0, 2.0, 6.0)),
      (Some("second"), Array(1.0, 3.0, 8.0)),
      (Some("third"), Array(1.0, 4.0, 6.0)),
      (Some("fourth"), Array(1.0, 2.0, 1.0)),
      (Some("fifth"), Array(1.0, 1.0, 2.0))
    )
    val rdd = sc.parallelize(points)

    // K-means|| initialization should place all clusters into distinct centers because
    // it will make at least five passes, and it will give non-zero probability to each
    // unselected point as long as it hasn't yet selected all of them

    var model = KMeans.train(new FastEuOps[String])(rdd, k = 5, maxIterations = 1, runs = 1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.raw}, points.map(_._2))

  }

  test("k-means|| initialization on FastEuOps on multiple runs or multiple iterations") {

    val points = Array[(Option[String],Array[Double])](
      (Some("first"), Array(1.0, 2.0, 6.0)),
      (Some("second"), Array(1.0, 3.0, 8.0)),
      (Some("third"), Array(1.0, 4.0, 6.0)),
      (Some("fourth"), Array(1.0, 2.0, 1.0)),
      (Some("fifth"), Array(1.0, 1.0, 1.0)))
    val rdd = sc.parallelize(points)

    // K-means|| initialization should place all clusters into distinct centers because
    // it will make at least five passes, and it will give non-zero probability to each
    // unselected point as long as it hasn't yet selected all of them


    // Iterations of Lloyd's should not change the answer either
    var model = KMeans.train(new FastEuOps[String])(rdd, k = 5, maxIterations = 10, runs = 1, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.raw}, points.map(_._2))

    // Neither should more runs
    model = KMeans.train(new FastEuOps[String])(rdd, k = 5, maxIterations = 10, runs = 5, initializationMode = KMeans.K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters.map{_.raw}, points.map(_._2))
  }

}
