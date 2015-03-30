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

import com.massivedatascience.linalg.WeightedVector
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.FunSuite

class KMeansPlusPlusSuite extends FunSuite with LocalSparkContext {

  test("k-means plus plus chooses only point with non-zero weight") {

    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val implementation = new KMeansPlusPlus(ops)
    val seed = 0
    val data = Array(
      WeightedVector(Vectors.dense(1.0, 2.0, 6.0), 2.0),
      WeightedVector(Vectors.dense(1.0, 3.0, 0.0), 4.0),
      WeightedVector(Vectors.dense(1.0, 4.0, 6.0), 8.0)
    )

    val candidateCenters = data.map(ops.toCenter)
    val weights = Array(1.0, 0.0, 0.0)
    val totalRequested = 1
    val perRound = 1
    val numPreselected = 0

    val goodCenters = implementation.goodCenters(
      seed,
      candidateCenters,
      weights,
      totalRequested,
      perRound,
      numPreselected)

    assert(goodCenters.length == 1)
    assert(goodCenters.head == candidateCenters(0))
  }

  test("k-means plus plus keep pre-selected centers") {

    val ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
    val implementation = new KMeansPlusPlus(ops)
    val seed = 0
    val data = Array(
      WeightedVector(Vectors.dense(1.0, 2.0, 6.0), 2.0),
      WeightedVector(Vectors.dense(1.0, 3.0, 0.0), 4.0),
      WeightedVector(Vectors.dense(1.0, 4.0, 6.0), 8.0),
      WeightedVector(Vectors.dense(1.0, 4.0, 10.0), 10.0)

    )

    val candidateCenters = data.map(ops.toCenter)
    val weights = Array(0.0, 0.0, 1.0, 0.0)
    val totalRequested = 2
    val perRound = 1
    val numPreselected = 1

    val goodCenters = implementation.goodCenters(
      seed,
      candidateCenters,
      weights,
      totalRequested,
      perRound,
      numPreselected)

    assert(goodCenters.length == 2)
    assert(goodCenters.head == candidateCenters(0))
    assert(goodCenters(1) == candidateCenters(2))
  }

}