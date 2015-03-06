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

import com.massivedatascience.clusterer.KMeans.RunConfig
import com.massivedatascience.divergence.BregmanDivergence
import com.massivedatascience.linalg.WeightedVector
import com.massivedatascience.transforms.Embedding
import com.massivedatascience.transforms.Embedding._

import scala.util.Random

import org.scalatest.FunSuite

import org.apache.spark.mllib.linalg.{ Vector, Vectors }

import com.massivedatascience.clusterer.TestingUtils._
import com.massivedatascience.clusterer.BregmanPointOps._

class KMeansSuite extends FunSuite with LocalSparkContext {

  import com.massivedatascience.clusterer.KMeansSelector._

  test("sparse vector iterator") {

    import com.massivedatascience.linalg._

    val v = Vectors.sparse(10, Seq((2, 30.2), (4, 42.0)))

    val iter = v.iterator

    println(iter.toString)

    assert(iter.hasNext)
    val firstIndex = iter.index
    assert(firstIndex == 2)
    val firstValue = iter.value
    assert(firstValue == 30.2)

    assert(iter.hasNext)
    iter.advance()

    val secondIndex = iter.index
    assert(secondIndex == 4)
    val secondValue = iter.value
    assert(secondValue == 42)
    iter.advance()
    assert(!iter.hasNext)
  }

  test("sparse negative vector iterator") {

    import com.massivedatascience.linalg._

    val v = Vectors.sparse(10, Seq((2, 30.2), (4, 42.0)))

    val iter = v.negativeIterator

    assert(iter.hasNext)
    val firstIndex = iter.index
    assert(firstIndex == 2)
    val firstValue = iter.value
    assert(firstValue == -30.2)

    assert(iter.hasNext)
    iter.advance()

    val secondIndex = iter.index
    assert(secondIndex == 4)
    val secondValue = iter.value
    assert(secondValue == -42)
    iter.advance()
    assert(!iter.hasNext)

    println(iter.toString)
  }

  test("dense vector iterator") {

    import com.massivedatascience.linalg._

    val v = Vectors.dense(5.0, 9.0)

    val iter = v.iterator

    println(iter.toString)

    assert(iter.hasNext)
    val firstIndex = iter.index
    assert(firstIndex == 0)
    val firstValue = iter.value
    assert(firstValue == 5.0)

    assert(iter.hasNext)
    iter.advance()

    val secondIndex = iter.index
    assert(secondIndex == 1)
    val secondValue = iter.value
    assert(secondValue == 9.0)
    iter.advance()
    assert(!iter.hasNext)
  }

  test("dense negative vector iterator") {

    import com.massivedatascience.linalg._

    val v = Vectors.dense(5.0, 9.0)

    val iter = v.negativeIterator

    println(iter.toString)

    assert(iter.hasNext)
    val firstIndex = iter.index
    assert(firstIndex == 0)
    val firstValue = iter.value
    assert(firstValue == -5.0)

    assert(iter.hasNext)
    iter.advance()

    val secondIndex = iter.index
    assert(secondIndex == 1)
    val secondValue = iter.value
    assert(secondValue == -9.0)
    iter.advance()
    assert(!iter.hasNext)
  }

  test("coverage") {

    val seed = 0
    val r = new Random(seed)

    val data = sc.parallelize(Array.fill(1000)(Vectors.dense(Array.fill(20)(r.nextDouble()))))

    KMeans.train(data, k = 20, maxIterations = 10)

    KMeans.train(data, k = 20, maxIterations = 10, runs = 2)

    KMeans.train(data, k = 20, maxIterations = 10, runs = 1, mode = KMeansSelector.RANDOM)

    KMeans.train(data, k = 20, maxIterations = 10, runs = 1, distanceFunctionNames = Seq(BregmanPointOps.EUCLIDEAN))

    KMeans.train(data, k = 20, maxIterations = 10, runs = 1,
      distanceFunctionNames = Seq(BregmanPointOps.EUCLIDEAN),
      clustererName = MultiKMeansClusterer.CHANGE_TRACKING)

    KMeans.timeSeriesTrain(
      new RunConfig(20, 1, 0, 10),
      data.map(WeightedVector.apply),
      KMeansSelector(KMeansSelector.K_MEANS_PARALLEL),
      BregmanPointOps(BregmanPointOps.EUCLIDEAN),
      MultiKMeansClusterer(MultiKMeansClusterer.COLUMN_TRACKING),
      Embedding(HAAR_EMBEDDING))

    KMeans.train(data, k = 20, maxIterations = 10, runs = 1, clustererName = MultiKMeansClusterer.MINI_BATCH_10)
  }

  test("single cluster") {
    val data = sc.parallelize(Array(
      Vectors.dense(1.0, 2.0, 6.0),
      Vectors.dense(1.0, 3.0, 0.0),
      Vectors.dense(1.0, 4.0, 6.0)
    ))

    val center = Vectors.dense(1.0, 3.0, 4.0)

    // No matter how many runs or iterations we use, we should get one cluster,
    // centered at the mean of the points

    var model = KMeans.train(data, k = 1, maxIterations = 1)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 2)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = RANDOM)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(
      data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL)
    assert(model.clusterCenters.head ~== center absTol 1E-5)
  }

  test("no distinct points") {
    val data = sc.parallelize(
      Array(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(1.0, 2.0, 3.0)),
      2)
    val center = Vectors.dense(1.0, 2.0, 3.0)

    // Make sure code runs.
    var model = KMeans.train(data, k = 2, maxIterations = 1)
    assert(model.clusterCenters.size === 1)
  }

  test("more clusters than points") {
    val data = sc.parallelize(
      Array(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(1.0, 3.0, 4.0)),
      2)

    // Make sure code runs.
    var model = KMeans.train(data, k = 3, maxIterations = 1)
    assert(model.clusterCenters.size === 2, s"${model.clusterCenters.size} != 2")
  }

  test("single cluster with big dataset") {
    val smallData = Array(
      Vectors.dense(1.0, 2.0, 6.0),
      Vectors.dense(1.0, 3.0, 0.0),
      Vectors.dense(1.0, 4.0, 6.0)
    )
    val data = sc.parallelize((1 to 100).flatMap(_ => smallData), 4)

    // No matter how many runs or iterations we use, we should get one cluster,
    // centered at the mean of the points

    val center = Vectors.dense(1.0, 3.0, 4.0)

    var model = KMeans.train(data, k = 1, maxIterations = 1)
    assert(model.clusterCenters.size === 1)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 2)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = RANDOM)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL)
    assert(model.clusterCenters.head ~== center absTol 1E-5)
  }

  test("single cluster with sparse data") {

    val n = 10000
    val data = sc.parallelize((1 to 100).flatMap { i =>
      val x = i / 1000.0
      Array(
        Vectors.sparse(n, Seq((0, 1.0 + x), (1, 2.0), (2, 6.0))),
        Vectors.sparse(n, Seq((0, 1.0 - x), (1, 2.0), (2, 6.0))),
        Vectors.sparse(n, Seq((0, 1.0), (1, 3.0 + x))),
        Vectors.sparse(n, Seq((0, 1.0), (1, 3.0 - x))),
        Vectors.sparse(n, Seq((0, 1.0), (1, 4.0), (2, 6.0 + x))),
        Vectors.sparse(n, Seq((0, 1.0), (1, 4.0), (2, 6.0 - x)))
      )
    }, 4)

    data.persist()

    // No matter how many runs or iterations we use, we should get one cluster,
    // centered at the mean of the points

    val center = Vectors.sparse(n, Seq((0, 1.0), (1, 3.0), (2, 4.0)))

    var model = KMeans.train(data, k = 1, maxIterations = 1)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 2)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 5)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = RANDOM)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL)
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL,
      distanceFunctionNames = Seq(BregmanPointOps.RELATIVE_ENTROPY))
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL,
      distanceFunctionNames = Seq(EUCLIDEAN))
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL,
      distanceFunctionNames = Seq(DISCRETE_KL))
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    model = KMeans.train(data, k = 1, maxIterations = 1, runs = 1, mode = K_MEANS_PARALLEL,
      distanceFunctionNames = Seq(SPARSE_SMOOTHED_KL))
    assert(model.clusterCenters.head ~== center absTol 1E-5)

    data.unpersist()
  }

  test("k-means|| initialization") {

    case class VectorWithCompare(x: Vector) extends Ordered[VectorWithCompare] {
      @Override def compare(that: VectorWithCompare): Int = {
        if (this.x.toArray.foldLeft[Double](0.0)((acc, x) => acc + x * x) >
          that.x.toArray.foldLeft[Double](0.0)((acc, x) => acc + x * x)) -1
        else 1
      }
    }

    val points = Seq(
      Vectors.dense(1.0, 2.0, 6.0),
      Vectors.dense(1.0, 3.0, 0.0),
      Vectors.dense(1.0, 4.0, 6.0),
      Vectors.dense(1.0, 0.0, 1.0),
      Vectors.dense(1.0, 1.0, 1.0)
    )
    val rdd = sc.parallelize(points)

    // K-means|| initialization should place all clusters into distinct centers because
    // it will make at least five passes, and it will give non-zero probability to each
    // unselected point as long as it hasn't yet selected all of them

    var model = KMeans.train(rdd, k = 5, maxIterations = 1)

    assert(model.clusterCenters.sortBy(VectorWithCompare)
      .zip(points.sortBy(VectorWithCompare)).forall(x => x._1 ~== x._2 absTol 1E-5))

    // Iterations of Lloyd's should not change the answer either
    model = KMeans.train(rdd, k = 5, maxIterations = 10)
    assert(model.clusterCenters.sortBy(VectorWithCompare)
      .zip(points.sortBy(VectorWithCompare)).forall(x => x._1 ~== x._2 absTol 1E-5))

    // Neither should more runs
    model = KMeans.train(rdd, k = 5, maxIterations = 10, runs = 5)
    assert(model.clusterCenters.sortBy(VectorWithCompare)
      .zip(points.sortBy(VectorWithCompare)).forall(x => x._1 ~== x._2 absTol 1E-5))
  }

  test("two clusters") {
    val points = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.0, 0.1),
      Vectors.dense(0.1, 0.0),
      Vectors.dense(9.0, 0.0),
      Vectors.dense(9.0, 0.2),
      Vectors.dense(9.2, 0.0)
    )
    val rdd = sc.parallelize(points, 3)

    for (initMode <- Seq(RANDOM, K_MEANS_PARALLEL)) {
      // Two iterations are sufficient no matter where the initial centers are.
      val model = KMeans.train(rdd, k = 2, maxIterations = 2, runs = 1, mode = initMode)

      val predicts = model.predict(rdd).collect()

      assert(predicts(0) === predicts(1))
      assert(predicts(0) === predicts(2))
      assert(predicts(3) === predicts(4))
      assert(predicts(3) === predicts(5))
      assert(predicts(0) != predicts(3))
    }
  }
}

