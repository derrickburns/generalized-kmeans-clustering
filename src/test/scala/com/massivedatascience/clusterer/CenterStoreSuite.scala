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
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.scalatest.matchers.should.Matchers

class CenterStoreSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("CenterStoreSuite")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) {
        spark.stop()
      }
    } finally {
      super.afterAll()
    }
  }

  // Helper to create centers - directly construct BregmanCenter
  // For testing purposes, we use simple Euclidean-like values
  def makeCenter(values: Array[Double], weight: Double = 1.0): BregmanCenter = {
    val homogeneous = Vectors.dense(values.map(_ * weight))
    val gradient = Vectors.dense(values)  // For Euclidean: gradient = inhomogeneous
    val dotGradMinusF = 0.5 * values.map(x => x * x).sum  // For Euclidean: F(x) = 0.5 ||x||^2
    BregmanCenter(homogeneous, weight, dotGradMinusF, gradient)
  }

  test("ArrayCenterStore should store centers") {
    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0)),
      makeCenter(Array(3.0, 4.0)),
      makeCenter(Array(5.0, 6.0))
    )

    val store = ArrayCenterStore(centers)

    assert(store.count == 3)
    assert(store.nonEmpty)
  }

  test("ArrayCenterStore should provide indexed access") {
    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0)),
      makeCenter(Array(3.0, 4.0))
    )

    val store = ArrayCenterStore(centers)

    assert(store(0) == centers(0))
    assert(store(1) == centers(1))
  }

  test("ArrayCenterStore should throw on invalid index") {
    val centers = IndexedSeq(makeCenter(Array(1.0, 2.0)))
    val store   = ArrayCenterStore(centers)

    intercept[IllegalArgumentException] {
      store(-1)
    }

    intercept[IllegalArgumentException] {
      store(1)
    }
  }

  test("ArrayCenterStore should support update") {
    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0)),
      makeCenter(Array(3.0, 4.0))
    )

    val store      = ArrayCenterStore(centers)
    val newCenter  = makeCenter(Array(10.0, 20.0))
    val newStore   = store.updated(0, newCenter)

    assert(newStore(0) == newCenter)
    assert(newStore(1) == centers(1))
    assert(store(0) == centers(0)) // Original unchanged
  }

  test("ArrayCenterStore should filter non-empty centers") {
    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0), weight = 1.0),
      makeCenter(Array(3.0, 4.0), weight = 0.0), // Empty center
      makeCenter(Array(5.0, 6.0), weight = 2.0)
    )

    val store    = ArrayCenterStore(centers)
    val filtered = store.filterNonEmpty()

    assert(filtered.count == 2)
    assert(filtered(0).weight == 1.0)
    assert(filtered(1).weight == 2.0)
  }

  test("ArrayCenterStore should detect empty centers") {
    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0), weight = 1.0),
      makeCenter(Array(3.0, 4.0), weight = 0.0)
    )

    val store = ArrayCenterStore(centers)

    assert(!store.isEmpty(0))
    assert(store.isEmpty(1))
  }

  test("ArrayCenterStore should convert to array") {
    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0)),
      makeCenter(Array(3.0, 4.0))
    )

    val store = ArrayCenterStore(centers)
    val array = store.toArray

    assert(array.length == 2)
    assert(array(0) == centers(0))
    assert(array(1) == centers(1))
  }

  test("ArrayCenterStore should convert to sequence") {
    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0)),
      makeCenter(Array(3.0, 4.0))
    )

    val store = ArrayCenterStore(centers)
    val seq   = store.toSeq

    assert(seq.length == 2)
    assert(seq(0) == centers(0))
    assert(seq(1) == centers(1))
  }

  test("ArrayCenterStore should support map") {
    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0), weight = 1.0),
      makeCenter(Array(3.0, 4.0), weight = 2.0)
    )

    val store = ArrayCenterStore(centers)
    val mapped = store.map { c =>
      // Double the weight
      makeCenter(c.inhomogeneous.toArray, c.weight * 2.0)
    }

    assert(mapped(0).weight == 2.0)
    assert(mapped(1).weight == 4.0)
  }

  test("ArrayCenterStore should support foldLeft") {
    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0), weight = 1.0),
      makeCenter(Array(3.0, 4.0), weight = 2.0),
      makeCenter(Array(5.0, 6.0), weight = 3.0)
    )

    val store       = ArrayCenterStore(centers)
    val totalWeight = store.foldLeft(0.0)((acc, c) => acc + c.weight)

    assert(totalWeight == 6.0)
  }

  test("ArrayCenterStore should convert to DataFrame") {
    val sparkSession = spark
    import sparkSession.implicits._

    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0)),
      makeCenter(Array(3.0, 4.0))
    )

    val store = ArrayCenterStore(centers)
    val df    = store.toDataFrame(spark)

    assert(df.count() == 2)
    assert(df.columns.contains("cluster_id"))
    assert(df.columns.contains("center"))

    val rows = df.orderBy("cluster_id").collect()
    assert(rows(0).getInt(0) == 0)
    assert(rows(1).getInt(0) == 1)
  }

  test("CenterStore.fromArray should create store") {
    val centers = Array(
      makeCenter(Array(1.0, 2.0)),
      makeCenter(Array(3.0, 4.0))
    )

    val store = CenterStore.fromArray(centers)

    assert(store.count == 2)
    assert(store(0) == centers(0))
  }

  test("CenterStore.fromSeq should create store") {
    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0)),
      makeCenter(Array(3.0, 4.0))
    )

    val store = CenterStore.fromSeq(centers)

    assert(store.count == 2)
    assert(store(0) == centers(0))
  }

  test("CenterStore.empty should create empty store") {
    val store = CenterStore.empty

    assert(store.count == 0)
    assert(!store.nonEmpty)
  }

  // Note: fromDataFrame tests require BregmanPointOps which would need a full clustering setup
  // These tests are deferred until CenterStore is integrated into the main codebase

  test("ArrayCenterStore should maintain stable ordering") {
    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0)),
      makeCenter(Array(3.0, 4.0)),
      makeCenter(Array(5.0, 6.0))
    )

    val store = ArrayCenterStore(centers)

    // Multiple calls to toSeq should return same ordering
    val seq1 = store.toSeq
    val seq2 = store.toSeq

    assert(seq1 == seq2)
    assert(seq1.indices.forall(i => seq1(i) == centers(i)))
  }

  test("ArrayCenterStore should handle single center") {
    val center = makeCenter(Array(1.0, 2.0))
    val store  = ArrayCenterStore(IndexedSeq(center))

    assert(store.count == 1)
    assert(store(0) == center)
    assert(store.nonEmpty)
  }

  test("ArrayCenterStore should be immutable") {
    val centers = IndexedSeq(
      makeCenter(Array(1.0, 2.0)),
      makeCenter(Array(3.0, 4.0))
    )

    val store1 = ArrayCenterStore(centers)
    val store2 = store1.updated(0, makeCenter(Array(10.0, 20.0)))

    // Original store unchanged
    assert(store1(0) == centers(0))
    assert(store2(0) != centers(0))
  }
}
