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

class ReseedPolicySuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  @transient var spark: SparkSession  = _
  @transient var ops: BregmanPointOps = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("ReseedPolicySuite")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    // Create point operations
    ops = BregmanPointOps(BregmanPointOps.EUCLIDEAN)
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

  // Helper to create points
  def makePoint(values: Array[Double]): BregmanPoint = {
    ops.toPoint(WeightedVector.fromInhomogeneousWeighted(Vectors.dense(values), 1.0))
  }

  // Helper to create centers
  def makeCenter(values: Array[Double]): BregmanCenter = {
    ops.toCenter(WeightedVector.fromInhomogeneousWeighted(Vectors.dense(values), 1.0))
  }

  test("NoReseedPolicy should not modify centers") {
    val policy = NoReseedPolicy

    assert(policy.name == "none")
    assert(!policy.requiresFullScan)
    assert(policy.costCategory == ReseedCostCategory.Constant)

    val data = spark.sparkContext.parallelize(
      Seq(
        makePoint(Array(1.0, 2.0)),
        makePoint(Array(3.0, 4.0))
      )
    )

    val centers = IndexedSeq(
      makeCenter(Array(0.0, 0.0)),
      makeCenter(Array(5.0, 5.0)),
      makeCenter(Array(10.0, 10.0))
    )

    val result = policy.reseedEmptyClusters(
      data = data,
      centers = centers,
      emptyClusters = Set(1),
      ops = ops,
      seed = 42
    )

    assert(result == centers) // Unchanged
  }

  test("RandomReseedPolicy should reseed empty clusters") {
    val policy = RandomReseedPolicy(sampleSize = 10)

    assert(policy.name == "random(sample=10)")
    assert(!policy.requiresFullScan)
    assert(policy.costCategory == ReseedCostCategory.Linear)

    val data = spark.sparkContext.parallelize(
      Seq(
        makePoint(Array(1.0, 2.0)),
        makePoint(Array(3.0, 4.0)),
        makePoint(Array(5.0, 6.0))
      )
    )

    val centers = IndexedSeq(
      makeCenter(Array(0.0, 0.0)),
      makeCenter(Array(10.0, 10.0)), // Empty cluster to reseed
      makeCenter(Array(20.0, 20.0))
    )

    val result = policy.reseedEmptyClusters(
      data = data,
      centers = centers,
      emptyClusters = Set(1),
      ops = ops,
      seed = 42
    )

    // Center at index 1 should be different
    assert(result.size == centers.size)
    assert(result(0) == centers(0)) // Unchanged
    assert(result(1) != centers(1)) // Reseeded
    assert(result(2) == centers(2)) // Unchanged
  }

  test("RandomReseedPolicy should handle multiple empty clusters") {
    val policy = RandomReseedPolicy(sampleSize = 10)

    val data = spark.sparkContext.parallelize(
      Seq(
        makePoint(Array(1.0, 2.0)),
        makePoint(Array(3.0, 4.0))
      )
    )

    val centers = IndexedSeq(
      makeCenter(Array(0.0, 0.0)),
      makeCenter(Array(10.0, 10.0)),
      makeCenter(Array(20.0, 20.0))
    )

    val result = policy.reseedEmptyClusters(
      data = data,
      centers = centers,
      emptyClusters = Set(1, 2),
      ops = ops,
      seed = 42
    )

    assert(result.size == 3)
    assert(result(0) == centers(0))
    assert(result(1) != centers(1))
    assert(result(2) != centers(2))
  }

  test("RandomReseedPolicy should handle empty data") {
    val policy = RandomReseedPolicy()

    val data    = spark.sparkContext.parallelize(Seq.empty[BregmanPoint])
    val centers = IndexedSeq(makeCenter(Array(0.0, 0.0)))

    val result = policy.reseedEmptyClusters(
      data = data,
      centers = centers,
      emptyClusters = Set(0),
      ops = ops,
      seed = 42
    )

    assert(result == centers) // Can't reseed without data
  }

  test("FarthestPointReseedPolicy should have correct properties") {
    val policy = FarthestPointReseedPolicy(numCandidates = 50)

    assert(policy.name == "farthest(candidates=50)")
    assert(policy.requiresFullScan)
    assert(policy.costCategory == ReseedCostCategory.FullScan)
  }

  test("FarthestPointReseedPolicy should reseed with outliers") {
    val policy = FarthestPointReseedPolicy(numCandidates = 3)

    val data = spark.sparkContext.parallelize(
      Seq(
        makePoint(Array(1.0, 1.0)),
        makePoint(Array(2.0, 2.0)),
        makePoint(Array(100.0, 100.0)) // Outlier
      )
    )

    val centers = IndexedSeq(
      makeCenter(Array(1.5, 1.5)),
      makeCenter(Array(50.0, 50.0)) // Empty, far from data
    )

    val result = policy.reseedEmptyClusters(
      data = data,
      centers = centers,
      emptyClusters = Set(1),
      ops = ops,
      seed = 42
    )

    // Should reseed with a point (likely the outlier at 100, 100)
    assert(result.size == 2)
    assert(result(0) == centers(0))
    assert(result(1) != centers(1))
  }

  test("SplitLargestReseedPolicy should have correct properties") {
    val policy = SplitLargestReseedPolicy(perturbation = 0.2)

    assert(policy.name == "splitLargest(perturb=0.2)")
    assert(!policy.requiresFullScan)
    assert(policy.costCategory == ReseedCostCategory.SinglePass)
  }

  test("ReseedPolicy.fromString should parse policy names") {
    assert(ReseedPolicy.fromString("none").isInstanceOf[NoReseedPolicy.type])
    assert(ReseedPolicy.fromString("random").isInstanceOf[RandomReseedPolicy])
    assert(ReseedPolicy.fromString("farthest").isInstanceOf[FarthestPointReseedPolicy])
    assert(ReseedPolicy.fromString("splitLargest").isInstanceOf[SplitLargestReseedPolicy])
  }

  test("ReseedPolicy.fromString should be case-insensitive") {
    assert(ReseedPolicy.fromString("NONE").isInstanceOf[NoReseedPolicy.type])
    assert(ReseedPolicy.fromString("Random").isInstanceOf[RandomReseedPolicy])
    assert(ReseedPolicy.fromString("FARTHEST").isInstanceOf[FarthestPointReseedPolicy])
  }

  test("ReseedPolicy.fromString should normalize names") {
    assert(ReseedPolicy.fromString("no-reseed").isInstanceOf[NoReseedPolicy.type])
    assert(ReseedPolicy.fromString("split_largest").isInstanceOf[SplitLargestReseedPolicy])
  }

  test("ReseedPolicy.fromString should throw on unknown policy") {
    intercept[IllegalArgumentException] {
      ReseedPolicy.fromString("unknown_policy")
    }
  }

  test("ReseedPolicy factory methods should work") {
    assert(ReseedPolicy.none == NoReseedPolicy)
    assert(ReseedPolicy.random().isInstanceOf[RandomReseedPolicy])
    assert(ReseedPolicy.farthest().isInstanceOf[FarthestPointReseedPolicy])
    assert(ReseedPolicy.splitLargest().isInstanceOf[SplitLargestReseedPolicy])
    assert(ReseedPolicy.default.isInstanceOf[RandomReseedPolicy])
  }

  test("ReseedPolicy should be serializable") {
    val policy: ReseedPolicy = RandomReseedPolicy()

    val stream = new java.io.ByteArrayOutputStream()
    val oos    = new java.io.ObjectOutputStream(stream)
    oos.writeObject(policy)
    oos.close()

    val bytes = stream.toByteArray
    assert(bytes.nonEmpty)
  }

  test("ReseedPolicy with same seed should be deterministic") {
    val policy = RandomReseedPolicy(sampleSize = 10)

    val data = spark.sparkContext.parallelize(
      Seq(
        makePoint(Array(1.0, 2.0)),
        makePoint(Array(3.0, 4.0)),
        makePoint(Array(5.0, 6.0))
      )
    )

    val centers = IndexedSeq(
      makeCenter(Array(0.0, 0.0)),
      makeCenter(Array(10.0, 10.0))
    )

    val result1 = policy.reseedEmptyClusters(data, centers, Set(1), ops, seed = 123)
    val result2 = policy.reseedEmptyClusters(data, centers, Set(1), ops, seed = 123)

    assert(result1 == result2)
  }

  test("ReseedPolicy with different seeds should vary") {
    val policy = RandomReseedPolicy(sampleSize = 10)

    val data = spark.sparkContext.parallelize(
      Seq(
        makePoint(Array(1.0, 2.0)),
        makePoint(Array(3.0, 4.0)),
        makePoint(Array(5.0, 6.0)),
        makePoint(Array(7.0, 8.0))
      )
    )

    val centers = IndexedSeq(
      makeCenter(Array(0.0, 0.0)),
      makeCenter(Array(10.0, 10.0))
    )

    val result1 = policy.reseedEmptyClusters(data, centers, Set(1), ops, seed = 123)
    val result2 = policy.reseedEmptyClusters(data, centers, Set(1), ops, seed = 456)

    // Results may differ (not guaranteed, but likely with enough data)
    // At minimum, check that both produced valid results
    assert(result1.size == 2)
    assert(result2.size == 2)
  }

  test("RandomReseedPolicy should validate sample size") {
    intercept[IllegalArgumentException] {
      RandomReseedPolicy(sampleSize = 0)
    }

    intercept[IllegalArgumentException] {
      RandomReseedPolicy(sampleSize = -1)
    }
  }

  test("FarthestPointReseedPolicy should validate num candidates") {
    intercept[IllegalArgumentException] {
      FarthestPointReseedPolicy(numCandidates = 0)
    }
  }

  test("SplitLargestReseedPolicy should validate perturbation") {
    intercept[IllegalArgumentException] {
      SplitLargestReseedPolicy(perturbation = 0.0)
    }

    intercept[IllegalArgumentException] {
      SplitLargestReseedPolicy(perturbation = -0.1)
    }
  }

  test("ReseedPolicy should handle no empty clusters") {
    val policy = RandomReseedPolicy()

    val data = spark.sparkContext.parallelize(
      Seq(
        makePoint(Array(1.0, 2.0))
      )
    )

    val centers = IndexedSeq(makeCenter(Array(0.0, 0.0)))

    val result = policy.reseedEmptyClusters(
      data = data,
      centers = centers,
      emptyClusters = Set.empty,
      ops = ops,
      seed = 42
    )

    assert(result == centers) // No reseeding needed
  }
}
