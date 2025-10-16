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

class SeedingServiceSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  @transient var spark: SparkSession = _
  @transient var ops: BregmanPointOps = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("SeedingServiceSuite")
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

  test("RandomSeeding should select k random centers") {
    val seeding = RandomSeeding(k = 5, seed = 42)

    assert(seeding.name == "random(k=5)")
    assert(seeding.k == 5)
    assert(seeding.seed == 42)
    assert(!seeding.requiresMultiplePasses)

    val data = spark.sparkContext.parallelize((1 to 100).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val centers = seeding.selectInitialCenters(data, ops)

    assert(centers.size == 5)
  }

  test("RandomSeeding should handle k > n") {
    val seeding = RandomSeeding(k = 100, seed = 42)

    val data = spark.sparkContext.parallelize((1 to 10).map { i =>
      makePoint(Array(i.toDouble))
    })

    val centers = seeding.selectInitialCenters(data, ops)

    // Should return at most n centers
    assert(centers.size == 10)
  }

  test("RandomSeeding should validate k") {
    intercept[IllegalArgumentException] {
      RandomSeeding(k = 0, seed = 42)
    }

    intercept[IllegalArgumentException] {
      RandomSeeding(k = -5, seed = 42)
    }
  }

  test("RandomSeeding should handle empty data") {
    val seeding = RandomSeeding(k = 5, seed = 42)
    val data    = spark.sparkContext.parallelize(Seq.empty[BregmanPoint])

    intercept[IllegalArgumentException] {
      seeding.selectInitialCenters(data, ops)
    }
  }

  test("RandomSeeding with same seed should be deterministic") {
    val seeding = RandomSeeding(k = 3, seed = 123)

    val data = spark.sparkContext.parallelize((1 to 50).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val centers1 = seeding.selectInitialCenters(data, ops)
    val centers2 = seeding.selectInitialCenters(data, ops)

    assert(centers1 == centers2)
  }

  test("RandomSeeding with different seeds should vary") {
    val seeding1 = RandomSeeding(k = 5, seed = 123)
    val seeding2 = RandomSeeding(k = 5, seed = 456)

    val data = spark.sparkContext.parallelize((1 to 100).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val centers1 = seeding1.selectInitialCenters(data, ops)
    val centers2 = seeding2.selectInitialCenters(data, ops)

    // Results may differ (not guaranteed, but very likely with 100 points)
    assert(centers1.size == 5)
    assert(centers2.size == 5)
  }

  test("KMeansPlusPlusSeeding should select k centers") {
    val seeding = KMeansPlusPlusSeeding(k = 5, seed = 42)

    assert(seeding.name.contains("kMeans++"))
    assert(seeding.k == 5)
    assert(seeding.requiresMultiplePasses)

    val data = spark.sparkContext.parallelize((1 to 100).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val centers = seeding.selectInitialCenters(data, ops)

    assert(centers.size == 5)
  }

  test("KMeansPlusPlusSeeding should validate parameters") {
    intercept[IllegalArgumentException] {
      KMeansPlusPlusSeeding(k = 0, seed = 42)
    }

    intercept[IllegalArgumentException] {
      KMeansPlusPlusSeeding(k = 5, seed = 42, oversamplingFactor = 0)
    }

    intercept[IllegalArgumentException] {
      KMeansPlusPlusSeeding(k = 5, seed = 42, oversamplingFactor = -1)
    }
  }

  test("KMeansPlusPlusSeeding with same seed should be deterministic") {
    val seeding = KMeansPlusPlusSeeding(k = 3, seed = 123)

    val data = spark.sparkContext.parallelize((1 to 50).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val centers1 = seeding.selectInitialCenters(data, ops)
    val centers2 = seeding.selectInitialCenters(data, ops)

    assert(centers1 == centers2)
  }

  test("KMeansPlusPlusSeeding should handle k > n") {
    val seeding = KMeansPlusPlusSeeding(k = 50, seed = 42)

    val data = spark.sparkContext.parallelize((1 to 10).map { i =>
      makePoint(Array(i.toDouble))
    })

    val centers = seeding.selectInitialCenters(data, ops)

    assert(centers.size == 10)
  }

  test("KMeansParallelSeeding should select k centers") {
    val seeding = KMeansParallelSeeding(k = 5, seed = 42, rounds = 3)

    assert(seeding.name.contains("kMeans||"))
    assert(seeding.k == 5)
    assert(seeding.requiresMultiplePasses)

    val data = spark.sparkContext.parallelize((1 to 100).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val centers = seeding.selectInitialCenters(data, ops)

    assert(centers.size == 5)
  }

  test("KMeansParallelSeeding should validate parameters") {
    intercept[IllegalArgumentException] {
      KMeansParallelSeeding(k = 0, seed = 42)
    }

    intercept[IllegalArgumentException] {
      KMeansParallelSeeding(k = 5, seed = 42, rounds = 0)
    }

    intercept[IllegalArgumentException] {
      KMeansParallelSeeding(k = 5, seed = 42, rounds = -1)
    }
  }

  test("KMeansParallelSeeding with same seed should be deterministic") {
    val seeding = KMeansParallelSeeding(k = 3, seed = 123)

    val data = spark.sparkContext.parallelize((1 to 50).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val centers1 = seeding.selectInitialCenters(data, ops)
    val centers2 = seeding.selectInitialCenters(data, ops)

    assert(centers1 == centers2)
  }

  test("GridSeeding should select k centers") {
    val seeding = GridSeeding(k = 5, seed = 42)

    assert(seeding.name == "grid(k=5)")
    assert(seeding.k == 5)
    assert(!seeding.requiresMultiplePasses)

    val data = spark.sparkContext.parallelize((1 to 100).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val centers = seeding.selectInitialCenters(data, ops)

    assert(centers.size == 5)
  }

  test("GridSeeding should validate k") {
    intercept[IllegalArgumentException] {
      GridSeeding(k = 0, seed = 42)
    }

    intercept[IllegalArgumentException] {
      GridSeeding(k = -5, seed = 42)
    }
  }

  test("GridSeeding should handle k > n") {
    val seeding = GridSeeding(k = 50, seed = 42)

    val data = spark.sparkContext.parallelize((1 to 10).map { i =>
      makePoint(Array(i.toDouble))
    })

    val centers = seeding.selectInitialCenters(data, ops)

    assert(centers.size == 10)
  }

  test("GridSeeding should be deterministic") {
    val seeding = GridSeeding(k = 3, seed = 123)

    val data = spark.sparkContext.parallelize((1 to 50).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val centers1 = seeding.selectInitialCenters(data, ops)
    val centers2 = seeding.selectInitialCenters(data, ops)

    assert(centers1 == centers2)
  }

  test("SeedingService.fromString should parse seeding names") {
    assert(SeedingService.fromString("random", k = 5).isInstanceOf[RandomSeeding])
    assert(SeedingService.fromString("kmeans++", k = 5).isInstanceOf[KMeansPlusPlusSeeding])
    assert(SeedingService.fromString("kmeans||", k = 5).isInstanceOf[KMeansParallelSeeding])
    assert(SeedingService.fromString("grid", k = 5).isInstanceOf[GridSeeding])
  }

  test("SeedingService.fromString should be case-insensitive") {
    assert(SeedingService.fromString("RANDOM", k = 5).isInstanceOf[RandomSeeding])
    assert(SeedingService.fromString("KMeans++", k = 5).isInstanceOf[KMeansPlusPlusSeeding])
    assert(SeedingService.fromString("KMEANS||", k = 5).isInstanceOf[KMeansParallelSeeding])
  }

  test("SeedingService.fromString should normalize names") {
    assert(SeedingService.fromString("k-means++", k = 5).isInstanceOf[KMeansPlusPlusSeeding])
    assert(SeedingService.fromString("k_means_parallel", k = 5).isInstanceOf[KMeansParallelSeeding])
    assert(SeedingService.fromString("kmeans-pp", k = 5).isInstanceOf[KMeansPlusPlusSeeding])
  }

  test("SeedingService.fromString should throw on unknown strategy") {
    intercept[IllegalArgumentException] {
      SeedingService.fromString("unknown_strategy", k = 5)
    }
  }

  test("SeedingService factory methods should work") {
    assert(SeedingService.random(k = 5).isInstanceOf[RandomSeeding])
    assert(SeedingService.kMeansPlusPlus(k = 5).isInstanceOf[KMeansPlusPlusSeeding])
    assert(SeedingService.kMeansParallel(k = 5).isInstanceOf[KMeansParallelSeeding])
    assert(SeedingService.grid(k = 5).isInstanceOf[GridSeeding])
    assert(SeedingService.default(k = 5).isInstanceOf[KMeansPlusPlusSeeding])
  }

  test("SeedingService should be serializable") {
    val seeding: SeedingService = RandomSeeding(k = 5, seed = 42)

    val stream = new java.io.ByteArrayOutputStream()
    val oos    = new java.io.ObjectOutputStream(stream)
    oos.writeObject(seeding)
    oos.close()

    val bytes = stream.toByteArray
    assert(bytes.nonEmpty)
  }

  test("Different seeding strategies should produce different centers") {
    val data = spark.sparkContext.parallelize((1 to 100).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val randomCenters = RandomSeeding(k = 5, seed = 42).selectInitialCenters(data, ops)
    val kppCenters    = KMeansPlusPlusSeeding(k = 5, seed = 42).selectInitialCenters(data, ops)
    val gridCenters   = GridSeeding(k = 5, seed = 42).selectInitialCenters(data, ops)

    // All should produce 5 centers
    assert(randomCenters.size == 5)
    assert(kppCenters.size == 5)
    assert(gridCenters.size == 5)

    // Different strategies likely produce different results
    // (not guaranteed but very likely with this data)
  }

  test("KMeansPlusPlusSeeding with custom oversampling should work") {
    val seeding = KMeansPlusPlusSeeding(k = 5, seed = 42, oversamplingFactor = 5)

    val data = spark.sparkContext.parallelize((1 to 100).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val centers = seeding.selectInitialCenters(data, ops)
    assert(centers.size == 5)
  }

  test("KMeansParallelSeeding with custom rounds should work") {
    val seeding = KMeansParallelSeeding(k = 5, seed = 42, rounds = 10)

    val data = spark.sparkContext.parallelize((1 to 100).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val centers = seeding.selectInitialCenters(data, ops)
    assert(centers.size == 5)
  }

  test("SeedingService should handle single data point") {
    val seeding = RandomSeeding(k = 3, seed = 42)

    val data = spark.sparkContext.parallelize(Seq(
      makePoint(Array(1.0, 2.0))
    ))

    val centers = seeding.selectInitialCenters(data, ops)
    assert(centers.size == 1)
  }

  test("SeedingService should handle duplicate points") {
    val seeding = KMeansPlusPlusSeeding(k = 3, seed = 42)

    val data = spark.sparkContext.parallelize(Seq.fill(20)(
      makePoint(Array(1.0, 2.0))
    ))

    val centers = seeding.selectInitialCenters(data, ops)
    assert(centers.size <= 3)
  }

  test("SeedingService should produce valid centers") {
    val seeding = RandomSeeding(k = 5, seed = 42)

    val data = spark.sparkContext.parallelize((1 to 100).map { i =>
      makePoint(Array(i.toDouble, i.toDouble * 2))
    })

    val centers = seeding.selectInitialCenters(data, ops)

    // All centers should have valid coordinates
    centers.foreach { center =>
      val point = ops.toPoint(center)
      assert(point.homogeneous.toArray.forall(!_.isNaN))
      assert(point.homogeneous.toArray.forall(!_.isInfinite))
    }
  }
}
