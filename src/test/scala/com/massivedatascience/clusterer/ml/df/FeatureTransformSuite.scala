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

package com.massivedatascience.clusterer.ml.df

import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.scalatest.matchers.should.Matchers

class FeatureTransformSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("FeatureTransformSuite")
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

  // Helper to compare vectors with tolerance
  def vectorsEqual(v1: Vector, v2: Vector, tol: Double = 1e-10): Boolean = {
    if (v1.size != v2.size) return false
    v1.toArray.zip(v2.toArray).forall { case (a, b) => math.abs(a - b) < tol }
  }

  test("NoOpTransform should be identity") {
    val v        = Vectors.dense(1.0, 2.0, 3.0)
    val result   = NoOpTransform(v)
    val inverted = NoOpTransform.inverseCenter(result)

    assert(vectorsEqual(v, result))
    assert(vectorsEqual(v, inverted))
  }

  test("NoOpTransform on DataFrame should work") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(3.0, 4.0))
    ).toDF("features")

    val transformed = NoOpTransform(df, "features", "output")
    assert(transformed.columns.contains("output"))
    assert(transformed.count() == 2)
  }

  test("Log1pTransform should compute log(1 + x)") {
    val v      = Vectors.dense(0.0, 1.0, 2.0)
    val result = Log1pTransform(v)

    val expected = Vectors.dense(math.log1p(0.0), math.log1p(1.0), math.log1p(2.0))
    assert(vectorsEqual(result, expected))
  }

  test("Log1pTransform should be invertible") {
    val v        = Vectors.dense(1.0, 2.0, 3.0)
    val result   = Log1pTransform(v)
    val inverted = Log1pTransform.inverseCenter(result)

    assert(vectorsEqual(v, inverted, tol = 1e-9))
  }

  test("EpsilonShiftTransform should add epsilon") {
    val epsilon = 0.01
    val v       = Vectors.dense(1.0, 2.0, 3.0)
    val result  = EpsilonShiftTransform(epsilon)(v)

    val expected = Vectors.dense(1.01, 2.01, 3.01)
    assert(vectorsEqual(result, expected))
  }

  test("EpsilonShiftTransform should be invertible") {
    val epsilon   = 0.01
    val v         = Vectors.dense(1.0, 2.0, 3.0)
    val transform = EpsilonShiftTransform(epsilon)
    val result    = transform(v)
    val inverted  = transform.inverseCenter(result)

    assert(vectorsEqual(v, inverted, tol = 1e-9))
  }

  test("NormalizeL2Transform should create unit vectors") {
    val v      = Vectors.dense(3.0, 4.0) // ||v|| = 5
    val result = NormalizeL2Transform()(v)

    val expected = Vectors.dense(0.6, 0.8)
    assert(vectorsEqual(result, expected))

    // Check unit norm
    val norm = math.sqrt(result.toArray.map(x => x * x).sum)
    assert(math.abs(norm - 1.0) < 1e-10)
  }

  test("NormalizeL2Transform should handle zero vectors") {
    val v      = Vectors.dense(0.0, 0.0)
    val result = NormalizeL2Transform()(v)

    // Should return original for zero-norm vectors
    assert(vectorsEqual(result, v))
  }

  test("NormalizeL1Transform should create probability vectors") {
    val v      = Vectors.dense(1.0, 2.0, 3.0) // sum = 6
    val result = NormalizeL1Transform()(v)

    val expected = Vectors.dense(1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0)
    assert(vectorsEqual(result, expected))

    // Check sum = 1
    val sum = result.toArray.sum
    assert(math.abs(sum - 1.0) < 1e-10)
  }

  test("StandardScalingTransform should center and scale") {
    val mean      = Vectors.dense(1.0, 2.0)
    val stddev    = Vectors.dense(0.5, 1.0)
    val transform = StandardScalingTransform(mean, stddev)

    val v      = Vectors.dense(1.5, 4.0)
    val result = transform(v)

    // (1.5 - 1.0) / 0.5 = 1.0
    // (4.0 - 2.0) / 1.0 = 2.0
    val expected = Vectors.dense(1.0, 2.0)
    assert(vectorsEqual(result, expected))
  }

  test("StandardScalingTransform should be invertible") {
    val mean      = Vectors.dense(1.0, 2.0)
    val stddev    = Vectors.dense(0.5, 1.0)
    val transform = StandardScalingTransform(mean, stddev)

    val v        = Vectors.dense(1.5, 4.0)
    val result   = transform(v)
    val inverted = transform.inverseCenter(result)

    assert(vectorsEqual(v, inverted, tol = 1e-9))
  }

  test("ComposedTransform should apply transforms sequentially") {
    val epsilon  = 0.01
    val composed = EpsilonShiftTransform(epsilon).andThen(Log1pTransform)

    val v      = Vectors.dense(0.0, 1.0)
    val result = composed(v)

    // First: add epsilon -> [0.01, 1.01]
    // Then: log1p -> [log1p(0.01), log1p(1.01)]
    val expected = Vectors.dense(math.log1p(0.01), math.log1p(1.01))
    assert(vectorsEqual(result, expected))
  }

  test("ComposedTransform should invert correctly") {
    val epsilon  = 0.01
    val composed = EpsilonShiftTransform(epsilon).andThen(Log1pTransform)

    val v        = Vectors.dense(1.0, 2.0)
    val result   = composed(v)
    val inverted = composed.inverseCenter(result)

    assert(vectorsEqual(v, inverted, tol = 1e-9))
  }

  test("forKL should combine epsilon shift and log1p") {
    val transform = FeatureTransform.forKL(0.01)
    assert(transform.name.contains("epsilon_shift"))
    assert(transform.name.contains("log1p"))
  }

  test("fromString should parse transform names") {
    assert(FeatureTransform.fromString("identity") == NoOpTransform)
    assert(FeatureTransform.fromString("log1p") == Log1pTransform)
    assert(FeatureTransform.fromString("normalize_l2").isInstanceOf[NormalizeL2Transform])
  }

  test("fromString should throw on unknown transform") {
    intercept[IllegalArgumentException] {
      FeatureTransform.fromString("unknown_transform")
    }
  }

  test("compatibleWith should check divergence compatibility") {
    assert(Log1pTransform.compatibleWith("kl"))
    assert(Log1pTransform.compatibleWith("squaredEuclidean"))
    // Log1p returns compatibility based on a set list
    assert(Log1pTransform.compatibleWith("euclidean"))

    assert(EpsilonShiftTransform().compatibleWith("kl"))
    assert(EpsilonShiftTransform().compatibleWith("generalizedI"))
    assert(!EpsilonShiftTransform().compatibleWith("squaredEuclidean"))

    assert(NormalizeL2Transform().compatibleWith("squaredEuclidean"))
    assert(!NormalizeL2Transform().compatibleWith("kl"))
  }

  test("Transform on DataFrame should preserve row count") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(3.0, 4.0)),
      Tuple1(Vectors.dense(5.0, 6.0))
    ).toDF("features")

    val transformed = Log1pTransform(df, "features", "transformed")
    assert(transformed.count() == 3)
    assert(transformed.columns.contains("transformed"))
  }

  test("In-place transform should work") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(1.0, 2.0))
    ).toDF("features")

    val transformed = Log1pTransform(df, "features", "features")
    assert(transformed.count() == 1)
    assert(transformed.columns.contains("features"))
  }

  test("Composed transform on DataFrame should work") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(1.0, 2.0))
    ).toDF("features")

    val composed    = EpsilonShiftTransform(0.01).andThen(Log1pTransform)
    val transformed = composed(df, "features", "output")

    assert(transformed.count() == 1)
    assert(transformed.columns.contains("output"))
    assert(!transformed.columns.contains(s"output_tmp")) // Temporary column should be cleaned up
  }
}
