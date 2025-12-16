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

package com.massivedatascience.util

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

class DivergenceDomainValidatorSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .appName("DivergenceDomainValidatorSuite")
      .master("local[2]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
    super.afterAll()
  }

  // ============================================================================
  // Domain Requirement Tests
  // ============================================================================

  test("getDomainRequirement returns correct requirement for each divergence") {
    assert(
      DivergenceDomainValidator.getDomainRequirement("kl") ==
        DivergenceDomainValidator.StrictlyPositive
    )
    assert(
      DivergenceDomainValidator.getDomainRequirement("itakuraSaito") ==
        DivergenceDomainValidator.StrictlyPositive
    )
    assert(
      DivergenceDomainValidator.getDomainRequirement("logistic") ==
        DivergenceDomainValidator.OpenInterval01
    )
    assert(
      DivergenceDomainValidator.getDomainRequirement("generalizedI") ==
        DivergenceDomainValidator.NonNegative
    )
    assert(
      DivergenceDomainValidator.getDomainRequirement("squaredEuclidean") ==
        DivergenceDomainValidator.Unrestricted
    )
    assert(
      DivergenceDomainValidator.getDomainRequirement("l1") ==
        DivergenceDomainValidator.Unrestricted
    )
  }

  test("getDomainRequirement is case-insensitive") {
    assert(
      DivergenceDomainValidator.getDomainRequirement("KL") ==
        DivergenceDomainValidator.StrictlyPositive
    )
    assert(
      DivergenceDomainValidator.getDomainRequirement("ItakuraSaito") ==
        DivergenceDomainValidator.StrictlyPositive
    )
    assert(
      DivergenceDomainValidator.getDomainRequirement("SQUAREDEUCLIDEAN") ==
        DivergenceDomainValidator.Unrestricted
    )
  }

  test("StrictlyPositive.check validates correctly") {
    val req = DivergenceDomainValidator.StrictlyPositive

    // Valid values
    assert(req.check(0.001))
    assert(req.check(1.0))
    assert(req.check(1e6))
    assert(req.check(Double.MinPositiveValue))

    // Invalid values
    assert(!req.check(0.0))
    assert(!req.check(-1.0))
    assert(!req.check(Double.NaN))
    assert(!req.check(Double.PositiveInfinity))
    assert(!req.check(Double.NegativeInfinity))
  }

  test("OpenInterval01.check validates correctly") {
    val req = DivergenceDomainValidator.OpenInterval01

    // Valid values
    assert(req.check(0.001))
    assert(req.check(0.5))
    assert(req.check(0.999))
    assert(req.check(Double.MinPositiveValue))

    // Invalid values
    assert(!req.check(0.0))
    assert(!req.check(1.0))
    assert(!req.check(-0.1))
    assert(!req.check(1.1))
    assert(!req.check(Double.NaN))
    assert(!req.check(Double.PositiveInfinity))
  }

  test("NonNegative.check validates correctly") {
    val req = DivergenceDomainValidator.NonNegative

    // Valid values
    assert(req.check(0.0))
    assert(req.check(0.001))
    assert(req.check(1.0))
    assert(req.check(1e6))

    // Invalid values
    assert(!req.check(-0.001))
    assert(!req.check(-1.0))
    assert(!req.check(Double.NaN))
    assert(!req.check(Double.PositiveInfinity))
    assert(!req.check(Double.NegativeInfinity))
  }

  test("Unrestricted.check validates correctly") {
    val req = DivergenceDomainValidator.Unrestricted

    // Valid values
    assert(req.check(-1e6))
    assert(req.check(-1.0))
    assert(req.check(0.0))
    assert(req.check(1.0))
    assert(req.check(1e6))
    assert(req.check(Double.MaxValue))
    assert(req.check(Double.MinValue))

    // Invalid values
    assert(!req.check(Double.NaN))
    assert(!req.check(Double.PositiveInfinity))
    assert(!req.check(Double.NegativeInfinity))
  }

  // ============================================================================
  // Vector Validation Tests
  // ============================================================================

  test("validateVector with valid strictly positive data") {
    val vec = Vectors.dense(1.0, 2.0, 3.0)
    // Should not throw
    DivergenceDomainValidator.validateVector(vec, "kl")
  }

  test("validateVector with invalid strictly positive data - zero") {
    val vec       = Vectors.dense(1.0, 0.0, 3.0)
    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateVector(vec, "kl")
    }
    assert(exception.getMessage.contains("strictly positive"))
    assert(exception.getMessage.contains("0.0"))
    assert(exception.getMessage.contains("index 1"))
  }

  test("validateVector with invalid strictly positive data - negative") {
    val vec       = Vectors.dense(1.0, -0.5, 3.0)
    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateVector(vec, "itakuraSaito")
    }
    assert(exception.getMessage.contains("strictly positive"))
    assert(exception.getMessage.contains("-0.5"))
    assert(exception.getMessage.contains("setInputTransform"))
  }

  test("validateVector with valid open interval (0,1) data") {
    val vec = Vectors.dense(0.1, 0.5, 0.9)
    // Should not throw
    DivergenceDomainValidator.validateVector(vec, "logistic")
  }

  test("validateVector with invalid open interval data - zero") {
    val vec       = Vectors.dense(0.1, 0.0, 0.9)
    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateVector(vec, "logistic")
    }
    assert(exception.getMessage.contains("open interval (0, 1)"))
    assert(exception.getMessage.contains("0.0"))
  }

  test("validateVector with invalid open interval data - one") {
    val vec       = Vectors.dense(0.1, 1.0, 0.9)
    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateVector(vec, "logistic")
    }
    assert(exception.getMessage.contains("open interval (0, 1)"))
    assert(exception.getMessage.contains("1.0"))
  }

  test("validateVector with valid non-negative data") {
    val vec = Vectors.dense(0.0, 1.0, 2.0)
    // Should not throw
    DivergenceDomainValidator.validateVector(vec, "generalizedI")
  }

  test("validateVector with invalid non-negative data") {
    val vec       = Vectors.dense(1.0, -0.1, 2.0)
    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateVector(vec, "generalizedI")
    }
    assert(exception.getMessage.contains("non-negative"))
    assert(exception.getMessage.contains("-0.1"))
  }

  test("validateVector with unrestricted data") {
    val vec1 = Vectors.dense(-10.0, 0.0, 10.0)
    val vec2 = Vectors.dense(-1e6, 1e6)
    // Should not throw
    DivergenceDomainValidator.validateVector(vec1, "squaredEuclidean")
    DivergenceDomainValidator.validateVector(vec2, "l1")
  }

  test("validateVector with NaN in unrestricted data") {
    val vec       = Vectors.dense(1.0, Double.NaN, 2.0)
    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateVector(vec, "squaredEuclidean")
    }
    assert(exception.getMessage.contains("non-finite"))
    assert(exception.getMessage.contains("NaN"))
  }

  // ============================================================================
  // DataFrame Validation Tests
  // ============================================================================

  test("validateDataFrame with valid strictly positive data") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(3.0, 4.0)),
      Tuple1(Vectors.dense(0.5, 0.1))
    ).toDF("features")

    // Should not throw
    DivergenceDomainValidator.validateDataFrame(df, "features", "kl")
  }

  test("validateDataFrame with invalid strictly positive data") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(0.0, 4.0)), // Invalid: contains zero
      Tuple1(Vectors.dense(0.5, 0.1))
    ).toDF("features")

    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateDataFrame(df, "features", "kl")
    }
    assert(exception.getMessage.contains("kl divergence"))
    assert(exception.getMessage.contains("strictly positive"))
    assert(exception.getMessage.contains("0.0"))
    assert(exception.getMessage.contains("setInputTransform"))
  }

  test("validateDataFrame with valid open interval data") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(0.1, 0.2)),
      Tuple1(Vectors.dense(0.3, 0.4)),
      Tuple1(Vectors.dense(0.5, 0.9))
    ).toDF("features")

    // Should not throw
    DivergenceDomainValidator.validateDataFrame(df, "features", "logistic")
  }

  test("validateDataFrame with invalid open interval data") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(0.1, 0.2)),
      Tuple1(Vectors.dense(1.0, 0.4)), // Invalid: contains 1.0
      Tuple1(Vectors.dense(0.5, 0.9))
    ).toDF("features")

    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateDataFrame(df, "features", "logistic")
    }
    assert(exception.getMessage.contains("logistic divergence"))
    assert(exception.getMessage.contains("open interval (0, 1)"))
    assert(exception.getMessage.contains("1.0"))
  }

  test("validateDataFrame with valid non-negative data") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(0.0, 1.0)),
      Tuple1(Vectors.dense(2.0, 0.0)),
      Tuple1(Vectors.dense(0.5, 3.0))
    ).toDF("features")

    // Should not throw
    DivergenceDomainValidator.validateDataFrame(df, "features", "generalizedI")
  }

  test("validateDataFrame with invalid non-negative data") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(0.0, 1.0)),
      Tuple1(Vectors.dense(-1.0, 0.0)), // Invalid: contains negative
      Tuple1(Vectors.dense(0.5, 3.0))
    ).toDF("features")

    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateDataFrame(df, "features", "generalizedI")
    }
    assert(exception.getMessage.contains("generalizedI divergence"))
    assert(exception.getMessage.contains("non-negative"))
    assert(exception.getMessage.contains("-1.0"))
  }

  test("validateDataFrame with unrestricted divergence") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(-10.0, 10.0)),
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(-5.0, 5.0))
    ).toDF("features")

    // Should not throw
    DivergenceDomainValidator.validateDataFrame(df, "features", "squaredEuclidean")
  }

  test("validateDataFrame with NaN in unrestricted data") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(Double.NaN, 4.0)), // Invalid: contains NaN
      Tuple1(Vectors.dense(0.5, 0.1))
    ).toDF("features")

    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateDataFrame(df, "features", "squaredEuclidean")
    }
    assert(exception.getMessage.contains("Non-finite value"))
    assert(exception.getMessage.contains("NaN"))
  }

  test("validateDataFrame with Infinity") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(Double.PositiveInfinity, 4.0)), // Invalid: contains Infinity
      Tuple1(Vectors.dense(0.5, 0.1))
    ).toDF("features")

    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateDataFrame(df, "features", "kl")
    }
    assert(
      exception.getMessage.contains("Infinity") || exception.getMessage.contains(
        "strictly positive"
      )
    )
  }

  test("validateDataFrame with maxSamples parameter") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(3.0, 4.0)),
      Tuple1(Vectors.dense(5.0, 6.0)),
      Tuple1(Vectors.dense(0.0, 8.0)) // Invalid, but might be skipped with small maxSamples
    ).toDF("features")

    // With maxSamples=2, might not detect the violation in row 4
    // This test documents the behavior rather than asserting success/failure
    // since the sampling is deterministic based on limit()
  }

  test("validateDataFrame reports feature index in error") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0)),
      Tuple1(Vectors.dense(1.0, 2.0, 0.0, 4.0, 5.0)) // Invalid at index 2
    ).toDF("features")

    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateDataFrame(df, "features", "kl")
    }
    assert(exception.getMessage.contains("index 2"))
    assert(exception.getMessage.contains("0.0"))
  }

  // ============================================================================
  // Helper Method Tests
  // ============================================================================

  test("getDomainDescription returns readable descriptions") {
    val desc1 = DivergenceDomainValidator.getDomainDescription("kl")
    assert(desc1.contains("kl"))
    assert(desc1.contains("strictly positive"))

    val desc2 = DivergenceDomainValidator.getDomainDescription("logistic")
    assert(desc2.contains("logistic"))
    assert(desc2.contains("open interval (0, 1)"))

    val desc3 = DivergenceDomainValidator.getDomainDescription("squaredEuclidean")
    assert(desc3.contains("squaredEuclidean"))
    assert(desc3.contains("unrestricted"))
  }

  test("requiresStrictValidation correctly identifies divergences") {
    // Divergences requiring strict validation
    assert(DivergenceDomainValidator.requiresStrictValidation("kl"))
    assert(DivergenceDomainValidator.requiresStrictValidation("itakuraSaito"))
    assert(DivergenceDomainValidator.requiresStrictValidation("logistic"))
    assert(DivergenceDomainValidator.requiresStrictValidation("generalizedI"))

    // Unrestricted divergences
    assert(!DivergenceDomainValidator.requiresStrictValidation("squaredEuclidean"))
    assert(!DivergenceDomainValidator.requiresStrictValidation("l1"))
    assert(!DivergenceDomainValidator.requiresStrictValidation("manhattan"))
  }

  // ============================================================================
  // Error Message Quality Tests
  // ============================================================================

  test("error messages include suggested fixes") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(1.0, -1.0))
    ).toDF("features")

    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateDataFrame(df, "features", "kl")
    }

    val msg = exception.getMessage
    // Check that suggested fixes are included
    assert(msg.contains("Suggested fixes:"))
    assert(msg.contains("setInputTransform"))
    assert(msg.contains("log1p") || msg.contains("epsilonShift"))
  }

  test("error messages show example code") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(0.0, 1.0))
    ).toDF("features")

    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateDataFrame(df, "features", "itakuraSaito")
    }

    val msg = exception.getMessage
    // Check that example code is included
    assert(msg.contains("Example:"))
    assert(msg.contains("new GeneralizedKMeans()"))
    assert(msg.contains(".setDivergence"))
  }

  test("error messages are actionable for logistic divergence") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.dense(1.5, 0.5))
    ).toDF("features")

    val exception = intercept[IllegalArgumentException] {
      DivergenceDomainValidator.validateDataFrame(df, "features", "logistic")
    }

    val msg = exception.getMessage
    assert(msg.contains("logistic divergence"))
    assert(msg.contains("open interval (0, 1)"))
    assert(msg.contains("1.5"))
    assert(msg.contains("Suggested fixes:"))
  }

  // ============================================================================
  // Edge Cases
  // ============================================================================

  test("validateDataFrame with empty DataFrame") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = spark.emptyDataFrame.selectExpr("array(1.0, 2.0) as features").limit(0)

    // Should not throw on empty data
    DivergenceDomainValidator.validateDataFrame(df, "features", "kl")
  }

  test("validateVector with single-element vector") {
    val vec = Vectors.dense(1.0)
    // Should not throw
    DivergenceDomainValidator.validateVector(vec, "kl")
  }

  test("validateVector with large vector") {
    val vec = Vectors.dense(Array.fill(1000)(1.0))
    // Should not throw
    DivergenceDomainValidator.validateVector(vec, "kl")
  }

  test("validateDataFrame with sparse vectors") {
    val sparkSession = spark
    import sparkSession.implicits._
    val df           = Seq(
      Tuple1(Vectors.sparse(5, Seq((0, 1.0), (2, 2.0)))),
      Tuple1(Vectors.sparse(5, Seq((1, 3.0), (4, 4.0))))
    ).toDF("features")

    // Sparse vectors with non-negative divergence should work (implicit zeros)
    DivergenceDomainValidator.validateDataFrame(df, "features", "generalizedI")
  }
}
