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

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.scalatest.matchers.should.Matchers

class ValidatorSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  private val sparkSession: SparkSession = SparkSession
    .builder()
    .master("local[2]")
    .appName("ValidatorSuite")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()

  sparkSession.sparkContext.setLogLevel("WARN")

  import sparkSession.implicits._

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  override def afterAll(): Unit = {
    try {
      if (sparkSession != null) {
        sparkSession.stop()
      }
    } finally {
      super.afterAll()
    }
  }

  // Helper to create DataFrame with vectors
  def makeDF(vectors: Seq[org.apache.spark.ml.linalg.Vector], colName: String = "features") = {
    sparkSession.createDataFrame(vectors.map(Tuple1.apply)).toDF(colName)
  }

  test("ValidationResult.success should indicate valid") {
    val result = ValidationResult.success
    assert(result.isValid)
    assert(result.violations.isEmpty)
    assert(result.message == "Validation passed")
  }

  test("ValidationResult.failure should indicate invalid") {
    val violation = ValidationViolation(
      message = "Test violation",
      field = "testField",
      violationType = "test"
    )
    val result    = ValidationResult.failure(violation)

    assert(!result.isValid)
    assert(result.violations.size == 1)
    assert(result.violations.head == violation)
  }

  test("ValidationResult.and should combine results") {
    val result1 = ValidationResult.success
    val result2 = ValidationResult.success

    val combined = result1.and(result2)
    assert(combined.isValid)

    val violation = ValidationViolation("error", "field", "type")
    val result3   = ValidationResult.failure(violation)
    val combined2 = result1.and(result3)

    assert(!combined2.isValid)
    assert(combined2.violations.size == 1)
  }

  test("ValidationResult.getOrThrow should throw on failure") {
    val violation = ValidationViolation("error", "field", "type")
    val result    = ValidationResult.failure(violation)

    intercept[IllegalArgumentException] {
      result.getOrThrow()
    }

    // Should not throw on success
    ValidationResult.success.getOrThrow()
  }

  test("NoNaNValidator should pass for valid vectors") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(4.0, 5.0, 6.0)
      )
    )

    val validator = Validator.noNaN("features")
    val result    = validator.validate(df)

    assert(result.isValid)
  }

  test("NoNaNValidator should fail for vectors with NaN") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(Double.NaN, 5.0, 6.0),
        Vectors.dense(4.0, Double.NaN, 6.0)
      )
    )

    val validator = Validator.noNaN("features")
    val result    = validator.validate(df)

    assert(!result.isValid)
    assert(result.violations.size == 1)
    assert(result.violations.head.violationType == "nan")
    assert(result.violations.head.field == "features")
  }

  test("NoNaNValidator should work with scalar columns") {
    val df = Seq(1.0, 2.0, Double.NaN, 4.0).toDF("value")

    val validator = Validator.noNaN("value")
    val result    = validator.validate(df)

    assert(!result.isValid)
    assert(result.violations.head.violationType == "nan")
  }

  test("FiniteValidator should pass for finite vectors") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(-1.0, 0.0, 100.0)
      )
    )

    val validator = Validator.finite("features")
    val result    = validator.validate(df)

    assert(result.isValid)
  }

  test("FiniteValidator should fail for infinite vectors") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(Double.PositiveInfinity, 5.0, 6.0)
      )
    )

    val validator = FiniteValidator("features")
    val result    = validator.validate(df)

    assert(!result.isValid)
    assert(result.violations.head.violationType == "infinite")
  }

  test("PositiveValidator should pass for positive vectors") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(0.1, 0.5, 100.0)
      )
    )

    val validator = Validator.positive("features")
    val result    = validator.validate(df)

    assert(result.isValid)
  }

  test("PositiveValidator should fail for negative vectors") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(-1.0, 5.0, 6.0)
      )
    )

    val validator = Validator.positive("features", strict = true)
    val result    = validator.validate(df)

    assert(!result.isValid)
    assert(result.violations.head.violationType == "negative")
  }

  test("PositiveValidator with strict=false should allow small negatives") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(-1e-15, 5.0, 6.0) // Very small negative (numerical error)
      )
    )

    val validator = Validator.positive("features", strict = false)
    val result    = validator.validate(df)

    assert(result.isValid) // Should tolerate tiny negatives
  }

  test("NotNullValidator should pass for non-null values") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0),
        Vectors.dense(3.0, 4.0)
      )
    )

    val validator = Validator.notNull("features")
    val result    = validator.validate(df)

    assert(result.isValid)
  }

  test("ConsistentDimensionValidator should pass for consistent dimensions") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(4.0, 5.0, 6.0),
        Vectors.dense(7.0, 8.0, 9.0)
      )
    )

    val validator = Validator.consistentDimension("features")
    val result    = validator.validate(df)

    assert(result.isValid)
  }

  test("ConsistentDimensionValidator should fail for inconsistent dimensions") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(4.0, 5.0), // Wrong dimension
        Vectors.dense(7.0, 8.0, 9.0)
      )
    )

    val validator = Validator.consistentDimension("features")
    val result    = validator.validate(df)

    assert(!result.isValid)
    assert(result.violations.head.violationType == "dimension")
  }

  test("ConsistentDimensionValidator with expected dimension should validate") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(4.0, 5.0, 6.0)
      )
    )

    val validatorCorrect = Validator.consistentDimension("features", Some(3))
    assert(validatorCorrect.validate(df).isValid)

    val validatorWrong = Validator.consistentDimension("features", Some(5))
    assert(!validatorWrong.validate(df).isValid)
  }

  test("NotEmptyValidator should pass for non-empty DataFrame") {
    val df = Seq(1, 2, 3).toDF("value")

    val validator = Validator.notEmpty
    val result    = validator.validate(df)

    assert(result.isValid)
  }

  test("NotEmptyValidator should fail for empty DataFrame") {
    val df = Seq.empty[Int].toDF("value")

    val validator = Validator.notEmpty
    val result    = validator.validate(df)

    assert(!result.isValid)
    assert(result.violations.head.violationType == "empty")
  }

  test("KernelCompatibilityValidator should pass for Euclidean with any features") {
    val df = makeDF(
      Seq(
        Vectors.dense(-1.0, 2.0, 3.0),
        Vectors.dense(4.0, -5.0, 6.0)
      )
    )

    val validator = Validator.kernelCompatibility("squaredEuclidean", "features")
    val result    = validator.validate(df)

    assert(result.isValid)
  }

  test("KernelCompatibilityValidator should require positive features for KL") {
    val dfValid = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(4.0, 5.0, 6.0)
      )
    )

    val dfInvalid = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(-1.0, 5.0, 6.0) // Negative value
      )
    )

    val validator = Validator.kernelCompatibility("kl", "features")

    assert(validator.validate(dfValid).isValid)
    assert(!validator.validate(dfInvalid).isValid)
  }

  test("Validator.and should combine validators") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(4.0, 5.0, 6.0)
      )
    )

    val validator = Validator.notNull("features") and
      Validator.finite("features") and
      Validator.consistentDimension("features")

    val result = validator.validate(df)
    assert(result.isValid)
  }

  test("Validator.features should validate features column") {
    val dfValid = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(4.0, 5.0, 6.0)
      )
    )

    val dfInvalid = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(Double.NaN, 5.0, 6.0)
      )
    )

    val validator = Validator.features("features")

    assert(validator.validate(dfValid).isValid)
    assert(!validator.validate(dfInvalid).isValid)
  }

  test("Validator.weight should validate weight column") {
    val dfValid   = Seq(1.0, 2.0, 3.0).toDF("weight")
    val dfInvalid = Seq(1.0, -1.0, 3.0).toDF("weight")

    val validator = Validator.weight("weight")

    assert(validator.validate(dfValid).isValid)
    assert(!validator.validate(dfInvalid).isValid)
  }

  test("Validator.kmeansInput should validate full input") {
    val dfValid = Seq(
      (Vectors.dense(1.0, 2.0, 3.0), 1.0),
      (Vectors.dense(4.0, 5.0, 6.0), 2.0)
    ).toDF("features", "weight")

    val dfEmpty = Seq.empty[(org.apache.spark.ml.linalg.Vector, Double)].toDF("features", "weight")

    val dfBadFeatures = Seq(
      (Vectors.dense(1.0, 2.0, 3.0), 1.0),
      (Vectors.dense(Double.NaN, 5.0, 6.0), 2.0)
    ).toDF("features", "weight")

    val dfBadWeight = Seq(
      (Vectors.dense(1.0, 2.0, 3.0), 1.0),
      (Vectors.dense(4.0, 5.0, 6.0), -1.0)
    ).toDF("features", "weight")

    val validator = Validator.kmeansInput(
      featuresCol = "features",
      weightCol = Some("weight")
    )

    assert(validator.validate(dfValid).isValid)
    assert(!validator.validate(dfEmpty).isValid)
    assert(!validator.validate(dfBadFeatures).isValid)
    assert(!validator.validate(dfBadWeight).isValid)
  }

  test("Validator.kmeansInput should work without weight column") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(4.0, 5.0, 6.0)
      )
    )

    val validator = Validator.kmeansInput(featuresCol = "features")
    val result    = validator.validate(df)

    assert(result.isValid)
  }

  test("Validator.kmeansInput should validate kernel compatibility") {
    val dfNegative = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(-1.0, 5.0, 6.0)
      )
    )

    val validatorKL = Validator.kmeansInput(
      featuresCol = "features",
      kernelName = "kl"
    )

    assert(!validatorKL.validate(dfNegative).isValid)

    val validatorEuclidean = Validator.kmeansInput(
      featuresCol = "features",
      kernelName = "squaredEuclidean"
    )

    assert(validatorEuclidean.validate(dfNegative).isValid)
  }

  test("ValidationViolation should include sample rows") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0, 3.0),
        Vectors.dense(Double.NaN, 5.0, 6.0),
        Vectors.dense(7.0, Double.NaN, 9.0)
      )
    )

    val validator = Validator.noNaN("features", maxSampleRows = 2)
    val result    = validator.validate(df)

    assert(!result.isValid)
    assert(result.violations.head.sampleRows.nonEmpty)
    assert(result.violations.head.sampleRows.size <= 2)
  }

  test("Validator should be serializable") {
    val validator: Validator = Validator.finite("features")

    val stream = new java.io.ByteArrayOutputStream()
    val oos    = new java.io.ObjectOutputStream(stream)
    oos.writeObject(validator)
    oos.close()

    val bytes = stream.toByteArray
    assert(bytes.nonEmpty)
  }

  test("CombinedValidator should report all violations") {
    val df = makeDF(
      Seq(
        Vectors.dense(Double.NaN, 2.0, 3.0),             // NaN
        Vectors.dense(4.0, Double.PositiveInfinity, 6.0) // Infinite
      )
    )

    val validator = Validator.noNaN("features") and FiniteValidator("features")
    val result    = validator.validate(df)

    assert(!result.isValid)
    // Should have violations for both NaN and infinite (noNaN catches NaN, Finite catches Infinite)
    assert(result.violations.size >= 2)
    assert(result.violations.exists(_.violationType == "nan"))
    assert(result.violations.exists(_.violationType == "infinite"))
  }

  test("Validator error messages should be clear") {
    val df = makeDF(
      Seq(
        Vectors.dense(1.0, 2.0),
        Vectors.dense(Double.NaN, 4.0)
      )
    )

    val validator = Validator.noNaN("features")
    val result    = validator.validate(df)

    assert(!result.isValid)
    assert(result.message.contains("Validation failed"))
    assert(result.message.contains("NaN"))
  }
}
