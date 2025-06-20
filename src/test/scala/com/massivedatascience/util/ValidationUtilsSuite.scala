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

import com.massivedatascience.clusterer.TestingUtils._
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite

class ValidationUtilsSuite extends AnyFunSuite {

  test("requirePositiveWeight with valid weights") {
    // Should pass without exception
    ValidationUtils.requirePositiveWeight(1.0)
    ValidationUtils.requirePositiveWeight(0.001)
    ValidationUtils.requirePositiveWeight(Double.MaxValue)
    ValidationUtils.requirePositiveWeight(1e-100)
    ValidationUtils.requirePositiveWeight(1e100)
  }

  test("requirePositiveWeight with invalid weights") {
    val exception1 = intercept[IllegalArgumentException] {
      ValidationUtils.requirePositiveWeight(0.0)
    }
    assert(exception1.getMessage.contains("must be positive"))

    val exception2 = intercept[IllegalArgumentException] {
      ValidationUtils.requirePositiveWeight(-1.0)
    }
    assert(exception2.getMessage.contains("must be positive"))

    val exception3 = intercept[IllegalArgumentException] {
      ValidationUtils.requirePositiveWeight(Double.MinValue)
    }
    assert(exception3.getMessage.contains("must be positive"))

    val exception4 = intercept[IllegalArgumentException] {
      ValidationUtils.requirePositiveWeight(Double.NegativeInfinity)
    }
    assert(exception4.getMessage.contains("must be positive"))
  }

  test("requirePositiveWeight with custom context") {
    val exception = intercept[IllegalArgumentException] {
      ValidationUtils.requirePositiveWeight(-1.0, "Custom weight parameter")
    }
    assert(exception.getMessage.contains("Custom weight parameter must be positive"))
  }

  test("requireNonNegativeWeight with valid weights") {
    // Should pass without exception
    ValidationUtils.requireNonNegativeWeight(0.0)
    ValidationUtils.requireNonNegativeWeight(1.0)
    ValidationUtils.requireNonNegativeWeight(Double.MaxValue)
    ValidationUtils.requireNonNegativeWeight(1e-100)
  }

  test("requireNonNegativeWeight with invalid weights") {
    val exception1 = intercept[IllegalArgumentException] {
      ValidationUtils.requireNonNegativeWeight(-1.0)
    }
    assert(exception1.getMessage.contains("cannot be negative"))

    val exception2 = intercept[IllegalArgumentException] {
      ValidationUtils.requireNonNegativeWeight(Double.MinValue)
    }
    assert(exception2.getMessage.contains("cannot be negative"))

    val exception3 = intercept[IllegalArgumentException] {
      ValidationUtils.requireNonNegativeWeight(Double.NegativeInfinity)
    }
    assert(exception3.getMessage.contains("cannot be negative"))
  }

  test("requireValidProbability with valid probabilities") {
    // Should pass without exception
    ValidationUtils.requireValidProbability(0.0)
    ValidationUtils.requireValidProbability(1.0)
    ValidationUtils.requireValidProbability(0.5)
    ValidationUtils.requireValidProbability(0.001)
    ValidationUtils.requireValidProbability(0.999)
  }

  test("requireValidProbability with invalid probabilities") {
    val exception1 = intercept[IllegalArgumentException] {
      ValidationUtils.requireValidProbability(-0.1)
    }
    assert(exception1.getMessage.contains("must be between 0 and 1"))

    val exception2 = intercept[IllegalArgumentException] {
      ValidationUtils.requireValidProbability(1.1)
    }
    assert(exception2.getMessage.contains("must be between 0 and 1"))

    val exception3 = intercept[IllegalArgumentException] {
      ValidationUtils.requireValidProbability(Double.NaN)
    }
    assert(exception3.getMessage.contains("must be between 0 and 1"))

    val exception4 = intercept[IllegalArgumentException] {
      ValidationUtils.requireValidProbability(Double.PositiveInfinity)
    }
    assert(exception4.getMessage.contains("must be between 0 and 1"))
  }

  test("requirePositiveVector with valid vectors") {
    // Should pass without exception
    ValidationUtils.requirePositiveVector(Vectors.dense(1.0, 2.0, 3.0))
    ValidationUtils.requirePositiveVector(Vectors.dense(0.001, 1e100))
    // Sparse vectors only check explicitly stored values (which are positive)
    ValidationUtils.requirePositiveVector(Vectors.sparse(5, Seq((1, 2.0), (3, 4.0))))
  }

  test("requirePositiveVector with invalid vectors") {
    val exception1 = intercept[IllegalArgumentException] {
      ValidationUtils.requirePositiveVector(Vectors.dense(1.0, 0.0, 2.0))
    }
    assert(exception1.getMessage.contains("Vector elements must be positive"))

    val exception2 = intercept[IllegalArgumentException] {
      ValidationUtils.requirePositiveVector(Vectors.dense(1.0, -1.0, 2.0))
    }
    assert(exception2.getMessage.contains("Vector elements must be positive"))

    val exception3 = intercept[IllegalArgumentException] {
      ValidationUtils.requirePositiveVector(Vectors.sparse(5, Seq((1, 0.0), (3, 4.0))))
    }
    assert(exception3.getMessage.contains("Vector elements must be positive"))
  }

  test("requireNonNegativeVector with valid vectors") {
    // Should pass without exception
    ValidationUtils.requireNonNegativeVector(Vectors.dense(0.0, 1.0, 2.0))
    ValidationUtils.requireNonNegativeVector(Vectors.dense(0.0, 0.0, 0.0))
    ValidationUtils.requireNonNegativeVector(Vectors.sparse(5, Seq((1, 0.0), (3, 4.0))))
  }

  test("requireNonNegativeVector with invalid vectors") {
    val exception1 = intercept[IllegalArgumentException] {
      ValidationUtils.requireNonNegativeVector(Vectors.dense(1.0, -1.0, 2.0))
    }
    assert(exception1.getMessage.contains("Vector elements cannot be negative"))

    val exception2 = intercept[IllegalArgumentException] {
      ValidationUtils.requireNonNegativeVector(Vectors.sparse(5, Seq((1, -2.0), (3, 4.0))))
    }
    assert(exception2.getMessage.contains("Vector elements cannot be negative"))
  }

  test("requireValidArrayIndex with valid indices") {
    // Should pass without exception
    ValidationUtils.requireValidArrayIndex(0, 5)
    ValidationUtils.requireValidArrayIndex(4, 5)
    ValidationUtils.requireValidArrayIndex(0, 1)
  }

  test("requireValidArrayIndex with invalid indices") {
    val exception1 = intercept[IllegalArgumentException] {
      ValidationUtils.requireValidArrayIndex(-1, 5)
    }
    assert(exception1.getMessage.contains("Index must be between 0 and 4"))

    val exception2 = intercept[IllegalArgumentException] {
      ValidationUtils.requireValidArrayIndex(5, 5)
    }
    assert(exception2.getMessage.contains("Index must be between 0 and 4"))

    val exception3 = intercept[IllegalArgumentException] {
      ValidationUtils.requireValidArrayIndex(1, 0)
    }
    assert(exception3.getMessage.contains("Index must be between"))
  }

  test("requireSameDimensions with matching vectors") {
    // Should pass without exception
    ValidationUtils.requireSameDimensions(
      Vectors.dense(1.0, 2.0), 
      Vectors.dense(3.0, 4.0)
    )
    ValidationUtils.requireSameDimensions(
      Vectors.sparse(5, Seq((1, 2.0))), 
      Vectors.sparse(5, Seq((3, 4.0)))
    )
  }

  test("requireSameDimensions with mismatched vectors") {
    val exception1 = intercept[IllegalArgumentException] {
      ValidationUtils.requireSameDimensions(
        Vectors.dense(1.0, 2.0), 
        Vectors.dense(1.0, 2.0, 3.0)
      )
    }
    assert(exception1.getMessage.contains("Vectors must have same dimensions"))
    assert(exception1.getMessage.contains("2 and 3"))

    val exception2 = intercept[IllegalArgumentException] {
      ValidationUtils.requireSameDimensions(
        Vectors.sparse(3, Seq((1, 2.0))), 
        Vectors.sparse(5, Seq((3, 4.0)))
      )
    }
    assert(exception2.getMessage.contains("Vectors must have same dimensions"))
    assert(exception2.getMessage.contains("3 and 5"))
  }

  test("requireNonEmpty with valid collections") {
    // Should pass without exception
    ValidationUtils.requireNonEmpty(Seq(1, 2, 3))
    ValidationUtils.requireNonEmpty(List("a"))
    ValidationUtils.requireNonEmpty(Array(1.0))
  }

  test("requireNonEmpty with empty collections") {
    val exception1 = intercept[IllegalArgumentException] {
      ValidationUtils.requireNonEmpty(Seq.empty[Int])
    }
    assert(exception1.getMessage.contains("Collection must not be empty"))

    val exception2 = intercept[IllegalArgumentException] {
      ValidationUtils.requireNonEmpty(List.empty[String], "Custom collection")
    }
    assert(exception2.getMessage.contains("Custom collection must not be empty"))
  }

  test("requirePositive int with valid values") {
    // Should pass without exception
    ValidationUtils.requirePositive(1)
    ValidationUtils.requirePositive(Int.MaxValue)
    ValidationUtils.requirePositive(1000)
  }

  test("requirePositive int with invalid values") {
    val exception1 = intercept[IllegalArgumentException] {
      ValidationUtils.requirePositive(0)
    }
    assert(exception1.getMessage.contains("Value must be positive"))

    val exception2 = intercept[IllegalArgumentException] {
      ValidationUtils.requirePositive(-1)
    }
    assert(exception2.getMessage.contains("Value must be positive"))

    val exception3 = intercept[IllegalArgumentException] {
      ValidationUtils.requirePositive(Int.MinValue, "Custom value")
    }
    assert(exception3.getMessage.contains("Custom value must be positive"))
  }

  test("requireNonNegative int with valid values") {
    // Should pass without exception
    ValidationUtils.requireNonNegative(0)
    ValidationUtils.requireNonNegative(1)
    ValidationUtils.requireNonNegative(Int.MaxValue)
  }

  test("requireNonNegative int with invalid values") {
    val exception1 = intercept[IllegalArgumentException] {
      ValidationUtils.requireNonNegative(-1)
    }
    assert(exception1.getMessage.contains("Value cannot be negative"))

    val exception2 = intercept[IllegalArgumentException] {
      ValidationUtils.requireNonNegative(Int.MinValue, "Custom value")
    }
    assert(exception2.getMessage.contains("Custom value cannot be negative"))
  }

  test("requireFinite with valid values") {
    // Should pass without exception
    ValidationUtils.requireFinite(0.0)
    ValidationUtils.requireFinite(Double.MaxValue)
    ValidationUtils.requireFinite(Double.MinValue)
    ValidationUtils.requireFinite(-1e100)
    ValidationUtils.requireFinite(1e-100)
  }

  test("requireFinite with invalid values") {
    val exception1 = intercept[IllegalArgumentException] {
      ValidationUtils.requireFinite(Double.NaN)
    }
    assert(exception1.getMessage.contains("Value must be finite"))

    val exception2 = intercept[IllegalArgumentException] {
      ValidationUtils.requireFinite(Double.PositiveInfinity)
    }
    assert(exception2.getMessage.contains("Value must be finite"))

    val exception3 = intercept[IllegalArgumentException] {
      ValidationUtils.requireFinite(Double.NegativeInfinity, "Custom value")
    }
    assert(exception3.getMessage.contains("Custom value must be finite"))
  }

  test("requireNotNull with valid values") {
    // Should pass without exception and return the value
    val result1 = ValidationUtils.requireNotNull("test")
    assert(result1 == "test")

    val list = List(1, 2, 3)
    val result2 = ValidationUtils.requireNotNull(list)
    assert(result2 == list)
  }

  test("requireNotNull with null values") {
    val exception1 = intercept[IllegalArgumentException] {
      ValidationUtils.requireNotNull(null)
    }
    assert(exception1.getMessage.contains("Value must not be null"))

    val exception2 = intercept[IllegalArgumentException] {
      ValidationUtils.requireNotNull(null, "Custom parameter")
    }
    assert(exception2.getMessage.contains("Custom parameter must not be null"))
  }

  test("error messages contain context information") {
    val exception = intercept[IllegalArgumentException] {
      ValidationUtils.requirePositiveWeight(-1.0, "Learning rate")
    }
    assert(exception.getMessage.contains("Learning rate"))
    assert(exception.getMessage.contains("-1.0"))
  }

  test("validation with extreme values") {
    // Test with very small positive values
    ValidationUtils.requirePositiveWeight(Double.MinPositiveValue)
    ValidationUtils.requireNonNegativeWeight(Double.MinPositiveValue)
    ValidationUtils.requireFinite(Double.MinPositiveValue)

    // Test with very large values
    ValidationUtils.requirePositiveWeight(Double.MaxValue)
    ValidationUtils.requireNonNegativeWeight(Double.MaxValue)
    ValidationUtils.requireFinite(Double.MaxValue)

    // Test probability edge cases
    ValidationUtils.requireValidProbability(0.0)
    ValidationUtils.requireValidProbability(1.0)
    ValidationUtils.requireValidProbability(Double.MinPositiveValue)
    ValidationUtils.requireValidProbability(1.0 - Double.MinPositiveValue)
  }
}