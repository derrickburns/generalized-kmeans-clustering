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

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class GKMErrorSuite extends AnyFunSuite with Matchers {

  // ========================================
  // Validation Errors
  // ========================================

  test("InvalidK should have correct message and category") {
    val error = InvalidK(k = 0, n = 100)

    assert(error.message.contains("Invalid k=0"))
    assert(error.message.contains("100"))
    assert(error.category == ErrorCategory.Validation)
    assert(error.toException.isInstanceOf[IllegalArgumentException])
  }

  test("InvalidTolerance should have correct message") {
    val error = InvalidTolerance(tolerance = -0.1)

    assert(error.message.contains("-0.1"))
    assert(error.message.contains("must be >= 0.0"))
    assert(error.category == ErrorCategory.Validation)
  }

  test("InvalidMaxIterations should have correct message") {
    val error = InvalidMaxIterations(maxIter = 0)

    assert(error.message.contains("0"))
    assert(error.message.contains("must be >= 1"))
    assert(error.category == ErrorCategory.Validation)
  }

  test("InvalidSeed should have correct message") {
    val error = InvalidSeed(seed = -1)

    assert(error.message.contains("-1"))
    assert(error.category == ErrorCategory.Validation)
  }

  test("InvalidWeight should format message with row index") {
    val errorWithRow = InvalidWeight(weight = -1.0, rowIndex = Some(42))
    assert(errorWithRow.message.contains("-1.0"))
    assert(errorWithRow.message.contains("row 42"))

    val errorWithoutRow = InvalidWeight(weight = 0.0, rowIndex = None)
    assert(errorWithoutRow.message.contains("0.0"))
    assert(!errorWithoutRow.message.contains("row"))
  }

  // ========================================
  // Configuration Errors
  // ========================================

  test("UnknownKernel should list supported kernels") {
    val error = UnknownKernel("unknown", Seq("euclidean", "kl", "manhattan"))

    assert(error.message.contains("unknown"))
    assert(error.message.contains("euclidean"))
    assert(error.message.contains("kl"))
    assert(error.message.contains("manhattan"))
    assert(error.category == ErrorCategory.Configuration)
  }

  test("UnknownInitMethod should list supported methods") {
    val error = UnknownInitMethod("unknown", Seq("random", "kmeans++"))

    assert(error.message.contains("unknown"))
    assert(error.message.contains("random"))
    assert(error.message.contains("kmeans++"))
    assert(error.category == ErrorCategory.Configuration)
  }

  test("IncompatibleTransform should explain reason") {
    val error = IncompatibleTransform("log1p", "kl", "KL requires positive features")

    assert(error.message.contains("log1p"))
    assert(error.message.contains("kl"))
    assert(error.message.contains("KL requires positive features"))
    assert(error.category == ErrorCategory.Configuration)
  }

  test("MissingColumn should list available columns") {
    val error = MissingColumn("features", Seq("id", "label", "weight"))

    assert(error.message.contains("features"))
    assert(error.message.contains("id"))
    assert(error.message.contains("label"))
    assert(error.message.contains("weight"))
    assert(error.category == ErrorCategory.Configuration)
  }

  // ========================================
  // Data Errors
  // ========================================

  test("EmptyDataset should format with context") {
    val errorWithContext = EmptyDataset("after filtering")
    assert(errorWithContext.message.contains("after filtering"))

    val errorWithoutContext = EmptyDataset()
    assert(errorWithoutContext.message == "Empty dataset")
    assert(errorWithoutContext.category == ErrorCategory.Data)
  }

  test("InvalidFeatures should format with all context") {
    val error1 = InvalidFeatures("contains NaN", Some(42), Some("features"))
    assert(error1.message.contains("contains NaN"))
    assert(error1.message.contains("row 42"))
    assert(error1.message.contains("features"))

    val error2 = InvalidFeatures("negative values", Some(10), None)
    assert(error2.message.contains("row 10"))
    assert(!error2.message.contains("column"))

    val error3 = InvalidFeatures("infinite values", None, Some("data"))
    assert(error3.message.contains("data"))
    assert(!error3.message.contains("row"))

    val error4 = InvalidFeatures("zeros", None, None)
    assert(!error4.message.contains("row"))
    assert(!error4.message.contains("column"))
  }

  test("DimensionMismatch should format with row index") {
    val errorWithRow = DimensionMismatch(10, 5, Some(42))
    assert(errorWithRow.message.contains("expected 10"))
    assert(errorWithRow.message.contains("got 5"))
    assert(errorWithRow.message.contains("row 42"))

    val errorWithoutRow = DimensionMismatch(10, 5, None)
    assert(!errorWithoutRow.message.contains("row"))
    assert(errorWithoutRow.category == ErrorCategory.Data)
  }

  // ========================================
  // Convergence Errors
  // ========================================

  test("ConvergenceFailure should format with metrics") {
    val error = ConvergenceFailure(maxIter = 100, finalCost = 123.45, costDelta = 0.001)

    assert(error.message.contains("100"))
    assert(error.message.contains("123.45"))
    assert(error.message.contains("0.001"))
    assert(error.category == ErrorCategory.Convergence)
  }

  test("AllClustersEmpty should include iteration") {
    val error = AllClustersEmpty(iteration = 5)

    assert(error.message.contains("5"))
    assert(error.category == ErrorCategory.Convergence)
  }

  test("CostIncreased should show delta") {
    val error = CostIncreased(iteration = 3, previousCost = 100.0, currentCost = 105.0)

    assert(error.message.contains("3"))
    assert(error.message.contains("100.0"))
    assert(error.message.contains("105.0"))
    assert(error.message.contains("5.0")) // delta
    assert(error.category == ErrorCategory.Convergence)
  }

  // ========================================
  // Internal Errors
  // ========================================

  test("UnexpectedNull should include context") {
    val error = UnexpectedNull("cluster centers array")

    assert(error.message.contains("cluster centers array"))
    assert(error.category == ErrorCategory.Internal)
  }

  test("InvalidState should include description") {
    val error = InvalidState("center count != k")

    assert(error.message.contains("center count != k"))
    assert(error.category == ErrorCategory.Internal)
  }

  test("AssertionFailed should format with context") {
    val errorWithContext = AssertionFailed("k > 0", "during initialization")
    assert(errorWithContext.message.contains("k > 0"))
    assert(errorWithContext.message.contains("during initialization"))

    val errorWithoutContext = AssertionFailed("n > 0")
    assert(errorWithoutContext.message.contains("n > 0"))
    assert(!errorWithoutContext.message.contains("("))
  }

  // ========================================
  // Error Result Type
  // ========================================

  test("GKMResult.success should create Success") {
    val result = GKMResult.success(42)

    assert(result.isSuccess)
    assert(!result.isFailure)
    assert(result.get == 42)
    assert(result.getOrElse(0) == 42)
    assert(result.error.isEmpty)
  }

  test("GKMResult.failure should create Failure") {
    val error  = InvalidK(0, 100)
    val result = GKMResult.failure[Int](error)

    assert(!result.isSuccess)
    assert(result.isFailure)
    assert(result.getOrElse(42) == 42)
    assert(result.error.contains(error))
  }

  test("GKMResult.Success.get should return value") {
    val result = GKMResult.success("hello")
    assert(result.get == "hello")
  }

  test("GKMResult.Failure.get should throw exception") {
    val error  = InvalidK(0, 100)
    val result = GKMResult.failure[Int](error)

    intercept[IllegalArgumentException] {
      result.get
    }
  }

  test("GKMResult.map should transform success") {
    val result = GKMResult.success(5)
    val mapped = result.map(_ * 2)

    assert(mapped.isSuccess)
    assert(mapped.get == 10)
  }

  test("GKMResult.map should preserve failure") {
    val error  = InvalidK(0, 100)
    val result = GKMResult.failure[Int](error)
    val mapped = result.map(_ * 2)

    assert(mapped.isFailure)
    assert(mapped.error.contains(error))
  }

  test("GKMResult.flatMap should chain operations") {
    val result1 = GKMResult.success(5)
    val result2 = result1.flatMap(x => GKMResult.success(x * 2))

    assert(result2.isSuccess)
    assert(result2.get == 10)
  }

  test("GKMResult.flatMap should short-circuit on failure") {
    val error   = InvalidK(0, 100)
    val result1 = GKMResult.failure[Int](error)
    val result2 = result1.flatMap(x => GKMResult.success(x * 2))

    assert(result2.isFailure)
    assert(result2.error.contains(error))
  }

  test("GKMResult.flatMap should propagate failure from mapper") {
    val error   = InvalidK(0, 100)
    val result1 = GKMResult.success(5)
    val result2 = result1.flatMap(_ => GKMResult.failure[Int](error))

    assert(result2.isFailure)
    assert(result2.error.contains(error))
  }

  test("GKMResult.fromOption should handle Some") {
    val result = GKMResult.fromOption(Some(42), InvalidK(0, 100))

    assert(result.isSuccess)
    assert(result.get == 42)
  }

  test("GKMResult.fromOption should handle None") {
    val error  = InvalidK(0, 100)
    val result = GKMResult.fromOption(None, error)

    assert(result.isFailure)
    assert(result.error.contains(error))
  }

  test("GKMResult.attempt should handle success") {
    val result = GKMResult.attempt(42)(e => UnexpectedNull(e.getMessage))

    assert(result.isSuccess)
    assert(result.get == 42)
  }

  test("GKMResult.attempt should handle exception") {
    val result = GKMResult.attempt {
      throw new RuntimeException("test error")
    }(e => UnexpectedNull(e.getMessage))

    assert(result.isFailure)
    assert(result.error.exists(_.message.contains("test error")))
  }

  test("GKMResult for-comprehension should work") {
    val result = for {
      x <- GKMResult.success(5)
      y <- GKMResult.success(10)
      z <- GKMResult.success(x + y)
    } yield z

    assert(result.isSuccess)
    assert(result.get == 15)
  }

  test("GKMResult for-comprehension should short-circuit") {
    val error  = InvalidK(0, 100)
    val result = for {
      x <- GKMResult.success(5)
      y <- GKMResult.failure[Int](error)
      z <- GKMResult.success(x + y)
    } yield z

    assert(result.isFailure)
    assert(result.error.contains(error))
  }

  test("Error categories should convert to appropriate exceptions") {
    assert(InvalidK(0, 100).toException.isInstanceOf[IllegalArgumentException])
    assert(UnknownKernel("test", Seq()).toException.isInstanceOf[IllegalStateException])
    assert(ConvergenceFailure(100, 1.0, 0.001).toException.isInstanceOf[RuntimeException])
    assert(EmptyDataset().toException.isInstanceOf[IllegalArgumentException])
    assert(UnexpectedNull("test").toException.isInstanceOf[RuntimeException])
  }

  test("GKMError should be serializable") {
    val error: GKMError = InvalidK(0, 100)

    val stream = new java.io.ByteArrayOutputStream()
    val oos    = new java.io.ObjectOutputStream(stream)
    oos.writeObject(error)
    oos.close()

    val bytes = stream.toByteArray
    assert(bytes.nonEmpty)
  }

  test("GKMResult should be serializable") {
    val result: GKMResult[Int] = GKMResult.success(42)

    val stream = new java.io.ByteArrayOutputStream()
    val oos    = new java.io.ObjectOutputStream(stream)
    oos.writeObject(result)
    oos.close()

    val bytes = stream.toByteArray
    assert(bytes.nonEmpty)
  }

  test("Pattern matching on GKMResult should work") {
    def process(result: GKMResult[Int]): String = result match {
      case GKMResult.Success(value) => s"Got $value"
      case GKMResult.Failure(error) => s"Error: ${error.message}"
    }

    assert(process(GKMResult.success(42)) == "Got 42")
    assert(process(GKMResult.failure(InvalidK(0, 100))).startsWith("Error:"))
  }
}
