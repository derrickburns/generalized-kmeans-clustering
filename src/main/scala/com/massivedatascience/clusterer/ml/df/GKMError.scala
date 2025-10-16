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

/** Typed errors for generalized k-means clustering.
  *
  * This ADT provides a structured way to represent and handle errors that can occur during clustering. Each error type
  * carries relevant context and provides clear error messages.
  *
  * Design principles:
  * - Exhaustive pattern matching for error handling
  * - Context-rich error messages for debugging
  * - Type-safe error propagation
  * - Easy to extend with new error types
  */
sealed trait GKMError {

  /** Human-readable error message */
  def message: String

  /** Error category for grouping */
  def category: ErrorCategory

  /** Convert to exception for throwing */
  def toException: Exception = category match {
    case ErrorCategory.Validation   => new IllegalArgumentException(message)
    case ErrorCategory.Configuration => new IllegalStateException(message)
    case ErrorCategory.Convergence  => new RuntimeException(message)
    case ErrorCategory.Data         => new IllegalArgumentException(message)
    case ErrorCategory.Internal     => new RuntimeException(message)
  }
}

/** Error categories for classification */
sealed trait ErrorCategory

object ErrorCategory {

  /** Validation errors (invalid input parameters) */
  case object Validation extends ErrorCategory

  /** Configuration errors (invalid algorithm configuration) */
  case object Configuration extends ErrorCategory

  /** Convergence errors (algorithm failed to converge) */
  case object Convergence extends ErrorCategory

  /** Data errors (invalid or malformed data) */
  case object Data extends ErrorCategory

  /** Internal errors (unexpected algorithm state) */
  case object Internal extends ErrorCategory
}

// ========================================
// Validation Errors
// ========================================

/** Invalid value for k (number of clusters).
  *
  * @param k
  *   the invalid k value
  * @param n
  *   the dataset size
  */
case class InvalidK(k: Int, n: Long) extends GKMError {
  override def message: String = s"Invalid k=$k: must be positive and <= dataset size ($n)"
  override def category: ErrorCategory = ErrorCategory.Validation
}

/** Invalid tolerance value.
  *
  * @param tolerance
  *   the invalid tolerance value
  */
case class InvalidTolerance(tolerance: Double) extends GKMError {
  override def message: String = s"Invalid tolerance=$tolerance: must be >= 0.0"
  override def category: ErrorCategory = ErrorCategory.Validation
}

/** Invalid max iterations value.
  *
  * @param maxIter
  *   the invalid max iterations value
  */
case class InvalidMaxIterations(maxIter: Int) extends GKMError {
  override def message: String = s"Invalid maxIter=$maxIter: must be >= 1"
  override def category: ErrorCategory = ErrorCategory.Validation
}

/** Invalid seed value.
  *
  * @param seed
  *   the invalid seed value
  */
case class InvalidSeed(seed: Long) extends GKMError {
  override def message: String = s"Invalid seed=$seed: must be >= 0"
  override def category: ErrorCategory = ErrorCategory.Validation
}

/** Invalid weight value.
  *
  * @param weight
  *   the invalid weight
  * @param rowIndex
  *   optional row index where error occurred
  */
case class InvalidWeight(weight: Double, rowIndex: Option[Long] = None) extends GKMError {
  override def message: String = {
    val location = rowIndex.map(i => s" at row $i").getOrElse("")
    s"Invalid weight=$weight$location: must be > 0.0 and finite"
  }
  override def category: ErrorCategory = ErrorCategory.Validation
}

// ========================================
// Configuration Errors
// ========================================

/** Unknown divergence/kernel name.
  *
  * @param name
  *   the unknown kernel name
  * @param supported
  *   list of supported kernel names
  */
case class UnknownKernel(name: String, supported: Seq[String]) extends GKMError {
  override def message: String = s"Unknown kernel '$name'. Supported: ${supported.mkString(", ")}"
  override def category: ErrorCategory = ErrorCategory.Configuration
}

/** Unknown initialization method.
  *
  * @param method
  *   the unknown initialization method
  * @param supported
  *   list of supported methods
  */
case class UnknownInitMethod(method: String, supported: Seq[String]) extends GKMError {
  override def message: String = s"Unknown initialization method '$method'. Supported: ${supported.mkString(", ")}"
  override def category: ErrorCategory = ErrorCategory.Configuration
}

/** Incompatible feature transform for the given kernel.
  *
  * @param transform
  *   the transform name
  * @param kernel
  *   the kernel name
  * @param reason
  *   explanation of incompatibility
  */
case class IncompatibleTransform(transform: String, kernel: String, reason: String) extends GKMError {
  override def message: String = s"Transform '$transform' incompatible with kernel '$kernel': $reason"
  override def category: ErrorCategory = ErrorCategory.Configuration
}

/** Missing required column in DataFrame.
  *
  * @param columnName
  *   the missing column name
  * @param availableColumns
  *   list of available columns
  */
case class MissingColumn(columnName: String, availableColumns: Seq[String]) extends GKMError {
  override def message: String = s"Missing required column '$columnName'. Available: ${availableColumns.mkString(", ")}"
  override def category: ErrorCategory = ErrorCategory.Configuration
}

// ========================================
// Data Errors
// ========================================

/** Empty dataset error.
  *
  * @param context
  *   additional context about where empty dataset was encountered
  */
case class EmptyDataset(context: String = "") extends GKMError {
  override def message: String = {
    if (context.nonEmpty) s"Empty dataset: $context"
    else "Empty dataset"
  }
  override def category: ErrorCategory = ErrorCategory.Data
}

/** Invalid feature vector.
  *
  * @param reason
  *   explanation of what's invalid (e.g., "contains NaN", "negative values")
  * @param rowIndex
  *   optional row index where error occurred
  * @param columnName
  *   optional column name
  */
case class InvalidFeatures(reason: String, rowIndex: Option[Long] = None, columnName: Option[String] = None)
    extends GKMError {
  override def message: String = {
    val location = (rowIndex, columnName) match {
      case (Some(row), Some(col)) => s" in column '$col' at row $row"
      case (Some(row), None)      => s" at row $row"
      case (None, Some(col))      => s" in column '$col'"
      case (None, None)           => ""
    }
    s"Invalid features$location: $reason"
  }
  override def category: ErrorCategory = ErrorCategory.Data
}

/** Inconsistent feature dimensionality.
  *
  * @param expected
  *   expected dimension
  * @param actual
  *   actual dimension found
  * @param rowIndex
  *   optional row index where mismatch occurred
  */
case class DimensionMismatch(expected: Int, actual: Int, rowIndex: Option[Long] = None) extends GKMError {
  override def message: String = {
    val location = rowIndex.map(i => s" at row $i").getOrElse("")
    s"Dimension mismatch$location: expected $expected, got $actual"
  }
  override def category: ErrorCategory = ErrorCategory.Data
}

// ========================================
// Convergence Errors
// ========================================

/** Failed to converge within max iterations.
  *
  * @param maxIter
  *   maximum iterations reached
  * @param finalCost
  *   final clustering cost
  * @param costDelta
  *   final cost change
  */
case class ConvergenceFailure(maxIter: Int, finalCost: Double, costDelta: Double) extends GKMError {
  override def message: String =
    f"Failed to converge after $maxIter iterations (final cost=$finalCost%.4f, delta=$costDelta%.6f)"
  override def category: ErrorCategory = ErrorCategory.Convergence
}

/** All clusters became empty (degenerate clustering).
  *
  * @param iteration
  *   iteration where this occurred
  */
case class AllClustersEmpty(iteration: Int) extends GKMError {
  override def message: String = s"All clusters became empty at iteration $iteration"
  override def category: ErrorCategory = ErrorCategory.Convergence
}

/** Cost increased instead of decreasing (algorithm divergence).
  *
  * @param iteration
  *   iteration where cost increased
  * @param previousCost
  *   cost from previous iteration
  * @param currentCost
  *   cost from current iteration
  */
case class CostIncreased(iteration: Int, previousCost: Double, currentCost: Double) extends GKMError {
  override def message: String =
    f"Cost increased at iteration $iteration: $previousCost%.4f -> $currentCost%.4f (delta=${currentCost - previousCost}%.4f)"
  override def category: ErrorCategory = ErrorCategory.Convergence
}

// ========================================
// Internal Errors
// ========================================

/** Unexpected null value in internal computation.
  *
  * @param context
  *   where the null was encountered
  */
case class UnexpectedNull(context: String) extends GKMError {
  override def message: String = s"Unexpected null value: $context"
  override def category: ErrorCategory = ErrorCategory.Internal
}

/** Invalid internal state.
  *
  * @param description
  *   description of the invalid state
  */
case class InvalidState(description: String) extends GKMError {
  override def message: String = s"Invalid internal state: $description"
  override def category: ErrorCategory = ErrorCategory.Internal
}

/** Assertion failure in internal logic.
  *
  * @param assertion
  *   the failed assertion
  * @param context
  *   additional context
  */
case class AssertionFailed(assertion: String, context: String = "") extends GKMError {
  override def message: String = {
    if (context.nonEmpty) s"Assertion failed: $assertion ($context)"
    else s"Assertion failed: $assertion"
  }
  override def category: ErrorCategory = ErrorCategory.Internal
}

// ========================================
// Error Result Type
// ========================================

/** Result type for operations that can fail with typed errors.
  *
  * This provides a functional approach to error handling, allowing errors to be composed and transformed without
  * throwing exceptions.
  *
  * Example usage:
  * {{{
  *   def validateK(k: Int, n: Long): GKMResult[Int] = {
  *     if (k <= 0 || k > n) GKMResult.failure(InvalidK(k, n))
  *     else GKMResult.success(k)
  *   }
  *
  *   validateK(5, 100) match {
  *     case GKMResult.Success(validK) => println(s"Valid k: $validK")
  *     case GKMResult.Failure(error) => println(error.message)
  *   }
  * }}}
  */
sealed trait GKMResult[+A] {

  /** Check if this is a success */
  def isSuccess: Boolean

  /** Check if this is a failure */
  def isFailure: Boolean = !isSuccess

  /** Get value or throw exception */
  def get: A

  /** Get value or return default */
  def getOrElse[B >: A](default: => B): B

  /** Transform success value */
  def map[B](f: A => B): GKMResult[B]

  /** Chain operations that can fail */
  def flatMap[B](f: A => GKMResult[B]): GKMResult[B]

  /** Get error if failure */
  def error: Option[GKMError]
}

object GKMResult {

  /** Successful result */
  case class Success[A](value: A) extends GKMResult[A] {
    override def isSuccess: Boolean = true
    override def get: A = value
    override def getOrElse[B >: A](default: => B): B = value
    override def map[B](f: A => B): GKMResult[B] = Success(f(value))
    override def flatMap[B](f: A => GKMResult[B]): GKMResult[B] = f(value)
    override def error: Option[GKMError] = None
  }

  /** Failed result */
  case class Failure[A](err: GKMError) extends GKMResult[A] {
    override def isSuccess: Boolean = false
    override def get: A = throw err.toException
    override def getOrElse[B >: A](default: => B): B = default
    override def map[B](f: A => B): GKMResult[B] = Failure(err)
    override def flatMap[B](f: A => GKMResult[B]): GKMResult[B] = Failure(err)
    override def error: Option[GKMError] = Some(err)
  }

  /** Create a successful result */
  def success[A](value: A): GKMResult[A] = Success(value)

  /** Create a failed result */
  def failure[A](error: GKMError): GKMResult[A] = Failure(error)

  /** Create result from option */
  def fromOption[A](opt: Option[A], error: => GKMError): GKMResult[A] = {
    opt.map(Success(_)).getOrElse(Failure(error))
  }

  /** Create result from try block */
  def attempt[A](f: => A)(onError: Throwable => GKMError): GKMResult[A] = {
    try {
      Success(f)
    } catch {
      case e: Throwable => Failure(onError(e))
    }
  }
}
