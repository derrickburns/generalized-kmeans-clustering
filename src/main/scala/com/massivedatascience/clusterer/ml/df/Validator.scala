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

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{ DataFrame, Row }
import org.apache.spark.sql.functions._

/** Result of validation containing any violations found.
  *
  * @param isValid
  *   true if no violations were found
  * @param violations
  *   list of validation violations with details
  */
case class ValidationResult(
    isValid: Boolean,
    violations: Seq[ValidationViolation]
) {

  /** Combine this result with another result. */
  def and(other: ValidationResult): ValidationResult = {
    ValidationResult(
      isValid = this.isValid && other.isValid,
      violations = this.violations ++ other.violations
    )
  }

  /** Get summary message of all violations. */
  def message: String = {
    if (isValid) {
      "Validation passed"
    } else {
      s"Validation failed with ${violations.size} violations:\n" +
        violations.map(v => s"  - ${v.message}").mkString("\n")
    }
  }

  /** Throw exception if validation failed. */
  def getOrThrow(): Unit = {
    if (!isValid) {
      throw new IllegalArgumentException(message)
    }
  }
}

object ValidationResult {

  /** Create a successful validation result. */
  def success: ValidationResult = ValidationResult(isValid = true, violations = Seq.empty)

  /** Create a failed validation result. */
  def failure(violation: ValidationViolation): ValidationResult = {
    ValidationResult(isValid = false, violations = Seq(violation))
  }

  /** Create a failed validation result with multiple violations. */
  def failures(violations: Seq[ValidationViolation]): ValidationResult = {
    ValidationResult(isValid = false, violations = violations)
  }
}

/** A validation violation with details about what failed.
  *
  * @param message
  *   human-readable error message
  * @param field
  *   name of field that failed validation
  * @param violationType
  *   type of violation (e.g., "nan", "negative", "infinite")
  * @param sampleRows
  *   sample rows that violated the constraint (up to 5)
  */
case class ValidationViolation(
    message: String,
    field: String,
    violationType: String,
    sampleRows: Seq[Row] = Seq.empty
)

/** Validator trait for composable validation rules.
  *
  * Validators can be combined using `and` to build complex validation chains. Each validator checks
  * a single concern and returns a ValidationResult.
  *
  * Example usage:
  * {{{
  *   val validator = Validator.finite("features") and Validator.positive("weight")
  *   val result = validator.validate(df)
  *   result.getOrThrow() // Throws if validation fails
  * }}}
  */
trait Validator extends Serializable {

  /** Name of this validator */
  def name: String

  /** Validate the DataFrame and return result.
    *
    * @param df
    *   DataFrame to validate
    * @return
    *   validation result with any violations
    */
  def validate(df: DataFrame): ValidationResult

  /** Combine this validator with another validator. */
  def and(other: Validator): Validator = {
    CombinedValidator(this, other)
  }
}

/** Validator that combines two validators. */
case class CombinedValidator(first: Validator, second: Validator) extends Validator {
  override def name: String = s"${first.name} and ${second.name}"

  override def validate(df: DataFrame): ValidationResult = {
    val result1 = first.validate(df)
    val result2 = second.validate(df)
    result1.and(result2)
  }
}

/** Validator that checks if a numeric column contains no NaN values. */
case class NoNaNValidator(columnName: String, maxSampleRows: Int = 5) extends Validator {
  override def name: String = s"noNaN($columnName)"

  override def validate(df: DataFrame): ValidationResult = {
    try {
      // For vector columns, check each element
      if (df.schema(columnName).dataType.typeName == "vector") {
        val hasNaN = df
          .withColumn(
            "_has_nan",
            udf((v: Vector) => v.toArray.exists(_.isNaN)).apply(col(columnName))
          )
          .filter(col("_has_nan"))

        val count = hasNaN.count()
        if (count > 0) {
          val samples = hasNaN.drop("_has_nan").take(maxSampleRows).toIndexedSeq
          ValidationResult.failure(
            ValidationViolation(
              message = s"Column '$columnName' contains $count rows with NaN values",
              field = columnName,
              violationType = "nan",
              sampleRows = samples
            )
          )
        } else {
          ValidationResult.success
        }
      } else {
        // For scalar columns
        val hasNaN = df.filter(col(columnName).isNaN)
        val count  = hasNaN.count()
        if (count > 0) {
          val samples = hasNaN.take(maxSampleRows).toIndexedSeq
          ValidationResult.failure(
            ValidationViolation(
              message = s"Column '$columnName' contains $count NaN values",
              field = columnName,
              violationType = "nan",
              sampleRows = samples
            )
          )
        } else {
          ValidationResult.success
        }
      }
    } catch {
      case e: Exception =>
        ValidationResult.failure(
          ValidationViolation(
            message = s"Failed to validate column '$columnName': ${e.getMessage}",
            field = columnName,
            violationType = "error"
          )
        )
    }
  }
}

/** Validator that checks if a numeric column contains no infinite values. */
case class FiniteValidator(columnName: String, maxSampleRows: Int = 5) extends Validator {
  override def name: String = s"finite($columnName)"

  override def validate(df: DataFrame): ValidationResult = {
    try {
      // For vector columns, check each element
      if (df.schema(columnName).dataType.typeName == "vector") {
        val hasInf = df
          .withColumn(
            "_has_inf",
            udf((v: Vector) => v.toArray.exists(_.isInfinite)).apply(col(columnName))
          )
          .filter(col("_has_inf"))

        val count = hasInf.count()
        if (count > 0) {
          val samples = hasInf.drop("_has_inf").take(maxSampleRows).toIndexedSeq
          ValidationResult.failure(
            ValidationViolation(
              message = s"Column '$columnName' contains $count rows with infinite values",
              field = columnName,
              violationType = "infinite",
              sampleRows = samples
            )
          )
        } else {
          ValidationResult.success
        }
      } else {
        // For scalar columns
        val hasInf = df.filter(col(columnName).isNaN || col(columnName).isNull)
        val count  = hasInf.count()
        if (count > 0) {
          val samples = hasInf.take(maxSampleRows).toIndexedSeq
          ValidationResult.failure(
            ValidationViolation(
              message = s"Column '$columnName' contains $count infinite values",
              field = columnName,
              violationType = "infinite",
              sampleRows = samples
            )
          )
        } else {
          ValidationResult.success
        }
      }
    } catch {
      case e: Exception =>
        ValidationResult.failure(
          ValidationViolation(
            message = s"Failed to validate column '$columnName': ${e.getMessage}",
            field = columnName,
            violationType = "error"
          )
        )
    }
  }
}

/** Validator that checks if a numeric column contains only positive values. */
case class PositiveValidator(columnName: String, strict: Boolean = false, maxSampleRows: Int = 5)
    extends Validator {
  override def name: String =
    if (strict) s"strictlyPositive($columnName)" else s"positive($columnName)"

  override def validate(df: DataFrame): ValidationResult = {
    try {
      val threshold =
        if (strict) 0.0 else -1e-10 // Allow small negative values for numerical stability

      // For vector columns, check each element
      if (df.schema(columnName).dataType.typeName == "vector") {
        val hasNegative = df
          .withColumn(
            "_has_neg",
            udf((v: Vector) => v.toArray.exists(_ <= threshold)).apply(col(columnName))
          )
          .filter(col("_has_neg"))

        val count = hasNegative.count()
        if (count > 0) {
          val samples = hasNegative.drop("_has_neg").take(maxSampleRows).toIndexedSeq
          ValidationResult.failure(
            ValidationViolation(
              message = s"Column '$columnName' contains $count rows with non-positive values",
              field = columnName,
              violationType = "negative",
              sampleRows = samples
            )
          )
        } else {
          ValidationResult.success
        }
      } else {
        // For scalar columns
        val hasNegative = df.filter(col(columnName) <= threshold)
        val count       = hasNegative.count()
        if (count > 0) {
          val samples = hasNegative.take(maxSampleRows).toIndexedSeq
          ValidationResult.failure(
            ValidationViolation(
              message = s"Column '$columnName' contains $count non-positive values",
              field = columnName,
              violationType = "negative",
              sampleRows = samples
            )
          )
        } else {
          ValidationResult.success
        }
      }
    } catch {
      case e: Exception =>
        ValidationResult.failure(
          ValidationViolation(
            message = s"Failed to validate column '$columnName': ${e.getMessage}",
            field = columnName,
            violationType = "error"
          )
        )
    }
  }
}

/** Validator that checks if a column is not null. */
case class NotNullValidator(columnName: String, maxSampleRows: Int = 5) extends Validator {
  override def name: String = s"notNull($columnName)"

  override def validate(df: DataFrame): ValidationResult = {
    try {
      val hasNull = df.filter(col(columnName).isNull)
      val count   = hasNull.count()
      if (count > 0) {
        val samples = hasNull.take(maxSampleRows).toIndexedSeq
        ValidationResult.failure(
          ValidationViolation(
            message = s"Column '$columnName' contains $count null values",
            field = columnName,
            violationType = "null",
            sampleRows = samples
          )
        )
      } else {
        ValidationResult.success
      }
    } catch {
      case e: Exception =>
        ValidationResult.failure(
          ValidationViolation(
            message = s"Failed to validate column '$columnName': ${e.getMessage}",
            field = columnName,
            violationType = "error"
          )
        )
    }
  }
}

/** Validator that checks if vectors in a column have consistent dimensionality. */
case class ConsistentDimensionValidator(
    columnName: String,
    expectedDim: Option[Int] = None,
    maxSampleRows: Int = 5
) extends Validator {

  override def name: String = expectedDim match {
    case Some(d) => s"dimension($columnName, $d)"
    case None    => s"consistentDimension($columnName)"
  }

  override def validate(df: DataFrame): ValidationResult = {
    try {
      if (df.schema(columnName).dataType.typeName != "vector") {
        return ValidationResult.failure(
          ValidationViolation(
            message = s"Column '$columnName' is not a vector column",
            field = columnName,
            violationType = "type"
          )
        )
      }

      // Get actual dimensions
      val dims = df
        .select(udf((v: Vector) => v.size).apply(col(columnName)).as("_dim"))
        .distinct()
        .collect()
        .map(_.getInt(0))

      // Check if dimensions are consistent
      if (dims.length > 1) {
        val dimCounts = df
          .groupBy(udf((v: Vector) => v.size).apply(col(columnName)).as("_dim"))
          .count()
          .collect()
          .map(r => (r.getInt(0), r.getLong(1)))
          .sortBy(-_._2)

        ValidationResult.failure(
          ValidationViolation(
            message = s"Column '$columnName' has inconsistent dimensions: ${dimCounts.map {
                case (d, c) => s"$d ($c rows)"
              }.mkString(", ")}",
            field = columnName,
            violationType = "dimension"
          )
        )
      } else if (expectedDim.isDefined && dims.headOption != expectedDim) {
        ValidationResult.failure(
          ValidationViolation(
            message = s"Column '$columnName' has dimension ${dims.headOption
                .getOrElse("unknown")}, expected ${expectedDim.get}",
            field = columnName,
            violationType = "dimension"
          )
        )
      } else {
        ValidationResult.success
      }
    } catch {
      case e: Exception =>
        ValidationResult.failure(
          ValidationViolation(
            message = s"Failed to validate column '$columnName': ${e.getMessage}",
            field = columnName,
            violationType = "error"
          )
        )
    }
  }
}

/** Validator that checks if a DataFrame is not empty. */
case object NotEmptyValidator extends Validator {
  override def name: String = "notEmpty"

  override def validate(df: DataFrame): ValidationResult = {
    val count = df.count()
    if (count == 0) {
      ValidationResult.failure(
        ValidationViolation(
          message = "DataFrame is empty",
          field = "",
          violationType = "empty"
        )
      )
    } else {
      ValidationResult.success
    }
  }
}

/** Validator that checks kernel-specific requirements. */
case class KernelCompatibilityValidator(kernelName: String, featuresCol: String) extends Validator {
  override def name: String = s"kernelCompatibility($kernelName)"

  override def validate(df: DataFrame): ValidationResult = {
    val normalized = kernelName.toLowerCase.replaceAll("[\\s-_]", "")

    // KL divergence and Itakura-Saito require positive features
    if (normalized == "kl" || normalized == "itakurasaito" || normalized == "generalizedI") {
      PositiveValidator(featuresCol, strict = true).validate(df)
    } else {
      ValidationResult.success
    }
  }
}

/** Factory methods for common validators. */
object Validator {

  /** Validate that a column contains no NaN values. */
  def noNaN(columnName: String, maxSampleRows: Int = 5): Validator = {
    NoNaNValidator(columnName, maxSampleRows)
  }

  /** Validate that a column contains only finite values (no NaN or Inf). */
  def finite(columnName: String, maxSampleRows: Int = 5): Validator = {
    NoNaNValidator(columnName, maxSampleRows) and FiniteValidator(columnName, maxSampleRows)
  }

  /** Validate that a column contains only positive values. */
  def positive(columnName: String, strict: Boolean = false, maxSampleRows: Int = 5): Validator = {
    PositiveValidator(columnName, strict, maxSampleRows)
  }

  /** Validate that a column is not null. */
  def notNull(columnName: String, maxSampleRows: Int = 5): Validator = {
    NotNullValidator(columnName, maxSampleRows)
  }

  /** Validate that vector column has consistent dimensionality. */
  def consistentDimension(
      columnName: String,
      expectedDim: Option[Int] = None,
      maxSampleRows: Int = 5
  ): Validator = {
    ConsistentDimensionValidator(columnName, expectedDim, maxSampleRows)
  }

  /** Validate that DataFrame is not empty. */
  def notEmpty: Validator = NotEmptyValidator

  /** Validate kernel-specific requirements (e.g., KL requires positive features). */
  def kernelCompatibility(kernelName: String, featuresCol: String): Validator = {
    KernelCompatibilityValidator(kernelName, featuresCol)
  }

  /** Validate features column for k-means (finite, consistent dimension). */
  def features(columnName: String, expectedDim: Option[Int] = None): Validator = {
    notNull(columnName) and
      finite(columnName) and
      consistentDimension(columnName, expectedDim)
  }

  /** Validate weight column (positive, finite, not null). */
  def weight(columnName: String): Validator = {
    notNull(columnName) and
      finite(columnName) and
      positive(columnName)
  }

  /** Validate full k-means input (features, optional weight, non-empty). */
  def kmeansInput(
      featuresCol: String,
      weightCol: Option[String] = None,
      kernelName: String = "squaredEuclidean",
      expectedDim: Option[Int] = None
  ): Validator = {
    val baseValidator = notEmpty and
      features(featuresCol, expectedDim) and
      kernelCompatibility(kernelName, featuresCol)

    weightCol match {
      case Some(wCol) => baseValidator and weight(wCol)
      case None       => baseValidator
    }
  }
}
