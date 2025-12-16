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

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Dataset

/** Domain validation helpers for Bregman divergences with actionable error messages.
  *
  * Different divergences have different domain requirements:
  *   - KL Divergence: requires strictly positive values (x > 0)
  *   - Itakura-Saito: requires strictly positive values (x > 0)
  *   - Logistic Loss: requires values in open interval (0, 1)
  *   - Generalized I: requires non-negative values (x ≥ 0)
  *   - Squared Euclidean: no restrictions (x ∈ ℝ)
  *   - L1/Manhattan: no restrictions (x ∈ ℝ)
  *
  * This validator provides clear error messages with suggested fixes using inputTransform options.
  */
object DivergenceDomainValidator {

  /** Domain requirements for each divergence type. */
  sealed trait DomainRequirement {
    def name: String
    def check(value: Double): Boolean
    def errorMessage(invalidValue: Double, divergenceName: String): String
    def suggestedFixes: Seq[String]
  }

  case object StrictlyPositive extends DomainRequirement {
    val name = "strictly positive (x > 0)"

    def check(value: Double): Boolean = value > 0.0 && java.lang.Double.isFinite(value)

    def errorMessage(invalidValue: Double, divergenceName: String): String = {
      val fixesStr = suggestedFixes.mkString("\n  - ")
      s"""$divergenceName divergence requires strictly positive values, but found: $invalidValue
         |
         |The $divergenceName divergence is only defined for positive data.
         |
         |Suggested fixes:
         |  - $fixesStr
         |
         |Example:
         |  new GeneralizedKMeans()
         |    .setDivergence("$divergenceName")
         |    .setInputTransform("log1p")  // Transforms data using log(1 + x)
         |    .setMaxIter(20)
         |""".stripMargin
    }

    def suggestedFixes: Seq[String] = Seq(
      "Use .setInputTransform(\"log1p\") to transform data using log(1 + x), which maps [0, ∞) → [0, ∞)",
      "Use .setInputTransform(\"epsilonShift\") with .setShiftValue(1e-6) to add a small constant",
      "Pre-process your data to ensure all values are positive",
      "Consider using Squared Euclidean divergence (.setDivergence(\"squaredEuclidean\")) which has no domain restrictions"
    )
  }

  case object OpenInterval01 extends DomainRequirement {
    val name = "open interval (0, 1)"

    def check(value: Double): Boolean =
      value > 0.0 && value < 1.0 && java.lang.Double.isFinite(value)

    def errorMessage(invalidValue: Double, divergenceName: String): String = {
      val fixesStr = suggestedFixes.mkString("\n  - ")
      s"""$divergenceName divergence requires values in the open interval (0, 1), but found: $invalidValue
         |
         |The $divergenceName divergence models probability distributions and requires 0 < x < 1.
         |
         |Suggested fixes:
         |  - $fixesStr
         |
         |Example:
         |  new GeneralizedKMeans()
         |    .setDivergence("$divergenceName")
         |    .setInputTransform("epsilonShift")
         |    .setShiftValue(1e-6)
         |    .setMaxIter(20)
         |""".stripMargin
    }

    def suggestedFixes: Seq[String] = Seq(
      "Normalize your data to [0, 1] range, then use .setInputTransform(\"epsilonShift\") with .setShiftValue(1e-6)",
      "Pre-process your data to ensure all values are probabilities (0 < x < 1)",
      "Consider using KL divergence (.setDivergence(\"kl\")) if your data represents positive values",
      "Consider using Squared Euclidean divergence (.setDivergence(\"squaredEuclidean\")) which has no domain restrictions"
    )
  }

  case object NonNegative extends DomainRequirement {
    val name = "non-negative (x ≥ 0)"

    def check(value: Double): Boolean = value >= 0.0 && java.lang.Double.isFinite(value)

    def errorMessage(invalidValue: Double, divergenceName: String): String = {
      val fixesStr = suggestedFixes.mkString("\n  - ")
      s"""$divergenceName divergence requires non-negative values, but found: $invalidValue
         |
         |The $divergenceName divergence requires x ≥ 0.
         |
         |Suggested fixes:
         |  - $fixesStr
         |
         |Example:
         |  new GeneralizedKMeans()
         |    .setDivergence("$divergenceName")
         |    .setInputTransform("none")
         |    .setMaxIter(20)
         |""".stripMargin
    }

    def suggestedFixes: Seq[String] = Seq(
      "Pre-process your data to ensure all values are non-negative (e.g., take absolute values)",
      "Apply a feature transformation that maps to non-negative values",
      "Consider using Squared Euclidean divergence (.setDivergence(\"squaredEuclidean\")) which has no domain restrictions"
    )
  }

  case object Unrestricted extends DomainRequirement {
    val name = "unrestricted (x ∈ ℝ)"

    def check(value: Double): Boolean = java.lang.Double.isFinite(value)

    def errorMessage(invalidValue: Double, divergenceName: String): String =
      s"$divergenceName divergence found non-finite value: $invalidValue (should be finite)"

    def suggestedFixes: Seq[String] = Seq(
      "Remove or impute NaN/Infinity values from your data"
    )
  }

  /** Get domain requirement for a divergence name. */
  def getDomainRequirement(divergence: String): DomainRequirement = {
    divergence.toLowerCase match {
      case "kl" | "kullbackleibler"                       => StrictlyPositive
      case "itakurasaito" | "is"                          => StrictlyPositive
      case "logistic" | "logisticloss"                    => OpenInterval01
      case "generalizedi" | "geni"                        => NonNegative
      case "squaredeuclidean" | "euclidean" | "se" | "l2" => Unrestricted
      case "l1" | "manhattan"                             => Unrestricted
      case _                                              => Unrestricted // Default to unrestricted for unknown divergences
    }
  }

  /** Validate a single vector against domain requirements.
    *
    * @param vector
    *   Vector to validate
    * @param divergence
    *   Divergence name (e.g., "kl", "itakuraSaito")
    * @param context
    *   Context string for error message (e.g., "training data", "center initialization")
    * @throws IllegalArgumentException
    *   if validation fails
    */
  def validateVector(vector: Vector, divergence: String, context: String = "data"): Unit = {
    val requirement = getDomainRequirement(divergence)
    val values      = vector.toArray

    values.zipWithIndex.foreach { case (value, idx) =>
      if (!requirement.check(value)) {
        throw new IllegalArgumentException(
          s"""Invalid value in $context at index $idx:
             |${requirement.errorMessage(value, divergence)}
             |""".stripMargin
        )
      }
    }
  }

  /** Validate a DataFrame column containing vectors against domain requirements.
    *
    * This performs a distributed check across the entire dataset and reports the first violation
    * found.
    *
    * @param dataset
    *   DataFrame to validate
    * @param featuresCol
    *   Name of the features column
    * @param divergence
    *   Divergence name
    * @param maxSamples
    *   Maximum number of samples to check (for performance). Default: check all rows.
    * @throws IllegalArgumentException
    *   if validation fails
    */
  def validateDataFrame(
      dataset: Dataset[_],
      featuresCol: String,
      divergence: String,
      maxSamples: Option[Long] = None
  ): Unit = {
    val requirement = getDomainRequirement(divergence)

    // For unrestricted divergences, only check for NaN/Inf
    if (requirement == Unrestricted) {
      validateFiniteValues(dataset, featuresCol, divergence, maxSamples)
      return
    }

    // Sample if requested
    val dataToCheck = maxSamples match {
      case Some(n) if n < dataset.count() => dataset.limit(n.toInt)
      case _                              => dataset
    }

    // Check domain requirements
    val violationOpt = dataToCheck
      .select(featuresCol)
      .rdd
      .flatMap { row =>
        val vector = row.getAs[Vector](0)
        val values = vector.toArray

        values.zipWithIndex.collectFirst {
          case (value, idx) if !requirement.check(value) =>
            (value, idx, vector.size)
        }
      }
      .take(1)
      .headOption

    violationOpt.foreach { case (invalidValue, idx, dim) =>
      throw new IllegalArgumentException(
        s"""Invalid value found in column '$featuresCol' at feature index $idx/$dim:
           |${requirement.errorMessage(invalidValue, divergence)}
           |""".stripMargin
      )
    }
  }

  /** Check for NaN/Infinity values in the dataset. */
  private def validateFiniteValues(
      dataset: Dataset[_],
      featuresCol: String,
      divergence: String,
      maxSamples: Option[Long]
  ): Unit = {
    val dataToCheck = maxSamples match {
      case Some(n) if n < dataset.count() => dataset.limit(n.toInt)
      case _                              => dataset
    }

    val violationOpt = dataToCheck
      .select(featuresCol)
      .rdd
      .flatMap { row =>
        val vector = row.getAs[Vector](0)
        val values = vector.toArray

        values.zipWithIndex.collectFirst {
          case (value, idx) if !java.lang.Double.isFinite(value) =>
            (value, idx, vector.size)
        }
      }
      .take(1)
      .headOption

    violationOpt.foreach { case (invalidValue, idx, dim) =>
      throw new IllegalArgumentException(
        s"""Non-finite value found in column '$featuresCol' at feature index $idx/$dim: $invalidValue
           |
           |All divergences require finite values. Found: $invalidValue
           |
           |Suggested fixes:
           |  - Remove rows with NaN/Infinity values from your data
           |  - Impute missing values before clustering
           |  - Check for numerical overflow in feature engineering
           |""".stripMargin
      )
    }
  }

  /** Get a user-friendly description of domain requirements for a divergence.
    *
    * Useful for documentation and error messages.
    */
  def getDomainDescription(divergence: String): String = {
    val requirement = getDomainRequirement(divergence)
    s"$divergence divergence requires ${requirement.name}"
  }

  /** Check if a divergence requires domain validation beyond finite values. */
  def requiresStrictValidation(divergence: String): Boolean = {
    getDomainRequirement(divergence) != Unrestricted
  }
}
