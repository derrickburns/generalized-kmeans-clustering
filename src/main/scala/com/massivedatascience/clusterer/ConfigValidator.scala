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

/** Trait providing standardized configuration validation methods.
  *
  * This trait eliminates duplicate validation logic across configuration case classes and provides
  * consistent error messages.
  */
trait ConfigValidator {

  /** Require that a value is positive (> 0).
    */
  protected def requirePositive(value: Double, name: String): Unit = {
    require(value > 0.0, s"$name must be positive, got: $value")
  }

  /** Require that an integer value is positive (> 0).
    */
  protected def requirePositive(value: Int, name: String): Unit = {
    require(value > 0, s"$name must be positive, got: $value")
  }

  /** Require that a value is non-negative (>= 0).
    */
  protected def requireNonNegative(value: Double, name: String): Unit = {
    require(value >= 0.0, s"$name must be non-negative, got: $value")
  }

  /** Require that an integer value is non-negative (>= 0).
    */
  protected def requireNonNegative(value: Int, name: String): Unit = {
    require(value >= 0, s"$name must be non-negative, got: $value")
  }

  /** Require that a value is in a specified range [min, max].
    */
  protected def requireInRange(value: Double, min: Double, max: Double, name: String): Unit = {
    require(value >= min && value <= max, s"$name must be in [$min, $max], got: $value")
  }

  /** Require that an integer value is in a specified range [min, max].
    */
  protected def requireInRange(value: Int, min: Int, max: Int, name: String): Unit = {
    require(value >= min && value <= max, s"$name must be in [$min, $max], got: $value")
  }

  /** Require that a value is one of a specified set of options.
    */
  protected def requireOneOf[T](value: T, options: Seq[T], name: String): Unit = {
    require(
      options.contains(value),
      s"$name must be one of ${options.mkString("[", ", ", "]")}, got: $value"
    )
  }

  /** Require that a value is greater than another value.
    */
  protected def requireGreaterThan(value: Double, threshold: Double, name: String): Unit = {
    require(value > threshold, s"$name must be > $threshold, got: $value")
  }

  /** Require that an integer value is greater than another value.
    */
  protected def requireGreaterThan(value: Int, threshold: Int, name: String): Unit = {
    require(value > threshold, s"$name must be > $threshold, got: $value")
  }

  /** Require that a value is less than another value.
    */
  protected def requireLessThan(value: Double, threshold: Double, name: String): Unit = {
    require(value < threshold, s"$name must be < $threshold, got: $value")
  }

  /** Require that a value is at least a certain amount.
    */
  protected def requireAtLeast(value: Double, minimum: Double, name: String): Unit = {
    require(value >= minimum, s"$name must be >= $minimum, got: $value")
  }

  /** Require that an integer value is at least a certain amount.
    */
  protected def requireAtLeast(value: Int, minimum: Int, name: String): Unit = {
    require(value >= minimum, s"$name must be >= $minimum, got: $value")
  }

  /** Require a probability value (in [0, 1]).
    */
  protected def requireProbability(value: Double, name: String): Unit = {
    requireInRange(value, 0.0, 1.0, name)
  }

  /** Require a valid percentage (in [0, 100]).
    */
  protected def requirePercentage(value: Double, name: String): Unit = {
    requireInRange(value, 0.0, 100.0, name)
  }
}
