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

/** Numeric validation utilities for clustering operations.
  *
  * These guards help detect and diagnose numeric issues early, preventing silent failures and
  * providing actionable error messages. They work together with the GKMError type system to provide
  * clear, type-safe error handling.
  *
  * Key features:
  *   - Early detection of NaN/Inf propagation
  *   - Domain validation for divergence-specific constraints
  *   - Clear, actionable error messages
  *   - Minimal performance overhead (while loops, no allocations)
  */
object NumericGuards {

  /** Check that vector contains only finite values (no NaN or Inf).
    *
    * @param v
    *   Vector to check
    * @param context
    *   Description of where this check is being performed (for error messages)
    * @throws InvalidFeatures
    *   if vector contains NaN or Inf
    */
  def checkFinite(v: Vector, context: String): Unit = {
    val arr = v.toArray
    var i   = 0
    while (i < arr.length) {
      val x = arr(i)
      if (x.isNaN) {
        val preview = formatVectorPreview(arr)
        throw InvalidFeatures(
          s"$context: Vector contains NaN at index $i. " +
            s"Vector preview: $preview. " +
            s"This may indicate:\n" +
            s"  - Numerical instability in divergence calculations\n" +
            s"  - Division by zero in center updates\n" +
            s"  - Missing smoothing parameter for KL/IS divergence"
        ).toException
      }
      if (x.isInfinite) {
        val preview = formatVectorPreview(arr)
        throw InvalidFeatures(
          s"$context: Vector contains Inf at index $i. " +
            s"Vector preview: $preview. " +
            s"This may indicate:\n" +
            s"  - Overflow in distance computations\n" +
            s"  - Extremely large input values\n" +
            s"  - Numerical instability"
        ).toException
      }
      i += 1
    }
  }

  /** Check that vector contains only positive values (for divergences requiring positivity).
    *
    * KL and Itakura-Saito divergences require strictly positive values. This guard helps catch
    * violations early with clear guidance on how to fix them.
    *
    * @param v
    *   Vector to check
    * @param context
    *   Description of where this check is being performed
    * @param epsilon
    *   Tolerance for near-zero values (values > -epsilon are considered acceptable)
    * @throws InvalidFeatures
    *   if vector contains negative values beyond tolerance
    */
  def checkPositive(v: Vector, context: String, epsilon: Double = 1e-10): Unit = {
    val arr = v.toArray
    var i   = 0
    while (i < arr.length) {
      val x = arr(i)
      if (x < -epsilon) {
        val preview = formatVectorPreview(arr)
        throw InvalidFeatures(
          s"$context: Vector contains negative value $x at index $i. " +
            s"Vector preview: $preview. " +
            s"KL/Itakura-Saito divergences require positive values. " +
            s"Solutions:\n" +
            s"  - Use .setSmoothing(1e-6) to add epsilon shift\n" +
            s"  - Transform input data to ensure positivity\n" +
            s"  - Consider using Squared Euclidean or L1 divergence instead"
        ).toException
      }
      i += 1
    }
  }

  /** Check that vector contains values in (0, 1) for logistic loss.
    *
    * @param v
    *   Vector to check
    * @param context
    *   Description of where this check is being performed
    * @param epsilon
    *   Small value to ensure strict bounds (0+eps, 1-eps)
    * @throws InvalidFeatures
    *   if vector contains values outside (epsilon, 1-epsilon)
    */
  def checkProbability(v: Vector, context: String, epsilon: Double = 1e-10): Unit = {
    val arr = v.toArray
    var i   = 0
    while (i < arr.length) {
      val x = arr(i)
      if (x <= epsilon || x >= 1.0 - epsilon) {
        val preview = formatVectorPreview(arr)
        throw InvalidFeatures(
          s"$context: Vector contains value $x at index $i outside (0,1). " +
            s"Vector preview: $preview. " +
            s"Logistic loss requires values in open interval (0,1). " +
            s"Solutions:\n" +
            s"  - Use .setSmoothing($epsilon) to ensure strict bounds\n" +
            s"  - Transform input to probabilities using softmax or normalization"
        ).toException
      }
      i += 1
    }
  }

  /** Check that a scalar value is finite (not NaN or Inf).
    *
    * @param value
    *   Value to check
    * @param context
    *   Description of where this check is being performed
    * @throws RuntimeException
    *   if value is NaN or Inf
    */
  def checkFiniteScalar(value: Double, context: String): Unit = {
    if (value.isNaN) {
      throw new RuntimeException(
        s"$context: Computed value is NaN. " +
          s"This may indicate:\n" +
          s"  - Numerical instability in cost calculation\n" +
          s"  - Invalid divergence computation\n" +
          s"  - Missing smoothing parameter"
      )
    }
    if (value.isInfinite) {
      throw new RuntimeException(
        s"$context: Computed value is Inf. " +
          s"This may indicate:\n" +
          s"  - Overflow in distance/cost computation\n" +
          s"  - Extremely large cluster spread\n" +
          s"  - Numerical instability"
      )
    }
  }

  /** Check that weight is positive and finite.
    *
    * @param weight
    *   Weight value to check
    * @param context
    *   Description of where this check is being performed
    * @throws InvalidWeight
    *   if weight is not finite or non-positive
    */
  def checkWeight(weight: Double, context: String): Unit = {
    if (weight.isNaN || weight.isInfinite) {
      throw InvalidWeight(weight).toException
    }
    if (weight <= 0.0) {
      throw InvalidWeight(weight).toException
    }
  }

  /** Format vector for error messages (show first 10 elements).
    */
  private def formatVectorPreview(arr: Array[Double]): String = {
    if (arr.length <= 10) {
      arr.mkString("[", ", ", "]")
    } else {
      arr.take(10).mkString("[", ", ", ", ...]")
    }
  }

  /** Safe vector addition with overflow detection.
    *
    * @param v1
    *   First vector
    * @param v2
    *   Second vector
    * @param context
    *   Context for error messages
    * @return
    *   v1 + v2
    * @throws InvalidFeatures
    *   if result contains NaN or Inf
    */
  def safeAdd(v1: Vector, v2: Vector, context: String): Vector = {
    require(v1.size == v2.size, s"Vector dimensions must match: ${v1.size} vs ${v2.size}")
    val arr1   = v1.toArray
    val arr2   = v2.toArray
    val result = new Array[Double](arr1.length)
    var i      = 0
    while (i < arr1.length) {
      result(i) = arr1(i) + arr2(i)
      if (result(i).isNaN || result(i).isInfinite) {
        throw InvalidFeatures(
          s"$context: Overflow in vector addition at index $i: ${arr1(i)} + ${arr2(i)} = ${result(i)}"
        ).toException
      }
      i += 1
    }
    org.apache.spark.ml.linalg.Vectors.dense(result)
  }

  /** Safe scalar multiplication with overflow detection.
    *
    * @param v
    *   Vector to scale
    * @param scalar
    *   Scalar multiplier
    * @param context
    *   Context for error messages
    * @return
    *   v * scalar
    * @throws InvalidFeatures
    *   if result contains NaN or Inf
    */
  def safeScale(v: Vector, scalar: Double, context: String): Vector = {
    checkFiniteScalar(scalar, s"$context: scalar multiplier")
    val arr    = v.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = arr(i) * scalar
      if (result(i).isNaN || result(i).isInfinite) {
        throw InvalidFeatures(
          s"$context: Overflow in scalar multiplication at index $i: ${arr(i)} * $scalar = ${result(i)}"
        ).toException
      }
      i += 1
    }
    org.apache.spark.ml.linalg.Vectors.dense(result)
  }
}
