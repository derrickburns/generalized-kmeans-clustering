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

import com.massivedatascience.linalg.WeightedVector
import org.apache.spark.rdd.RDD

/**
 * Utility for sanitizing weighted vectors to ensure numerical stability
 * and avoid edge cases with extreme weight values.
 */
object WeightSanitizer {
  /** Minimum valid weight to avoid numerical precision issues */
  val MIN_VALID_WEIGHT = 1e-100

  /** Maximum valid weight to avoid overflow issues */
  val MAX_VALID_WEIGHT = 1e100

  /**
   * Sanitize an RDD of weighted vectors by filtering out zero weights
   * and clamping extreme values to safe ranges.
   *
   * @param data input RDD of weighted vectors
   * @param removeZeroWeights if true, filter out vectors with weight below MIN_VALID_WEIGHT
   * @return sanitized RDD
   */
  def sanitize(
    data: RDD[WeightedVector],
    removeZeroWeights: Boolean = true): RDD[WeightedVector] = {

    val filtered = if (removeZeroWeights) {
      data.filter(_.weight >= MIN_VALID_WEIGHT)
    } else {
      data
    }

    filtered.map { v =>
      val originalWeight = v.weight
      val clampedWeight = math.min(math.max(originalWeight, MIN_VALID_WEIGHT), MAX_VALID_WEIGHT)

      if (clampedWeight != originalWeight) {
        WeightedVector.fromInhomogeneousWeighted(v.inhomogeneous, clampedWeight)
      } else {
        v
      }
    }
  }

  /**
   * Sanitize a single weighted vector by clamping weight to safe range.
   *
   * @param vector input weighted vector
   * @return sanitized weighted vector
   */
  def sanitize(vector: WeightedVector): WeightedVector = {
    val originalWeight = vector.weight
    val clampedWeight = math.min(math.max(originalWeight, MIN_VALID_WEIGHT), MAX_VALID_WEIGHT)

    if (clampedWeight != originalWeight) {
      WeightedVector.fromInhomogeneousWeighted(vector.inhomogeneous, clampedWeight)
    } else {
      vector
    }
  }

  /**
   * Check if a weight value is valid (within acceptable bounds).
   *
   * @param weight weight value to check
   * @return true if weight is valid
   */
  def isValidWeight(weight: Double): Boolean = {
    weight >= MIN_VALID_WEIGHT &&
    weight <= MAX_VALID_WEIGHT &&
    java.lang.Double.isFinite(weight)
  }

  /**
   * Check if a weighted vector is valid (has valid weight and finite coordinates).
   *
   * @param vector weighted vector to check
   * @return true if vector is valid
   */
  def isValid(vector: WeightedVector): Boolean = {
    isValidWeight(vector.weight) &&
    vector.inhomogeneous.toArray.forall(java.lang.Double.isFinite)
  }
}
