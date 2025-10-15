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

object ValidationUtils {

  def requirePositiveWeight(weight: Double, context: String = "Weight"): Unit = {
    if (!java.lang.Double.isFinite(weight) || weight <= 0.0) {
      throw new IllegalArgumentException(s"$context must be positive, got: $weight")
    }
  }

  def requireNonNegativeWeight(weight: Double, context: String = "Weight"): Unit = {
    if (!java.lang.Double.isFinite(weight) || weight < 0.0) {
      throw new IllegalArgumentException(s"$context cannot be negative, got: $weight")
    }
  }

  def requireValidProbability(value: Double, context: String = "Probability"): Unit = {
    if (java.lang.Double.isNaN(value) || value < 0.0 || value > 1.0) {
      throw new IllegalArgumentException(s"$context must be between 0 and 1, got: $value")
    }
  }

  def requirePositiveVector(vector: Vector, context: String = "Vector elements"): Unit = {
    // For sparse vectors, only check explicitly stored non-zero values
    // For dense vectors, check all values
    vector match {
      case sparse: org.apache.spark.ml.linalg.SparseVector =>
        sparse.values.foreach { value =>
          if (value <= 0.0) {
            throw new IllegalArgumentException(s"$context must be positive, found: $value")
          }
        }
      case _ =>
        val values = vector.toArray
        values.foreach { value =>
          if (value <= 0.0) {
            throw new IllegalArgumentException(s"$context must be positive, found: $value")
          }
        }
    }
  }

  def requireNonNegativeVector(vector: Vector, context: String = "Vector elements"): Unit = {
    val values = vector.toArray
    values.foreach { value =>
      if (value < 0.0) {
        throw new IllegalArgumentException(s"$context cannot be negative, found: $value")
      }
    }
  }

  def requireValidArrayIndex(index: Int, arrayLength: Int, context: String = "Index"): Unit = {
    if (index < 0 || index >= arrayLength) {
      throw new IllegalArgumentException(
        s"$context must be between 0 and ${arrayLength - 1}, got: $index"
      )
    }
  }

  def requireSameDimensions(v1: Vector, v2: Vector, context: String = "Vectors"): Unit = {
    if (v1.size != v2.size) {
      throw new IllegalArgumentException(
        s"$context must have same dimensions, got: ${v1.size} and ${v2.size}"
      )
    }
  }

  def requireNonEmpty[T](collection: Traversable[T], context: String = "Collection"): Unit = {
    if (collection.isEmpty) {
      throw new IllegalArgumentException(s"$context must not be empty")
    }
  }

  def requirePositive(value: Int, context: String = "Value"): Unit = {
    if (value <= 0) {
      throw new IllegalArgumentException(s"$context must be positive, got: $value")
    }
  }

  def requireNonNegative(value: Int, context: String = "Value"): Unit = {
    if (value < 0) {
      throw new IllegalArgumentException(s"$context cannot be negative, got: $value")
    }
  }

  def requireFinite(value: Double, context: String = "Value"): Unit = {
    if (!java.lang.Double.isFinite(value)) {
      throw new IllegalArgumentException(s"$context must be finite, got: $value")
    }
  }

  def requireNotNull[T <: AnyRef](value: T, context: String = "Value"): T = {
    if (value == null) {
      throw new IllegalArgumentException(s"$context must not be null")
    }
    value
  }
}
