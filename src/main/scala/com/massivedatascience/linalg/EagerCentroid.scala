/*
 * Licensed to the Massive Data Science and Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Massive Data Science and Derrick R. Burns licenses this file to You under the
 * Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
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

package com.massivedatascience.linalg

import org.apache.spark.mllib.linalg.{ DenseVector, Vector }

/**
 * This centroid eagerly adds new vectors to the centroid. Consequently,
 * it is appropriate for use with dense vectors.
 */
class EagerCentroid extends MutableWeightedVector with Serializable {

  import com.massivedatascience.linalg.EagerCentroid.empty

  def homogeneous = raw

  def asImmutable = WeightedVector(raw, weight)

  private var raw: Vector = empty

  var weight: Double = 0.0

  def add(p: WeightedVector): this.type = add(p.homogeneous, p.weight, 1.0)

  def sub(p: WeightedVector): this.type = add(p.homogeneous, p.weight, -1.0)

  def add(p: MutableWeightedVector): this.type = add(p.homogeneous, p.weight, 1.0)

  def sub(p: MutableWeightedVector): this.type = add(p.homogeneous, p.weight, -1.0)

  def scale(alpha: Double): this.type = {
    if (raw != empty) {
      BLAS.scal(alpha, raw)
      weight = weight * alpha
    }
    this
  }

  /**
   * Add in a vector, preserving the sparsity of the original/first vector.
   * @param r   vector to add
   * @param w   weight of vector to add
   * @param direction whether to add or subtract
   * @return
   */
  private[this] def add(r: Vector, w: Double, direction: Double): this.type = {
    if (w > 0.0) {
      if (raw == empty) {
        assert(r != empty)
        raw = r.copy
        weight = w
      } else {
        raw = BLAS.axpy(direction, r, raw)
        weight = weight + direction * w
      }
    }
    this
  }

}

object EagerCentroid {
  val empty: DenseVector = new DenseVector(Array[Double]())
}
