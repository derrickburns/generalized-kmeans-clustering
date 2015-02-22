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

package com.massivedatascience.clusterer

import org.apache.spark.mllib.linalg.{Vectors, Vector}

trait WeightedVector extends Serializable {
  def weight: Double

  def inhomogeneous: Vector

  def homogeneous: Vector
}


trait MutableWeightedVector extends WeightedVector {
  def add(p: WeightedVector): this.type

  def sub(p: WeightedVector): this.type

  def asImmutable: WeightedVector

}

object ImmutableInhomogeneousVector {
  def apply(v: WeightedVector): WeightedVector = new ImmutableInhomogeneousVector(v.inhomogeneous, v.weight)

  def apply(v: Vector): WeightedVector = new ImmutableInhomogeneousVector(v, 1.0)

  def apply(v: Array[Double]): WeightedVector = new ImmutableInhomogeneousVector(Vectors.dense(v), 1.0)

}

object ImmutableHomogeneousVector {
  def apply(v: WeightedVector): WeightedVector = new ImmutableHomogeneousVector(v.homogeneous, v.weight)

  def apply(v: Vector): WeightedVector = new ImmutableHomogeneousVector(v, 1.0)
}

class ImmutableInhomogeneousVector(v: Vector, val weight: Double) extends WeightedVector {
  override val inhomogeneous = v.copy
  override lazy val homogeneous = asHomogeneous(v, weight)
}

class ImmutableHomogeneousVector(v: Vector, val weight: Double) extends WeightedVector {
  override lazy val inhomogeneous: Vector = asInhomogeneous(v, weight)
  override val homogeneous: Vector = v.copy
}

