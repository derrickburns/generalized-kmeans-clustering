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

import com.massivedatascience.clusterer
import org.apache.spark.mllib.linalg.Vector

trait WeightedVector extends Serializable {
  def weight: Double

  def inhomogeneous: Vector

  def homogeneous: Vector

  def asInhomogeneous: Vector = clusterer.asInhomogeneous(homogeneous, weight)

  def asHomogeneous: Vector = clusterer.asHomogeneous(inhomogeneous, weight)

  override def toString: String = weight + "," + homogeneous.toString

  def asImmutable: WeightedVector = new ImmutableHomogeneousVector(homogeneous.copy, weight)

}

trait MutableWeightedVector extends WeightedVector {
  def add(p: WeightedVector): this.type

  def sub(p: WeightedVector): this.type

  def asImmutable: WeightedVector

}

class ImmutableInhomogeneousVector(raw: Vector, val weight: Double) extends WeightedVector {
  override val inhomogeneous = raw
  override lazy val homogeneous = asHomogeneous
}

class ImmutableHomogeneousVector(raw: Vector, val weight: Double) extends WeightedVector {
  override lazy val inhomogeneous: Vector = asInhomogeneous
  override val homogeneous: Vector = raw
}

