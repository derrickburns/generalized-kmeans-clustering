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

import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
 * An IMMUTABLE weighted vector.
 *
 */
trait WeightedVector extends Serializable {
  def weight: Double

  def inhomogeneous: Vector

  def homogeneous: Vector

  def size: Int = homogeneous.size
}


object WeightedVector {

  private[this] class ImmutableInhomogeneousVector(val weight: Double, v: Vector) extends WeightedVector {
    override def toString: String = s"InhomogeneousVector($weight, $v)"
    override val inhomogeneous: Vector = v
    override lazy val homogeneous: Vector = asHomogeneous(v, weight)
  }

  private[this] class ImmutableHomogeneousVector(val weight: Double, v: Vector) extends WeightedVector {
    override def toString: String = s"HomogeneousVector($weight, $v)"
    override lazy val inhomogeneous: Vector = asInhomogeneous(v, weight)
    override val homogeneous: Vector = v
  }

  def apply(v: Vector): WeightedVector = new ImmutableInhomogeneousVector(1.0, v)

  def apply(v: Array[Double]): WeightedVector = new ImmutableInhomogeneousVector(1.0, Vectors.dense(v))

  def apply(v: Vector, weight: Double): WeightedVector = new ImmutableHomogeneousVector(weight, v)

  def apply(v: Array[Double], weight: Double): WeightedVector = new ImmutableHomogeneousVector(weight, Vectors.dense(v))

  def fromInhomogeneousWeighted(v: Array[Double], weight: Double): WeightedVector = new ImmutableInhomogeneousVector(weight, Vectors.dense(v))

  def fromInhomogeneousWeighted(v: Vector, weight: Double): WeightedVector = new ImmutableInhomogeneousVector(weight, v)
}



