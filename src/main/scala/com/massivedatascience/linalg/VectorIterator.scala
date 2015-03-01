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

import org.apache.spark.mllib.linalg.{ DenseVector, SparseVector, Vector }

trait VectorIterator extends Serializable {
  def hasNext: Boolean
  def advance(): Unit
  def index: Int
  def value: Double
  val underlying: Vector
}

class SparseVectorIterator(val underlying: SparseVector) extends VectorIterator {
  override def toString = s"$i, $underlying"
  protected var i = 0
  def hasNext: Boolean = i < underlying.values.length
  def advance(): Unit = i = i + 1
  def index: Int = underlying.indices(i)
  def value: Double = underlying.values(i)
}

class DenseVectorIterator(val underlying: DenseVector) extends VectorIterator {
  override def toString = s"$i, $underlying"
  protected var i = 0
  def hasNext: Boolean = i < underlying.values.length
  def advance(): Unit = i = i + 1
  def index: Int = i
  def value: Double = underlying.values(i)
}

class NegativeSparseVectorIterator(underlying: SparseVector) extends SparseVectorIterator(underlying) {
  override def value: Double = -underlying.values(i)
}

class NegativeDenseVectorIterator(underlying: DenseVector) extends DenseVectorIterator(underlying) {
  override def value: Double = -underlying.values(i)
}