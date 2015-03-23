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

package com.massivedatascience

import com.massivedatascience.linalg.BLAS._
import org.apache.spark.mllib.linalg.{ DenseVector, SparseVector, Vector }

package object linalg {
  def asInhomogeneous(homogeneous: Vector, weight: Double): Vector = {
    val x = homogeneous.copy
    scal(1.0 / weight, x)
    x
  }

  def asHomogeneous(inhomogeneous: Vector, weight: Double): Vector = {
    val x = inhomogeneous.copy
    scal(weight, x)
    x
  }

  implicit class RichSeq(val v: IndexedSeq[(Int, Double)]) extends AnyVal {
    def iterator: VectorIterator = {
      new SparseSeqIterator(v)
    }
  }

  implicit class RichVector(val v: Vector) extends AnyVal {
    def iterator: VectorIterator = {
      v match {
        case s: SparseVector => new SparseVectorIterator(s)
        case d: DenseVector => new DenseVectorIterator(d)
      }
    }

    def negativeIterator: VectorIterator = {
      v match {
        case s: SparseVector => new NegativeSparseVectorIterator(s)
        case d: DenseVector => new NegativeDenseVectorIterator(d)
      }
    }
  }

}
