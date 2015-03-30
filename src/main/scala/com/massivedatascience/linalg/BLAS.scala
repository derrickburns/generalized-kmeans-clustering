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

package com.massivedatascience.linalg

import com.github.fommil.netlib.{ BLAS => NetlibBLAS, F2jBLAS }
import org.apache.spark.mllib.linalg.{ DenseVector, SparseVector, Vector }

import scala.collection.mutable.ArrayBuffer

/**
 * This code comes from Spark.  Unfortunately, the Spark version is private to the mllib package,
 * so we copied the code to a separate package.
 */

/**
 * BLAS routines for MLlib's vectors and matrices.
 */
object BLAS extends Serializable {

  @transient private var _f2jBLAS: NetlibBLAS = _

  // For level-1 routines, we use Java implementation.
  private def f2jBLAS: NetlibBLAS = {
    if (_f2jBLAS == null) {
      _f2jBLAS = new F2jBLAS
    }
    _f2jBLAS
  }

  /**
   * y += a * x
   */
  def axpy(a: Double, x: Vector, y: Vector): Vector = {
    y match {
      case dy: DenseVector =>
        x match {
          case sx: SparseVector =>
            axpy(a, sx, dy)
          case dx: DenseVector =>
            denseAxpy(a, dx, dy)
          case _ =>
            throw new UnsupportedOperationException(
              s"axpy doesn't support x type ${x.getClass}.")
        }
      case sy: SparseVector =>
        x match {
          case sx: SparseVector =>
            axpy(a, sx, sy)
          case dx: DenseVector =>
            axpy(a, new SparseVector(dx.size, (0 until dx.size).toArray, dx.values), sy)
          case _ =>
            throw new UnsupportedOperationException(
              s"axpy doesn't support x type ${x.getClass}.")

        }
    }
  }

  def trans(v: Vector, func: (Double => Double)): Vector = {
    def transform(v: Array[Double]): Array[Double] = {
      var i = 0
      while (i < v.length) {
        v(i) = func(v(i))
        i = i + 1
      }
      v
    }

    val transformed = v.copy
    transformed match {
      case y: DenseVector => transform(y.values)
      case y: SparseVector => transform(y.values)
    }
    transformed
  }

  def merge(x: Vector, y: Vector, op: ((Double, Double) => Double)): Vector = {
    y match {
      case dy: DenseVector =>
        x match {
          case dx: DenseVector =>
            merge(dx, dy, op)
          case _ =>
            throw new UnsupportedOperationException(
              s"axpy doesn't support x type ${x.getClass}.")
        }
      case sy: SparseVector =>
        x match {
          case sx: SparseVector =>
            merge(sx, sy, op)
          case _ =>
            throw new UnsupportedOperationException(
              s"merge doesn't support x type ${x.getClass}.")
        }
    }
  }

  private def axpy(a: Double, x: SparseVector, y: SparseVector): Vector = {
    merge(x, y, (xv: Double, yv: Double) => a * xv + yv)
  }

  private def merge(x: DenseVector, y: DenseVector, op: ((Double, Double) => Double)): DenseVector = {
    var i = 0
    val results = new Array[Double](x.size)
    while (i < x.size) {
      results(i) = op(x(i), y(i))
      i = i + 1
    }
    new DenseVector(results)
  }

  private def merge(x: SparseVector, y: SparseVector, op: ((Double, Double) => Double)): SparseVector = {
    var i = 0
    var j = 0
    val xIndices = x.indices
    val yIndices = y.indices
    val xValues = x.values
    val yValues = y.values

    val maxLength = x.indices.length + y.indices.length
    val indexBuffer = new ArrayBuffer[Int](maxLength)
    val valueBuffer = new ArrayBuffer[Double](maxLength)

    @inline
    def append(value: Double, index: Int): Unit = {
      if (value != 0.0) {
        indexBuffer += index
        valueBuffer += value
      }
    }

    while (i < xIndices.length && j < yIndices.length) {
      if (xIndices(i) < yIndices(j)) {
        append(op(xValues(i), 0.0), xIndices(i))
        i = i + 1
      } else if (yIndices(j) < xIndices(i)) {
        append(op(0.0, yValues(j)), yIndices(j))
        j = j + 1
      } else {
        append(op(xValues(i), yValues(j)), xIndices(i))
        i = i + 1
        j = j + 1
      }
    }
    while (i < xIndices.length) {
      append(op(xValues(i), 0.0), xIndices(i))
      i = i + 1
    }
    while (j < yIndices.length) {
      append(op(0.0, yValues(j)), yIndices(j))
      j = j + 1
    }

    val indices = indexBuffer.toArray
    val values = valueBuffer.toArray
    new SparseVector(x.size, indices, values)
  }

  /**
   * y += a * x
   */
  def denseAxpy(a: Double, x: DenseVector, y: DenseVector): Vector = {
    val n = x.size
    f2jBLAS.daxpy(n, a, x.values, 1, y.values, 1)
    y
  }

  /**
   * y += a * x
   */
  private def axpy(a: Double, x: SparseVector, y: DenseVector): Vector = {
    val nnz = x.indices.length
    if (a == 1.0) {
      var k = 0
      while (k < nnz) {
        y.values(x.indices(k)) += x.values(k)
        k += 1
      }
    } else {
      var k = 0
      while (k < nnz) {
        y.values(x.indices(k)) += a * x.values(k)
        k += 1
      }
    }
    y
  }

  /**
   * dot(x, y)
   */
  def dot(x: Vector, y: Vector): Double = {
    if (x.size != y.size) {
      require(x.size == y.size, s"${x.size} vs ${y.size}")
    }

    y match {
      case dy: DenseVector =>
        x match {
          case sx: SparseVector =>
            dot(sx, dy)
          case dx: DenseVector =>
            denseDot(dx, dy)
          case _ =>
            throw new UnsupportedOperationException(
              s"sum doesn't support x type ${x.getClass}.")
        }
      case sy: SparseVector =>
        x match {
          case sx: SparseVector =>
            dot(sx, sy)
          case dx: DenseVector =>
            dot(new SparseVector(dx.size, (0 until dx.size).toArray, dx.values), sy)
          case _ =>
            throw new UnsupportedOperationException(
              s"sum doesn't support x type ${x.getClass}.")

        }
    }
  }

  /**
   * dot(x, y)
   */
  @inline
  private def denseDot(x: DenseVector, y: DenseVector): Double = {
    val n = x.size
    f2jBLAS.ddot(n, x.values, 1, y.values, 1)
  }

  /**
   * dot(x, y)
   */

  private def dot(x: SparseVector, y: DenseVector): Double = {
    val nnz = x.indices.length
    var sum = 0.0
    var k = 0
    while (k < nnz) {
      sum += x.values(k) * y.values(x.indices(k))
      k += 1
    }
    sum
  }

  /**
   * dot(x, y)
   */
  private def dot(x: SparseVector, y: SparseVector): Double = {
    var kx = 0
    val nnzx = x.indices.length
    var ky = 0
    val nnzy = y.indices.length
    var sum = 0.0
    // y catching x
    while (kx < nnzx && ky < nnzy) {
      val ix = x.indices(kx)
      while (ky < nnzy && y.indices(ky) < ix) {
        ky += 1
      }
      if (ky < nnzy && y.indices(ky) == ix) {
        sum += x.values(kx) * y.values(ky)
        ky += 1
      }
      kx += 1
    }
    sum
  }

  /**
   * y = x
   */
  def copy(x: Vector, y: Vector): Unit = {
    val n = y.size
    require(x.size == n)
    y match {
      case dy: DenseVector =>
        x match {
          case sx: SparseVector =>
            var i = 0
            var k = 0
            val nnz = sx.indices.length
            while (k < nnz) {
              val j = sx.indices(k)
              while (i < j) {
                dy.values(i) = 0.0
                i += 1
              }
              dy.values(i) = sx.values(k)
              i += 1
              k += 1
            }
            while (i < n) {
              dy.values(i) = 0.0
              i += 1
            }
          case dx: DenseVector =>
            Array.copy(dx.values, 0, dy.values, 0, n)
        }
      case _ =>
        throw new IllegalArgumentException(s"y must be dense in copy but got ${y.getClass}")
    }
  }

  /**
   * x = a * x
   */
  def scal(a: Double, x: Vector): Unit = {
    x match {
      case sx: SparseVector =>
        f2jBLAS.dscal(sx.values.length, a, sx.values, 1)
      case dx: DenseVector =>
        f2jBLAS.dscal(dx.values.length, a, dx.values, 1)
      case _ =>
        throw new IllegalArgumentException(s"scal doesn't support vector type ${x.getClass}.")
    }
  }

  private def doSum(v: Array[Double]): Double = {
    var i = 0
    var s = 0.0
    while (i < v.length) {
      s = s + v(i)
      i = i + 1
    }
    s
  }

  private def doAdd(v: Array[Double], c: Double): Array[Double] = {
    var i = 0
    while (i < v.length) {
      v(i) = v(i) + c
      i = i + 1
    }
    v
  }

  private def doMax(v: Array[Double]): Double = {
    var i = 0
    var maximum = Double.MinValue
    while (i < v.length) {
      if (v(i) < maximum) maximum = v(i)
      i = i + 1
    }
    maximum
  }

  def add(v: Vector, c: Double): Vector = {
    v match {
      case x: DenseVector => new DenseVector(doAdd(x.values.clone(), c))
      case x: SparseVector => new SparseVector(x.size, x.indices, doAdd(x.values.clone(), c))
    }
  }

  def sum(v: Vector): Double = {
    v match {
      case x: DenseVector => doSum(x.values)
      case y: SparseVector => doSum(y.values)
    }
  }

  def max(v: Vector): Double = {
    v match {
      case x: DenseVector => doMax(x.values)
      case y: SparseVector => doMax(y.values)
    }
  }
}
