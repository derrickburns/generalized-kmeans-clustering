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

import java.util.Comparator

import com.massivedatascience.clusterer.util.BLAS._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import com.google.common.collect.MinMaxPriorityQueue


import scala.collection.mutable.ArrayBuffer

/**
 * K-Means algorithms need a method to construct a median or centroid value.
 * This trait abstracts the type of the object used to create the centroid.
 */
trait ClusterFactory {
  def getCentroid: MutableWeightedVector
}

/**
 * This centroid eagerly adds new vectors to the centroid. Consequently,
 * it is appropriate for use with dense vectors.
 */
class EagerCentroid extends MutableWeightedVector with Serializable {
  def homogeneous = raw

  def inhomogeneous = asInhomogeneous

  def isEmpty = weight == 0.0

  private var raw: Vector = empty

  var weight: Double = 0.0

  def add(p: WeightedVector): this.type = add(p.homogeneous, p.weight, 1.0)

  def sub(p: WeightedVector): this.type = add(p.homogeneous, p.weight, -1.0)

  /**
   * Add in a vector, preserving the sparsity of the original/first vector.
   * @param r   vector to add
   * @param w   weight of vector to add
   * @param direction whether to add or subtract
   * @return
   */
  private def add(r: Vector, w: Double, direction: Double): this.type = {
    if (w > 0.0) {
      if (weight == 0.0) {
        raw = r.copy
        weight = w
      } else {
        raw = axpy(direction, r, raw)
        weight = weight + w
      }
    }
    this
  }
}


trait Collector {
  def add(index: Int, value: Double): Unit
  def result(size: Int): Vector
}

trait FullCollector extends Collector {
  val indices = new ArrayBuffer[Int]
  val values = new ArrayBuffer[Double]

  @inline
  def add(index: Int, value: Double) = {
    indices += index
    values += value
  }

  def result(size: Int): Vector = Vectors.sparse(size, indices.toArray, values.toArray)
}


/**
 * Retains only the top k most heavily weighted items
 *
 * Note: this collector is not associative!
 */
trait TopKCollector extends Collector {

  import com.google.common.collect.MinMaxPriorityQueue

  private[this] val heap: MinMaxPriorityQueue[(Int, Double)] = MinMaxPriorityQueue.orderedBy(
    new Comparator[(Int, Double)]() {
      def compare(x: (Int, Double), y: (Int, Double)): Int = (y._2 - x._2).toInt
    }
  ).maximumSize(numberToRetain).create()

  def numberToRetain: Int = 128

  @inline
  def add(index: Int, value: Double) = {
    heap.add((index, value))
    println( s"added $index $value")
  }

  def result(size: Int): Vector = {
    Vectors.sparse(size, heap.toArray[(Int, Double)](new Array[(Int, Double)](heap.size())))
  }
}

/**
 * Implements a centroid where the points are assumed to be sparse.
 *
 * The calculation of the centroid is deferred until all points are added to the
 * cluster. When the calculation is performed, a priority queue is used to sort the entries
 * by index.
 *
 */
trait LateCentroid extends MutableWeightedVector with Serializable {
  this: Collector =>

  import com.massivedatascience.clusterer.RichVector

  final val empty = Vectors.zeros(1)
  private[this] val container = new ArrayBuffer[VectorIterator]()
  var weight: Double = 0.0

  override lazy val homogeneous = asHomogeneous

  override def asHomogeneous = {
    if (container.isEmpty) {
      empty
    } else if (container.size == 1) {
      container.remove(0).underlying
    } else {
      val pq: MinMaxPriorityQueue[VectorIterator] = MinMaxPriorityQueue.orderedBy[VectorIterator](
        new Comparator[VectorIterator]() {
          def compare(x: VectorIterator, y: VectorIterator): Int = x.index - y.index
        }
      ).create()

      for( c <- container) pq.add(c)
      val peek = pq.peek()
      val size = peek.underlying.size
      var total = 0.0
      var lastIndex = peek.index
      while (pq.size > 0) {
        val head = pq.remove()
        val index = head.index
        val value = head.value
        if (index == lastIndex) {
          total = total + value
        } else {
          add(lastIndex, total)
          total = value
          lastIndex = index
        }
        head.advance()

        if (head.hasNext) pq.add(head)
      }
      if (total != 0.0) add(lastIndex, total)
      result(size)
    }
  }

  def inhomogeneous = asInhomogeneous

  def isEmpty = weight == 0.0

  def add(p: WeightedVector): this.type = {
    if (p.weight > 0.0) {
      val iterator = p.homogeneous.iterator
      container += iterator
      weight = weight + p.weight
    }
    this
  }

  def sub(p: WeightedVector): this.type = {
    if (p.weight > 0.0) {
      container += p.homogeneous.negativeIterator
      weight = weight + p.weight
    }
    this
  }
}

trait DenseClusterFactory extends ClusterFactory {
  def getCentroid: MutableWeightedVector = new EagerCentroid
}

trait SparseClusterFactory extends ClusterFactory {
  def getCentroid: MutableWeightedVector = new LateCentroid with FullCollector
}

trait SparseTopKClusterFactory extends ClusterFactory {
  def getCentroid: MutableWeightedVector = new LateCentroid with TopKCollector
}