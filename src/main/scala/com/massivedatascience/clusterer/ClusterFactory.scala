package com.massivedatascience.clusterer

import java.util.Comparator

import com.massivedatascience.clusterer.util.BLAS._
import org.apache.spark.mllib.linalg.{Vectors, Vector}

import scala.collection.mutable
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

  def result(size: Int): Vector = Vectors.sparse(Int.MaxValue, indices.toArray, values.toArray)
}


/**
 * Retains only the top k most heavily weighted items
 */
trait TopKCollector extends Collector {

  import com.google.common.collect.MinMaxPriorityQueue

  val heap: MinMaxPriorityQueue[(Int, Double)] = MinMaxPriorityQueue.orderedBy(
    new Comparator[(Int, Double)]() {
      def compare(x: (Int, Double), y: (Int, Double)): Int = (y._2 - x._2).toInt
    }
  ).maximumSize(numberToRetain).create()

  def numberToRetain: Int = 128

  @inline
  def add(index: Int, value: Double) = heap.add((index, value))

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

  implicit val ordering = new Ordering[VectorIterator]  {
    override def compare(x: VectorIterator, y: VectorIterator): Int = x.index - y.index
  }

  val empty = Vectors.zeros(1)
  val pq = new mutable.PriorityQueue[VectorIterator]()
  var weight: Double = 0.0

  def homogeneous = {
    if (pq.nonEmpty) {
      val size = pq.head.underlying.size
      if (pq.length == 1) {
        pq.dequeue().underlying
      } else {
        var total = 0.0
        var lastIndex = pq.head.index
        while (pq.nonEmpty) {
          val head = pq.dequeue()
          val index = head.index
          val value = head.value
          if (index == lastIndex) {
            total = total + value
          } else {
            add(lastIndex, total)
            total = 0.0
            lastIndex = index
          }
          head.forward()
          if (head.hasNext) pq.enqueue(head)
        }
        if (total != 0.0) add(lastIndex, total)
        result(size)
      }
    } else {
      empty
    }
  }

  def inhomogeneous = asInhomogeneous

  def isEmpty = weight == 0.0

  def add(p: WeightedVector): this.type = {
    if (p.weight > 0.0) {
      pq += p.homogeneous.iterator
      weight = weight + p.weight
    }
    this
  }

  def sub(p: WeightedVector): this.type = {
    if (p.weight > 0.0) {
      pq += p.homogeneous.negativeIterator
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