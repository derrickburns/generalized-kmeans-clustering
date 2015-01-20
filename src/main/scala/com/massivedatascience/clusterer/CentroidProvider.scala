package com.massivedatascience.clusterer

import java.util.Comparator

import com.massivedatascience.clusterer.util.BLAS._
import org.apache.spark.mllib.linalg.{Vectors, Vector}

import scala.collection.mutable.ArrayBuffer

/**
 * K-Means algorithms need a method to construct a median or centroid value.
 * This trait abstracts the type of the object used to create the centroid.
 */
trait CentroidProvider {
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
        axpy(direction, r, raw)
        weight = weight + w
      }
    }
    this
  }
}


trait Collector {
  def add(index: Int, value: Double): Unit

  def result(): WeightedVector
}

trait FullCollector extends Collector {
  val indices = new ArrayBuffer[Int]
  val values = new ArrayBuffer[Double]

  def add(index: Int, value: Double) = {
    indices += index
    values += value
  }

  def result(): WeightedVector = {
    val weight = values.sum
    new ImmutableHomogeneousVector(Vectors.sparse(Int.MaxValue, indices.toArray, values.toArray), weight)
  }
}


/**
 * Retains only the top k most heavily weighted items
 */
trait TopCollector extends Collector {

  import com.google.common.collect.MinMaxPriorityQueue

  val heap: MinMaxPriorityQueue[(Int, Double)] = MinMaxPriorityQueue.orderedBy(
    new Comparator[(Int, Double)]() {
      def compare(x: (Int, Double), y: (Int, Double)): Int = (y._2 - x._2).toInt
    }
  ).maximumSize(numberToRetain).create()

  def numberToRetain: Int = 128

  def add(index: Int, value: Double) = heap.add((index, value))

  def result(): WeightedVector = {
    val entries = heap.toArray[(Int, Double)](new Array[(Int, Double)](heap.size()))
    val weight = entries.map(_._2).sum
    new ImmutableHomogeneousVector(Vectors.sparse(Int.MaxValue, entries), weight)
  }
}

trait LateCentroid extends MutableWeightedVector with Serializable {
  this: Collector =>

  import com.massivedatascience.clusterer.RichVector

  var iterators = new ArrayBuffer[VectorIterator]()

  var weight: Double = 0.0

  def homogeneous = {
    iterators = iterators.filter(_.hasNext)
    while (iterators.nonEmpty) {
      var total: Double = 0.0
      val minIndex = iterators.minBy(_.index).index
      for (x <- iterators) {
        if (x.index == minIndex) {
          total = total + x.value
          x.forward()
        }
      }
      iterators = iterators.filter(_.hasNext)
      add(minIndex, total)
    }
    val x = result()
    weight = x.weight
    x.homogeneous
  }

  def inhomogeneous = asInhomogeneous

  def isEmpty = weight == 0.0

  def add(p: WeightedVector): this.type = {
    iterators += p.homogeneous.iterator
    weight = weight + p.weight
    this
  }

  def sub(p: WeightedVector): this.type = {
    iterators += p.homogeneous.negativeIterator
    weight = weight - p.weight
    this
  }
}

trait DenseCentroidProvider {
  def getCentroid: MutableWeightedVector = new EagerCentroid
}

trait SparseCentroidProvider {
  def getCentroid: MutableWeightedVector = new LateCentroid with FullCollector
}

trait TopKCentroidProvider {
  def getCentroid: MutableWeightedVector = new LateCentroid with TopCollector
}