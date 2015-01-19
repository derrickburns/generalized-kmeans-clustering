package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.util.BLAS._
import org.apache.spark.mllib.linalg.Vector

trait CentroidProvider {
  def getCentroid: MutableWeightedVector
}

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

trait DenseCentroidProvider {
  def getCentroid: MutableWeightedVector = new EagerCentroid
}

trait SparseCentroidProvider {
  def getCentroid: MutableWeightedVector = new EagerCentroid
}