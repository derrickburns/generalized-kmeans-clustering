package com.massivedatascience.linalg

import org.apache.spark.mllib.linalg.{DenseVector, Vector}

/**
 * This centroid eagerly adds new vectors to the centroid. Consequently,
 * it is appropriate for use with dense vectors.
 */
class EagerCentroid extends MutableWeightedVector with Serializable {

  import com.massivedatascience.linalg.EagerCentroid.empty

  def homogeneous = raw

  def asImmutable = WeightedVector(raw, weight)

  private var raw: Vector = empty

  var weight: Double = 0.0

  def add(p: WeightedVector): this.type = add(p.homogeneous, p.weight, 1.0)

  def sub(p: WeightedVector): this.type = add(p.homogeneous, p.weight, -1.0)

  def add(p: MutableWeightedVector): this.type = add(p.homogeneous, p.weight, 1.0)

  def sub(p: MutableWeightedVector): this.type = add(p.homogeneous, p.weight, -1.0)

  def scale(alpha: Double): this.type = {
    if (raw != empty) {
      BLAS.scal(alpha, raw)
      weight = weight * alpha
    }
    this
  }

  /**
   * Add in a vector, preserving the sparsity of the original/first vector.
   * @param r   vector to add
   * @param w   weight of vector to add
   * @param direction whether to add or subtract
   * @return
   */
  private def add(r: Vector, w: Double, direction: Double): this.type = {
    if (w > 0.0) {
      if (raw == empty) {
        assert(r != empty)
        raw = r.copy
        weight = w
      } else {
        raw = BLAS.axpy(direction, r, raw)
        weight = weight + direction * w
      }
    }
    this
  }

}

object EagerCentroid {
  val empty: DenseVector = new DenseVector(Array[Double]())
}
