package com.massivedatascience.clusterer

import com.massivedatascience.clusterer
import org.apache.spark.mllib.linalg.Vector

trait WeightedVector extends Serializable {
  def weight: Double

  def inhomogeneous: Vector

  def homogeneous: Vector

  def asInhomogeneous = clusterer.asInhomogeneous(homogeneous, weight)

  def asHomogeneous = clusterer.asHomogeneous(inhomogeneous, weight)

  override def toString: String = weight + "," + homogeneous.toString

  def asImmutable = new ImmutableHomogeneousVector(homogeneous, weight).asInstanceOf[WeightedVector]
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

