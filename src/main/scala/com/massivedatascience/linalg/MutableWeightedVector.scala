package com.massivedatascience.linalg

import org.apache.spark.mllib.linalg.Vector


trait MutableWeightedVector {

  def weight: Double

  def homogeneous: Vector

  def add(p: WeightedVector): this.type

  def sub(p: WeightedVector): this.type

  def add(p: MutableWeightedVector): this.type

  def sub(p: MutableWeightedVector): this.type

  def asImmutable: WeightedVector

}
