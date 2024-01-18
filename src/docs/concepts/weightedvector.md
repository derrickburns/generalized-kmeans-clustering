# WeightedVector

Often, data points that are clustered have varying significance, i.e. they are weighted. This clusterer operates on weighted vectors. Use these `WeightedVector` companion object to construct weighted vectors.

```scala
package com.massivedatascience.linalg

trait WeightedVector extends Serializable {
  def weight: Double

  def inhomogeneous: Vector

  def homogeneous: Vector

  def size: Int = homogeneous.size
}

object WeightedVector {

  def apply(v: Vector): WeightedVector = ???

  def apply(v: Array[Double]): WeightedVector = ???

  def apply(v: Vector, weight: Double): WeightedVector = ???

  def apply(v: Array[Double], weight: Double): WeightedVector = ???

  def fromInhomogeneousWeighted(v: Array[Double], weight: Double): WeightedVector = ???

  def fromInhomogeneousWeighted(v: Vector, weight: Double): WeightedVector = ???
}
```
