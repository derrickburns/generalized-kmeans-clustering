package com.rincaro.clusterer.metrics

import com.rincaro.clusterer.base.{FPoint, Centroid, PointOps}

class FastEUPoint[T](raw: Array[Double], weight: Double, name: Option[T] = None) extends FPoint(raw, weight, name) {
  val norm = {
    if( weight == 0.0 ) 0.0 else {
      var i = 0
      val end = raw.length
      var x = 0.0D
      while( i < end ) {
        x = x + raw(i) * raw(i)
        i = i + 1
      }
      x / (weight * weight)
    }
  }
}

/**
 * Euclidean distance measure - expedited by pre-computing vector norms
 */
class FastEuOps[T] extends PointOps[FastEUPoint[T], FastEUPoint[T], T] with Serializable {

  val epsilon = 1e-4

  def distance(p: FastEUPoint[T], c: FastEUPoint[T]): Double = {
    if( p.weight == 0.0 || c.weight == 0.0 ) {
      p.norm + c.norm
    } else {
      var i = 0
      val end = p.raw.length
      var x = 0.0D
      while( i < end ) {
        x = x + p.raw(i) * c.raw(i)
        i = i + 1
      }
      p.norm + c.norm - 2.0 * x / (p.weight * c.weight)
    }
  }

  def userToPoint(raw: Array[Double], index: Option[T]) = new FastEUPoint(raw, 1.0, index)

  def centerToPoint(v: FastEUPoint[T]) = new FastEUPoint(v.raw, v.weight, v.index)

  def centroidToPoint(v: Centroid) = new FastEUPoint(v.raw, v.weight, None)

  def pointToCenter(v: FastEUPoint[T]) = new FastEUPoint(v.raw, v.weight, v.index)

  def centroidToCenter(v: Centroid) = new FastEUPoint(v.raw, v.weight)

  def centerMoved(v: FastEUPoint[T], w: FastEUPoint[T]): Boolean = distance(v, w) > epsilon * epsilon

}

