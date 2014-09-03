package com.rincaro.clusterer

package object base {

  trait FP[T] extends Serializable {
    val weight: Double
    val index: Option[T]
    val raw : Array[Double]
  }

  class FPoint[T](val raw: Array[Double], val weight: Double, val index: Option[T]) extends FP[T] {
    override def toString: String = weight + "," + (raw mkString ",")

    lazy val inh = {
      val end = raw.length
      val x = new Array[Double](end)
      var i = 0
      while (i != end) {
        x(i) = raw(i) / weight
        i = i + 1
      }
      x
    }
  }

  /**
   * A mutable point in homogeneous coordinates
   */
  class Centroid extends Serializable {
    override def toString: String = weight + "," + (raw mkString ",")

    var raw: Array[Double] = null
    var weight: Double = 0.0

    def add( p: Centroid ) : Centroid = add(p.raw, p.weight)

    def add( p: FP[_] ) : Centroid = add(p.raw, p.weight)

    private def add(r: Array[Double], w: Double): Centroid = {
      if (r != null) {
        if (raw == null) {
          raw = r.clone()
          weight = w
        } else {
          var i = 0
          val end = raw.length
          while (i < end) {
            raw(i) = raw(i) + r(i)
            i = i + 1
          }
          weight = weight + w
        }
      }
      this
    }
  }

  trait PointOps[P <: FP[T], C <: FP[T], T] {
    def distance(p: P, c: C): Double

    def userToPoint(v: Array[Double], index: Option[T]): P

    def centerToPoint(v: C): P

    def pointToCenter(v: P): C

    def centroidToCenter(v: Centroid): C

    def centroidToPoint(v: Centroid): P

    def centerMoved(v: P, w: C): Boolean
  }


}
