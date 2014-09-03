package com.rincaro.clusterer.metrics

import com.rincaro.clusterer.base.{Centroid, FPoint, PointOps}

/**
 * Euclidean distance measure - not optimized
 */
class EuOps[T] extends PointOps[FPoint[T], FPoint[T], T] with Serializable {

  val epsilon = 1e-4

  def distance(p: FPoint[T], c: FPoint[T]): Double = {
    p.inh.zip(c.inh).foldLeft(0.0D) { case (d: Double, (a: Double, b: Double)) => d + (a - b) * (a - b)}
  }

  def userToPoint(raw: Array[Double], name: Option[T]) = new FPoint[T](raw, 1, name)

  def centerToPoint(v: FPoint[T]) = new FPoint[T](v.raw, v.weight, v.index)

  def centroidToPoint(v: Centroid) = new FPoint[T](v.raw, v.weight, None)

  def pointToCenter(v: FPoint[T]) = new FPoint[T](v.raw, v.weight, v.index)

  def centroidToCenter(v: Centroid) = new FPoint[T](v.raw, v.weight, None)

  def centerMoved(v: FPoint[T], w: FPoint[T]): Boolean = distance(v, w) > epsilon * epsilon

}

