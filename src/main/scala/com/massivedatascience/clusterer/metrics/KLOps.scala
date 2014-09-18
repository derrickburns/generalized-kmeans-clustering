/*
 * Licensed to the Massive Data Science and Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Massive Data Science and Derrick R. Burns licenses this file to You under the
 * Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.massivedatascience.clusterer.metrics

import breeze.linalg.{sum, DenseVector => BDV, Vector => BV}
import com.massivedatascience.clusterer.base._
import org.apache.spark.mllib.linalg.{DenseVector, Vector}


class KLPoint(raw: BV[Double], weight: Double) extends FPoint(raw, weight) {
  /**
   * the negative of the entropy times the weight of the point
   */
  val weightedNegEntropy: Double = {
    var d = Zero
    var i = 0
    while (i < raw.length) {
      val x = raw(i)
      d = d + x * KLOps.fastLog(x)
      i = i + 1
    }
    assert(d >= Zero)
    d - weight * KLOps.fastLog(weight)
  }
}

class KLCenter(r: BV[Double], w: Double) extends FPoint(r, w) {
  private val logWeight = KLOps.fastLog(weight)
  val getLog = raw map {
    KLOps.fastLog(_) - logWeight
  }
}

object KLOps extends Serializable with LogTable

/**
 * Kullback-Leibler - return an lower bound on the distance
 */
class KLOps extends PointOps[KLPoint, KLCenter] with Serializable {

  type C = KLCenter
  type P = KLPoint

  val epsilon = 1e-4

  /**
   * Distance from point to cluster, if that distance is less than a given bound.
   *
   * A cluster with zero weight (an empty cluster) is infinitely far away.
   * A point with zero weight is co-incident with any non-empty cluster.
   *
   * @param p point
   * @param c center
   * @param upperBound upper bound on distance
   * @return distance to cluster
   */
  def distance(p: P, c: C, upperBound: Double = Infinity): Double = {
    assert(upperBound >= Zero)
    if (c.weight <= epsilon) return Infinity
    if (p.weight <= epsilon) return Zero
    val weightedNegativeCrossEntropy = c.getLog.dot(p.raw)
    val kl = (p.weightedNegEntropy - weightedNegativeCrossEntropy) / p.weight
    if (kl < Zero) Zero else if (kl < upperBound) kl else upperBound
  }

  def arrayToPoint(x: Array[Double]): P = {
    val bv = BDV(x)
    new KLPoint(bv, sum(bv))
  }

  def vectorToPoint(x: Vector): P = {
    val raw = BDV(x.toArray)
    new KLPoint(raw, sum(raw))
  }

  def centerToPoint(v: C) = new P(v.raw :- 1.0, v.weight - v.raw.length)

  def centroidToPoint(v: Centroid) = new KLPoint(v.raw, v.weight)

  def pointToCenter(v: P) = new C(v.raw :+ 1.0, v.weight + v.raw.length)

  def centroidToCenter(v: Centroid) = new C(v.raw :+ 1.0, v.weight + v.raw.length)

  def centerMoved(v: P, w: KLCenter): Boolean = distance(v, w) > epsilon * epsilon

  def centerToVector(c: C) = new DenseVector(c.inh)

}
