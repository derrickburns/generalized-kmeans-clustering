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

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.util.BLAS._
import org.apache.spark.mllib.linalg.Vector


/**
 * A point with an additional single Double value that is used in distance computation.
 *
 * @param inh  inhomogeneous coordinates of point
 * @param weight weight of point
 * @param f f(point)
 */
class BregmanPoint(inh: Vector, weight: Double, val f: Double)
  extends ImmutableInhomogeneousVector(inh, weight)

/**
 * A cluster center with two additional Double values that are used in distance computation.
 *
 * @param h vector in homogeneous coordinates
 * @param weight weight of vector
 * @param dotGradMinusF  center dot gradient(center) - f(center)
 * @param gradient gradient of center
 */
class BregmanCenter(h: Vector, weight: Double, val dotGradMinusF: Double, val gradient: Vector)
  extends ImmutableHomogeneousVector(h, weight)


trait BregmanPointOps extends PointOps[BregmanPoint, BregmanCenter] {
  this: BregmanDivergence =>
  val weightThreshold = 1e-4
  val distanceThreshold = 1e-8

  /**
   * Bregman distance function
   *
   * The distance function is called in the innermost loop of the K-Means clustering algorithm.
   * Therefore, we seek to make the operation as efficient as possible.
   *
   * @param p point
   * @param c center
   * @return
   */
  def distance(p: BregmanPoint, c: BregmanCenter): Double = {
    if (c.weight <= weightThreshold) {
      Infinity
    } else if (p.weight <= weightThreshold) {
      0.0
    } else {
      val d = p.f + c.dotGradMinusF - dot(c.gradient, p.inhomogeneous)
      if (d < 0.0) 0.0 else d
    }
  }

  def homogeneousToPoint(h: Vector, weight: Double): BregmanPoint = {
    val inh = asInhomogeneous(h, weight)
    new BregmanPoint(inh, weight, F(inh))
  }

  def inhomogeneousToPoint(inh: Vector, weight: Double): BregmanPoint =
    new BregmanPoint(inh, weight, F(inh))

  def toCenter(v: WeightedVector): BregmanCenter = {
    val h = v.homogeneous
    val w = v.weight
    val df = gradF(h, w)
    new BregmanCenter(h, w, dot(h, df) / w - F(h, w), df)
  }

  def toPoint(v: WeightedVector): BregmanPoint = {
    val inh = v.inhomogeneous
    new BregmanPoint(inh, v.weight, F(inh))
  }

  def centerMoved(v: BregmanPoint, w: BregmanCenter): Boolean =
    distance(v, w) > distanceThreshold
}

object KullbackLeiblerPointOps extends KullbackLeiblerDivergence with BregmanPointOps

object GeneralizedIPointOps extends GeneralizedIDivergence with BregmanPointOps

object SquaredEuclideanPointOps extends SquaredEuclideanDistanceDivergence with BregmanPointOps

object LogisticLossPointOps extends LogisticLossDivergence with BregmanPointOps
