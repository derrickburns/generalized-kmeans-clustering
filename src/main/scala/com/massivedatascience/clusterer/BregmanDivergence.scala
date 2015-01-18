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
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
 * Bregman Divergence defines a convex function F and its gradient.  
 *
 * For convenience, we also provide methods for evaluating F and its 
 * gradient when points are provided using homogeneous coordinates.  Evaluating F and
 * gradientF on homogeneous coordinates is often more efficient that mapping
 * the homogeneous coordinates to inhomogeneous coordinates and then evaluating F and gradientF.
 *
 * http://en.wikipedia.org/wiki/Bregman_divergence
 *
 * Bregman Divergences are used in K-Means clustering:
 *
 * http://jmlr.csail.mit.edu/papers/volume6/banerjee05b/banerjee05b.pdf
 */

trait BregmanDivergence {
  @inline
  def log(x: Double) = if (x == 0.0) 0.0 else Math.log(x)

  /**
   * F is any convex function.
   *
   * @param v input
   * @return F(v)
   */
  def F(v: Vector): Double

  /**
   * Gradient of F
   *
   * @param v input
   * @return  gradient of F when at v
   */
  def gradF(v: Vector): Vector

  /**
   * F applied to homogeneous coordinates.
   *
   * @param v input
   * @param w weight
   * @return  F(v/w)
   */
  def F(v: Vector, w: Double): Double

  /**
   * Gradient of F, applied to homogeneous coordinates
   * @param v input
   * @param w weight
   * @return  gradient(v/w)
   */
  def gradF(v: Vector, w: Double): Vector
}

/**
 * The squared Euclidean distance function is defined on points in R**n
 */
trait SquaredEuclideanDistanceDivergence extends BregmanDivergence {

  /**
   * Squared L^2 norm
   * @param v input
   * @return F(v)
   */
  def F(v: Vector): Double = dot(v, v)

  def F(v: Vector, w: Double) = dot(v, v) / (w * w)

  def gradF(v: Vector): Vector = {
    val c = v.copy
    scal(2.0, c)
    c
  }

  def gradF(v: Vector, w: Double): Vector = {
    val c = v.copy
    scal(2.0 / w, c)
    c
  }
}

/**
 * The Kullback-Leibler divergence is defined on points on a simplex in R+ ** n
 *
 */
trait KullbackLeiblerSimplexDivergence extends BregmanDivergence {
  val invLog2 = 1.0 / Math.log(2)

  @inline def logBase2(x: Double) = log(x) * invLog2

  def F(v: Vector): Double = dot(trans(v, logBase2), v)

  def F(v: Vector, w: Double) = {
    val logBase2w = logBase2(w)
    dot(trans(v, logBase2(_) - logBase2w), v) / w
  }

  def gradF(v: Vector): Vector = {
    trans(v, invLog2 + logBase2(_))
  }

  def gradF(v: Vector, w: Double): Vector = {
    val c = invLog2 - logBase2(w)
    trans(v, c + logBase2(_))
  }
}

/**
 * The geenralized Kullback-Leibler divergence is defined on points on R+ ** n
 *
 */
trait KullbackLeiblerDivergence extends BregmanDivergence {
  val invLog2 = 1.0 / Math.log(2)

  @inline def logBase2Minus1(x: Double) = log(x) * invLog2 - 1

  def F(v: Vector): Double = dot(trans(v, logBase2Minus1), v)

  def F(v: Vector, w: Double) = {
    val logBase2w = logBase2Minus1(w)
    dot(trans(v, logBase2Minus1(_) - logBase2w), v) / w
  }

  def gradF(v: Vector): Vector = {
    trans(v, invLog2 + logBase2Minus1(_))
  }

  def gradF(v: Vector, w: Double): Vector = {
    val c = invLog2 - logBase2Minus1(w)
    trans(v, c + logBase2Minus1(_))
  }
}

/**
 * The generalized I-Divergence is defined on points in R**n
 */
trait GeneralizedIDivergence extends BregmanDivergence {

  def F(v: Vector): Double = dot(trans(v, log), v)

  def F(v: Vector, w: Double): Double = {
    val logW = log(w)
    dot(v, trans(v, log(_) - logW)) / w
  }

  def gradF(v: Vector): Vector = {
    trans(v, log)
  }

  def gradF(v: Vector, w: Double): Vector = {
    val c = -log(w)
    trans(v, c + log(_))
  }
}

/**
 * The Logistic loss divergence is defined on points in R
 */
trait LogisticLossDivergence extends BregmanDivergence {

  def F(v: Vector): Double = {
    val x = v(0)
    x * log(x) + (1.0 - x) * log(1.0 - x)
  }

  def F(v: Vector, w: Double): Double = {
    val x = v(0) / w
    x * log(x) + (1.0 - x) * log(1.0 - x)
  }

  def gradF(v: Vector): Vector = {
    val x = v(0)
    Vectors.dense(log(x) - log(1.0 - x))
  }

  def gradF(v: Vector, w: Double): Vector = {
    val x = v(0) / w
    Vectors.dense(log(x) - log(1.0 - x))
  }
}


/**
 * The Itakura-Saito Divergence is defined on points in R+ ** n
 */
trait ItakuraSaitoDivergence extends BregmanDivergence {

  /**
   * Burg entropy
   *
   * @param v input
   * @return F(v)
   */
  def F(v: Vector): Double = {
    -sum(trans(v, log))
  }

  def F(v: Vector, w: Double): Double = {
    log(w) - sum(trans(v, log))
  }

  def gradF(v: Vector): Vector = {
    trans(v, x => -1.0 / x)
  }

  def gradF(v: Vector, w: Double): Vector = {
    val c = 1.0 - log(w)
    trans(v, x => -w / x)
  }
}

