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
 * http://mark.reid.name/blog/meet-the-bregman-divergences.html
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
 *
 * http://en.wikipedia.org/wiki/Euclidean_distance
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
 * If we know that the points are on the simplex, then we may simplify the implementation
 * of KL divergence.  This trait implements that simplification.
 *
 * http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
 *
 * KL divergence is usually defined using log base 2.  By using natural log,
 * we get distances that are related to the usual definition by a constant. Therefore,
 * the clustering using these distances are identical.
 */
trait KullbackLeiblerSimplexDivergence extends BregmanDivergence {

  def F(v: Vector): Double = dot(trans(v, log), v)

  def F(v: Vector, w: Double) = {
    val logW = log(w)
    dot(trans(v, log(_) - logW), v) / w
  }

  def gradF(v: Vector): Vector = {
    trans(v, 1.0 + log(_))
  }

  def gradF(v: Vector, w: Double): Vector = {
    val c = 1.0 - log(w)
    trans(v, c + log(_))
  }
}

/**
 * The generalized Kullback-Leibler divergence is defined on points on R+ ** n
 *
 * http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
 *
 */
trait KullbackLeiblerDivergence extends BregmanDivergence {

  @inline def logMinusOne(x: Double) = log(x) - 1

  def F(v: Vector): Double = dot(trans(v, logMinusOne), v)

  def F(v: Vector, w: Double) = {
    val logWMinusOne = logMinusOne(w)
    dot(trans(v, logMinusOne(_) - logWMinusOne), v) / w
  }

  def gradF(v: Vector): Vector = {
    trans(v, 1.0 + logMinusOne(_))
  }

  def gradF(v: Vector, w: Double): Vector = {
    val c = 1.0 - logMinusOne(w)
    trans(v, c + logMinusOne(_))
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
 * The Logistic loss divergence is defined on points in (0.0,1.0)
 *
 * Logistic loss is the same as KL Divergence with the embedding into R**2
 *
 *    x => (x, 1.0 - x)
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

trait LogTable {
  private val logTable = new Array[Double](4096 * 1000)

  def fastLog(d: Double): Double = {
    if (d == 0.0 || d == 1.0) {
      0.0
    } else {
      if (d < logTable.length ) {
        val x = d.toInt
        if (x.toDouble == d) {
          if (logTable(x) == 0.0) logTable(x) = Math.log(x)
          logTable(x)
        } else {
          Math.log(d)
        }
      }  else {
        Math.log(d)
      }
    }
  }
}

/**
 * An implementation of Kullback Leibler divergence that is most efficient
 * for vectors whose values are integral frequencies and whose weight is the
 * sum of those frequencies.
 */
trait DiscreteKullbackLeiblerDivergence extends BregmanDivergence with LogTable {

  def F(v: Vector): Double = dot(trans(v, fastLog), v)

  def F(v: Vector, w: Double) = {
    val logW = fastLog(w)
    dot(trans(v, fastLog(_) - logW), v) / w
  }

  def gradF(v: Vector): Vector = {
    trans(v, 1.0 + fastLog(_))
  }

  def gradF(v: Vector, w: Double): Vector = {
    val c = 1.0 - fastLog(w)
    trans(v, c + fastLog(_))
  }
}


