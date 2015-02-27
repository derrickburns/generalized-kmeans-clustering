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
 * gradient of F on homogeneous coordinates is often more efficient that mapping
 * the homogeneous coordinates to inhomogeneous coordinates and then evaluating F and gradient F.
 *
 * http://mark.reid.name/blog/meet-the-bregman-divergences.html
 *
 * http://en.wikipedia.org/wiki/Bregman_divergence
 *
 * Bregman Divergences are used in K-Means clustering:
 *
 * http://jmlr.csail.mit.edu/papers/volume6/banerjee05b/banerjee05b.pdf
 */

private[clusterer]
trait BregmanDivergence extends Serializable {

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
 * The Kullback-Leibler divergence is defined on points on a simplex in R+ ** n
 *
 * If we know that the points are on the simplex, then we may simplify the implementation of KL
 * divergence. This trait implements that simplification.
 *
 * http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
 *
 */
private[clusterer]
trait KullbackLeiblerSimplexDivergence extends BregmanDivergence {

  val logFunc: MathLog

  def F(v: Vector): Double = dot(trans(v, logFunc.log), v)

  def F(v: Vector, w: Double) = {
    F(v) / w - logFunc.log(w) - 1.0
  }

  def gradF(v: Vector): Vector = {
    trans(v, logFunc.log)
  }

  def gradF(v: Vector, w: Double): Vector = {
    val c = logFunc.log(w)
    trans(v, x => logFunc.log(x) - c)
  }
}

/**
 * The generalized Kullback-Leibler divergence is defined on points on R+ ** n
 *
 * http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
 *
 */
private[clusterer]
trait KullbackLeiblerDivergence extends BregmanDivergence {

  val logFunc: MathLog

  def F(v: Vector): Double = dot(trans(v, x => logFunc.log(x) - 1), v)

  def F(v: Vector, w: Double) = {
    val c = v.copy
    scal(1.0 / w, c)
    F(c)
  }

  def gradF(v: Vector): Vector = {
    trans(v, logFunc.log)
  }

  def gradF(v: Vector, w: Double): Vector = {
    val c = logFunc.log(w)
    trans(v, x => logFunc.log(x) - c)
  }
}

/**
 * The squared Euclidean distance function is defined on points in R**n
 *
 * http://en.wikipedia.org/wiki/Euclidean_distance
 */
private[clusterer]
case object SquaredEuclideanDistanceDivergence extends BregmanDivergence {

  /**
   * Squared L ** 2 norm
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

private[clusterer]
case object RealKullbackLeiblerSimplexDivergence extends KullbackLeiblerSimplexDivergence {
  val logFunc: MathLog = GeneralLog
}

private[clusterer]
case object NaturalKLSimplexDivergence extends KullbackLeiblerSimplexDivergence {
  val logFunc: MathLog = DiscreteLog
}

private[clusterer]
case object RealKLDivergence extends KullbackLeiblerDivergence {
  val logFunc: MathLog = GeneralLog
}

private[clusterer]
case object NaturalKLDivergence extends KullbackLeiblerDivergence {
  val logFunc: MathLog = DiscreteLog
}

/**
 * The generalized I-Divergence is defined on points in R**n
 */
private[clusterer]
case object GeneralizedIDivergence extends BregmanDivergence {

  val logFunc: MathLog = DiscreteLog

  def F(v: Vector): Double = dot(trans(v, logFunc.log), v)

  def F(v: Vector, w: Double): Double = {
    val logW = logFunc.log(w)
    dot(v, trans(v, logFunc.log(_) - logW)) / w
  }

  def gradF(v: Vector): Vector = {
    trans(v, logFunc.log)
  }

  def gradF(v: Vector, w: Double): Vector = {
    val c = -logFunc.log(w)
    trans(v, c + logFunc.log(_))
  }
}

/**
 * The Logistic loss divergence is defined on points in (0.0,1.0)
 *
 * Logistic loss is the same as KL Divergence with the embedding into R**2
 *
 * x => (x, 1.0 - x)
 */
private[clusterer]
case object LogisticLossDivergence extends BregmanDivergence {

  val log = GeneralLog.log _

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
 *
 * http://en.wikipedia.org/wiki/Itakura%E2%80%93Saito_distance
 */
private[clusterer]
case object ItakuraSaitoDivergence extends BregmanDivergence {

  val logFunc: MathLog = GeneralLog

  /**
   * Burg entropy
   *
   * @param v input
   * @return F(v)
   */
  def F(v: Vector): Double = {
    -sum(trans(v, logFunc.log))
  }

  def F(v: Vector, w: Double): Double = {
    logFunc.log(w) - sum(trans(v, logFunc.log))
  }

  def gradF(v: Vector): Vector = {
    trans(v, x => -1.0 / x)
  }

  def gradF(v: Vector, w: Double): Vector = {
    val c = 1.0 - logFunc.log(w)
    trans(v, x => -w / x)
  }
}

object BregmanDivergence {
  /**
   * Create a Bregman Divergence from
   * @param f any continuously-differentiable real-valued and strictly
   *          convex function defined on a closed convex set in R^^N
   * @param gradientF the gradient of f
   * @return a Bregman Divergence on that function
   */
  def apply(f: (Vector) => Double, gradientF: (Vector) => Vector): BregmanDivergence =
    new BregmanDivergence {
      def F(v: Vector): Double = f(v)

      def F(v: Vector, w: Double) = {
        val c = v.copy
        scal(1.0 / w, c)
        F(c)
      }

      def gradF(v: Vector): Vector = gradientF(v)

      def gradF(v: Vector, w: Double): Vector = {
        val c = v.copy
        scal(1.0 / w, c)
        gradientF(c)
      }
    }
}
