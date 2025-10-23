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

package com.massivedatascience.divergence

import com.massivedatascience.linalg.BLAS._
import com.massivedatascience.util.{ DiscreteLog, GeneralLog, MathLog, ValidationUtils }
import org.apache.spark.ml.linalg.{ Vector, Vectors }

/** Bregman Divergence defines a convex function F and its gradient.
  *
  * For convenience, we also provide methods for evaluating F and its gradient when points are
  * provided using homogeneous coordinates. Evaluating F and gradient of F on homogeneous
  * coordinates is often more efficient that mapping the homogeneous coordinates to inhomogeneous
  * coordinates and then evaluating F and gradient F.
  *
  * @see
  *   <a href="http://en.wikipedia.org/wiki/Bregman_divergence">Definition</a>
  * @see
  *   <a href="http://mark.reid.name/blog/meet-the-bregman-divergences.html">A gentle
  *   introduction</a>
  * @see
  *   <a href="http://jmlr.csail.mit.edu/papers/volume6/banerjee05b/banerjee05b.pdf">Clustering with
  *   Bregman Divergences</a>
  */

trait BregmanDivergence extends Serializable {

  /** F is any convex function.
    *
    * @param v
    *   input
    * @return
    *   F(v)
    */
  def convex(v: Vector): Double

  /** Gradient of F
    *
    * @param v
    *   input
    * @return
    *   gradient of F when at v
    */
  def gradientOfConvex(v: Vector): Vector

  /** F applied to homogeneous coordinates.
    *
    * @param v
    *   input
    * @param w
    *   weight
    * @return
    *   F(v/w)
    */
  def convexHomogeneous(v: Vector, w: Double): Double

  /** Gradient of F, applied to homogeneous coordinates
    * @param v
    *   input
    * @param w
    *   weight
    * @return
    *   gradient(v/w)
    */
  def gradientOfConvexHomogeneous(v: Vector, w: Double): Vector
}

/** The Kullback-Leibler divergence is defined on points on a simplex in R+ ** n
  *
  * If we know that the points are on the simplex, then we may simplify the implementation of KL
  * divergence. This trait implements that simplification.
  *
  * @see
  *   <a href="http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Definition</a>
  */
trait KullbackLeiblerSimplexDivergence extends BregmanDivergence {

  protected val logFunc: MathLog

  def convex(v: Vector): Double = dot(trans(v, logFunc.log), v)

  def convexHomogeneous(v: Vector, w: Double): Double = {
    ValidationUtils.requirePositiveWeight(w, "Weight for KL simplex divergence")
    convex(v) / w - logFunc.log(w)
  }

  def gradientOfConvex(v: Vector): Vector = {
    trans(v, logFunc.log)
  }

  def gradientOfConvexHomogeneous(v: Vector, w: Double): Vector = {
    ValidationUtils.requirePositiveWeight(w, "Weight for KL simplex divergence gradient")
    ValidationUtils.requirePositiveVector(v, "Vector elements for KL simplex divergence gradient")
    val c = logFunc.log(w)
    trans(v, x => logFunc.log(x) - c)
  }
}

/** The generalized Kullback-Leibler divergence is defined on points on R+ ** n
  *
  * @see
  *   <a href="http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Definition</a>
  */
trait KullbackLeiblerDivergence extends BregmanDivergence {

  protected val logFunc: MathLog

  def convex(v: Vector): Double = dot(trans(v, x => logFunc.log(x) - 1), v)

  def convexHomogeneous(v: Vector, w: Double): Double = {
    ValidationUtils.requirePositiveWeight(w, "Weight for KL divergence")
    val c = v.copy
    scal(1.0 / w, c)
    convex(c)
  }

  def gradientOfConvex(v: Vector): Vector = {
    trans(v, logFunc.log)
  }

  def gradientOfConvexHomogeneous(v: Vector, w: Double): Vector = {
    ValidationUtils.requirePositiveWeight(w, "Weight for KL divergence gradient")
    ValidationUtils.requirePositiveVector(v, "Vector elements for KL divergence gradient")
    val c = logFunc.log(w)
    trans(v, x => logFunc.log(x) - c)
  }
}

/** The squared Euclidean distance function is defined on points in R**n
  *
  * @see
  *   <a href="http://en.wikipedia.org/wiki/Euclidean_distance">Definition</a>
  */
case object SquaredEuclideanDistanceDivergence extends BregmanDivergence {

  /** Squared L ** 2 norm
    * @param v
    *   input
    * @return
    *   F(v)
    */
  def convex(v: Vector): Double = dot(v, v)

  def convexHomogeneous(v: Vector, w: Double): Double = {
    if (w == 0.0) {
      0.0 // For Euclidean distance, allow zero weight and return 0
    } else {
      dot(v, v) / (w * w)
    }
  }

  def gradientOfConvex(v: Vector): Vector = {
    val c = v.copy
    scal(2.0, c)
    c
  }

  def gradientOfConvexHomogeneous(v: Vector, w: Double): Vector = {
    require(w != 0.0, "Weight must be nonzero for Euclidean gradient")
    val c = v.copy
    // Return ∇F(v/w) = 2(v/w) to match the pattern used by other divergences
    // The caller will apply dot(v, ∇F(v/w))/w to get ⟨v/w, ∇F(v/w)⟩
    scal(2.0 / w, c)
    c
  }
}

case object RealKullbackLeiblerSimplexDivergence extends KullbackLeiblerSimplexDivergence {
  protected val logFunc: MathLog = GeneralLog
}

case object NaturalKLSimplexDivergence extends KullbackLeiblerSimplexDivergence {
  protected val logFunc: MathLog = DiscreteLog
}

case object RealKLDivergence extends KullbackLeiblerDivergence {
  protected val logFunc: MathLog = GeneralLog
}

case object NaturalKLDivergence extends KullbackLeiblerDivergence {
  protected val logFunc: MathLog = DiscreteLog
}

/** The generalized I-Divergence is defined on points in R**n
  */
case object GeneralizedIDivergence extends BregmanDivergence {
  protected val logFunc: MathLog = GeneralLog

  // F(x) = sum_i x_i (log x_i - 1)
  def convex(v: Vector): Double =
    dot(trans(v, x => logFunc.log(x) - 1.0), v)

  // F(v/w) with input given in homogeneous coords
  def convexHomogeneous(v: Vector, w: Double): Double = {
    ValidationUtils.requirePositiveWeight(w, "Weight for generalized I-divergence")
    ValidationUtils.requirePositiveVector(v, "Vector elements for generalized I-divergence")
    val c = v.copy
    scal(1.0 / w, c)
    convex(c)
  }

  // ∇F(x) = log x
  def gradientOfConvex(v: Vector): Vector =
    trans(v, logFunc.log)

  // ∇F(v/w) = log(v/w) = log v - log w
  def gradientOfConvexHomogeneous(v: Vector, w: Double): Vector = {
    ValidationUtils.requirePositiveWeight(w, "Weight for generalized I-divergence gradient")
    ValidationUtils.requirePositiveVector(
      v,
      "Vector elements for generalized I-divergence gradient"
    )
    val c = logFunc.log(w)
    trans(v, x => logFunc.log(x) - c)
  }
}

/** The Logistic loss divergence is defined on points in (0.0,1.0)
  *
  * Logistic loss is the same as KL Divergence with the embedding into R**2
  *
  * x => (x, 1.0 - x)
  */
case object LogisticLossDivergence extends BregmanDivergence {

  protected val log = GeneralLog.log _

  def convex(v: Vector): Double = {
    val x = v(0)
    x * log(x) + (1.0 - x) * log(1.0 - x)
  }

  def convexHomogeneous(v: Vector, w: Double): Double = {
    ValidationUtils.requirePositiveWeight(w, "Weight for logistic loss divergence")
    val x = v(0) / w
    ValidationUtils.requireValidProbability(x, "Normalized value for logistic loss")
    if (x <= 0.0 || x >= 1.0) {
      throw new IllegalArgumentException(s"Value must be in (0,1), got: $x")
    }
    x * log(x) + (1.0 - x) * log(1.0 - x)
  }

  def gradientOfConvex(v: Vector): Vector = {
    val x = v(0)
    Vectors.dense(log(x) - log(1.0 - x))
  }

  def gradientOfConvexHomogeneous(v: Vector, w: Double): Vector = {
    ValidationUtils.requirePositiveWeight(w, "Weight for logistic loss gradient")
    val x = v(0) / w
    ValidationUtils.requireValidProbability(x, "Normalized value for logistic loss gradient")
    if (x <= 0.0 || x >= 1.0) {
      throw new IllegalArgumentException(s"Value must be in (0,1), got: $x")
    }
    Vectors.dense(log(x) - log(1.0 - x))
  }
}

/** The Itakura-Saito Divergence is defined on points in R+ ** n
  *
  * @see
  *   <a href="http://en.wikipedia.org/wiki/Itakura%E2%80%93Saito_distance">Definition</a>
  */
case object ItakuraSaitoDivergence extends BregmanDivergence {

  protected val logFunc: MathLog = GeneralLog

  /** Burg entropy
    *
    * @param v
    *   input
    * @return
    *   F(v)
    */
  def convex(v: Vector): Double = {
    -sum(trans(v, logFunc.log))
  }

  def convexHomogeneous(v: Vector, w: Double): Double = {
    ValidationUtils.requirePositiveWeight(w, "Weight for Itakura-Saito divergence")
    ValidationUtils.requirePositiveVector(v, "Vector elements for Itakura-Saito divergence")
    logFunc.log(w) - sum(trans(v, logFunc.log))
  }

  def gradientOfConvex(v: Vector): Vector = {
    trans(v, x => -1.0 / x)
  }

  def gradientOfConvexHomogeneous(v: Vector, w: Double): Vector = {
    ValidationUtils.requirePositiveWeight(w, "Weight for Itakura-Saito gradient")
    ValidationUtils.requirePositiveVector(v, "Vector elements for Itakura-Saito gradient")
    trans(v, x => -w / x)
  }
}

object BregmanDivergence {

  def apply(name: String): BregmanDivergence = {
    name match {
      case "SquaredEuclideanDistanceDivergence"   => SquaredEuclideanDistanceDivergence
      case "RealKullbackLeiblerSimplexDivergence" => RealKullbackLeiblerSimplexDivergence
      case "NaturalKLSimplexDivergence"           => NaturalKLSimplexDivergence
      case "RealKLDivergence"                     => RealKLDivergence
      case "NaturalKLDivergence"                  => NaturalKLDivergence
      case "LogisticLossDivergence"               => LogisticLossDivergence
      case "ItakuraSaitoDivergence"               => ItakuraSaitoDivergence
      case "GeneralizedIDivergence"               => GeneralizedIDivergence
    }
  }

  /** Create a Bregman Divergence from
    * @param f
    *   any continuously-differentiable real-valued and strictly convex function defined on a closed
    *   convex set in R^^N
    * @param gradientF
    *   the gradient of f
    * @return
    *   a Bregman Divergence on that function
    */
  def apply(f: (Vector) => Double, gradientF: (Vector) => Vector): BregmanDivergence =
    new BregmanDivergence {
      def convex(v: Vector): Double = f(v)

      def convexHomogeneous(v: Vector, w: Double): Double = {
        ValidationUtils.requirePositiveWeight(w, "Weight for custom divergence")
        val c = v.copy
        scal(1.0 / w, c)
        convex(c)
      }

      def gradientOfConvex(v: Vector): Vector = gradientF(v)

      def gradientOfConvexHomogeneous(v: Vector, w: Double): Vector = {
        ValidationUtils.requirePositiveWeight(w, "Weight for custom divergence gradient")
        val c = v.copy
        scal(1.0 / w, c)
        gradientF(c)
      }
    }
}
