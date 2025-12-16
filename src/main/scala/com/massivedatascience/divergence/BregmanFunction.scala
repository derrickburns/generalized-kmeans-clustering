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

import org.apache.spark.ml.linalg.{ Vector, Vectors }

/** Core trait for Bregman functions, providing a single source of truth for divergence computations
  * across both RDD and DataFrame APIs.
  *
  * A Bregman function F: S → ℝ is a strictly convex, continuously differentiable function defined
  * on a closed convex set S ⊆ ℝ^d. The Bregman divergence D_F is defined as:
  *
  * D_F(x, y) = F(x) - F(y) - ⟨∇F(y), x - y⟩
  *
  * This trait provides:
  *   - F(x): The convex function value
  *   - ∇F(x): Gradient (maps to natural parameters)
  *   - (∇F)^(-1)(θ): Inverse gradient (maps natural to expectation parameters)
  *   - D_F(x, y): Direct divergence computation
  *   - validate(x): Domain validation
  *
  * ==Design Rationale==
  *
  * This trait unifies two previously separate implementations:
  *   - `BregmanDivergence` (RDD API): Focused on F and ∇F with homogeneous coordinates
  *   - `BregmanKernel` (DataFrame API): Focused on grad/invGrad for Lloyd's algorithm
  *
  * By consolidating the mathematical definitions, we:
  *   1. Ensure consistent divergence formulas across APIs 2. Reduce maintenance burden (single
  *      place to fix bugs) 3. Enable sharing of implementations between APIs
  *
  * ==Usage==
  *
  * DataFrame API (primary):
  * {{{
  * val kernel = BregmanFunctions.squaredEuclidean
  * val dist = kernel.divergence(point, center)
  * val newCenter = kernel.invGradF(meanGradient)
  * }}}
  *
  * RDD API (legacy, via adapter):
  * {{{
  * val divergence = BregmanFunctionAdapter.asBregmanDivergence(BregmanFunctions.kl)
  * }}}
  *
  * @see
  *   [[https://en.wikipedia.org/wiki/Bregman_divergence Bregman Divergence (Wikipedia)]]
  * @see
  *   [[http://jmlr.csail.mit.edu/papers/volume6/banerjee05b/banerjee05b.pdf Clustering with Bregman Divergences]]
  */
trait BregmanFunction extends Serializable {

  /** Evaluate the convex function F at point x.
    *
    * @param x
    *   input point in the domain S
    * @return
    *   F(x)
    */
  def F(x: Vector): Double

  /** Compute the gradient ∇F(x) at point x.
    *
    * The gradient maps points to natural parameters in the dual space.
    *
    * @param x
    *   input point
    * @return
    *   gradient vector θ = ∇F(x)
    */
  def gradF(x: Vector): Vector

  /** Compute the inverse gradient: given θ = ∇F(x), recover x.
    *
    * This maps natural parameters back to expectation parameters. For k-means: center =
    * invGradF(weighted_mean_of_gradients)
    *
    * @param theta
    *   natural parameters
    * @return
    *   point x such that ∇F(x) = θ
    */
  def invGradF(theta: Vector): Vector

  /** Compute the Bregman divergence D_F(x, y).
    *
    * D_F(x, y) = F(x) - F(y) - ⟨∇F(y), x - y⟩
    *
    * Default implementation uses the definition. Override for efficiency or when the direct formula
    * is simpler (e.g., squared Euclidean).
    *
    * @param x
    *   first point
    * @param y
    *   second point (typically the cluster center)
    * @return
    *   D_F(x, y) ≥ 0
    */
  def divergence(x: Vector, y: Vector): Double = {
    val fx      = F(x)
    val fy      = F(y)
    val gradY   = gradF(y)
    val xArr    = x.toArray
    val yArr    = y.toArray
    val gradArr = gradY.toArray

    var dot = 0.0
    var i   = 0
    while (i < xArr.length) {
      dot += gradArr(i) * (xArr(i) - yArr(i))
      i += 1
    }

    fx - fy - dot
  }

  /** Validate that point x is in the valid domain of F.
    *
    * @param x
    *   point to validate
    * @return
    *   true if x is in valid domain, false otherwise
    */
  def validate(x: Vector): Boolean

  /** Human-readable name for this Bregman function.
    */
  def name: String

  /** Whether this function supports SQL expression-based optimization.
    *
    * Only Squared Euclidean can use the fast cross-join path.
    */
  def supportsExpressionOptimization: Boolean = false
}

/** Squared Euclidean Bregman function: F(x) = ½||x||²
  *
  * Properties:
  *   - ∇F(x) = x
  *   - (∇F)^(-1)(θ) = θ
  *   - D_F(x, y) = ½||x - y||²
  */
class SquaredEuclideanFunction extends BregmanFunction {

  override def F(x: Vector): Double = {
    val arr = x.toArray
    var sum = 0.0
    var i   = 0
    while (i < arr.length) {
      sum += arr(i) * arr(i)
      i += 1
    }
    sum * 0.5
  }

  override def gradF(x: Vector): Vector = x

  override def invGradF(theta: Vector): Vector = theta

  override def divergence(x: Vector, y: Vector): Double = {
    val xArr = x.toArray
    val yArr = y.toArray
    var sum  = 0.0
    var i    = 0
    while (i < xArr.length) {
      val diff = xArr(i) - yArr(i)
      sum += diff * diff
      i += 1
    }
    sum * 0.5
  }

  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    arr.forall(v => java.lang.Double.isFinite(v))
  }

  override def name: String = "SquaredEuclidean"

  override def supportsExpressionOptimization: Boolean = true
}

/** KL Divergence Bregman function: F(x) = ∑_i x_i log(x_i)
  *
  * For probability distributions and count data. Domain: x_i > 0 for all i
  *
  * Properties:
  *   - ∇F(x) = [log(x_i) + 1]
  *   - (∇F)^(-1)(θ) = [exp(θ_i - 1)]
  *   - D_F(x, y) = ∑_i x_i log(x_i / y_i) - x_i + y_i (generalized KL)
  *
  * @param smoothing
  *   small constant to avoid log(0)
  */
class KLFunction(smoothing: Double = 1e-10) extends BregmanFunction {
  require(smoothing > 0, "Smoothing must be positive")

  override def F(x: Vector): Double = {
    val arr = x.toArray
    var sum = 0.0
    var i   = 0
    while (i < arr.length) {
      val xi = arr(i) + smoothing
      sum += xi * math.log(xi)
      i += 1
    }
    sum
  }

  override def gradF(x: Vector): Vector = {
    val arr    = x.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = math.log(arr(i) + smoothing) + 1.0
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGradF(theta: Vector): Vector = {
    val arr    = theta.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = math.exp(arr(i) - 1.0)
      i += 1
    }
    Vectors.dense(result)
  }

  override def divergence(x: Vector, y: Vector): Double = {
    val xArr = x.toArray
    val yArr = y.toArray
    var sum  = 0.0
    var i    = 0
    while (i < xArr.length) {
      val xi = xArr(i) + smoothing
      val yi = yArr(i) + smoothing
      sum += xi * math.log(xi / yi)
      i += 1
    }
    sum
  }

  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    arr.forall(v => java.lang.Double.isFinite(v) && v >= 0.0)
  }

  override def name: String = s"KL(smoothing=$smoothing)"
}

/** Itakura-Saito Bregman function: F(x) = -∑_i log(x_i)
  *
  * For spectral analysis and audio. Domain: x_i > 0
  *
  * Properties:
  *   - ∇F(x) = [-1/x_i]
  *   - (∇F)^(-1)(θ) = [-1/θ_i]
  *   - D_F(x, y) = ∑_i (x_i/y_i - log(x_i/y_i) - 1)
  *
  * @param smoothing
  *   small constant to avoid division by zero
  */
class ItakuraSaitoFunction(smoothing: Double = 1e-10) extends BregmanFunction {
  require(smoothing > 0, "Smoothing must be positive")

  override def F(x: Vector): Double = {
    val arr = x.toArray
    var sum = 0.0
    var i   = 0
    while (i < arr.length) {
      sum -= math.log(arr(i) + smoothing)
      i += 1
    }
    sum
  }

  override def gradF(x: Vector): Vector = {
    val arr    = x.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = -1.0 / (arr(i) + smoothing)
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGradF(theta: Vector): Vector = {
    val arr    = theta.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = -1.0 / arr(i)
      i += 1
    }
    Vectors.dense(result)
  }

  override def divergence(x: Vector, y: Vector): Double = {
    val xArr = x.toArray
    val yArr = y.toArray
    var sum  = 0.0
    var i    = 0
    while (i < xArr.length) {
      val xi = xArr(i) + smoothing
      val yi = yArr(i) + smoothing
      sum += xi / yi - math.log(xi / yi) - 1.0
      i += 1
    }
    sum
  }

  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    arr.forall(v => java.lang.Double.isFinite(v) && v > 0.0)
  }

  override def name: String = s"ItakuraSaito(smoothing=$smoothing)"
}

/** Generalized I-Divergence Bregman function: F(x) = ∑_i x_i log(x_i) - x_i
  *
  * Similar to KL but without normalization constraint.
  *
  * @param smoothing
  *   small constant to avoid log(0)
  */
class GeneralizedIFunction(smoothing: Double = 1e-10) extends BregmanFunction {
  require(smoothing > 0, "Smoothing must be positive")

  override def F(x: Vector): Double = {
    val arr = x.toArray
    var sum = 0.0
    var i   = 0
    while (i < arr.length) {
      val xi = arr(i) + smoothing
      sum += xi * math.log(xi) - xi
      i += 1
    }
    sum
  }

  override def gradF(x: Vector): Vector = {
    val arr    = x.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = math.log(arr(i) + smoothing)
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGradF(theta: Vector): Vector = {
    val arr    = theta.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = math.exp(arr(i))
      i += 1
    }
    Vectors.dense(result)
  }

  override def divergence(x: Vector, y: Vector): Double = {
    val xArr = x.toArray
    val yArr = y.toArray
    var sum  = 0.0
    var i    = 0
    while (i < xArr.length) {
      val xi = xArr(i) + smoothing
      val yi = yArr(i) + smoothing
      sum += xi * math.log(xi / yi) - xi + yi
      i += 1
    }
    sum
  }

  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    arr.forall(v => java.lang.Double.isFinite(v) && v >= 0.0)
  }

  override def name: String = s"GeneralizedI(smoothing=$smoothing)"
}

/** Logistic Loss Bregman function: F(x) = ∑_i (x_i log(x_i) + (1-x_i)log(1-x_i))
  *
  * For binary probability vectors. Domain: x_i ∈ (0, 1)
  *
  * @param smoothing
  *   small constant to keep values away from 0 and 1
  */
class LogisticLossFunction(smoothing: Double = 1e-10) extends BregmanFunction {
  require(smoothing > 0 && smoothing < 0.5, "Smoothing must be in (0, 0.5)")

  override def F(x: Vector): Double = {
    val arr = x.toArray
    var sum = 0.0
    var i   = 0
    while (i < arr.length) {
      val xi = math.max(smoothing, math.min(1.0 - smoothing, arr(i)))
      sum += xi * math.log(xi) + (1.0 - xi) * math.log(1.0 - xi)
      i += 1
    }
    sum
  }

  override def gradF(x: Vector): Vector = {
    val arr    = x.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      val xi = math.max(smoothing, math.min(1.0 - smoothing, arr(i)))
      result(i) = math.log(xi / (1.0 - xi))
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGradF(theta: Vector): Vector = {
    val arr    = theta.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = 1.0 / (1.0 + math.exp(-arr(i)))
      i += 1
    }
    Vectors.dense(result)
  }

  override def divergence(x: Vector, y: Vector): Double = {
    val xArr = x.toArray
    val yArr = y.toArray
    var sum  = 0.0
    var i    = 0
    while (i < xArr.length) {
      val xi = math.max(smoothing, math.min(1.0 - smoothing, xArr(i)))
      val yi = math.max(smoothing, math.min(1.0 - smoothing, yArr(i)))
      sum += xi * math.log(xi / yi) + (1.0 - xi) * math.log((1.0 - xi) / (1.0 - yi))
      i += 1
    }
    sum
  }

  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    arr.forall(v => java.lang.Double.isFinite(v) && v >= 0.0 && v <= 1.0)
  }

  override def name: String = s"LogisticLoss(smoothing=$smoothing)"
}

/** L1 (Manhattan) distance function.
  *
  * NOT a Bregman divergence (non-differentiable at 0), but provided for K-Medians. Uses sign
  * function as subgradient.
  */
class L1Function extends BregmanFunction {

  override def F(x: Vector): Double = {
    val arr = x.toArray
    var sum = 0.0
    var i   = 0
    while (i < arr.length) {
      sum += math.abs(arr(i))
      i += 1
    }
    sum
  }

  override def gradF(x: Vector): Vector = {
    val arr    = x.toArray
    val result = new Array[Double](arr.length)
    var i      = 0
    while (i < arr.length) {
      result(i) = if (arr(i) > 0) 1.0 else if (arr(i) < 0) -1.0 else 0.0
      i += 1
    }
    Vectors.dense(result)
  }

  override def invGradF(theta: Vector): Vector = theta // Identity for K-Medians

  override def divergence(x: Vector, y: Vector): Double = {
    val xArr = x.toArray
    val yArr = y.toArray
    var sum  = 0.0
    var i    = 0
    while (i < xArr.length) {
      sum += math.abs(xArr(i) - yArr(i))
      i += 1
    }
    sum
  }

  override def validate(x: Vector): Boolean = {
    val arr = x.toArray
    arr.forall(v => java.lang.Double.isFinite(v))
  }

  override def name: String = "L1"
}

/** Spherical (cosine similarity) function.
  *
  * NOT a Bregman divergence, but provided for spherical K-Means. Automatically normalizes vectors
  * to unit length.
  */
class SphericalFunction extends BregmanFunction {

  private def normalize(x: Vector): Vector = {
    val arr  = x.toArray
    var norm = 0.0
    var i    = 0
    while (i < arr.length) {
      norm += arr(i) * arr(i)
      i += 1
    }
    norm = math.sqrt(norm)

    if (norm > 1e-10) {
      val result = new Array[Double](arr.length)
      i = 0
      while (i < arr.length) {
        result(i) = arr(i) / norm
        i += 1
      }
      Vectors.dense(result)
    } else {
      x
    }
  }

  override def F(x: Vector): Double = 0.0 // Not used for spherical

  override def gradF(x: Vector): Vector = normalize(x)

  override def invGradF(theta: Vector): Vector = normalize(theta)

  override def divergence(x: Vector, y: Vector): Double = {
    val xNorm = normalize(x)
    val yNorm = normalize(y)
    val xArr  = xNorm.toArray
    val yArr  = yNorm.toArray

    var dot = 0.0
    var i   = 0
    while (i < xArr.length) {
      dot += xArr(i) * yArr(i)
      i += 1
    }

    1.0 - math.max(-1.0, math.min(1.0, dot))
  }

  override def validate(x: Vector): Boolean = {
    val arr  = x.toArray
    var norm = 0.0
    var i    = 0
    while (i < arr.length) {
      val v = arr(i)
      if (!java.lang.Double.isFinite(v)) return false
      norm += v * v
      i += 1
    }
    norm > 1e-20
  }

  override def name: String = "Spherical"
}

/** Factory for standard Bregman functions.
  */
object BregmanFunctions {
  val squaredEuclidean: BregmanFunction                        = new SquaredEuclideanFunction()
  def kl(smoothing: Double = 1e-10): BregmanFunction           = new KLFunction(smoothing)
  def itakuraSaito(smoothing: Double = 1e-10): BregmanFunction = new ItakuraSaitoFunction(
    smoothing
  )
  def generalizedI(smoothing: Double = 1e-10): BregmanFunction = new GeneralizedIFunction(
    smoothing
  )
  def logisticLoss(smoothing: Double = 1e-10): BregmanFunction = new LogisticLossFunction(
    smoothing
  )
  val l1: BregmanFunction                                      = new L1Function()
  val spherical: BregmanFunction                               = new SphericalFunction()

  /** Get a BregmanFunction by name.
    *
    * @param name
    *   divergence name (squaredEuclidean, kl, itakuraSaito, generalizedI, logistic, l1, spherical,
    *   cosine)
    * @param smoothing
    *   smoothing parameter for KL, IS, GenI, Logistic
    * @return
    *   corresponding BregmanFunction
    */
  def apply(name: String, smoothing: Double = 1e-10): BregmanFunction = name.toLowerCase match {
    case "squaredeuclidean" | "se" | "euclidean" => squaredEuclidean
    case "kl" | "kullbackleibler"                => kl(smoothing)
    case "itakurasaito" | "is"                   => itakuraSaito(smoothing)
    case "generalizedi" | "geni"                 => generalizedI(smoothing)
    case "logistic" | "logisticloss"             => logisticLoss(smoothing)
    case "l1" | "manhattan"                      => l1
    case "spherical" | "cosine"                  => spherical
    case _                                       =>
      throw new IllegalArgumentException(s"Unknown divergence: $name")
  }
}
