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

package com.massivedatascience.clusterer.ml.df

/** Information-theoretic utilities for clustering.
  *
  * Provides entropy, mutual information, and KL divergence computations for discrete probability
  * distributions. These are used by the Information Bottleneck algorithm.
  *
  * ==Definitions==
  *
  *   - '''Entropy:''' H(X) = -Σ p(x) log p(x)
  *   - '''Conditional Entropy:''' H(X|Y) = -Σ p(x,y) log p(x|y)
  *   - '''Mutual Information:''' I(X;Y) = H(X) - H(X|Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
  *   - '''KL Divergence:''' D_KL(P||Q) = Σ p(x) log(p(x) / q(x))
  *
  * All logarithms are natural (base e), giving results in nats. For bits, divide by log(2).
  */
object MutualInformation {

  private val LOG2 = math.log(2.0)

  /** Compute Shannon entropy H(p) = -Σ p_i log(p_i).
    *
    * @param p
    *   probability distribution (must sum to 1)
    * @param smoothing
    *   small value to add for numerical stability
    * @return
    *   entropy in nats (natural units)
    */
  def entropy(p: Array[Double], smoothing: Double = 1e-10): Double = {
    var sum = 0.0
    var i   = 0
    while (i < p.length) {
      val pi = p(i) + smoothing
      if (pi > smoothing) {
        sum -= pi * math.log(pi)
      }
      i += 1
    }
    sum
  }

  /** Compute entropy in bits. */
  def entropyBits(p: Array[Double], smoothing: Double = 1e-10): Double =
    entropy(p, smoothing) / LOG2

  /** Compute KL divergence D_KL(P||Q) = Σ p_i log(p_i / q_i).
    *
    * @param p
    *   first distribution (typically the "true" distribution)
    * @param q
    *   second distribution (typically the "model" distribution)
    * @param smoothing
    *   small value to add for numerical stability
    * @return
    *   KL divergence (non-negative, 0 iff p == q)
    */
  def klDivergence(p: Array[Double], q: Array[Double], smoothing: Double = 1e-10): Double = {
    require(p.length == q.length, "Distributions must have same length")
    var sum = 0.0
    var i   = 0
    while (i < p.length) {
      val pi = p(i) + smoothing
      val qi = q(i) + smoothing
      if (pi > smoothing) {
        sum += pi * math.log(pi / qi)
      }
      i += 1
    }
    sum
  }

  /** Compute Jensen-Shannon divergence (symmetric, bounded).
    *
    * JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M) where M = 0.5 * (P + Q)
    *
    * @return
    *   JS divergence in [0, log(2)]
    */
  def jsDivergence(p: Array[Double], q: Array[Double], smoothing: Double = 1e-10): Double = {
    require(p.length == q.length, "Distributions must have same length")
    val m = new Array[Double](p.length)
    var i = 0
    while (i < p.length) {
      m(i) = 0.5 * (p(i) + q(i))
      i += 1
    }
    0.5 * klDivergence(p, m, smoothing) + 0.5 * klDivergence(q, m, smoothing)
  }

  /** Compute mutual information I(X;Y) from joint probability matrix.
    *
    * I(X;Y) = Σ_x,y p(x,y) log(p(x,y) / (p(x) p(y)))
    *
    * @param joint
    *   joint probability matrix p(x,y), shape [|X|, |Y|]
    * @param smoothing
    *   small value for numerical stability
    * @return
    *   mutual information in nats
    */
  def mutualInformation(joint: Array[Array[Double]], smoothing: Double = 1e-10): Double = {
    val numX = joint.length
    val numY = joint(0).length

    // Compute marginals
    val pX = new Array[Double](numX)
    val pY = new Array[Double](numY)

    var i = 0
    while (i < numX) {
      var j = 0
      while (j < numY) {
        pX(i) += joint(i)(j)
        pY(j) += joint(i)(j)
        j += 1
      }
      i += 1
    }

    // Compute MI
    var mi = 0.0
    i = 0
    while (i < numX) {
      var j = 0
      while (j < numY) {
        val pxy = joint(i)(j) + smoothing
        val px  = pX(i) + smoothing
        val py  = pY(j) + smoothing
        if (pxy > smoothing) {
          mi += pxy * math.log(pxy / (px * py))
        }
        j += 1
      }
      i += 1
    }
    math.max(0.0, mi) // Ensure non-negative due to numerical errors
  }

  /** Compute conditional entropy H(X|Y) from joint probability matrix.
    *
    * H(X|Y) = -Σ_x,y p(x,y) log(p(x|y))
    *
    * @param joint
    *   joint probability matrix p(x,y)
    * @param smoothing
    *   small value for numerical stability
    * @return
    *   conditional entropy in nats
    */
  def conditionalEntropy(joint: Array[Array[Double]], smoothing: Double = 1e-10): Double = {
    val numX = joint.length
    val numY = joint(0).length

    // Compute p(Y)
    val pY = new Array[Double](numY)
    var i  = 0
    while (i < numX) {
      var j = 0
      while (j < numY) {
        pY(j) += joint(i)(j)
        j += 1
      }
      i += 1
    }

    // Compute H(X|Y)
    var hXgivenY = 0.0
    i = 0
    while (i < numX) {
      var j = 0
      while (j < numY) {
        val pxy    = joint(i)(j) + smoothing
        val py     = pY(j) + smoothing
        val pXgivy = pxy / py
        if (pxy > smoothing) {
          hXgivenY -= pxy * math.log(pXgivy)
        }
        j += 1
      }
      i += 1
    }
    hXgivenY
  }

  /** Normalize an array to sum to 1. */
  def normalize(arr: Array[Double]): Array[Double] = {
    val sum = arr.sum
    if (sum > 0) arr.map(_ / sum) else arr.map(_ => 1.0 / arr.length)
  }

  /** Normalize a 2D array (joint distribution) to sum to 1. */
  def normalizeJoint(joint: Array[Array[Double]]): Array[Array[Double]] = {
    var sum = 0.0
    var i   = 0
    while (i < joint.length) {
      var j = 0
      while (j < joint(i).length) {
        sum += joint(i)(j)
        j += 1
      }
      i += 1
    }
    if (sum > 0) {
      joint.map(_.map(_ / sum))
    } else {
      val uniform = 1.0 / (joint.length * joint(0).length)
      joint.map(_.map(_ => uniform))
    }
  }

  /** Estimate joint distribution from data using histogram binning.
    *
    * @param xValues
    *   values for X variable (discrete labels 0..numX-1)
    * @param yValues
    *   values for Y variable (discrete labels 0..numY-1)
    * @param numX
    *   number of X categories
    * @param numY
    *   number of Y categories
    * @param weights
    *   optional sample weights
    * @return
    *   joint probability matrix p(x,y)
    */
  def estimateJoint(
      xValues: Array[Int],
      yValues: Array[Int],
      numX: Int,
      numY: Int,
      weights: Option[Array[Double]] = None
  ): Array[Array[Double]] = {
    require(xValues.length == yValues.length, "X and Y must have same length")
    val joint = Array.fill(numX, numY)(0.0)

    var i = 0
    while (i < xValues.length) {
      val w = weights.map(_(i)).getOrElse(1.0)
      joint(xValues(i))(yValues(i)) += w
      i += 1
    }

    normalizeJoint(joint)
  }

  /** Compute row-wise conditional distribution p(y|x) from joint p(x,y).
    *
    * @param joint
    *   joint probability matrix
    * @param smoothing
    *   small value for numerical stability
    * @return
    *   conditional probability matrix p(y|x)
    */
  def conditionalYgivenX(
      joint: Array[Array[Double]],
      smoothing: Double = 1e-10
  ): Array[Array[Double]] = {
    val numX   = joint.length
    val numY   = joint(0).length
    val result = Array.fill(numX, numY)(0.0)

    var i = 0
    while (i < numX) {
      var rowSum = 0.0
      var j      = 0
      while (j < numY) {
        rowSum += joint(i)(j) + smoothing
        j += 1
      }
      j = 0
      while (j < numY) {
        result(i)(j) = (joint(i)(j) + smoothing) / rowSum
        j += 1
      }
      i += 1
    }
    result
  }

  /** Compute column-wise conditional distribution p(x|y) from joint p(x,y). */
  def conditionalXgivenY(
      joint: Array[Array[Double]],
      smoothing: Double = 1e-10
  ): Array[Array[Double]] = {
    val numX   = joint.length
    val numY   = joint(0).length
    val result = Array.fill(numX, numY)(0.0)

    // Compute column sums (p(Y))
    val colSums = new Array[Double](numY)
    var i       = 0
    while (i < numX) {
      var j = 0
      while (j < numY) {
        colSums(j) += joint(i)(j) + smoothing
        j += 1
      }
      i += 1
    }

    i = 0
    while (i < numX) {
      var j = 0
      while (j < numY) {
        result(i)(j) = (joint(i)(j) + smoothing) / colSums(j)
        j += 1
      }
      i += 1
    }
    result
  }
}
