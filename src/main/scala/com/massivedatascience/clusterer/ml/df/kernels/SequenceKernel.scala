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

package com.massivedatascience.clusterer.ml.df.kernels

import org.apache.spark.ml.linalg.{ Vector, Vectors }

/** Kernel interface for sequence/time-series similarity.
  *
  * Extends MercerKernel with sequence-specific operations like alignment and barycenter
  * computation. Sequences are represented as Vectors where each element is a time point.
  *
  * ==Supported Kernels==
  *
  *   - '''DTW''' (Dynamic Time Warping) - Elastic alignment, handles time shifts/warping
  *   - '''SoftDTW''' - Differentiable DTW variant, enables gradient-based optimization
  *   - '''Derivative DTW''' - Shape-based, invariant to amplitude/offset
  *   - '''GAK''' (Global Alignment Kernel) - Positive-definite DTW-based kernel
  *
  * @see
  *   [[DTWKernel]] for standard DTW
  * @see
  *   [[SoftDTWKernel]] for differentiable version
  */
trait SequenceKernel extends MercerKernel {

  /** Compute alignment path between two sequences.
    *
    * @param x
    *   first sequence
    * @param y
    *   second sequence
    * @return
    *   sequence of (i, j) index pairs representing the optimal alignment
    */
  def alignmentPath(x: Vector, y: Vector): Seq[(Int, Int)]

  /** Compute barycenter (average) of multiple sequences.
    *
    * @param sequences
    *   sequences to average
    * @param weights
    *   optional weights for each sequence
    * @param maxIter
    *   maximum iterations for iterative methods
    * @return
    *   barycenter sequence
    */
  def barycenter(
      sequences: Array[Vector],
      weights: Option[Array[Double]] = None,
      maxIter: Int = 10
  ): Vector

  /** Supported distance metric for point-wise comparisons. */
  def pointwiseMetric: String = "euclidean"
}

/** Dynamic Time Warping (DTW) kernel for elastic sequence alignment.
  *
  * DTW allows non-linear alignment between sequences, handling time shifts, stretching, and
  * compression. This is essential for time-series that may be locally shifted or scaled.
  *
  * ==Algorithm==
  *
  * DTW finds the optimal alignment between sequences x[1..n] and y[1..m] by minimizing:
  * {{{
  * DTW(x, y) = min over all warping paths W: Σ d(x[w_i], y[w_j])
  * }}}
  *
  * Computed via dynamic programming in O(nm) time:
  * {{{
  * D[i,j] = d(x[i], y[j]) + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
  * }}}
  *
  * ==Parameters==
  *
  *   - '''window''' - Sakoe-Chiba band width (None = global, Some(w) = local constraint)
  *   - '''metric''' - Point-wise distance: "euclidean" (default), "squared", "manhattan"
  *
  * ==Example==
  *
  * {{{
  * val dtw = new DTWKernel(window = Some(10))
  * val dist = dtw.distance(seq1, seq2)
  * val path = dtw.alignmentPath(seq1, seq2)
  * }}}
  *
  * @param window
  *   Sakoe-Chiba band width constraint (None for unconstrained)
  * @param metric
  *   point-wise distance metric
  */
class DTWKernel(
    val window: Option[Int] = None,
    val metric: String = "euclidean"
) extends SequenceKernel {

  override def name: String = window match {
    case Some(w) => s"DTW(window=$w, metric=$metric)"
    case None    => s"DTW(metric=$metric)"
  }

  override def pointwiseMetric: String = metric

  /** Compute DTW distance between two sequences. */
  def distance(x: Vector, y: Vector): Double = {
    val (dtw, _) = computeDTWMatrix(x, y)
    dtw(x.size - 1)(y.size - 1)
  }

  override def apply(x: Vector, y: Vector): Double = {
    // Convert distance to similarity: k(x,y) = exp(-distance)
    math.exp(-distance(x, y))
  }

  override def squaredDistance(x: Vector, y: Vector): Double = {
    val d = distance(x, y)
    d * d
  }

  override def alignmentPath(x: Vector, y: Vector): Seq[(Int, Int)] = {
    val (_, backtrack) = computeDTWMatrix(x, y)
    extractPath(backtrack, x.size - 1, y.size - 1)
  }

  override def barycenter(
      sequences: Array[Vector],
      weights: Option[Array[Double]],
      maxIter: Int
  ): Vector = {
    DTWBarycenter.dba(sequences, weights, maxIter, this)
  }

  /** Compute DTW cost matrix and backtrack pointers. */
  private[kernels] def computeDTWMatrix(
      x: Vector,
      y: Vector
  ): (Array[Array[Double]], Array[Array[Int]]) = {
    val n = x.size
    val m = y.size

    val dtw       = Array.fill(n)(Array.fill(m)(Double.PositiveInfinity))
    val backtrack = Array.fill(n)(Array.fill(m)(-1)) // 0=diag, 1=left, 2=up

    // Initialize
    dtw(0)(0) = pointDistance(x(0), y(0))

    // First column
    for (i <- 1 until n) {
      if (inWindow(i, 0, n, m)) {
        dtw(i)(0) = dtw(i - 1)(0) + pointDistance(x(i), y(0))
        backtrack(i)(0) = 2 // up
      }
    }

    // First row
    for (j <- 1 until m) {
      if (inWindow(0, j, n, m)) {
        dtw(0)(j) = dtw(0)(j - 1) + pointDistance(x(0), y(j))
        backtrack(0)(j) = 1 // left
      }
    }

    // Fill matrix
    for (i <- 1 until n; j <- 1 until m) {
      if (inWindow(i, j, n, m)) {
        val cost       = pointDistance(x(i), y(j))
        val candidates = Array(
          dtw(i - 1)(j - 1), // diagonal
          dtw(i)(j - 1),     // left
          dtw(i - 1)(j)      // up
        )
        val minIdx     = candidates.zipWithIndex.minBy(_._1)._2
        dtw(i)(j) = cost + candidates(minIdx)
        backtrack(i)(j) = minIdx
      }
    }

    (dtw, backtrack)
  }

  /** Check if (i,j) is within Sakoe-Chiba window. */
  private def inWindow(i: Int, j: Int, n: Int, m: Int): Boolean = window match {
    case None    => true
    case Some(w) =>
      val diagI = (i.toDouble / n * m).toInt
      math.abs(j - diagI) <= w
  }

  /** Point-wise distance based on configured metric. */
  private[kernels] def pointDistance(a: Double, b: Double): Double = metric match {
    case "euclidean" => math.abs(a - b)
    case "squared"   => (a - b) * (a - b)
    case "manhattan" => math.abs(a - b)
    case _           => math.abs(a - b)
  }

  /** Extract alignment path from backtrack matrix. */
  private def extractPath(backtrack: Array[Array[Int]], i: Int, j: Int): Seq[(Int, Int)] = {
    val path = scala.collection.mutable.ArrayBuffer[(Int, Int)]()
    var ci   = i
    var cj   = j

    while (ci >= 0 && cj >= 0) {
      path.prepend((ci, cj))
      if (ci == 0 && cj == 0) {
        ci = -1; cj = -1
      } else if (ci == 0) {
        cj -= 1
      } else if (cj == 0) {
        ci -= 1
      } else {
        backtrack(ci)(cj) match {
          case 0 => ci -= 1; cj -= 1
          case 1 => cj -= 1
          case 2 => ci -= 1
          case _ => ci -= 1; cj -= 1
        }
      }
    }

    path.toSeq
  }
}

/** Soft-DTW kernel with differentiable minimum.
  *
  * Replaces the hard min in DTW with soft-min: softmin(a,b,c) = -γ log(exp(-a/γ) + exp(-b/γ) +
  * exp(-c/γ))
  *
  * This makes the distance function differentiable, enabling gradient-based optimization for
  * learning and averaging.
  *
  * ==Properties==
  *
  *   - As γ → 0, converges to standard DTW
  *   - Larger γ gives smoother distance surface
  *   - Enables efficient barycenter computation
  *
  * @param gamma
  *   smoothing parameter (γ > 0)
  * @param window
  *   optional Sakoe-Chiba band width
  */
class SoftDTWKernel(
    val gamma: Double = 1.0,
    val window: Option[Int] = None
) extends SequenceKernel {
  require(gamma > 0, s"gamma must be positive, got $gamma")

  override def name: String = s"SoftDTW(γ=$gamma)"

  /** Soft minimum: -γ log(Σ exp(-x_i/γ)) */
  private def softMin(values: Double*): Double = {
    val filtered = values.filter(_.isFinite)
    if (filtered.isEmpty) Double.PositiveInfinity
    else {
      val maxVal = filtered.max
      -gamma * math.log(filtered.map(v => math.exp(-(v - maxVal) / gamma)).sum) + maxVal
    }
  }

  /** Compute Soft-DTW distance. */
  def distance(x: Vector, y: Vector): Double = {
    val n   = x.size
    val m   = y.size
    val dtw = Array.fill(n)(Array.fill(m)(Double.PositiveInfinity))

    // Point-wise squared Euclidean distance
    def d(i: Int, j: Int): Double = {
      val diff = x(i) - y(j)
      diff * diff
    }

    dtw(0)(0) = d(0, 0)

    for (i <- 1 until n) dtw(i)(0) = dtw(i - 1)(0) + d(i, 0)
    for (j <- 1 until m) dtw(0)(j) = dtw(0)(j - 1) + d(0, j)

    for (i <- 1 until n; j <- 1 until m) {
      if (inWindow(i, j, n, m)) {
        dtw(i)(j) = d(i, j) + softMin(dtw(i - 1)(j - 1), dtw(i - 1)(j), dtw(i)(j - 1))
      }
    }

    dtw(n - 1)(m - 1)
  }

  private def inWindow(i: Int, j: Int, n: Int, m: Int): Boolean = window match {
    case None    => true
    case Some(w) =>
      val diagI = (i.toDouble / n * m).toInt
      math.abs(j - diagI) <= w
  }

  override def apply(x: Vector, y: Vector): Double = math.exp(-distance(x, y) / gamma)

  override def squaredDistance(x: Vector, y: Vector): Double = {
    val d = distance(x, y)
    d * d
  }

  override def alignmentPath(x: Vector, y: Vector): Seq[(Int, Int)] = {
    // For soft-DTW, use standard DTW path as approximation
    new DTWKernel(window).alignmentPath(x, y)
  }

  override def barycenter(
      sequences: Array[Vector],
      weights: Option[Array[Double]],
      maxIter: Int
  ): Vector = {
    // Use soft-DTW specific barycenter (gradient descent)
    DTWBarycenter.softDBA(sequences, weights, maxIter, gamma)
  }
}

/** Global Alignment Kernel (GAK) - positive-definite kernel based on DTW.
  *
  * Unlike raw DTW distance (which isn't a valid kernel), GAK is provably positive semi-definite:
  * {{{
  * GAK(x, y) = Σ over all alignments A: exp(-Σ_{(i,j)∈A} d(x_i, y_j) / σ²)
  * }}}
  *
  * @param sigma
  *   bandwidth parameter
  * @param triangular
  *   if true, use triangular kernel instead of Gaussian
  */
class GAKKernel(val sigma: Double = 1.0, val triangular: Boolean = false) extends SequenceKernel {
  require(sigma > 0, s"sigma must be positive, got $sigma")

  override def name: String = s"GAK(σ=$sigma)"

  override def apply(x: Vector, y: Vector): Double = {
    val n = x.size
    val m = y.size

    // Compute local kernel values
    def localKernel(xi: Double, yj: Double): Double = {
      val diff = xi - yj
      if (triangular) {
        math.max(0, 1 - math.abs(diff) / sigma)
      } else {
        math.exp(-diff * diff / (2 * sigma * sigma))
      }
    }

    // DP to sum over all alignments
    val gak = Array.fill(n + 1)(Array.fill(m + 1)(0.0))
    gak(0)(0) = 1.0

    for (i <- 1 to n; j <- 1 to m) {
      val k = localKernel(x(i - 1), y(j - 1))
      gak(i)(j) = (gak(i - 1)(j - 1) + gak(i - 1)(j) + gak(i)(j - 1)) * k
    }

    gak(n)(m)
  }

  override def squaredDistance(x: Vector, y: Vector): Double = {
    // d²(x,y) = k(x,x) - 2k(x,y) + k(y,y)
    apply(x, x) - 2 * apply(x, y) + apply(y, y)
  }

  override def alignmentPath(x: Vector, y: Vector): Seq[(Int, Int)] = {
    // Use DTW path as approximation
    new DTWKernel().alignmentPath(x, y)
  }

  override def barycenter(
      sequences: Array[Vector],
      weights: Option[Array[Double]],
      maxIter: Int
  ): Vector = {
    DTWBarycenter.dba(sequences, weights, maxIter, new DTWKernel())
  }
}

/** Derivative DTW - shape-based distance invariant to amplitude and offset.
  *
  * Instead of aligning raw values, aligns first derivatives (slopes):
  * {{{
  * d'(x) = [x[1]-x[0], x[2]-x[1], ..., x[n-1]-x[n-2]]
  * }}}
  *
  * This makes the distance invariant to:
  *   - Vertical offset (adding constant)
  *   - Amplitude scaling (multiplying by constant)
  *
  * @param window
  *   optional Sakoe-Chiba band
  */
class DerivativeDTWKernel(val window: Option[Int] = None) extends SequenceKernel {

  private val baseDTW = new DTWKernel(window)

  override def name: String = s"DerivativeDTW"

  /** Compute first derivative of sequence. */
  private def derivative(x: Vector): Vector = {
    if (x.size <= 1) Vectors.dense(Array(0.0))
    else {
      val arr = x.toArray
      val d   = new Array[Double](arr.length - 1)
      for (i <- 0 until d.length) {
        d(i) = arr(i + 1) - arr(i)
      }
      Vectors.dense(d)
    }
  }

  override def apply(x: Vector, y: Vector): Double = {
    baseDTW(derivative(x), derivative(y))
  }

  override def squaredDistance(x: Vector, y: Vector): Double = {
    baseDTW.squaredDistance(derivative(x), derivative(y))
  }

  override def alignmentPath(x: Vector, y: Vector): Seq[(Int, Int)] = {
    baseDTW.alignmentPath(derivative(x), derivative(y))
  }

  override def barycenter(
      sequences: Array[Vector],
      weights: Option[Array[Double]],
      maxIter: Int
  ): Vector = {
    // Compute barycenter of derivatives, then integrate
    val derivs          = sequences.map(derivative)
    val derivBarycenter = baseDTW.barycenter(derivs, weights, maxIter)

    // Integrate: cumulative sum starting from 0
    val arr        = derivBarycenter.toArray
    val integrated = new Array[Double](arr.length + 1)
    integrated(0) = 0.0
    for (i <- arr.indices) {
      integrated(i + 1) = integrated(i) + arr(i)
    }

    Vectors.dense(integrated)
  }
}

/** DTW Barycenter Averaging (DBA) algorithm.
  *
  * Computes the centroid of multiple sequences under DTW distance using iterative refinement:
  *
  *   1. Initialize barycenter (e.g., medoid or mean) 2. Align all sequences to current barycenter
  *      3. Update barycenter as weighted average of aligned points 4. Repeat until convergence
  *
  * @see
  *   Petitjean et al. "A global averaging method for DTW" (2011)
  */
object DTWBarycenter {

  /** Compute DBA barycenter using iterative refinement.
    *
    * @param sequences
    *   input sequences
    * @param weights
    *   optional weights (uniform if None)
    * @param maxIter
    *   maximum iterations
    * @param kernel
    *   DTW kernel for alignment
    * @return
    *   barycenter sequence
    */
  def dba(
      sequences: Array[Vector],
      weights: Option[Array[Double]],
      maxIter: Int,
      kernel: DTWKernel
  ): Vector = {
    if (sequences.isEmpty) return Vectors.dense(Array.empty[Double])
    if (sequences.length == 1) return sequences(0)

    val ws = weights.getOrElse(Array.fill(sequences.length)(1.0 / sequences.length))
    require(ws.length == sequences.length, "Weights must match sequences")

    // Initialize with medoid (sequence minimizing sum of distances)
    var barycenter = initMedoid(sequences, kernel)
    val targetLen  = barycenter.size

    for (_ <- 0 until maxIter) {
      val newValues = Array.fill(targetLen)(0.0)
      val counts    = Array.fill(targetLen)(0.0)

      // Align each sequence to barycenter and accumulate
      for (i <- sequences.indices) {
        val seq  = sequences(i)
        val w    = ws(i)
        val path = kernel.alignmentPath(barycenter, seq)

        for ((bi, si) <- path) {
          newValues(bi) += w * seq(si)
          counts(bi) += w
        }
      }

      // Update barycenter
      val newBarycenter = new Array[Double](targetLen)
      for (j <- 0 until targetLen) {
        newBarycenter(j) = if (counts(j) > 0) newValues(j) / counts(j) else barycenter(j)
      }

      barycenter = Vectors.dense(newBarycenter)
    }

    barycenter
  }

  /** Soft-DBA using gradient descent for soft-DTW.
    *
    * @param sequences
    *   input sequences
    * @param weights
    *   optional weights
    * @param maxIter
    *   maximum iterations
    * @param gamma
    *   soft-DTW smoothing parameter
    * @return
    *   barycenter sequence
    */
  def softDBA(
      sequences: Array[Vector],
      weights: Option[Array[Double]],
      maxIter: Int,
      gamma: Double
  ): Vector = {
    if (sequences.isEmpty) return Vectors.dense(Array.empty[Double])
    if (sequences.length == 1) return sequences(0)

    val ws  = weights.getOrElse(Array.fill(sequences.length)(1.0 / sequences.length))
    val dtw = new DTWKernel()

    // Initialize with DBA result
    var barycenter = dba(sequences, weights, maxIter / 2, dtw)
    val targetLen  = barycenter.size
    val lr         = 0.1 / sequences.length // Learning rate

    // Gradient descent refinement
    for (_ <- 0 until maxIter) {
      val gradient = Array.fill(targetLen)(0.0)

      // Accumulate gradients
      for (i <- sequences.indices) {
        val seq = sequences(i)
        val w   = ws(i)
        val g   = computeSoftDTWGradient(barycenter, seq, gamma)
        for (j <- 0 until targetLen) {
          gradient(j) += w * g(j)
        }
      }

      // Update
      val arr = barycenter.toArray
      for (j <- 0 until targetLen) {
        arr(j) -= lr * gradient(j)
      }
      barycenter = Vectors.dense(arr)
    }

    barycenter
  }

  /** Compute gradient of soft-DTW w.r.t. first sequence. */
  private def computeSoftDTWGradient(x: Vector, y: Vector, gamma: Double): Array[Double] = {
    val n = x.size
    val m = y.size

    // Forward pass
    val dtw                                              = Array.fill(n)(Array.fill(m)(Double.PositiveInfinity))
    def d(i: Int, j: Int): Double                        = { val diff = x(i) - y(j); diff * diff }
    def softMin(a: Double, b: Double, c: Double): Double = {
      val vals = Array(a, b, c).filter(_.isFinite)
      if (vals.isEmpty) Double.PositiveInfinity
      else {
        val maxV = vals.max
        -gamma * math.log(vals.map(v => math.exp(-(v - maxV) / gamma)).sum) + maxV
      }
    }

    dtw(0)(0) = d(0, 0)
    for (i <- 1 until n) dtw(i)(0) = dtw(i - 1)(0) + d(i, 0)
    for (j <- 1 until m) dtw(0)(j) = dtw(0)(j - 1) + d(0, j)
    for (i <- 1 until n; j <- 1 until m) {
      dtw(i)(j) = d(i, j) + softMin(dtw(i - 1)(j - 1), dtw(i - 1)(j), dtw(i)(j - 1))
    }

    // Backward pass for gradient
    val grad = Array.fill(n)(0.0)
    val e    = Array.fill(n)(Array.fill(m)(0.0))
    e(n - 1)(m - 1) = 1.0

    for (i <- (0 until n).reverse; j <- (0 until m).reverse) {
      val softMinVal =
        if (i == 0 && j == 0) 0.0
        else if (i == 0) dtw(0)(j - 1)
        else if (j == 0) dtw(i - 1)(0)
        else softMin(dtw(i - 1)(j - 1), dtw(i - 1)(j), dtw(i)(j - 1))

      if (i > 0) {
        val w = math.exp(-(dtw(i - 1)(j) - softMinVal) / gamma)
        e(i - 1)(j) += e(i)(j) * w
      }
      if (j > 0) {
        val w = math.exp(-(dtw(i)(j - 1) - softMinVal) / gamma)
        e(i)(j - 1) += e(i)(j) * w
      }
      if (i > 0 && j > 0) {
        val w = math.exp(-(dtw(i - 1)(j - 1) - softMinVal) / gamma)
        e(i - 1)(j - 1) += e(i)(j) * w
      }

      // Gradient contribution: d(cost)/d(x[i]) = 2(x[i] - y[j]) * e[i,j]
      grad(i) += 2 * (x(i) - y(j)) * e(i)(j)
    }

    grad
  }

  /** Find medoid (sequence with minimum total distance to others). */
  private def initMedoid(sequences: Array[Vector], kernel: DTWKernel): Vector = {
    if (sequences.length <= 2) return sequences(0)

    val distances = sequences.indices.map { i =>
      sequences.indices.filter(_ != i).map(j => kernel.distance(sequences(i), sequences(j))).sum
    }

    sequences(distances.zipWithIndex.minBy(_._1)._2)
  }
}

/** Factory for creating sequence kernels.
  */
object SequenceKernel {

  /** Create a sequence kernel by name.
    *
    * @param kernelType
    *   "dtw" | "softdtw" | "gak" | "derivative"
    * @param window
    *   optional Sakoe-Chiba band width
    * @param gamma
    *   smoothing parameter (for softdtw)
    * @param sigma
    *   bandwidth (for gak)
    */
  def create(
      kernelType: String,
      window: Option[Int] = None,
      gamma: Double = 1.0,
      sigma: Double = 1.0
  ): SequenceKernel = kernelType.toLowerCase match {
    case "dtw"                  => new DTWKernel(window)
    case "softdtw" | "soft_dtw" => new SoftDTWKernel(gamma, window)
    case "gak"                  => new GAKKernel(sigma)
    case "derivative" | "ddtw"  => new DerivativeDTWKernel(window)
    case other                  =>
      throw new IllegalArgumentException(
        s"Unknown sequence kernel: $other. Supported: dtw, softdtw, gak, derivative"
      )
  }

  /** Supported kernel types. */
  val supportedKernels: Seq[String] = Seq("dtw", "softdtw", "gak", "derivative")
}
