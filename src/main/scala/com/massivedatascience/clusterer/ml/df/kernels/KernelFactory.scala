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

/** Unified factory for creating Bregman kernels.
  *
  * This factory provides a single entry point for kernel creation with support for:
  *   - Dense kernels (standard implementation)
  *   - Sparse-optimized kernels (for high-dimensional sparse data)
  *   - Auto-selection based on data characteristics
  *
  * ==Supported Divergences==
  *
  * | Name             | Aliases         | Sparse Support | Domain  | Use Case                  |
  * |:-----------------|:----------------|:---------------|:--------|:--------------------------|
  * | squaredEuclidean | se, euclidean   | Yes            | R^n     | General clustering        |
  * | kl               | kullbackLeibler | Yes            | R+^n    | Probability distributions |
  * | itakuraSaito     | is              | No             | R+^n    | Audio/spectrum analysis   |
  * | generalizedI     | genI            | No             | R+^n    | Count data                |
  * | logistic         | -               | No             | [0,1]^n | Bounded probabilities     |
  * | l1               | manhattan       | Yes            | R^n     | Robust clustering         |
  * | spherical        | cosine          | Yes            | R^n     | Text/documents            |
  *
  * ==Example Usage==
  *
  * {{{
  * // Standard dense kernel
  * val seKernel = KernelFactory.create("squaredEuclidean")
  *
  * // Sparse-optimized kernel for text data
  * val klKernel = KernelFactory.create("kl", sparse = true)
  *
  * // Auto-select based on sparsity
  * val autoKernel = KernelFactory.forSparsity("squaredEuclidean", sparsityRatio = 0.1)
  * }}}
  *
  * @see
  *   [[BregmanKernel]] for the kernel interface
  * @see
  *   [[SparseBregmanKernel]] for sparse-optimized implementations
  */
object KernelFactory {

  /** Canonical divergence names. */
  object Divergence {
    val SquaredEuclidean: String = "squaredEuclidean"
    val KL: String               = "kl"
    val ItakuraSaito: String     = "itakuraSaito"
    val GeneralizedI: String     = "generalizedI"
    val Logistic: String         = "logistic"
    val L1: String               = "l1"
    val Spherical: String        = "spherical"

    /** All supported divergence names (canonical form). */
    val all: Seq[String] = Seq(
      SquaredEuclidean,
      KL,
      ItakuraSaito,
      GeneralizedI,
      Logistic,
      L1,
      Spherical
    )
  }

  /** Divergences with sparse-optimized implementations (lowercase for comparison). */
  val sparseSupported: Set[String] = Set(
    "squaredeuclidean",
    "se",
    "euclidean",
    "kl",
    "kullbackleibler",
    "l1",
    "manhattan",
    "spherical",
    "cosine"
  )

  /** Create a Bregman kernel for the specified divergence.
    *
    * @param divergence
    *   divergence name (case-insensitive)
    * @param sparse
    *   if true, use sparse-optimized implementation when available
    * @param smoothing
    *   smoothing parameter for divergences with domain constraints (KL, IS, etc.)
    * @return
    *   configured BregmanKernel instance
    * @throws IllegalArgumentException
    *   if divergence name is unknown
    */
  def create(
      divergence: String,
      sparse: Boolean = false,
      smoothing: Double = 1e-10
  ): BregmanKernel = {
    val normalized = divergence.toLowerCase.trim
    if (sparse && supportsSparse(normalized)) {
      createSparse(normalized, smoothing)
    } else {
      createDense(normalized, smoothing)
    }
  }

  /** Create a kernel with auto-selection based on data sparsity.
    *
    * Selects sparse implementation when sparsity ratio is below threshold and sparse implementation
    * is available.
    *
    * @param divergence
    *   divergence name
    * @param sparsityRatio
    *   fraction of non-zero elements (0.0 = all zeros, 1.0 = dense)
    * @param smoothing
    *   smoothing parameter
    * @param sparseThreshold
    *   use sparse when sparsityRatio < this value (default 0.3)
    * @return
    *   kernel optimized for the data sparsity
    */
  def forSparsity(
      divergence: String,
      sparsityRatio: Double,
      smoothing: Double = 1e-10,
      sparseThreshold: Double = 0.3
  ): BregmanKernel = {
    val useSparse = sparsityRatio < sparseThreshold && supportsSparse(divergence)
    create(divergence, sparse = useSparse, smoothing = smoothing)
  }

  /** Check if sparse optimization is available for the divergence.
    *
    * @param divergence
    *   divergence name (case-insensitive)
    * @return
    *   true if sparse-optimized implementation exists
    */
  def supportsSparse(divergence: String): Boolean =
    sparseSupported.contains(divergence.toLowerCase.trim)

  /** Normalize divergence name to canonical form.
    *
    * @param divergence
    *   any valid divergence name or alias
    * @return
    *   canonical divergence name
    */
  def normalize(divergence: String): String = divergence.toLowerCase.trim match {
    case "se" | "euclidean" => Divergence.SquaredEuclidean
    case "kullbackleibler"  => Divergence.KL
    case "is"               => Divergence.ItakuraSaito
    case "geni"             => Divergence.GeneralizedI
    case "manhattan"        => Divergence.L1
    case "cosine"           => Divergence.Spherical
    case other              => other
  }

  /** Create a dense (standard) kernel implementation. */
  private def createDense(divergence: String, smoothing: Double): BregmanKernel =
    divergence match {
      case "squaredeuclidean" | "se" | "euclidean" => new SquaredEuclideanKernel()
      case "kl" | "kullbackleibler"                => new KLDivergenceKernel(smoothing)
      case "itakurasaito" | "is"                   => new ItakuraSaitoKernel(smoothing)
      case "generalizedi" | "geni"                 => new GeneralizedIDivergenceKernel(smoothing)
      case "logistic"                              => new LogisticLossKernel(smoothing)
      case "l1" | "manhattan"                      => new L1Kernel()
      case "spherical" | "cosine"                  => new SphericalKernel()
      case other                                   =>
        throw new IllegalArgumentException(
          s"Unknown divergence: '$other'. Supported: ${Divergence.all.mkString(", ")}"
        )
    }

  /** Create a sparse-optimized kernel implementation. */
  private def createSparse(divergence: String, smoothing: Double): BregmanKernel =
    divergence match {
      case "squaredeuclidean" | "se" | "euclidean" => new SparseSEKernel()
      case "kl" | "kullbackleibler"                => new SparseKLKernel(smoothing)
      case "l1" | "manhattan"                      => new SparseL1Kernel()
      case "spherical" | "cosine"                  => new SparseSphericalKernel()
      // Fall back to dense for others (no sparse optimization available)
      case other                                   => createDense(other, smoothing)
    }
}
