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

import com.massivedatascience.clusterer.ml.df.kernels.MercerKernel
import org.apache.spark.ml.linalg.{ DenseMatrix, Vector, Vectors }

/** Graph construction and Laplacian computation for spectral clustering.
  *
  * ==Graph Laplacian Types==
  *
  *   - '''Unnormalized:''' L = D - W
  *   - '''Symmetric Normalized:''' L_sym = I - D^(-1/2) W D^(-1/2) = D^(-1/2) L D^(-1/2)
  *   - '''Random Walk:''' L_rw = I - D^(-1) W = D^(-1) L
  *
  * The symmetric normalized Laplacian is most commonly used (Ng, Jordan, Weiss 2002).
  *
  * ==Affinity Construction==
  *
  *   - '''Full:''' W_ij = k(x_i, x_j) using a Mercer kernel
  *   - '''k-NN:''' W_ij = k(x_i, x_j) if j ∈ kNN(i) or i ∈ kNN(j), else 0
  *   - '''ε-neighborhood:''' W_ij = k(x_i, x_j) if ||x_i - x_j|| < ε, else 0
  *
  * @see
  *   von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and Computing.
  */
object SpectralGraph {

  /** Laplacian type enumeration. */
  object LaplacianType {
    val Unnormalized: String        = "unnormalized"
    val SymmetricNormalized: String = "symmetric"
    val RandomWalk: String          = "randomWalk"

    val all: Seq[String] = Seq(Unnormalized, SymmetricNormalized, RandomWalk)
  }

  /** Affinity type enumeration. */
  object AffinityType {
    val Full: String            = "full"
    val KNN: String             = "knn"
    val EpsNeighborhood: String = "epsilon"

    val all: Seq[String] = Seq(Full, KNN, EpsNeighborhood)
  }

  /** Build a full affinity matrix using a Mercer kernel.
    *
    * @param points
    *   data points
    * @param kernel
    *   Mercer kernel for similarity computation
    * @return
    *   n × n affinity matrix W where W_ij = k(x_i, x_j)
    */
  def buildFullAffinity(points: Array[Vector], kernel: MercerKernel): DenseMatrix = {
    kernel.gramMatrix(points)
  }

  /** Build a k-nearest neighbor affinity matrix.
    *
    * Uses mutual k-NN: W_ij > 0 iff j ∈ kNN(i) OR i ∈ kNN(j)
    *
    * @param points
    *   data points
    * @param kernel
    *   Mercer kernel for similarity computation
    * @param k
    *   number of nearest neighbors
    * @return
    *   sparse affinity matrix (stored as dense for simplicity)
    */
  def buildKNNAffinity(points: Array[Vector], kernel: MercerKernel, k: Int): DenseMatrix = {
    val n      = points.length
    val values = new Array[Double](n * n)

    // First compute all pairwise distances
    val distances = Array.ofDim[Double](n, n)
    for (i <- 0 until n) {
      for (j <- i + 1 until n) {
        val d = kernel.squaredDistance(points(i), points(j))
        distances(i)(j) = d
        distances(j)(i) = d
      }
    }

    // Find k-nearest neighbors for each point
    val knnIndices = Array.ofDim[Set[Int]](n)
    for (i <- 0 until n) {
      val neighbors = (0 until n).filter(_ != i).sortBy(j => distances(i)(j)).take(k).toSet
      knnIndices(i) = neighbors
    }

    // Build symmetric affinity matrix (mutual k-NN)
    for (i <- 0 until n) {
      for (j <- i + 1 until n) {
        if (knnIndices(i).contains(j) || knnIndices(j).contains(i)) {
          val affinity = kernel(points(i), points(j))
          values(i * n + j) = affinity
          values(j * n + i) = affinity
        }
      }
    }

    new DenseMatrix(n, n, values)
  }

  /** Build an ε-neighborhood affinity matrix.
    *
    * W_ij = k(x_i, x_j) if ||x_i - x_j||² < ε², else 0
    *
    * @param points
    *   data points
    * @param kernel
    *   Mercer kernel for similarity computation
    * @param epsilon
    *   neighborhood radius
    * @return
    *   sparse affinity matrix (stored as dense)
    */
  def buildEpsilonAffinity(
      points: Array[Vector],
      kernel: MercerKernel,
      epsilon: Double
  ): DenseMatrix = {
    val n            = points.length
    val values       = new Array[Double](n * n)
    val epsSq        = epsilon * epsilon
    val linearKernel = new kernels.LinearKernel()

    for (i <- 0 until n) {
      for (j <- i + 1 until n) {
        val sqDist = linearKernel.squaredDistance(points(i), points(j))
        if (sqDist < epsSq) {
          val affinity = kernel(points(i), points(j))
          values(i * n + j) = affinity
          values(j * n + i) = affinity
        }
      }
    }

    new DenseMatrix(n, n, values)
  }

  /** Build affinity matrix based on type.
    *
    * @param points
    *   data points
    * @param kernel
    *   Mercer kernel
    * @param affinityType
    *   "full", "knn", or "epsilon"
    * @param kNeighbors
    *   k for k-NN (default 10)
    * @param epsilon
    *   radius for ε-neighborhood
    */
  def buildAffinity(
      points: Array[Vector],
      kernel: MercerKernel,
      affinityType: String,
      kNeighbors: Int = 10,
      epsilon: Double = 1.0
  ): DenseMatrix = affinityType.toLowerCase match {
    case AffinityType.Full            => buildFullAffinity(points, kernel)
    case AffinityType.KNN             => buildKNNAffinity(points, kernel, kNeighbors)
    case AffinityType.EpsNeighborhood => buildEpsilonAffinity(points, kernel, epsilon)
    case other                        =>
      throw new IllegalArgumentException(
        s"Unknown affinity type: $other. Supported: ${AffinityType.all.mkString(", ")}"
      )
  }

  /** Compute the degree vector from affinity matrix.
    *
    * d_i = Σ_j W_ij
    *
    * @param affinity
    *   affinity matrix W
    * @return
    *   degree vector d
    */
  def computeDegrees(affinity: DenseMatrix): Array[Double] = {
    val n       = affinity.numRows
    val degrees = new Array[Double](n)
    val values  = affinity.values

    for (i <- 0 until n) {
      var sum = 0.0
      for (j <- 0 until n) {
        sum += values(i * n + j)
      }
      degrees(i) = sum
    }

    degrees
  }

  /** Compute the unnormalized graph Laplacian L = D - W.
    *
    * @param affinity
    *   affinity matrix W
    * @return
    *   unnormalized Laplacian L
    */
  def unnormalizedLaplacian(affinity: DenseMatrix): DenseMatrix = {
    val n       = affinity.numRows
    val degrees = computeDegrees(affinity)
    val values  = new Array[Double](n * n)
    val W       = affinity.values

    for (i <- 0 until n) {
      for (j <- 0 until n) {
        if (i == j) {
          values(i * n + j) = degrees(i) - W(i * n + j)
        } else {
          values(i * n + j) = -W(i * n + j)
        }
      }
    }

    new DenseMatrix(n, n, values)
  }

  /** Compute the symmetric normalized Laplacian L_sym = I - D^(-1/2) W D^(-1/2).
    *
    * This is the most commonly used Laplacian for spectral clustering (Ng, Jordan, Weiss 2002).
    *
    * @param affinity
    *   affinity matrix W
    * @return
    *   symmetric normalized Laplacian L_sym
    */
  def symmetricNormalizedLaplacian(affinity: DenseMatrix): DenseMatrix = {
    val n       = affinity.numRows
    val degrees = computeDegrees(affinity)
    val values  = new Array[Double](n * n)
    val W       = affinity.values

    // Compute D^(-1/2)
    val dInvSqrt = degrees.map { d =>
      if (d > 1e-10) 1.0 / math.sqrt(d) else 0.0
    }

    // L_sym = I - D^(-1/2) W D^(-1/2)
    for (i <- 0 until n) {
      for (j <- 0 until n) {
        val normalizedW = dInvSqrt(i) * W(i * n + j) * dInvSqrt(j)
        if (i == j) {
          values(i * n + j) = 1.0 - normalizedW
        } else {
          values(i * n + j) = -normalizedW
        }
      }
    }

    new DenseMatrix(n, n, values)
  }

  /** Compute the random walk Laplacian L_rw = I - D^(-1) W.
    *
    * @param affinity
    *   affinity matrix W
    * @return
    *   random walk Laplacian L_rw
    */
  def randomWalkLaplacian(affinity: DenseMatrix): DenseMatrix = {
    val n       = affinity.numRows
    val degrees = computeDegrees(affinity)
    val values  = new Array[Double](n * n)
    val W       = affinity.values

    // Compute D^(-1)
    val dInv = degrees.map { d =>
      if (d > 1e-10) 1.0 / d else 0.0
    }

    // L_rw = I - D^(-1) W
    for (i <- 0 until n) {
      for (j <- 0 until n) {
        val normalizedW = dInv(i) * W(i * n + j)
        if (i == j) {
          values(i * n + j) = 1.0 - normalizedW
        } else {
          values(i * n + j) = -normalizedW
        }
      }
    }

    new DenseMatrix(n, n, values)
  }

  /** Compute the Laplacian matrix based on type.
    *
    * @param affinity
    *   affinity matrix
    * @param laplacianType
    *   "unnormalized", "symmetric", or "randomWalk"
    */
  def computeLaplacian(affinity: DenseMatrix, laplacianType: String): DenseMatrix =
    laplacianType.toLowerCase match {
      case "unnormalized" => unnormalizedLaplacian(affinity)
      case "symmetric"    => symmetricNormalizedLaplacian(affinity)
      case "randomwalk"   => randomWalkLaplacian(affinity)
      case other          =>
        throw new IllegalArgumentException(
          s"Unknown Laplacian type: $other. Supported: ${LaplacianType.all.mkString(", ")}"
        )
    }

  /** Compute eigenvectors of a symmetric matrix using power iteration with deflation.
    *
    * Returns the k smallest eigenvectors of the Laplacian (corresponding to the k largest
    * eigenvectors of (I - L), which is the similarity/affinity interpretation).
    *
    * @param matrix
    *   symmetric matrix (Laplacian)
    * @param k
    *   number of eigenvectors to compute
    * @param maxIter
    *   maximum iterations per eigenvector
    * @param tol
    *   convergence tolerance
    * @param seed
    *   random seed for initialization
    * @return
    *   (eigenvalues, eigenvectors as row vectors)
    */
  def computeSmallestEigenvectors(
      matrix: DenseMatrix,
      k: Int,
      maxIter: Int = 100,
      tol: Double = 1e-6,
      seed: Long = 42L
  ): (Array[Double], Array[Array[Double]]) = {
    val n   = matrix.numRows
    val rng = new scala.util.Random(seed)

    // For Laplacian, we want smallest eigenvalues
    // Use power iteration on (maxEig * I - L) to find largest, which correspond to smallest of L
    val maxDiag = (0 until n).map(i => matrix(i, i)).max + 1.0
    val shifted = shiftMatrix(matrix, maxDiag)

    val eigenvalues   = new Array[Double](k)
    val eigenvectors  = new Array[Array[Double]](k)
    var currentMatrix = shifted

    for (i <- 0 until k) {
      val (eigenvalue, eigenvector) = powerIteration(currentMatrix, maxIter, tol, rng)
      eigenvalues(i) = maxDiag - eigenvalue // Shift back
      eigenvectors(i) = eigenvector

      // Deflate: A' = A - λ v v^T
      currentMatrix = deflate(currentMatrix, eigenvalue, eigenvector)
    }

    (eigenvalues, eigenvectors)
  }

  /** Shift matrix: A' = shift * I - A */
  private def shiftMatrix(matrix: DenseMatrix, shift: Double): DenseMatrix = {
    val n      = matrix.numRows
    val values = new Array[Double](n * n)
    val M      = matrix.values

    for (i <- 0 until n) {
      for (j <- 0 until n) {
        if (i == j) {
          values(i * n + j) = shift - M(i * n + j)
        } else {
          values(i * n + j) = -M(i * n + j)
        }
      }
    }

    new DenseMatrix(n, n, values)
  }

  /** Power iteration to find dominant eigenvector. */
  private def powerIteration(
      matrix: DenseMatrix,
      maxIter: Int,
      tol: Double,
      rng: scala.util.Random
  ): (Double, Array[Double]) = {
    val n = matrix.numRows
    val M = matrix.values

    // Initialize random vector
    var v = Array.fill(n)(rng.nextGaussian())
    v = normalize(v)

    var eigenvalue = 0.0
    var converged  = false
    var iter       = 0

    while (iter < maxIter && !converged) {
      // Multiply: w = M * v
      val w = new Array[Double](n)
      for (i <- 0 until n) {
        var sum = 0.0
        for (j <- 0 until n) {
          sum += M(i * n + j) * v(j)
        }
        w(i) = sum
      }

      // Compute eigenvalue estimate (Rayleigh quotient)
      val newEigenvalue = dot(v, w)

      // Normalize
      val wNorm = normalize(w)

      // Check convergence
      if (math.abs(newEigenvalue - eigenvalue) < tol) {
        converged = true
      }

      eigenvalue = newEigenvalue
      v = wNorm
      iter += 1
    }

    (eigenvalue, v)
  }

  /** Deflate matrix: A' = A - λ v v^T */
  private def deflate(
      matrix: DenseMatrix,
      eigenvalue: Double,
      eigenvector: Array[Double]
  ): DenseMatrix = {
    val n      = matrix.numRows
    val values = new Array[Double](n * n)
    val M      = matrix.values

    for (i <- 0 until n) {
      for (j <- 0 until n) {
        values(i * n + j) = M(i * n + j) - eigenvalue * eigenvector(i) * eigenvector(j)
      }
    }

    new DenseMatrix(n, n, values)
  }

  /** Normalize a vector to unit length. */
  private def normalize(v: Array[Double]): Array[Double] = {
    var sumSq = 0.0
    var i     = 0
    while (i < v.length) {
      sumSq += v(i) * v(i)
      i += 1
    }
    val norm  = math.sqrt(sumSq)
    if (norm > 1e-10) v.map(_ / norm) else v
  }

  /** Dot product of two vectors. */
  private def dot(a: Array[Double], b: Array[Double]): Double = {
    var sum = 0.0
    var i   = 0
    while (i < a.length) {
      sum += a(i) * b(i)
      i += 1
    }
    sum
  }

  /** Build the spectral embedding: rows of U from smallest k eigenvectors.
    *
    * For symmetric normalized Laplacian, also row-normalize the embedding.
    *
    * @param eigenvectors
    *   k eigenvectors as arrays
    * @param laplacianType
    *   type of Laplacian used
    * @return
    *   n × k embedding matrix (each row is a point's embedding)
    */
  def buildEmbedding(
      eigenvectors: Array[Array[Double]],
      laplacianType: String
  ): Array[Vector] = {
    val k = eigenvectors.length
    val n = eigenvectors(0).length

    // Build n × k matrix (transpose of eigenvectors)
    val embedding = Array.ofDim[Double](n, k)
    for (i <- 0 until n) {
      for (j <- 0 until k) {
        embedding(i)(j) = eigenvectors(j)(i)
      }
    }

    // Row-normalize for symmetric normalized Laplacian (Ng, Jordan, Weiss)
    if (laplacianType.toLowerCase == LaplacianType.SymmetricNormalized) {
      for (i <- 0 until n) {
        var norm = 0.0
        for (j <- 0 until k) {
          norm += embedding(i)(j) * embedding(i)(j)
        }
        norm = math.sqrt(norm)
        if (norm > 1e-10) {
          for (j <- 0 until k) {
            embedding(i)(j) /= norm
          }
        }
      }
    }

    embedding.map(row => Vectors.dense(row))
  }

  /** Nyström approximation for large-scale spectral clustering.
    *
    * Approximates the full affinity matrix using a subset of landmark points: W ≈ A * B^(-1) * A^T
    *
    * where A is the n × m affinity between all points and m landmarks, and B is the m × m affinity
    * among landmarks.
    *
    * @param points
    *   all data points
    * @param kernel
    *   Mercer kernel
    * @param numLandmarks
    *   number of landmark points (m << n)
    * @param k
    *   number of eigenvectors
    * @param seed
    *   random seed for landmark selection
    * @return
    *   approximate embedding
    */
  def nystromApproximation(
      points: Array[Vector],
      kernel: MercerKernel,
      numLandmarks: Int,
      k: Int,
      seed: Long = 42L
  ): Array[Vector] = {
    val n   = points.length
    val m   = math.min(numLandmarks, n)
    val rng = new scala.util.Random(seed)

    // Sample landmarks uniformly
    val landmarkIndices = rng.shuffle((0 until n).toList).take(m).toArray.sorted
    val landmarks       = landmarkIndices.map(points)

    // Compute B: m × m affinity among landmarks
    val B = kernel.gramMatrix(landmarks)

    // Compute A: n × m affinity between all points and landmarks
    val A = new Array[Double](n * m)
    for (i <- 0 until n) {
      for (j <- 0 until m) {
        A(i * m + j) = kernel(points(i), landmarks(j))
      }
    }

    // Eigendecompose B
    val (eigVals, eigVecs) = computeSmallestEigenvectors(
      B,
      math.min(k + 1, m),
      maxIter = 50,
      seed = seed
    )

    // Nyström extension: U ≈ A * V * Λ^(-1/2)
    // where V, Λ are eigenvectors/values of B
    val embedding = Array.ofDim[Double](n, k)

    // Skip first eigenvector (constant for connected graph)
    val startIdx = if (eigVals.length > k) 1 else 0

    for (i <- 0 until n) {
      for (j <- 0 until k) {
        val eigIdx = startIdx + j
        if (eigIdx < eigVals.length && eigVals(eigIdx) > 1e-10) {
          var sum = 0.0
          for (l <- 0 until m) {
            sum += A(i * m + l) * eigVecs(eigIdx)(l)
          }
          embedding(i)(j) = sum / math.sqrt(eigVals(eigIdx))
        }
      }
    }

    // Row-normalize
    embedding.map { row =>
      var norm = 0.0
      for (j <- 0 until k) {
        norm += row(j) * row(j)
      }
      norm = math.sqrt(norm)
      if (norm > 1e-10) Vectors.dense(row.map(_ / norm))
      else Vectors.dense(row)
    }
  }
}
