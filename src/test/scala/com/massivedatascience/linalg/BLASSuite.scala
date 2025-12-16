package com.massivedatascience.linalg

import org.apache.spark.ml.linalg.{ DenseVector, SparseVector, Vectors }
import org.scalatest.funsuite.AnyFunSuite

/** Tests for vectorized BLAS operations. */
class BLASSuite extends AnyFunSuite {

  private val Epsilon = 1e-10

  // Test vectors
  private val denseVec1 = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0)
  private val denseVec2 = Vectors.dense(2.0, 3.0, 4.0, 5.0, 6.0)
  private val sparseVec = Vectors.sparse(5, Array(0, 2, 4), Array(1.0, 3.0, 5.0))
  private val zeroVec   = Vectors.dense(0.0, 0.0, 0.0)
  private val unitVec   = Vectors.dense(1.0, 0.0, 0.0)

  // Expected L2 norm of denseVec1: sqrt(1 + 4 + 9 + 16 + 25) = sqrt(55)
  private val expectedNorm1 = math.sqrt(55.0)

  // Expected L2 norm of sparseVec: sqrt(1 + 9 + 25) = sqrt(35)
  private val expectedSparseNorm = math.sqrt(35.0)

  // === nrm2 tests ===

  test("nrm2 computes correct L2 norm for dense vector") {
    val norm = BLAS.nrm2(denseVec1)
    assert(math.abs(norm - expectedNorm1) < Epsilon, s"Expected $expectedNorm1, got $norm")
  }

  test("nrm2 computes correct L2 norm for sparse vector") {
    val norm = BLAS.nrm2(sparseVec)
    assert(math.abs(norm - expectedSparseNorm) < Epsilon, s"Expected $expectedSparseNorm, got $norm")
  }

  test("nrm2 returns 0 for zero vector") {
    val norm = BLAS.nrm2(zeroVec)
    assert(math.abs(norm) < Epsilon)
  }

  test("nrm2 returns 1 for unit vector") {
    val norm = BLAS.nrm2(unitVec)
    assert(math.abs(norm - 1.0) < Epsilon)
  }

  test("nrm2 handles single element vector") {
    val vec  = Vectors.dense(5.0)
    val norm = BLAS.nrm2(vec)
    assert(math.abs(norm - 5.0) < Epsilon)
  }

  test("nrm2 handles negative values") {
    val vec  = Vectors.dense(-3.0, -4.0)
    val norm = BLAS.nrm2(vec)
    assert(math.abs(norm - 5.0) < Epsilon) // sqrt(9 + 16) = 5
  }

  // === squaredNorm tests ===

  test("squaredNorm computes correct squared L2 norm for dense vector") {
    val sqNorm = BLAS.squaredNorm(denseVec1)
    assert(math.abs(sqNorm - 55.0) < Epsilon, s"Expected 55.0, got $sqNorm")
  }

  test("squaredNorm computes correct squared L2 norm for sparse vector") {
    val sqNorm = BLAS.squaredNorm(sparseVec)
    assert(math.abs(sqNorm - 35.0) < Epsilon, s"Expected 35.0, got $sqNorm")
  }

  test("squaredNorm equals nrm2 squared") {
    val norm   = BLAS.nrm2(denseVec1)
    val sqNorm = BLAS.squaredNorm(denseVec1)
    assert(math.abs(sqNorm - norm * norm) < Epsilon)
  }

  test("squaredNorm returns 0 for zero vector") {
    val sqNorm = BLAS.squaredNorm(zeroVec)
    assert(math.abs(sqNorm) < Epsilon)
  }

  // === asum tests ===

  test("asum computes correct L1 norm for dense vector") {
    val l1Norm = BLAS.asum(denseVec1)
    assert(math.abs(l1Norm - 15.0) < Epsilon) // 1+2+3+4+5 = 15
  }

  test("asum computes correct L1 norm for sparse vector") {
    val l1Norm = BLAS.asum(sparseVec)
    assert(math.abs(l1Norm - 9.0) < Epsilon) // 1+3+5 = 9
  }

  test("asum handles negative values (sum of absolute values)") {
    val vec    = Vectors.dense(-1.0, 2.0, -3.0, 4.0)
    val l1Norm = BLAS.asum(vec)
    assert(math.abs(l1Norm - 10.0) < Epsilon) // |−1|+|2|+|−3|+|4| = 10
  }

  test("asum returns 0 for zero vector") {
    val l1Norm = BLAS.asum(zeroVec)
    assert(math.abs(l1Norm) < Epsilon)
  }

  // === normalize tests ===

  test("normalize produces unit L2 norm vector") {
    val normalized = BLAS.normalize(denseVec1)
    val norm       = BLAS.nrm2(normalized)
    assert(math.abs(norm - 1.0) < Epsilon, s"Expected norm 1.0, got $norm")
  }

  test("normalize preserves direction") {
    val normalized = BLAS.normalize(denseVec1)
    val expected   = denseVec1.toArray.map(_ / expectedNorm1)
    val actual     = normalized.toArray
    expected.zip(actual).foreach { case (e, a) =>
      assert(math.abs(e - a) < Epsilon, s"Direction not preserved: expected $e, got $a")
    }
  }

  test("normalize returns copy for near-zero norm vector") {
    val almostZero = Vectors.dense(1e-15, 1e-15, 1e-15)
    val normalized = BLAS.normalize(almostZero)
    // Should return copy without modification
    assert(BLAS.nrm2(normalized) < 1e-10)
  }

  test("normalize handles sparse vector") {
    val normalized = BLAS.normalize(sparseVec)
    val norm       = BLAS.nrm2(normalized)
    assert(math.abs(norm - 1.0) < Epsilon)
  }

  test("normalize is idempotent for unit vectors") {
    val unitVec    = Vectors.dense(1.0 / math.sqrt(3), 1.0 / math.sqrt(3), 1.0 / math.sqrt(3))
    val normalized = BLAS.normalize(unitVec)
    val doubleNorm = BLAS.normalize(normalized)

    normalized.toArray.zip(doubleNorm.toArray).foreach { case (n, d) =>
      assert(math.abs(n - d) < Epsilon)
    }
  }

  // === Integration tests ===

  test("squaredNorm useful for squared Euclidean distance") {
    // ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x'y
    val x = Vectors.dense(1.0, 2.0, 3.0)
    val y = Vectors.dense(4.0, 5.0, 6.0)

    val sqNormX = BLAS.squaredNorm(x)
    val sqNormY = BLAS.squaredNorm(y)
    val dotXY   = BLAS.dot(x, y)

    val sqDist = sqNormX + sqNormY - 2 * dotXY

    // Direct calculation: (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
    assert(math.abs(sqDist - 27.0) < Epsilon)
  }

  test("normalize useful for cosine similarity") {
    val x = Vectors.dense(1.0, 2.0, 3.0)
    val y = Vectors.dense(2.0, 4.0, 6.0) // Collinear with x

    val xNorm = BLAS.normalize(x)
    val yNorm = BLAS.normalize(y)

    val cosine = BLAS.dot(xNorm, yNorm)
    assert(math.abs(cosine - 1.0) < Epsilon, "Collinear vectors should have cosine 1.0")
  }

  test("normalize followed by squared Euclidean gives 2*(1-cos)") {
    val x = Vectors.dense(1.0, 0.0)
    val y = Vectors.dense(0.0, 1.0) // Orthogonal

    val xNorm = BLAS.normalize(x)
    val yNorm = BLAS.normalize(y)

    // ||xNorm - yNorm||^2 = 2 - 2*cos(theta) = 2 - 2*0 = 2
    val sqNormX = BLAS.squaredNorm(xNorm)
    val sqNormY = BLAS.squaredNorm(yNorm)
    val dotXY   = BLAS.dot(xNorm, yNorm)
    val sqDist  = sqNormX + sqNormY - 2 * dotXY

    assert(math.abs(sqDist - 2.0) < Epsilon)
  }

  // === Performance sanity tests ===

  test("nrm2 handles large vectors") {
    val largeVec = Vectors.dense(Array.fill(10000)(1.0))
    val norm     = BLAS.nrm2(largeVec)
    // sqrt(10000 * 1^2) = 100
    assert(math.abs(norm - 100.0) < Epsilon)
  }

  test("squaredNorm handles large vectors") {
    val largeVec = Vectors.dense(Array.fill(10000)(2.0))
    val sqNorm   = BLAS.squaredNorm(largeVec)
    // 10000 * 4 = 40000
    assert(math.abs(sqNorm - 40000.0) < Epsilon)
  }
}
