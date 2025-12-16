package com.massivedatascience.divergence

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.df._

/** Test suite for BregmanFunction unified trait.
  *
  * Verifies:
  *   1. Mathematical properties (non-negativity, identity)
  *   2. Consistency with existing BregmanKernel implementations
  */
class BregmanFunctionSuite extends AnyFunSuite with Matchers {

  val tolerance = 1e-9

  // Test vectors
  val v1 = Vectors.dense(1.0, 2.0, 3.0)
  val v2 = Vectors.dense(1.5, 2.5, 3.5)
  val vPositive = Vectors.dense(0.1, 0.2, 0.3)
  val vProbability = Vectors.dense(0.2, 0.3, 0.5)

  // ============ Squared Euclidean Tests ============

  test("SquaredEuclidean - divergence is non-negative") {
    val func = BregmanFunctions.squaredEuclidean
    func.divergence(v1, v2) should be >= 0.0
    func.divergence(v2, v1) should be >= 0.0
  }

  test("SquaredEuclidean - identity: D(x, x) = 0") {
    val func = BregmanFunctions.squaredEuclidean
    func.divergence(v1, v1) shouldBe 0.0 +- tolerance
  }

  test("SquaredEuclidean - gradient consistency: invGrad(grad(x)) = x") {
    val func = BregmanFunctions.squaredEuclidean
    val grad = func.gradF(v1)
    val recovered = func.invGradF(grad)
    recovered.toArray.zip(v1.toArray).foreach { case (a, b) =>
      a shouldBe b +- tolerance
    }
  }

  test("SquaredEuclidean - consistent with BregmanKernel") {
    val func = BregmanFunctions.squaredEuclidean
    val kernel = new SquaredEuclideanKernel()

    val funcDiv = func.divergence(v1, v2)
    val kernelDiv = kernel.divergence(v1, v2)
    funcDiv shouldBe kernelDiv +- tolerance

    val funcGrad = func.gradF(v1)
    val kernelGrad = kernel.grad(v1)
    funcGrad.toArray.zip(kernelGrad.toArray).foreach { case (a, b) =>
      a shouldBe b +- tolerance
    }
  }

  // ============ KL Divergence Tests ============

  test("KL - divergence is non-negative") {
    val func = BregmanFunctions.kl()
    func.divergence(vPositive, vPositive.copy) should be >= -tolerance // Allow small numerical error
  }

  test("KL - identity: D(x, x) ≈ 0") {
    val func = BregmanFunctions.kl()
    func.divergence(vPositive, vPositive) shouldBe 0.0 +- tolerance
  }

  test("KL - gradient consistency: invGrad(grad(x)) ≈ x") {
    val func = BregmanFunctions.kl()
    val grad = func.gradF(vPositive)
    val recovered = func.invGradF(grad)
    recovered.toArray.zip(vPositive.toArray).foreach { case (a, b) =>
      a shouldBe b +- 1e-6  // Slightly larger tolerance due to exp/log
    }
  }

  test("KL - consistent with BregmanKernel") {
    val func = BregmanFunctions.kl()
    val kernel = new KLDivergenceKernel()

    val v1Pos = Vectors.dense(0.5, 1.0, 1.5)
    val v2Pos = Vectors.dense(0.6, 1.1, 1.6)

    val funcDiv = func.divergence(v1Pos, v2Pos)
    val kernelDiv = kernel.divergence(v1Pos, v2Pos)
    funcDiv shouldBe kernelDiv +- tolerance
  }

  // ============ Itakura-Saito Tests ============

  test("ItakuraSaito - divergence is non-negative") {
    val func = BregmanFunctions.itakuraSaito()
    func.divergence(vPositive, vPositive.copy) should be >= -tolerance
  }

  test("ItakuraSaito - identity: D(x, x) ≈ 0") {
    val func = BregmanFunctions.itakuraSaito()
    func.divergence(vPositive, vPositive) shouldBe 0.0 +- tolerance
  }

  test("ItakuraSaito - consistent with BregmanKernel") {
    val func = BregmanFunctions.itakuraSaito()
    val kernel = new ItakuraSaitoKernel()

    val v1Pos = Vectors.dense(0.5, 1.0, 1.5)
    val v2Pos = Vectors.dense(0.6, 1.1, 1.6)

    val funcDiv = func.divergence(v1Pos, v2Pos)
    val kernelDiv = kernel.divergence(v1Pos, v2Pos)
    funcDiv shouldBe kernelDiv +- tolerance
  }

  // ============ Generalized-I Tests ============

  test("GeneralizedI - divergence is non-negative") {
    val func = BregmanFunctions.generalizedI()
    func.divergence(vPositive, vPositive.copy) should be >= -tolerance
  }

  test("GeneralizedI - consistent with BregmanKernel") {
    val func = BregmanFunctions.generalizedI()
    val kernel = new GeneralizedIDivergenceKernel()

    val v1Pos = Vectors.dense(0.5, 1.0, 1.5)
    val v2Pos = Vectors.dense(0.6, 1.1, 1.6)

    val funcDiv = func.divergence(v1Pos, v2Pos)
    val kernelDiv = kernel.divergence(v1Pos, v2Pos)
    funcDiv shouldBe kernelDiv +- tolerance
  }

  // ============ Logistic Loss Tests ============

  test("LogisticLoss - divergence is non-negative") {
    val func = BregmanFunctions.logisticLoss()
    func.divergence(vProbability, vProbability.copy) should be >= -tolerance
  }

  test("LogisticLoss - consistent with BregmanKernel") {
    val func = BregmanFunctions.logisticLoss()
    val kernel = new LogisticLossKernel()

    val funcDiv = func.divergence(vProbability, vProbability)
    val kernelDiv = kernel.divergence(vProbability, vProbability)
    funcDiv shouldBe kernelDiv +- tolerance
  }

  // ============ L1 Tests ============

  test("L1 - divergence is non-negative") {
    val func = BregmanFunctions.l1
    func.divergence(v1, v2) should be >= 0.0
  }

  test("L1 - identity: D(x, x) = 0") {
    val func = BregmanFunctions.l1
    func.divergence(v1, v1) shouldBe 0.0 +- tolerance
  }

  test("L1 - consistent with BregmanKernel") {
    val func = BregmanFunctions.l1
    val kernel = new L1Kernel()

    val funcDiv = func.divergence(v1, v2)
    val kernelDiv = kernel.divergence(v1, v2)
    funcDiv shouldBe kernelDiv +- tolerance
  }

  // ============ Spherical Tests ============

  test("Spherical - divergence is non-negative") {
    val func = BregmanFunctions.spherical
    func.divergence(v1, v2) should be >= 0.0
  }

  test("Spherical - identity: D(x, x) = 0") {
    val func = BregmanFunctions.spherical
    func.divergence(v1, v1) shouldBe 0.0 +- tolerance
  }

  test("Spherical - consistent with BregmanKernel") {
    val func = BregmanFunctions.spherical
    val kernel = new SphericalKernel()

    val funcDiv = func.divergence(v1, v2)
    val kernelDiv = kernel.divergence(v1, v2)
    funcDiv shouldBe kernelDiv +- tolerance
  }

  test("BregmanFunctions.apply returns correct types") {
    BregmanFunctions("squaredEuclidean").name should include("Euclidean")
    BregmanFunctions("kl").name should include("KL")
    BregmanFunctions("itakuraSaito").name should include("Itakura")
    BregmanFunctions("generalizedI").name should include("GeneralizedI")
    BregmanFunctions("logistic").name should include("Logistic")
    BregmanFunctions("l1").name shouldBe "L1"
    BregmanFunctions("spherical").name shouldBe "Spherical"
    BregmanFunctions("cosine").name shouldBe "Spherical"
  }

  test("all functions pass validation for valid inputs") {
    val functions = Seq(
      BregmanFunctions.squaredEuclidean,
      BregmanFunctions.l1,
      BregmanFunctions.spherical
    )

    functions.foreach { func =>
      func.validate(v1) shouldBe true
    }

    // Positive-domain functions
    val positiveFunctions = Seq(
      BregmanFunctions.kl(),
      BregmanFunctions.itakuraSaito(),
      BregmanFunctions.generalizedI()
    )

    positiveFunctions.foreach { func =>
      func.validate(vPositive) shouldBe true
    }

    // Probability-domain function
    BregmanFunctions.logisticLoss().validate(vProbability) shouldBe true
  }

  test("supportsExpressionOptimization is correct") {
    BregmanFunctions.squaredEuclidean.supportsExpressionOptimization shouldBe true
    BregmanFunctions.kl().supportsExpressionOptimization shouldBe false
    BregmanFunctions.spherical.supportsExpressionOptimization shouldBe false
    BregmanFunctions.l1.supportsExpressionOptimization shouldBe false
  }
}
