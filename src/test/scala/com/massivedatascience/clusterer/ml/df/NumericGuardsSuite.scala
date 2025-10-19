package com.massivedatascience.clusterer.ml.df

import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests for NumericGuards utility.
  *
  * These tests verify that numeric guards correctly detect and report:
  *   - NaN values
  *   - Inf values
  *   - Negative values (for divergences requiring positivity)
  *   - Out-of-range probability values
  *   - Invalid weights
  */
class NumericGuardsSuite extends AnyFunSuite with Matchers {

  test("checkFinite accepts finite values") {
    val v = Vectors.dense(1.0, 2.0, 3.0, -4.0, 0.0)
    noException should be thrownBy NumericGuards.checkFinite(v, "test")
  }

  test("checkFinite rejects NaN values") {
    val v  = Vectors.dense(1.0, Double.NaN, 3.0)
    val ex = intercept[Exception] {
      NumericGuards.checkFinite(v, "test context")
    }
    ex.getMessage should include("NaN")
    ex.getMessage should include("test context")
    ex.getMessage should include("index 1")
  }

  test("checkFinite rejects Inf values") {
    val v  = Vectors.dense(1.0, Double.PositiveInfinity, 3.0)
    val ex = intercept[Exception] {
      NumericGuards.checkFinite(v, "test context")
    }
    ex.getMessage should include("Inf")
    ex.getMessage should include("test context")
    ex.getMessage should include("index 1")
  }

  test("checkFinite rejects negative Inf") {
    val v  = Vectors.dense(1.0, Double.NegativeInfinity, 3.0)
    val ex = intercept[Exception] {
      NumericGuards.checkFinite(v, "test context")
    }
    ex.getMessage should include("Inf")
  }

  test("checkPositive accepts positive values") {
    val v = Vectors.dense(1.0, 2.0, 3.0, 0.001)
    noException should be thrownBy NumericGuards.checkPositive(v, "test")
  }

  test("checkPositive accepts zero") {
    val v = Vectors.dense(1.0, 0.0, 3.0)
    noException should be thrownBy NumericGuards.checkPositive(v, "test")
  }

  test("checkPositive rejects negative values") {
    val v  = Vectors.dense(1.0, -0.1, 3.0)
    val ex = intercept[Exception] {
      NumericGuards.checkPositive(v, "test context")
    }
    ex.getMessage should include("negative")
    ex.getMessage should include("test context")
    ex.getMessage should include("index 1")
    ex.getMessage should include("KL/Itakura-Saito")
  }

  test("checkPositive allows near-zero values within epsilon") {
    val v = Vectors.dense(1.0, -1e-12, 3.0)
    noException should be thrownBy NumericGuards.checkPositive(v, "test", epsilon = 1e-10)
  }

  test("checkPositive error message includes solutions") {
    val v  = Vectors.dense(1.0, -1.0, 3.0)
    val ex = intercept[Exception] {
      NumericGuards.checkPositive(v, "center update")
    }
    ex.getMessage should include("setSmoothing")
    ex.getMessage should include("Solutions")
  }

  test("checkProbability accepts values in (0, 1)") {
    val v = Vectors.dense(0.1, 0.5, 0.9, 0.001, 0.999)
    noException should be thrownBy NumericGuards.checkProbability(v, "test")
  }

  test("checkProbability rejects 0.0") {
    val v  = Vectors.dense(0.5, 0.0, 0.9)
    val ex = intercept[Exception] {
      NumericGuards.checkProbability(v, "test context")
    }
    ex.getMessage should include("outside (0,1)")
    ex.getMessage should include("test context")
  }

  test("checkProbability rejects 1.0") {
    val v  = Vectors.dense(0.5, 1.0, 0.9)
    val ex = intercept[Exception] {
      NumericGuards.checkProbability(v, "test context")
    }
    ex.getMessage should include("outside (0,1)")
    ex.getMessage should include("Logistic loss")
  }

  test("checkProbability rejects values > 1") {
    val v  = Vectors.dense(0.5, 1.1, 0.9)
    val ex = intercept[Exception] {
      NumericGuards.checkProbability(v, "test context")
    }
    ex.getMessage should include("outside (0,1)")
  }

  test("checkProbability rejects negative values") {
    val v  = Vectors.dense(0.5, -0.1, 0.9)
    val ex = intercept[Exception] {
      NumericGuards.checkProbability(v, "test context")
    }
    ex.getMessage should include("outside (0,1)")
  }

  test("checkFiniteScalar accepts finite values") {
    noException should be thrownBy NumericGuards.checkFiniteScalar(42.0, "test")
    noException should be thrownBy NumericGuards.checkFiniteScalar(-3.14, "test")
    noException should be thrownBy NumericGuards.checkFiniteScalar(0.0, "test")
  }

  test("checkFiniteScalar rejects NaN") {
    val ex = intercept[RuntimeException] {
      NumericGuards.checkFiniteScalar(Double.NaN, "cost calculation")
    }
    ex.getMessage should include("NaN")
    ex.getMessage should include("cost calculation")
  }

  test("checkFiniteScalar rejects Inf") {
    val ex = intercept[RuntimeException] {
      NumericGuards.checkFiniteScalar(Double.PositiveInfinity, "distance")
    }
    ex.getMessage should include("Inf")
    ex.getMessage should include("distance")
  }

  test("checkWeight accepts positive finite weights") {
    noException should be thrownBy NumericGuards.checkWeight(1.0, "test")
    noException should be thrownBy NumericGuards.checkWeight(0.001, "test")
    noException should be thrownBy NumericGuards.checkWeight(1000.0, "test")
  }

  test("checkWeight rejects zero") {
    val ex = intercept[Exception] {
      NumericGuards.checkWeight(0.0, "test context")
    }
    ex.getMessage should include("weight")
  }

  test("checkWeight rejects negative weights") {
    val ex = intercept[Exception] {
      NumericGuards.checkWeight(-1.0, "test context")
    }
    ex.getMessage should include("weight")
  }

  test("checkWeight rejects NaN weights") {
    val ex = intercept[Exception] {
      NumericGuards.checkWeight(Double.NaN, "test context")
    }
    ex.getMessage should include("weight")
  }

  test("checkWeight rejects Inf weights") {
    val ex = intercept[Exception] {
      NumericGuards.checkWeight(Double.PositiveInfinity, "test context")
    }
    ex.getMessage should include("weight")
  }

  test("safeAdd performs normal addition") {
    val v1     = Vectors.dense(1.0, 2.0, 3.0)
    val v2     = Vectors.dense(4.0, 5.0, 6.0)
    val result = NumericGuards.safeAdd(v1, v2, "test")
    result.toArray should contain theSameElementsInOrderAs Array(5.0, 7.0, 9.0)
  }

  test("safeAdd detects overflow to Inf") {
    val v1 = Vectors.dense(Double.MaxValue, 1.0)
    val v2 = Vectors.dense(Double.MaxValue, 1.0)
    val ex = intercept[Exception] {
      NumericGuards.safeAdd(v1, v2, "center update")
    }
    ex.getMessage should include("Overflow")
    ex.getMessage should include("center update")
  }

  test("safeAdd requires matching dimensions") {
    val v1 = Vectors.dense(1.0, 2.0)
    val v2 = Vectors.dense(3.0, 4.0, 5.0)
    val ex = intercept[IllegalArgumentException] {
      NumericGuards.safeAdd(v1, v2, "test")
    }
    ex.getMessage should include("dimensions must match")
  }

  test("safeScale performs normal scaling") {
    val v      = Vectors.dense(1.0, 2.0, 3.0)
    val result = NumericGuards.safeScale(v, 2.0, "test")
    result.toArray should contain theSameElementsInOrderAs Array(2.0, 4.0, 6.0)
  }

  test("safeScale detects overflow") {
    val v  = Vectors.dense(Double.MaxValue, 1.0)
    val ex = intercept[Exception] {
      NumericGuards.safeScale(v, 2.0, "test")
    }
    ex.getMessage should include("Overflow")
  }

  test("safeScale rejects NaN scalar") {
    val v  = Vectors.dense(1.0, 2.0)
    val ex = intercept[RuntimeException] {
      NumericGuards.safeScale(v, Double.NaN, "test")
    }
    ex.getMessage should include("NaN")
  }

  test("safeScale rejects Inf scalar") {
    val v  = Vectors.dense(1.0, 2.0)
    val ex = intercept[RuntimeException] {
      NumericGuards.safeScale(v, Double.PositiveInfinity, "test")
    }
    ex.getMessage should include("Inf")
  }

  test("error messages include vector preview for small vectors") {
    val v  = Vectors.dense(1.0, Double.NaN, 3.0)
    val ex = intercept[Exception] {
      NumericGuards.checkFinite(v, "test")
    }
    ex.getMessage should include("[1.0, NaN, 3.0]")
  }

  test("error messages truncate large vectors") {
    val arr = Array.fill(20)(1.0)
    arr(15) = Double.NaN
    val v   = Vectors.dense(arr)
    val ex  = intercept[Exception] {
      NumericGuards.checkFinite(v, "test")
    }
    ex.getMessage should include("...")
  }

  test("checkFinite provides actionable guidance for NaN") {
    val v  = Vectors.dense(1.0, Double.NaN)
    val ex = intercept[Exception] {
      NumericGuards.checkFinite(v, "test")
    }
    ex.getMessage should include("may indicate")
    ex.getMessage should include("instability")
  }

  test("checkPositive provides actionable guidance") {
    val v  = Vectors.dense(1.0, -1.0)
    val ex = intercept[Exception] {
      NumericGuards.checkPositive(v, "test")
    }
    ex.getMessage should include("Solutions")
    ex.getMessage should include("setSmoothing")
  }
}
