package com.massivedatascience.clusterer

import com.massivedatascience.divergence.BregmanDivergence
import com.massivedatascience.linalg.WeightedVector
import org.scalatest.FunSuite
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors

import scala.annotation.tailrec

class BregmanTestSuite extends FunSuite {

  import com.massivedatascience.clusterer.TestingUtils._

  test("custom Bregman divergence") {
    def f(d: Vector): Double = {
      d.toArray.foldLeft(0.0) { case (agg, c) => agg + c * c }
    }

    def g(d: Vector): Vector = {
      Vectors.dense(d.toArray.map { _ * 2.0 })
    }
    val div = BregmanDivergence(f, g)

    val input = Vectors.dense(1.0, 2.0, 4.0)

    val weightedInput = Vectors.dense(2.0, 4.0, 8.0)

    val x: Double = div.convex(input)

    assert(x ~= 21.0 absTol 1.0e-8)

    val xx: Double = div.convexHomogeneous(weightedInput, 2.0)

    assert(xx ~= 21.0 absTol 1.0e-8)

    val gradOutput = Vectors.dense(input.toArray.map { _ * 2.0 })

    val y: Vector = div.gradientOfConvex(Vectors.dense(1.0, 2.0, 4.0))

    assert(y ~= gradOutput absTol 1.0e-8)

    val z: Vector = div.gradientOfConvexHomogeneous(weightedInput, 2.0)

    assert(z ~= gradOutput absTol 1.0e-8)

    val ops = BregmanPointOps(div)

    val opsSmoothed = BregmanPointOps(div, 1.0)

  }

  test("DenseSquaredEuclideanPointOps") {

    val ops = SquaredEuclideanPointOps
    val v1 = WeightedVector(Vectors.dense(1.0, 2.0, 4.0))
    val v2 = WeightedVector(Vectors.dense(2.0, 2.0, 8.0))
    val p1 = ops.toPoint(v1)
    val p2 = ops.toPoint(v2)
    val c1 = ops.toCenter(p1)
    val c2 = ops.toCenter(p2)
    val distance11 = ops.distance(p1, c1)
    assert(distance11 ~= 0.0 absTol 1.0e-8)
    val distance12 = ops.distance(p1, c2)
    assert(distance12 ~= 17.0 absTol 1.0e-8)

    val euclideanDistanceSquared = divergence(v1.inhomogeneous.toArray, v2.inhomogeneous.toArray, x => x * x)
    assert(euclideanDistanceSquared ~= 17.0 absTol 1.0e-8)
  }

  test("DenseSimplexKLPointOps") {
    val ops = DiscreteSimplexSmoothedKLPointOps

    val v1 = WeightedVector(Vectors.dense(1.0, 2.0, 4.0), 7.0)
    val v2 = WeightedVector(Vectors.dense(2.0, 2.0, 8.0), 12.0)
    val p1 = ops.toPoint(v1)
    val p2 = ops.toPoint(v2)

    val c1 = ops.toCenter(p1)
    val c2 = ops.toCenter(p2)

    val distance11 = ops.distance(ops.toPoint(c1), c1)
    assert(distance11 ~= 0.0 absTol 1.0e-8)

    val distance12 = ops.distance(p1, c2)

    val kl1 = divergence(p1.inhomogeneous.toArray, p2.inhomogeneous.toArray, klf)
    val kl2 = kl(p1.inhomogeneous, p2.inhomogeneous)

    assert(kl1 ~= kl2 absTol 1.0e-7)
  }

  test("DenseKLPointOps") {
    val ops = DenseKLPointOps

    val v1 = WeightedVector(Vectors.dense(1.0, 2.0, 4.0), 7.0)
    val v2 = WeightedVector(Vectors.dense(2.0, 2.0, 8.0), 12.0)
    val p1 = ops.toPoint(v1)
    val p2 = ops.toPoint(v2)

    val c1 = ops.toCenter(p1)
    val c2 = ops.toCenter(p2)

    val distance11 = ops.distance(p1, c1)
    assert(distance11 ~= 0.0 absTol 1.0e-8)

    val distance12 = ops.distance(p1, c2)

    val kl1 = divergence(v1.inhomogeneous.toArray, v2.inhomogeneous.toArray, klf)
    val kl2 = kl(v1.inhomogeneous, v2.inhomogeneous)

    assert(kl1 ~= kl2 absTol 1.0e-7)

    assert(distance12 ~= kl1 absTol 1.0e-8)
  }

  test("DenseKLPointOps homogeneous") {
    val ops = DenseKLPointOps

    val v1 = WeightedVector(Vectors.dense(1.0, 2.0, 4.0), 7.0)
    val v2 = WeightedVector(Vectors.dense(2.0, 2.0, 8.0), 12.0)

    val p1 = ops.toPoint(v1)
    val p2 = ops.toPoint(v2)

    val c1 = ops.toCenter(p1)
    val c2 = ops.toCenter(p2)

    val distance11 = ops.distance(p1, c1)
    assert(distance11 ~= 0.0 absTol 1.0e-8)

    val distance12 = ops.distance(p1, c2)

    val kl1 = divergence(v1.inhomogeneous.toArray, v2.inhomogeneous.toArray, klf)
    val kl2 = kl(v1.inhomogeneous, v2.inhomogeneous)

    assert(kl1 ~= kl2 absTol 1.0e-7)
    assert(distance12 ~= kl1 absTol 1.0e-8)
  }

  test("gradient") {

    val x = gradient(2.0)(x => x)
    assert(x ~= 1.0 absTol 1.0e-8)

    val y = gradient(23.0)(x => 2.0 * x)
    assert(y ~= 2.0 absTol 1.0e-8)

    val z = gradient(23.0)(x => x * x)
    assert(z ~= 46.0 absTol 1.0e-8)
  }

  def klf(x: Double) = x * Math.log(x) - x

  def kl(v: Vector, w: Vector): Double = {
    v.toArray.zip(w.toArray).foldLeft(0.0) {
      case (agg, (x, y)) =>
        agg + x * Math.log(x / y) - x + y
    }
  }

  def divergence(v: Array[Double], w: Array[Double], f: (Double => Double)): Double = {
    F(v)(f) - F(w)(f) - dot(gradient(w)(f), diff(v, w))
  }

  def F(v: Array[Double])(implicit f: (Double => Double)): Double = {
    v.foldLeft(0.0) { case (agg, y) => agg + f(y) }
  }

  def dot(v: Array[Double], w: Array[Double]): Double = {
    v.zip(w).foldLeft(0.0) { case (agg, (x, y)) => agg + x * y }
  }

  def diff(v: Array[Double], w: Array[Double]): Array[Double] = {
    v.zip(w).map { case (x, y) => x - y }
  }

  def gradient(v: Array[Double])(f: (Double => Double)): Array[Double] = {
    v.map(x => gradient(x)(f))
  }

  def gradient(v: Double)(f: (Double => Double)): Double = {
    @tailrec
    def recurse(delta: Double, previous: Double): Double = {
      val grad = (f(v + delta) - f(v)) / delta
      if (Math.abs(grad - previous) > 1e-10) recurse(delta / 2.0, grad) else grad
    }

    recurse(1.0, Double.PositiveInfinity)
  }
}
