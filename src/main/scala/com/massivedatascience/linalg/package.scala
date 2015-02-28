package com.massivedatascience

import com.massivedatascience.linalg.BLAS._
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}


package object linalg {
  def asInhomogeneous(homogeneous: Vector, weight: Double) = {
    val x = homogeneous.copy
    scal(1.0 / weight, x)
    x
  }

  def asHomogeneous(inhomogeneous: Vector, weight: Double) = {
    val x = inhomogeneous.copy
    scal(weight, x)
    x
  }

  implicit class RichVector(val v: Vector) extends AnyVal {
    def iterator: VectorIterator = {
      v match {
        case s: SparseVector => new SparseVectorIterator(s)
        case d: DenseVector => new DenseVectorIterator(d)
      }
    }

    def negativeIterator: VectorIterator = {
      v match {
        case s: SparseVector => new NegativeSparseVectorIterator(s)
        case d: DenseVector => new NegativeDenseVectorIterator(d)
      }
    }
  }

}
