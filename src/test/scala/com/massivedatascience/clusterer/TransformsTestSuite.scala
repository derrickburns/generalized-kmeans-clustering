package com.massivedatascience.clusterer

import com.massivedatascience.linalg.WeightedVector
import com.massivedatascience.transforms.Embedding
import org.apache.spark.mllib.linalg.{ DenseVector, Vectors }
import org.scalatest._
import funsuite._

import com.holdenkarau.spark.testing.LocalSparkContext

class TransformsTestSuite extends AnyFunSuite with LocalSparkContext { 

  import com.massivedatascience.clusterer.TestingUtils._

  test("Dense Embedding") {

    val embedding = Embedding(Embedding.DENSE_EMBEDDING)

    val in = Vectors.sparse(10, Seq((2, 30.2), (4, 42.0)))

    val sp = WeightedVector(in, 1.0)

    val embedded = embedding.embed(sp)

    assert(Vectors.dense(embedded.inhomogeneous.toArray) ~= in absTol 1.0e-8)

  }

  test("Haar Embedding") {

    val embedding = Embedding(Embedding.HAAR_EMBEDDING)

    val in = Vectors.sparse(10, Seq((2, 30.0), (4, 42.0), (5, 8.0)))

    val out = Vectors.dense(0.0, 15.0, 25.0)

    val embedded = embedding.embed(WeightedVector(in))

    assert(embedded.inhomogeneous ~= out absTol 1.0e-8)

  }
}
