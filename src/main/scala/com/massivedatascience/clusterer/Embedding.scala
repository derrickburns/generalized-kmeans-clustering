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

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.util.BLAS._
import com.massivedatascience.clusterer.util.{HaarWavelet, XORShiftRandom}
import org.apache.spark.mllib.linalg.{Vectors, Vector, SparseVector, DenseVector}

trait Embedding extends Serializable {
  def embed(v: WeightedVector): WeightedVector
}

object IdentityEmbedding extends Embedding {
  def embed(v: WeightedVector): WeightedVector = v
}

object DenseEmbedding extends Embedding {
  def embed(v: WeightedVector): WeightedVector = {
   v match {
     case sv: SparseVector => WeightedVector(v.homogeneous.toArray, v.weight)
     case dv: DenseVector => dv
   }
  }
}

object HaarEmbedding extends Embedding {
  def embed(raw: WeightedVector): WeightedVector =
    WeightedVector(HaarWavelet.average(raw.homogeneous.toArray), raw.weight)
}

/**
 * One can create a symmetric version of any Kullback Leibler Divergence that can be clustered
 * by embedding the input points (which are a simplex in R+ ** n) into a new Euclidean space R ** N.
 *
 * See http://www-users.cs.umn.edu/~banerjee/papers/13/bregman-metric.pdf
 *
 * This one is
 *
 * distance(x,y) = KL(x,y) + KL(y,x) + (1/2) ||x-y||^2 + (1/2) || gradF(x) - gradF(y)||^2
 *
 * The embedding is simply
 *
 * x => x + gradF(x) (Lemma 1 with alpha = beta = 1)
 *
 */
class SymmetrizingEmbedding(divergence: BregmanDivergence) extends Embedding {
  def embed(v: WeightedVector): WeightedVector = {
    val embedded = v.homogeneous.copy
    WeightedVector(axpy(1.0, divergence.gradF(embedded), embedded), v.weight)
  }
}

object SymmetrizingKLEmbedding extends SymmetrizingEmbedding(RealKullbackLeiblerSimplexDivergence)

/**
 *
 * Embeds vectors in high dimensional spaces into dense vectors of low dimensional space
 * in a way that preserves similarity as per the Johnson-Lindenstrauss lemma.
 *
 * http://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss lemma
 *
 * This technique is called random indexing
 *
 * http://en.wikipedia.org/wiki/Random_indexing
 *
 * https://www.sics.se/~mange/papers/RI_intro.pdf
 *
 * @param dim number of dimensions to use for the dense vector
 * @param epsilon portion of dimensions with non-zero values
 */
case class RandomIndexEmbedding(dim: Int, epsilon: Double) extends Embedding {

  require(epsilon < 1.0)
  require(epsilon > 0.0)

  val on = Math.ceil(epsilon * dim).toInt & -2

  val tinySet = new TinyIntSet(on)

  def embed(v: WeightedVector): WeightedVector = {
    val iterator = v.homogeneous.iterator
    val rep = new Array[Double](dim)

    while (iterator.hasNext) {
      val count = iterator.value
      val random = new XORShiftRandom(iterator.index)
      iterator.advance()

      var remaining = on
      while (remaining > 0) {
        val ri = random.nextInt() % dim
        if (!tinySet.contains(ri)) {
          rep(ri) = if(random.nextBoolean()) rep(ri) + count else rep(ri) - count
          remaining = remaining - 1
          tinySet.add(ri)
        }
      }
      tinySet.clear()
    }
    WeightedVector(rep, v.weight)
  }
}

/**
 * A tiny set of integer values
 *
 * @param maxSize the maximum number of elements in the set
 */

private[clusterer] class TinyIntSet(maxSize: Int) {
  val taken = new Array[Int](maxSize)
  var size = 0

  def contains(ri: Int): Boolean = {
    var i = size
    while (i > 0) {
      i = i - 1
      if (taken(i) == ri) return true
    }
    false
  }

  def add(ri: Int): Unit = {
    require(size < maxSize)
    taken(size) = ri
    size = size + 1
  }

  def clear(): Unit = size = 0
}