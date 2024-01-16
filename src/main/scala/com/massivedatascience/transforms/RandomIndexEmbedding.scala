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

package com.massivedatascience.transforms

import com.massivedatascience.linalg._
import com.massivedatascience.util.XORShiftRandom
import org.apache.spark.ml.linalg.{ Vectors, Vector }
import org.joda.time.DateTime

object RandomIndexEmbedding {
  final val defaultLogInputDim: Int = 63
}

/**
 *
 * Embeds vectors in high dimensional spaces into dense vectors of low dimensional space
 * in a way that preserves similarity as per the
 * <a href= "http://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss">Johnson-Lindenstrauss lemma</a>.
 *
 * This technique is called <a href="http://en.wikipedia.org/wiki/Random_indexing">random indexing</a>.
 *
 * We use a family of <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.394.2077&rep=rep1&type=pdf">universal hash functions</a>
 * to avoid pre-computing the random projection.
 *
 * @see <a href="https://www.sics.se/~mange/papers/RI_intro.pdf">introduction to random indexing</a>
 *
 * @param outputDim number of dimensions to use for the dense vector
 * @param epsilon portion of dimensions with non-zero values
 */
class RandomIndexEmbedding(
    outputDim: Int,
    epsilon: Double,
    logInputDim: Int = RandomIndexEmbedding.defaultLogInputDim,
    seed: Long = new DateTime().getMillis) extends Embedding {
  require(epsilon < 1.0)
  require(epsilon > 0.0)

  private[this] val on = (Math.ceil(epsilon * outputDim).toInt & -1) * 2
  private[this] val logOutputDim = Math.log(outputDim).toInt
  private[this] val hashes = computeHashes(seed)
  private[this] val positive = hashes.take(on / 2)
  private[this] val negative = hashes.drop(on / 2)

  private[this] def computeHashes(seed: Long): Array[MultiplicativeHash] = {
    val rand = new XORShiftRandom(seed)
    Array.fill(on) {
      new MultiplicativeHash(rand.nextLong().abs, rand.nextLong().abs, logInputDim, logOutputDim)
    }
  }

  def embed(v: Vector): Vector = embed(v.iterator)

  def embed(iterator: VectorIterator): Vector = {
    val rep = new Array[Double](outputDim)
    while (iterator.hasNext) {
      val count = iterator.value
      val index = iterator.index
      for (p <- positive) {
        val hashKey = p.hash(index)
        rep(hashKey) += count
      }
      for (n <- negative) {
        val hashKey = n.hash(index)
        rep(hashKey) -= count
      }
      iterator.advance()
    }
    Vectors.dense(rep)
  }

}

/**
 * A simple <a href="http://en.wikipedia.org/wiki/Hash_function">multiplicative hash function</a>
 * that supports mapping from spaces that have a number of dimensions that is a power of two
 * to a space that has a dimension that is a (lower) power of two.
 *
 * @param seed multiplicative seed
 * @param offset additive seed
 * @param l log of the number of dimensions of the input
 * @param m log of the number of dimensions of the output
 */
class MultiplicativeHash(seed: Long, offset: Long, l: Int, m: Int) {
  require(m <= l)
  val mask = if (l >= RandomIndexEmbedding.defaultLogInputDim) Long.MaxValue else (1 << l) - 1
  val shift = l - m

  def hash(index: Int): Int = (((seed * index.toLong + offset) & mask) >> shift).toInt
}
