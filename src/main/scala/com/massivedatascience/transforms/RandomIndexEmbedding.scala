package com.massivedatascience.transforms

import com.massivedatascience.linalg.{WeightedVector, _}
import com.massivedatascience.util.XORShiftRandom
import org.joda.time.DateTime

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
 * We use a family of universal hash functions to avoid pre-computing the
 * random projection:
 *
 * http://www.velldal.net/erik/pubs/Velldal11a.ps
 *
 * @param outputDim number of dimensions to use for the dense vector
 * @param epsilon portion of dimensions with non-zero values
 */
case class RandomIndexEmbedding(
  outputDim: Int,
  epsilon: Double,
  logInputDim: Int = 63,
  seed: Long = new DateTime().getMillis) extends Embedding {
  require(epsilon < 1.0)
  require(epsilon > 0.0)

  private val on = Math.ceil(epsilon * outputDim).toInt & -2
  private val logOutputDim = Math.log(outputDim).toInt
  private val hashes = computeHashes(seed)
  private val positive = hashes.take(on / 2)
  private val negative = hashes.drop(on / 2)

  private def computeHashes(seed: Long): Array[MultiplicativeHash] = {
    val rand = new XORShiftRandom(seed)
    Array.fill(on) {
      new MultiplicativeHash(rand.nextLong().abs, rand.nextLong().abs, logInputDim, logOutputDim)
    }
  }


  def embed(v: WeightedVector): WeightedVector = {
    val iterator = v.homogeneous.iterator
    val rep = new Array[Double](outputDim)
    while (iterator.hasNext) {
      val count = iterator.value
      val index = iterator.index
      for (p <- positive) rep(p.hash(index)) += count
      for (n <- negative) rep(n.hash(index)) -= count
      iterator.advance()
    }
    WeightedVector(rep, v.weight)
  }
}


class MultiplicativeHash(seed: Long, offset: Long, l: Int, m: Int) {
  require(m <= l)
  val mask = if (l >= 63) Long.MaxValue else (1 << l) - 1
  val shift = l - m

  def hash(index: Int): Int = (((seed * index.toLong + offset) & mask) >> shift).toInt
}
