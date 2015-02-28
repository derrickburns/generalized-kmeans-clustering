package com.massivedatascience.transforms

import com.massivedatascience.divergence.{BregmanDivergence, RealKullbackLeiblerSimplexDivergence}
import com.massivedatascience.linalg.{WeightedVector, _}

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
    WeightedVector(BLAS.axpy(1.0, divergence.gradF(embedded), embedded), v.weight)
  }
}

case object SymmetrizingKLEmbedding extends SymmetrizingEmbedding(RealKullbackLeiblerSimplexDivergence)

