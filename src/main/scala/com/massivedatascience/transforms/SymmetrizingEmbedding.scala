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

import com.massivedatascience.divergence.{ BregmanDivergence, RealKullbackLeiblerSimplexDivergence }
import com.massivedatascience.linalg.{ WeightedVector, _ }

/**
 * One can create a symmetric version of any Bregman Divergence</a>
 *
 * This one is
 *
 *   distance(x,y) = KL(x,y) + KL(y,x) + (1/2) ||x-y||^2 + (1/2) || gradF(x) - gradF(y)||^2
 *
 * The embedding is simply
 *
 * x => x + gradF(x) (Lemma 1 with alpha = beta = 1)
 *
 * @see <a href="http://www-users.cs.umn.edu/~banerjee/papers/13/bregman-metric.pdf">Symmetrized Bregman Divergences and Metrics</a>
 *
 */
class SymmetrizingEmbedding(divergence: BregmanDivergence) extends Embedding {
  def embed(v: WeightedVector): WeightedVector = {
    val embedded = v.homogeneous.copy
    WeightedVector(BLAS.axpy(1.0, divergence.gradientOfConvex(embedded), embedded), v.weight)
  }
}

/**
 * A symmetric version of the standard Kullback-Leibler Divergence
 *
 *
 * @see <a href="http://www-users.cs.umn.edu/~banerjee/papers/13/bregman-metric.pdf">Symmetrized Bregman Divergences and Metrics</a>
 *
 */
case object SymmetrizingKLEmbedding extends SymmetrizingEmbedding(RealKullbackLeiblerSimplexDivergence)

