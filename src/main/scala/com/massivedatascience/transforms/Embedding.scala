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

import com.massivedatascience.linalg.{ WeightedVector, _ }
import org.apache.spark.mllib.linalg.{ DenseVector, SparseVector }

trait Embedding extends Serializable {
  def embed(v: WeightedVector): WeightedVector
}

case object IdentityEmbedding extends Embedding {
  def embed(v: WeightedVector): WeightedVector = v
}

case object DenseEmbedding extends Embedding {
  def embed(v: WeightedVector): WeightedVector = {
    v match {
      case sv: SparseVector => WeightedVector(v.homogeneous.toArray, v.weight)
      case dv: DenseVector => dv
    }
  }
}

case object HaarEmbedding extends Embedding {
  def embed(raw: WeightedVector): WeightedVector =
    WeightedVector(HaarWavelet.average(raw.homogeneous.toArray), raw.weight)
}

