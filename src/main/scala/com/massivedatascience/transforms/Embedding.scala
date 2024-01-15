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
import org.apache.spark.mllib.linalg.{ Vectors, Vector }

/**
 * An embedding of vectors into an alternative space.  Typically, embeddings are used
 * to lower the dimension of the data in such a way that preserves distances using the
 * given divergence so that clustering can proceed expeditiously.
 */
trait Embedding extends Serializable {
  /**
   * Tranform a weighted vector into another space
   * @param v the weighted vector
   * @return the transformed vector
   */
  def embed(v: WeightedVector): WeightedVector = WeightedVector(embed(v.homogeneous), v.weight)
  def embed(v: Vector): Vector
  def embed(v: VectorIterator): Vector
}

/**
 * The identity embedding
 */
case object IdentityEmbedding extends Embedding {
  def embed(v: VectorIterator): Vector = v.toVector
  def embed(v: Vector): Vector = v
}

/**
 * The embedding that transforms all points to dense vectors.
 */
case object DenseEmbedding extends Embedding {
  def embed(v: VectorIterator): Vector = Vectors.dense(v.toVector.toArray)
  def embed(v: Vector): Vector = Vectors.dense(v.toArray)
}

/**
 * The embedding that averages adjacent values, thereby lowering the dimension of
 * the data by two
 */
case object HaarEmbedding extends Embedding {
  def embed(v: VectorIterator): Vector = embed(v.toVector)
  def embed(v: Vector): Vector = Vectors.dense(HaarWavelet.average(v.toArray))
}

object Embedding {
  val IDENTITY_EMBEDDING = "IDENTITY"
  val DENSE_EMBEDDING = "DENSE"
  val HAAR_EMBEDDING = "HAAR"
  val SYMMETRIZING_KL_EMBEDDING = "SYMMETRIZING_KL_EMBEDDING"
  val LOW_DIMENSIONAL_RI = "LOW_DIMENSIONAL_RI"
  val MEDIUM_DIMENSIONAL_RI = "MEDIUM_DIMENSIONAL_RI"
  val HIGH_DIMENSIONAL_RI = "HIGH_DIMENSIONAL_RI"

  private[this] val lowDimension = 64
  private[this] val mediumDimension = 256
  private[this] val highDimension = 1024
  private[this] val epsilon = 0.01

  def apply(embeddingName: String): Embedding = {
    embeddingName match {
      case IDENTITY_EMBEDDING => IdentityEmbedding
      case DENSE_EMBEDDING => DenseEmbedding
      case LOW_DIMENSIONAL_RI => new RandomIndexEmbedding(lowDimension, epsilon, seed = 0)
      case MEDIUM_DIMENSIONAL_RI => new RandomIndexEmbedding(mediumDimension, epsilon, seed = 0)
      case HIGH_DIMENSIONAL_RI => new RandomIndexEmbedding(highDimension, epsilon, seed = 0)
      case HAAR_EMBEDDING => HaarEmbedding
      case SYMMETRIZING_KL_EMBEDDING => SymmetrizingKLEmbedding
      case _ => throw new RuntimeException(s"unknown embedding name $embeddingName")
    }
  }
}

