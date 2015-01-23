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

import com.massivedatascience.clusterer.util.XORShiftRandom
import org.apache.spark.mllib.linalg.{Vectors, Vector, SparseVector, DenseVector}

trait Embedding extends Serializable {
  def embed(v: Vector): Vector
}


object DirectEmbedding extends Embedding {
  def embed(v: Vector): Vector = {
   v match {
     case sv: SparseVector => Vectors.dense(v.toArray)
     case dv: DenseVector => dv
   }
  }
}

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
 * @param on number on positive and negative half-spaces to occupy
 */
class RandomIndexEmbedding(dim: Int, on: Int) extends Embedding {
  require(on * 2 < dim)

  val tinySet = new TinyIntSet(2 * on)

  def embed(v: Vector): Vector = {
    val iterator = v.iterator
    val rep = new Array[Double](dim)

    while (iterator.hasNext) {
      val count = iterator.value
      val random = new XORShiftRandom(iterator.index)
      iterator.advance()

      var remainingOn = on
      while (remainingOn > 0) {
        val ri = random.nextInt() % dim
        if (!tinySet.contains(ri)) {
          rep(ri) += count
          remainingOn = remainingOn - 1
          tinySet.add(ri)
        }
      }

      var remainingOff = on
      while (remainingOff > 0) {
        val ri = random.nextInt() % dim
        if (!tinySet.contains(ri)) {
          rep(ri) -= count
          remainingOff = remainingOff - 1
          tinySet.add(ri)
        }
      }
      tinySet.clear()
    }
    Vectors.dense(rep)
  }
}

/**
 * A tiny set of integer values
 *
 * @param maxSize the maximum number of elements in the set
 */

class TinyIntSet(maxSize: Int) {
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