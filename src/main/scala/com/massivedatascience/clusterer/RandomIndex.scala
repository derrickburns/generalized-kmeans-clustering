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
import org.apache.spark.mllib.linalg.{Vectors, Vector, SparseVector}


class TinyNaturalSet(on: Int) {
  val taken = new Array[Int](on)
  var size = 0

  def contains(ri: Int) : Boolean = {
    var i = size
    while( i > 0 ) {
      i = i - 1
      if( taken(i) == ri ) return true
    }
    false
  }

  def add(ri:Int) : Unit = {
    require(size < on)
    taken(size) = ri
    size = size + 1
  }

  def clear() : Unit = size = 0
}

class RandomIndex(dim: Int, on: Int) {
  require(on * 2 < dim)

  val tinySet = new TinyNaturalSet(on)

  def asDenseVector(v: SparseVector): Vector = {
    val iterator = v.iterator
    val rep = new Array[Double](dim)

    while (iterator.hasNext) {
      val index = iterator.index
      val value = iterator.value
      iterator.advance()
      val r = new XORShiftRandom(index)

      def spread(count: Double) {
        tinySet.clear()
        var remainingOn = on
        while (remainingOn > 0) {
          val ri = r.nextInt() % dim
          if (! tinySet.contains(ri)) {
            rep(ri) += count
            remainingOn = remainingOn - 1
            tinySet.add(ri)
          }
        }
      }
      spread(value)
      spread(-value)
    }
    Vectors.dense(rep)
  }
}
