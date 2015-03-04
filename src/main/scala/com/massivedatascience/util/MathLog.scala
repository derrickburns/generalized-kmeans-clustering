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

package com.massivedatascience.util

/**
 * A function that implements a variant of the natural logarithm
 */
sealed trait MathLog extends Serializable {

  /**
   *  The natural logarithm extended to be defined on 0.0.
   *
   * @param x  input
   * @return natural logarithm of x, or 0.0 if x == 0
   */
  def log(x: Double): Double
}

/**
 * The natural logarithm function implemented using that Math.log function.
 */
case object GeneralLog extends MathLog {
  @inline
  override def log(x: Double): Double = if (x == 0.0 || x == 1.0) 0.0 else Math.log(x)
}

/**
 * The natural logarithm, but defined as 0.0 on input 0.0 logarithm function implemented
 * using a table lookup on integral arguments,
 * and the Math.log function when entries are non-integral or not found.
 */
case object DiscreteLog extends MathLog {
  private[this] val cacheSize = 1000

  private val logTable = new Array[Double](cacheSize)

   def logg(d: Double): Double = {
    if (d == 0.0 || d == 1.0) {
      0.0
    } else {
      if (d < logTable.length) {
        val x = d.toInt
        if (x.toDouble == d) {
          if (logTable(x) == 0.0) logTable(x) = Math.log(x)
          logTable(x)
        } else {
          Math.log(d)
        }
      } else {
        Math.log(d)
      }
    }
  }

  override def log(d: Double): Double = {
    d match {
      case 0.0 => 0.0
      case 1.0 => 0.0
      case x if x < logTable.length =>
        val i = d.toInt
        if (i.toDouble == d) {
          if (logTable(i) == 0.0) logTable(i) = Math.log(d)
          logTable(i)
        } else {
          Math.log(d)
        }
      case x => Math.log(x)
    }
  }
}