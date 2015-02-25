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

package com.massivedatascience

import com.massivedatascience.clusterer.util.BLAS._
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.rdd.RDD

package object clusterer {

  val Infinity = Double.MaxValue
  val Unknown = -1.0
  val empty: DenseVector = new DenseVector(Array[Double]())

  def asInhomogeneous(homogeneous: Vector, weight: Double) = {
    val x = homogeneous.copy
    scal(1.0 / weight, x)
    x
  }

  def asHomogeneous(inhomogeneous: Vector, weight: Double) = {
    val x = inhomogeneous.copy
    scal(weight, x)
    x
  }

  implicit class RichVector(val v: Vector) extends AnyVal {
    def iterator: VectorIterator = {
      v match {
        case s: SparseVector => new SparseVectorIterator(s)
        case d: DenseVector => new DenseVectorIterator(d)
      }
    }

    def negativeIterator: VectorIterator = {
      v match {
        case s: SparseVector => new NegativeSparseVectorIterator(s)
        case d: DenseVector => new NegativeDenseVectorIterator(d)
      }
    }
  }

  type TerminationCondition = BasicStats => Boolean

  val DefaultTerminationCondition = { s: BasicStats => s.getRound > 40 ||
    s.getNonEmptyClusters == 0 ||
    s.getMovement / s.getNonEmptyClusters < 1.0E-5
  }

  trait PointOps[P <: WeightedVector, C <: WeightedVector] extends Serializable  with ClusterFactory {

    val weightThreshold : Double

    def distance(p: P, c: C): Double

    /**
     * converted a weighted vector to a point
     * @param v input vector
     * @return weighted vector
     */
    def toPoint(v: WeightedVector): P

    /**
     * converted a weighted vector to a center
     * @param v input vector
     * @return center
     */
    def toCenter(v: WeightedVector): C

    /**
     * determine if a center has moved appreciably
     * @param v point
     * @param w center
     * @return is the point distinct from the center
     */
    def centerMoved(v: P, w: C): Boolean

    /**
     * Return the index of the closest point in `centers` to `point`, as well as its distance.
     */
    def findClosest(centers: IndexedSeq[C], point: P): (Int, Double) = {
      var bestDistance = Infinity
      var bestIndex = 0
      var i = 0
      val end = centers.length
      while (i < end && bestDistance > 0.0) {
        val d = distance(point, centers(i))
        if (d < bestDistance) {
          bestIndex = i
          bestDistance = d
        }
        i = i + 1
      }
      (bestIndex, bestDistance)
    }

    def findClosestCluster(centers: IndexedSeq[C], point: P): Int = {
      var bestDistance = Infinity
      var bestIndex = 0
      var i = 0
      val end = centers.length
      while (i < end && bestDistance > 0.0) {
        val d = distance(point, centers(i))
        if (d < bestDistance) {
          bestIndex = i
          bestDistance = d
        }
        i = i + 1
      }
      bestIndex
    }

    def distortion(data: RDD[P], centers: IndexedSeq[C]) =
      data.aggregate(0.0)(_ + findClosest(centers, _)._2, _ + _)

    /**
     * Return the K-means cost of a given point against the given cluster centers.
     */
    def pointCost(centers: IndexedSeq[C], point: P): Double = findClosest(centers, point)._2
  }

  /**
   * A point with an additional single Double value that is used in distance computation.
   *
   */
  case class BregmanPoint(homogeneous: Vector, weight: Double, f: Double) extends WeightedVector {
    lazy val inhomogeneous = clusterer.asInhomogeneous(homogeneous, weight)
  }

  /**
   * A cluster center with an additional Double and an additional vector containing the gradient
   * that are used in distance computation.
   *
   * @param homogeneous point
   * @param dotGradMinusF  center dot gradient(center) - f(center)
   * @param gradient gradient of center
   */
  case class BregmanCenter(
    homogeneous: Vector,
    weight: Double,
    dotGradMinusF: Double,
    gradient: Vector) extends WeightedVector {
    lazy val inhomogeneous = clusterer.asInhomogeneous(homogeneous, weight)
  }

  object BregmanPoint {
    def apply(v: WeightedVector, f: Double): BregmanPoint =
      new BregmanPoint(v.homogeneous, v.weight, f)
  }

  object BregmanCenter {
    def apply(v: WeightedVector, dotGradMinusF: Double, gradient: Vector): BregmanCenter =
      new BregmanCenter(v.homogeneous, v.weight, dotGradMinusF, gradient)
  }

  trait BregmanPointOps extends PointOps[BregmanPoint, BregmanCenter]
}
