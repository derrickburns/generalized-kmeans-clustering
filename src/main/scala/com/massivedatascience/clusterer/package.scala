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


  def preview[T](data: RDD[T], header: String = ""): RDD[T] = {
    if (data.sparkContext.getConf.getBoolean("com.massivedatascience.clusterer.preview", false)) {
      println(header)
      val head = data.take(10)
      head.map(println)
    }
    data
  }


  val Infinity = Double.MaxValue
  val Unknown = -1.0
  val empty: Vector = new SparseVector(0, Array[Int](), Array[Double]())

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

  implicit class RichVector(v: Vector) {
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

  val DefaultTerminationCondition = { s: BasicStats => s.getRound > 20 ||
    s.getNonEmptyClusters == 0 ||
    s.getMovement / s.getNonEmptyClusters < 1.0E-5
  }

  trait PointOps[P <: WeightedVector, C <: WeightedVector] extends Serializable {
    def distance(p: P, c: C): Double

    /**
     * convert a vector in homogeneous coordinates into a point
     * @param v
     * @param weight
     * @return
     */
    def homogeneousToPoint(v: Vector, weight: Double): P


    /**
     * convert a vector in inhomogeneous coordinates into a point
     * @param v
     * @param weight
     * @return
     */
    def inhomogeneousToPoint(v: Vector, weight: Double): P

    /**
     * converted a weighted vector to a point
     * @param v
     * @return
     */
    def toPoint(v: WeightedVector): P

    /**
     * converted a weighted vector to a center
     * @param v
     * @return
     */
    def toCenter(v: WeightedVector): C

    /**
     * determine if a center has moved appreciably
     * @param v
     * @param w
     * @return
     */
    def centerMoved(v: P, w: C): Boolean


    /**
     * convert a weighted point to an inhomogeneous vector
     * @param c
     * @return
     */
    def toInhomogeneous(c: WeightedVector): Vector = c.inhomogeneous

    /**
     * convert a weighted point to an homogeneous vector and weight
     * @param c
     * @return
     */
    def toHomogeneous(c: WeightedVector): (Vector, Double) = (c.homogeneous, c.weight)

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

    def findClosestCluster(centers: Array[C], point: P): Int = {
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

    def distortion(data: RDD[P], centers: Array[C]) = {
      data.mapPartitions { points =>
        Array(points.foldLeft(0.0) { case (total, p) =>
          total + findClosest(centers, p)._2
        }).iterator
      }.reduce(_ + _)
    }

    /**
     * Return the K-means cost of a given point against the given cluster centers.
     */
    def pointCost(centers: IndexedSeq[C], point: P): Double = findClosest(centers, point)._2
  }

}
