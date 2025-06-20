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

import com.massivedatascience.divergence._
import com.massivedatascience.linalg._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory

/**
 * A point with an additional single Double value that is used in distance computation.
 *
 */
case class BregmanPoint(homogeneous: Vector, weight: Double, f: Double) extends WeightedVector {
  lazy val inhomogeneous = asInhomogeneous(homogeneous, weight)
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
  lazy val inhomogeneous = asInhomogeneous(homogeneous, weight)
}

object BregmanPoint {
  def apply(v: WeightedVector, f: Double): BregmanPoint =
    new BregmanPoint(v.homogeneous, v.weight, f)
}

object BregmanCenter {
  def apply(v: WeightedVector, dotGradMinusF: Double, gradient: Vector): BregmanCenter =
    new BregmanCenter(v.homogeneous, v.weight, dotGradMinusF, gradient)
}

/**
 * An enrichment to the Bregman divergence that supports expedient distance calculations.
 *
 * This is done by pre-computing and caching certain values.
 *
 * For points, these values are stored in BregmanPoint objects.
 *
 * For cluster centers, these values are stored in BregmanCenter objects.
 *
 */
trait BregmanPointOps extends Serializable with ClusterFactory {
  type P = BregmanPoint
  type C = BregmanCenter

  val divergence: BregmanDivergence

  @inline val weightThreshold = 1e-4

  val distanceThreshold = 1e-8

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
  def centerMoved(v: P, w: C): Boolean =
    distance(v, w) > distanceThreshold

  /**
   * Return the index of the closest point in `centers` to `point`, as well as its distance.
   *
   * N.B. This is inner loop code, so until the Scala compiler is smart enough, we eschew
   * the more functional approach.
   *
   * @param centers centers
   * @param point point
   * @param f function that creates the result
   * @param initialDistance initial best distance
   * @param initialIndex  initial index of best cluster
   * @tparam T   output type of function provided
   * @return best index and best distance after function f application
   */
  def findClosestInfo[T](
    centers: IndexedSeq[C],
    point: P,
    f: (Int, Double) => T,
    initialDistance: Double = Infinity,
    initialIndex: Int = -1): T = {

    var bestDistance = initialDistance
    var bestIndex = initialIndex
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
    f(bestIndex, bestDistance)
  }

  def findClosest(centers: IndexedSeq[C], point: P): (Int, Double) =
    findClosestInfo[(Int, Double)](centers, point, (i: Int, d: Double) => (i, d))

  def findClosestCluster(centers: IndexedSeq[C], point: P): Int =
    findClosestInfo[Int](centers, point, (i: Int, _: Double) => i)

  def findClosestDistance(centers: IndexedSeq[C], point: P): Double =
    findClosestInfo[Double](centers, point, (_: Int, d: Double) => d)

  def distortion(data: RDD[P], centers: IndexedSeq[C]): Double =
    data.aggregate(0.0)(_ + findClosestDistance(centers, _), _ + _)

  /**
   * Return the K-means cost of a given point against the given cluster centers.
   */
  def pointCost(centers: IndexedSeq[C], point: P): Double = findClosestDistance(centers, point)

  def make(index: Int = -1): MutableWeightedVector = DenseClusterFactory.make(index)

  /**
   * Bregman distance function
   *
   * The distance function is called in the innermost loop of the K-Means clustering algorithm.
   * Therefore, we seek to make the operation as efficient as possible.
   *
   * @param p point
   * @param c center
   * @return distance
   */
  def distance(p: P, c: C): Double = {
    weightThreshold match {
      case x: Double if c.weight <= x => Infinity
      case x: Double if p.weight <= x => 0.0
      case _ =>
        val d = p.f + c.dotGradMinusF - BLAS.dot(c.gradient, p.inhomogeneous)
        if (d < 0.0) {
          // Negative distances can indicate numerical instability or implementation issues
          // Log a warning for debugging purposes but return 0.0 as the minimum valid distance
          if (d < -1e-10) {
            // Only log if the negative value is significant (not just numerical precision issue)
            val logger = LoggerFactory.getLogger(getClass.getName)
            logger.warn(s"Negative distance computed: $d between point and center")
          }
          0.0
        } else d
    }
  }
}

/**
 * A factory for BregmanPoints and BregmanCenters given the Bregman divergence function.
 */
trait PointCenterFactory {

  /**
   * The divergence to use when manufacturing points and centers.
   */
  val divergence: BregmanDivergence

  /**
   * Create a BregmanPoint from a WeightedVector
   * @param v weighted vector
   * @return a BregmanPoint for the given Bregman divergence
   */
  def toPoint(v: WeightedVector): BregmanPoint

  /**
   * Create a BregmanCenter from a WeightedVector
   * @param v weighted vector
   * @return a BregmanCenter for the given Bregman divergence
   */
  def toCenter(v: WeightedVector): BregmanCenter
}

/**
 * Create BregmanCenters without smoothing.
 */
trait NonSmoothedPointCenterFactory extends PointCenterFactory {

  def toPoint(v: WeightedVector): BregmanPoint = {
    BregmanPoint(v, divergence.convexHomogeneous(v.homogeneous, v.weight))
  }

  def toCenter(v: WeightedVector): BregmanCenter = {
    val h = v.homogeneous
    val w = v.weight
    if (w < 0.0) {
      throw new IllegalArgumentException(s"Weight cannot be negative, got: $w")
    }
    val df = divergence.gradientOfConvexHomogeneous(h, w)
    val dotGradMinusF = if (w == 0.0) 0.0 else BLAS.dot(h, df) / w - divergence.convexHomogeneous(h, w)
    BregmanCenter(v, dotGradMinusF, df)
  }
}

/**
 * Create BregmanPoints and BregmanCenters by smoothing the centers.
 *
 * Smooth by adding one to each coordinate.
 *
 */
trait SmoothedPointCenterFactory extends PointCenterFactory {
  val smoothingFactor: Double

  def toPoint(v: WeightedVector): BregmanPoint = {
    BregmanPoint(v, divergence.convexHomogeneous(v.homogeneous, v.weight))
  }

  def toCenter(v: WeightedVector): BregmanCenter = {
    val h = BLAS.add(v.homogeneous, smoothingFactor)
    val w = v.weight + v.homogeneous.size * smoothingFactor
    if (w < 0.0) {
      throw new IllegalArgumentException(s"Smoothed weight cannot be negative, got: $w")
    }
    val df = divergence.gradientOfConvexHomogeneous(h, w)
    val dotGradMinusF = if (w == 0.0) 0.0 else BLAS.dot(h, df) / w - divergence.convexHomogeneous(h, w)
    BregmanCenter(h, w, dotGradMinusF, df)
  }
}

/**
 * Implements Kullback-Leibler divergence on dense vectors in R+ ** n
 */
private[clusterer] case object DenseKLPointOps extends BregmanPointOps
    with NonSmoothedPointCenterFactory {
  val divergence = RealKLDivergence
}

/**
 * Implements Generalized I-divergence on dense vectors in R+ ** n
 */
private[clusterer] case object GeneralizedIPointOps extends BregmanPointOps
    with NonSmoothedPointCenterFactory {
  val divergence = GeneralizedIDivergence
}

/**
 * Implements Squared Euclidean distance on dense vectors in R ** n
 */
private[clusterer] case object SquaredEuclideanPointOps extends BregmanPointOps
    with NonSmoothedPointCenterFactory {
  val divergence = SquaredEuclideanDistanceDivergence
}

/**
 * Implements logistic loss divergence on dense vectors in (0.0,1.0) ** n
 */

private[clusterer] case object LogisticLossPointOps extends BregmanPointOps
    with NonSmoothedPointCenterFactory {
  val divergence = LogisticLossDivergence
}

/**
 * Implements Itakura-Saito divergence on dense vectors in R+ ** n
 */
private[clusterer] case object ItakuraSaitoPointOps extends BregmanPointOps
    with NonSmoothedPointCenterFactory {
  val divergence = ItakuraSaitoDivergence
}

/**
 * Implements the Kullback-Leibler divergence for dense points are in N+ ** n,
 * i.e. the entries in each vector are positive integers.
 */
private[clusterer] case object DiscreteKLPointOps extends BregmanPointOps
    with NonSmoothedPointCenterFactory {
  val divergence = NaturalKLDivergence
}

/**
 * Implements Kullback-Leibler divergence with dense points in N ** n and whose
 * weights equal the sum of the frequencies.
 *
 * Because KL divergence is not defined on
 * zero values, we smooth the points by adding 1 to all entries.
 *
 */
private[clusterer] case object DiscreteSmoothedKLPointOps extends BregmanPointOps
    with SmoothedPointCenterFactory {
  val divergence = NaturalKLDivergence
  val smoothingFactor = 1.0
}

private[clusterer] case object DiscreteSimplexSmoothedKLPointOps extends BregmanPointOps
    with SmoothedPointCenterFactory {
  val divergence = NaturalKLSimplexDivergence
  val smoothingFactor = 1.0
}

object BregmanPointOps {
  val RELATIVE_ENTROPY = "DENSE_KL_DIVERGENCE"
  val DISCRETE_KL = "DISCRETE_DENSE_KL_DIVERGENCE"
  @deprecated("convert data to dense vectors first", "1.2.0")
  val SPARSE_SMOOTHED_KL = "SPARSE_SMOOTHED_KL_DIVERGENCE"
  val DISCRETE_SMOOTHED_KL = "DISCRETE_DENSE_SMOOTHED_KL_DIVERGENCE"
  val SIMPLEX_SMOOTHED_KL = "SIMPLEX_SMOOTHED_KL"
  val EUCLIDEAN = "EUCLIDEAN"
  val LOGISTIC_LOSS = "LOGISTIC_LOSS"
  val GENERALIZED_I = "GENERALIZED_I_DIVERGENCE"
  val ITAKURA_SAITO = "ITAKURA_SAITO"

  def apply(distanceFunction: String): BregmanPointOps = {
    distanceFunction match {
      case EUCLIDEAN => SquaredEuclideanPointOps
      case RELATIVE_ENTROPY => DenseKLPointOps
      case DISCRETE_KL => DiscreteKLPointOps
      case SIMPLEX_SMOOTHED_KL => DiscreteSimplexSmoothedKLPointOps
      case DISCRETE_SMOOTHED_KL => DiscreteSmoothedKLPointOps
      case LOGISTIC_LOSS => LogisticLossPointOps
      case GENERALIZED_I => GeneralizedIPointOps
      case ITAKURA_SAITO => ItakuraSaitoPointOps
      case _ => throw new RuntimeException(s"unknown distance function $distanceFunction")
    }
  }

  def apply(d: BregmanDivergence): BregmanPointOps =
    new BregmanPointOps with NonSmoothedPointCenterFactory {
      val divergence = d
    }

  def apply(d: BregmanDivergence, factor: Double): BregmanPointOps =
    new BregmanPointOps with SmoothedPointCenterFactory {
      val divergence = d
      val smoothingFactor = factor
    }
}
