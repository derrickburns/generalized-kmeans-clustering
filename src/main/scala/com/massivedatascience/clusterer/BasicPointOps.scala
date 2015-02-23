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

import com.massivedatascience.clusterer.util.BLAS._
import org.apache.spark.mllib.linalg.Vector

case class BasicPointOps(
  divergence: BregmanDivergence,
  clusterFactory: ClusterFactory = DenseClusterFactory) extends BregmanPointOps {

  @inline val weightThreshold = 1e-4
  val distanceThreshold = 1e-8

  def getCentroid = clusterFactory.getCentroid

  /**
   * Bregman distance function
   *
   * The distance function is called in the innermost loop of the K-Means clustering algorithm.
   * Therefore, we seek to make the operation as efficient as possible.
   *
   * @param p point
   * @param c center
   * @return
   */
  def distance(p: BregmanPoint, c: BregmanCenter): Double = {
    if (c.weight <= weightThreshold) {
      Infinity
    } else if (p.weight <= weightThreshold) {
      0.0
    } else {
      val d = p.f + c.dotGradMinusF - dot(c.gradient, p.inhomogeneous)
      if (d < 0.0) 0.0 else d
    }
  }

  def toCenter(v: WeightedVector): BregmanCenter = {
    val h = v.homogeneous
    val w = v.weight
    val df = divergence.gradF(h, w)
    BregmanCenter(v, dot(h, df) / w - divergence.F(h, w), df)
  }

  def toPoint(v: WeightedVector): BregmanPoint = {
    BregmanPoint(v, divergence.F(v.homogeneous, v.weight))
  }

  def centerMoved(v: BregmanPoint, w: BregmanCenter): Boolean =
    distance(v, w) > distanceThreshold
}

/**
 * Implements Kullback-Leibler divergence on dense vectors in R+ ** n
 */
object DenseKLPointOps extends BasicPointOps(RealKLDivergence)

/**
 * Implements Generalized I-divergence on dense vectors in R+ ** n
 */
object GeneralizedIPointOps extends BasicPointOps(new GeneralizedIDivergence(GeneralLog))

/**
 * Implements Squared Euclidean distance on dense vectors in R ** n
 */
object SquaredEuclideanPointOps extends BasicPointOps(SquaredEuclideanDistanceDivergence)


/**
 * Implements logistic loss divergence on dense vectors in (0.0,1.0) ** n
 */

object LogisticLossPointOps extends BasicPointOps(LogisticLossDivergence)

/**
 * Implements Itakura-Saito divergence on dense vectors in R+ ** n
 */
object ItakuraSaitoPointOps extends BasicPointOps(new ItakuraSaitoDivergence(GeneralLog))

/**
 * Implements Kullback-Leibler divergence for sparse points in R+ ** n
 *
 * We smooth the points by adding a constant to each dimension and then re-normalize the points
 * to get points on the simplex in R+ ** n.  This works fine with n is small and
 * known.  When n is large or unknown, one often uses sparse representations.  However, smoothing
 * turns a sparse vector into a dense one, and when n is large, this space is prohibitive.
 *
 * This implementation approximates smoothing by adding a penalty equal to the sum of the
 * values of the point along dimensions that are no represented in the cluster center.
 *
 * Also, with sparse data, the centroid can be of high dimension.  To address this, we limit the
 * density of the centroid by dropping low frequency entries in the SparseCentroidProvider
 */
object SparseRealKLPointOps extends BasicPointOps(RealKLDivergence) {
    /**
   * Smooth the center using a variant Laplacian smoothing.
   *
   * The distance is roughly the equivalent of adding 1 to the center for
   * each dimension of C that is zero in C but that is non-zero in P
   *
   * @return
   */
  override def distance(p: BregmanPoint, c: BregmanCenter): Double = {
    if (c.weight <= weightThreshold) {
      Infinity
    } else if (p.weight <= weightThreshold) {
      0.0
    } else {
      val smoothed = sumMissing(c.homogeneous, p.inhomogeneous)
      val d = p.f + c.dotGradMinusF - dot(c.gradient, p.inhomogeneous) + smoothed
      if (d < 0.0) 0.0 else d
    }
  }
}


object DiscreteDenseSimplexSmoothedKLPointOps extends BasicPointOps(NaturalKullbackLeiblerSimplexDivergence)

/**
 * Implements the Kullback-Leibler divergence for dense points are in N+ ** n,
 * i.e. the entries in each vector are positive integers.
 */
object DiscreteDenseKLPointOps extends BasicPointOps(NaturalKLDivergence)

/**
 * Implements Kullback-Leibler divergence with dense points in N ** n and whose
 * weights equal the sum of the frequencies.
 *
 * Because KL divergence is not defined on
 * zero values, we smooth the points by adding 1 to all entries.
 *
 */
object DiscreteDenseSmoothedKLPointOps extends BasicPointOps(NaturalKLDivergence) {

  override def toPoint(v: WeightedVector): BregmanPoint = {
    BregmanPoint(v, divergence.F(v.homogeneous, v.weight))
  }

  override def toCenter(v: WeightedVector): BregmanCenter = {
    val h = add(v.homogeneous, 1.0)
    val w = v.weight + v.homogeneous.size
    val df = divergence.gradF(h, w)
    BregmanCenter(h, w, dot(h, df) / w - divergence.F(h, w), df)
  }
}
