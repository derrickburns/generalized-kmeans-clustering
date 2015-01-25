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
  divergence: BregmanDivergence = SquaredEuclideanDistanceDivergence,
  clusterFactory: ClusterFactory = DenseClusterFactory,
  embedding: Embedding = IdentityEmbedding) extends BregmanPointOps {

  val weightThreshold = 1e-4
  val distanceThreshold = 1e-8

  def getCentroid = clusterFactory.getCentroid

  def embed(v: Vector): Vector = embedding.embed(v)

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

  def homogeneousToPoint(h: Vector, weight: Double): BregmanPoint = {
    val embedded = embed(asInhomogeneous(h, weight))
    new BregmanPoint(embedded, weight, divergence.F(embedded))
  }

  def inhomogeneousToPoint(inh: Vector, weight: Double): BregmanPoint = {
    val embedded = embed(inh)
    new BregmanPoint(embedded, weight, divergence.F(embedded))
  }

  def toCenter(v: WeightedVector): BregmanCenter = {
    val h = v.homogeneous
    val w = v.weight
    val df = divergence.gradF(h, w)
    new BregmanCenter(h, w, dot(h, df) / w - divergence.F(h, w), df)
  }

  def toPoint(v: WeightedVector): BregmanPoint = {
    val inh = v.inhomogeneous
    new BregmanPoint(inh, v.weight, divergence.F(inh))
  }

  def centerMoved(v: BregmanPoint, w: BregmanCenter): Boolean =
    distance(v, w) > distanceThreshold
}

class DelegatedPointOps(ops: BregmanPointOps, embedding: Embedding) extends BregmanPointOps  {
  val weightThreshold = ops.weightThreshold
  def embed(v: Vector): Vector = ops.embed(embedding.embed(v))
  def getCentroid = ops.getCentroid
  def distance(p: BregmanPoint, c: BregmanCenter) = ops.distance(p,c)
  def toCenter(v: WeightedVector) = ops.toCenter(v)
  def inhomogeneousToPoint(v: Vector, weight: Double) = ops. inhomogeneousToPoint(v,weight)
  def homogeneousToPoint(v: Vector, weight: Double) = ops.homogeneousToPoint(v,weight)
  def centerMoved(v: BregmanPoint, w: BregmanCenter) = ops.centerMoved(v,w)
  def toPoint(v: WeightedVector) = ops.toPoint(v)
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
object DenseSquaredEuclideanPointOps extends BasicPointOps()

/**
 * Implements Squared Euclidean distance on sparse vectors in R ** n
 */
object SparseSquaredEuclideanPointOps extends BasicPointOps(clusterFactory = SparseClusterFactory)

/**
 * Implements Squared Euclidean distance on sparse vectors in R ** n by
 * embedding the sparse vectors into a dense space using Random Indexing
 *
 */
class RISquaredEuclideanPointOps(dimension: Int, epsilon: Double = 0.01)
  extends BasicPointOps(embedding = new RandomIndexEmbedding(dimension, epsilon))

/**
 * Implements Squared Euclidean distance on sparse vectors in R ** n by
 * embedding the sparse vectors of various dimensions.
 *
 */
object LowDimensionalRISquaredEuclideanPointOps extends RISquaredEuclideanPointOps(128)

object MediumDimensionalRISquaredEuclideanPointOps extends RISquaredEuclideanPointOps(256)

object HighDimensionalRISquaredEuclideanPointOps extends RISquaredEuclideanPointOps(512)

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
object SparseRealKLPointOps extends BasicPointOps(RealKLDivergence, SparseClusterFactory) {
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
 * zero values, we smooth the centers by adding the unit vector to each center.
 *
 */
object DiscreteDenseSmoothedKLPointOps extends BasicPointOps(NaturalKLDivergence) {
  override def toCenter(v: WeightedVector): BregmanCenter = {
    val h = add(v.homogeneous, 1.0)
    val w = v.weight + v.homogeneous.size
    val df = divergence.gradF(h, w)
    new BregmanCenter(v.homogeneous, w, dot(h, df) / w - divergence.F(h, w), df)
  }
}


/**
 * One can create a symmetric version of the Kullback Leibler Divergence that can be clustered
 * by embedding the input points (which are a simplex in R+ ** n) into a new Euclidean space R ** N.
 *
 * See http://www-users.cs.umn.edu/~banerjee/papers/13/bregman-metric.pdf
 *
 * This one is
 *
 * distance(x,y) = KL(x,y) + KL(y,x) + (1/2) ||x-y||^2 + (1/2) || gradF(x) - gradF(y)||^2
 *
 * The embedding is simply
 *
 * x => x + gradF(x) (Lemma 1 with alpha = beta = 1)
 *
 */
object GeneralizedSymmetrizedKLPointOps extends BasicPointOps(RealKLDivergence) {
  override def embed(v: Vector): Vector = {
    val embedded = v.copy
    axpy(1.0, divergence.gradF(embedded), embedded)
  }
}
