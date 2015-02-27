package com.massivedatascience.clusterer

import org.apache.spark.rdd.RDD
import com.massivedatascience.clusterer.util.BLAS._

trait PointOps[P <: WeightedVector, C <: WeightedVector] extends Serializable with ClusterFactory {

  val weightThreshold: Double

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


private[clusterer]
trait BasicPointOps {

  val divergence: BregmanDivergence

  @inline val weightThreshold = 1e-4
  val distanceThreshold = 1e-8

  def getCentroid: MutableWeightedVector = DenseClusterFactory.getCentroid

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

  def centerMoved(v: BregmanPoint, w: BregmanCenter): Boolean =
    distance(v, w) > distanceThreshold
}

private[clusterer]
trait NonSmoothed {
  val divergence: BregmanDivergence

  def toPoint(v: WeightedVector): BregmanPoint = {
    BregmanPoint(v, divergence.F(v.homogeneous, v.weight))
  }

  def toCenter(v: WeightedVector): BregmanCenter = {
    val h = v.homogeneous
    val w = v.weight
    val df = divergence.gradF(h, w)
    BregmanCenter(v, dot(h, df) / w - divergence.F(h, w), df)
  }
}

private[clusterer]
trait Smoothed {
  val divergence: BregmanDivergence
  val smoothingFactor: Double

  def toPoint(v: WeightedVector): BregmanPoint = {
    BregmanPoint(v, divergence.F(v.homogeneous, v.weight))
  }

  def toCenter(v: WeightedVector): BregmanCenter = {
    val h = add(v.homogeneous, smoothingFactor)
    val w = v.weight + v.homogeneous.size * smoothingFactor
    val df = divergence.gradF(h, w)
    BregmanCenter(h, w, dot(h, df) / w - divergence.F(h, w), df)
  }
}

/**
 * Implements Kullback-Leibler divergence on dense vectors in R+ ** n
 */
private[clusterer]
case object DenseKLPointOps extends BregmanPointOps with BasicPointOps with NonSmoothed {
  val divergence = RealKLDivergence
}

/**
 * Implements Generalized I-divergence on dense vectors in R+ ** n
 */
private[clusterer]
case object GeneralizedIPointOps extends BregmanPointOps with BasicPointOps with NonSmoothed {
  val divergence = GeneralizedIDivergence
}

/**
 * Implements Squared Euclidean distance on dense vectors in R ** n
 */
private[clusterer]
case object SquaredEuclideanPointOps extends BregmanPointOps with BasicPointOps with NonSmoothed {
  val divergence = SquaredEuclideanDistanceDivergence
}


/**
 * Implements logistic loss divergence on dense vectors in (0.0,1.0) ** n
 */

private[clusterer]
case object LogisticLossPointOps extends BregmanPointOps with BasicPointOps with NonSmoothed {
  val divergence = LogisticLossDivergence
}

/**
 * Implements Itakura-Saito divergence on dense vectors in R+ ** n
 */
private[clusterer]
case object ItakuraSaitoPointOps extends BregmanPointOps with BasicPointOps with NonSmoothed {
  val divergence = ItakuraSaitoDivergence
}

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
private[clusterer]
case object SparseRealKLPointOps extends BregmanPointOps with BasicPointOps with NonSmoothed {

  val divergence = RealKLDivergence

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
private[clusterer]
case object DiscreteKLPointOps extends BregmanPointOps with BasicPointOps with NonSmoothed {
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
private[clusterer]
case object DiscreteSmoothedKLPointOps extends BregmanPointOps with BasicPointOps with Smoothed {
  val divergence = NaturalKLDivergence
  val smoothingFactor = 1.0
}

private[clusterer]
case object DiscreteSimplexSmoothedKLPointOps extends BregmanPointOps with BasicPointOps with Smoothed {
  val divergence = NaturalKLSimplexDivergence
  val smoothingFactor = 1.0
}

object PointOps {
  val RELATIVE_ENTROPY = "DENSE_KL_DIVERGENCE"
  val DISCRETE_KL = "DISCRETE_DENSE_KL_DIVERGENCE"
  val SPARSE_SMOOTHED_KL = "SPARSE_SMOOTHED_KL_DIVERGENCE"
  val DISCRETE_SMOOTHED_KL = "DISCRETE_DENSE_SMOOTHED_KL_DIVERGENCE"
  val SIMPLEX_SMOOTHED_KL = "SIMPLEX_SMOOTHED_KL"
  val EUCLIDEAN = "EUCLIDEAN"
  val LOGISTIC_LOSS = "LOGISTIC_LOSS"
  val GENERALIZED_I = "GENERALIZED_I_DIVERGENCE"

  def apply(distanceFunction: String): BregmanPointOps = {
    distanceFunction match {
      case EUCLIDEAN => SquaredEuclideanPointOps
      case RELATIVE_ENTROPY => DenseKLPointOps
      case DISCRETE_KL => DiscreteKLPointOps
      case SIMPLEX_SMOOTHED_KL => DiscreteSimplexSmoothedKLPointOps
      case DISCRETE_SMOOTHED_KL => DiscreteSmoothedKLPointOps
      case SPARSE_SMOOTHED_KL => SparseRealKLPointOps
      case LOGISTIC_LOSS => LogisticLossPointOps
      case GENERALIZED_I => GeneralizedIPointOps
      case _ => throw new RuntimeException(s"unknown distance function $distanceFunction")
    }
  }
}


