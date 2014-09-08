package com.rincaro.clusterer.base

import com.rincaro.clusterer.util.XORShiftRandom
import org.apache.spark.{SparkContext, Logging}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


/**
 *
 * The KMeans++ initialization algorithm
 *
 * @param pointOps distance function
 * @tparam T the user point type
 * @tparam P point type
 * @tparam C center type
 */
class KMeansPlusPlus[T, P <: FP[T] : ClassTag, C <: FP[T] : ClassTag](pointOps: PointOps[P, C, T]) extends Serializable with Logging {

  /**
   * We will maintain for each point the distance to its closest cluster center.  Since only one center is added on each
   * iteration, recomputing the closest cluster center only requires computing the distance to the new cluster center if
   * that distance is less than the closest cluster center.
   */
  type FatPoint = (P, Int, Double, Double)

  /**
   * K-means++ on the weighted point set `points`. This first does the K-means++
   * initialization procedure and then rounds of Lloyd's algorithm.
   */

  def cluster(
               sc: SparkContext,
               seed: Int,
               points: Array[C],
               weights: Array[Double],
               k: Int,
               maxIterations: Int,
               numPartitions: Int): Array[C] = {
    val centers = getCenters(sc, seed, points, weights, k, numPartitions, 1)
    val kmeans = new GeneralizedKMeans[T, P, C](pointOps)
    val (newCenters, _) = kmeans.cluster(sc.parallelize(points.map(pointOps.centerToPoint)), maxIterations, Array(centers))
    newCenters(0)
  }

  /**
   * Select centers in rounds.  On each round, select 'perRound' center, with probability of selection
   * equal to the as the product of the given weights and distance to the closest cluster center of the previous round.
   *
   * @param sc the Spark context
   * @param seed a random number seed
   * @param points  the candidate centers
   * @param weights  the weights on the candidate centers
   * @param k  the total number of centers to select
   * @param numPartitions the number of data partitions to use
   * @param perRound the number of centers to add per round
   * @return   an array of at most k cluster centers
   */
  def getCenters(sc: SparkContext, seed: Int, points: Array[C], weights: Array[Double], k: Int, numPartitions: Int, perRound: Int): Array[C] = {
    assert(points.length > 0)
    assert(k > 0)
    assert(numPartitions > 0)
    assert(perRound > 0)

    if (points.length < k) log.warn("number of clusters requested {} exceeds number of points {}", k, points.length)
    val centers = new ArrayBuffer[C](k)
    val rand = new XORShiftRandom(seed)
    centers += points(pickWeighted(rand, weights))
    log.info("starting kMeansPlusPlus initialization on {} points", points.length)

    var more = true
    var fatPoints = initialFatPoints(points, weights)
    fatPoints = updateDistances(fatPoints, centers, 0, 1)

    while (centers.length < k && more) {
      val chosen = choose(fatPoints, seed ^ (centers.length << 24), rand, perRound)
      val newCenters = chosen.map {
        points(_)
      }
      fatPoints = updateDistances(fatPoints, newCenters, 0, newCenters.length)
      log.info("chose {} points", chosen.length)
      for (index <- chosen) {
        log.info("  center({}) = points({})", centers.length, index)
        centers += points(index)
      }
      more = chosen.nonEmpty
    }
    val result = centers.take(k)
    log.info("completed kMeansPlusPlus initialization with {} centers of {} requested", result.length, k)
    result.toArray
  }

  /**
   * Choose points
   *
   * @param fatPoints points to choose from
   * @param seed  random number seed
   * @param rand  random number generator
   * @param count number of points to choose
   * @return indices of chosen points
   */
  def choose(fatPoints: Array[FatPoint], seed: Int, rand: XORShiftRandom, count: Int) =
    (0 until count).flatMap { x => pickCenter(rand, fatPoints.iterator)}.map { _._2}

  /**
   * Create initial fat points with weights given and infinite distance to closest cluster center.
   * @param points points
   * @param weights weights of points
   * @return fat points with given weighs and infinite distance to closest cluster center
   */
  def initialFatPoints(points: Array[C], weights: Array[Double]): Array[FatPoint] =
    points.map(pointOps.centerToPoint).zipWithIndex.zip(weights).map(x => (x._1._1, x._1._2, x._2, 1.0 / 0.0)).toArray

  /**
   * Update the distance of each point to its closest cluster center, given only the given cluster centers that were
   * modified.
   *
   * @param points set of candidate initial cluster centers
   * @param center new cluster center
   * @return  points with their distance to closest to cluster center updated
   */

  def updateDistances(points: Array[FatPoint], center: Seq[C], from: Int, to: Int): Array[FatPoint] =
    points.map { p =>
      var i = from
      var dist = p._4
      val point = p._1
      while (i < to) {
        val d = pointOps.distance(point, center(i))
        dist = if( d < dist ) d else dist
        i = i + 1
      }
      (p._1, p._2, p._3, dist)
    }

  /**
   * Pick a point at random, weighing the choices by the given weight vector.  Return -1 if all weights are 0.0
   *
   * @param rand  random number generator
   * @param weights  the weights of the points
   * @return the index of the point chosen
   */
  def pickWeighted(rand: XORShiftRandom, weights: Array[Double]): Int = {
    val r = rand.nextDouble() * weights.sum
    var i = 0
    var curWeight = 0.0
    while (i < weights.length && curWeight < r) {
      assert(weights(i) >= 0.0)
      curWeight += weights(i)
      i += 1
    }
    if (i == 0) throw new IllegalArgumentException("all weights are zero")
    i - 1
  }

  /**
   *
   * Select point randomly with probability weighted by the product of the weight and the distance
   *
   * @param rand random number generator
   * @return
   */
  def pickCenter(rand: XORShiftRandom, fatPoints: Iterator[FatPoint]): Array[FatPoint] = {
    var cumulative = 0.0
    val rangeAndIndexedPoints = fatPoints map { z =>
      val weightedDistance = z._3 * z._4
      val from = cumulative
      cumulative = cumulative + weightedDistance
      (from, cumulative, z._1, z._2)
    }
    val pts = rangeAndIndexedPoints.toArray
    val total = pts.last._2
    val r = rand.nextDouble() * total
    for (w <- pts if w._1 <= r && r < w._2) yield (w._3, w._4, 1.0, total)
  }
}