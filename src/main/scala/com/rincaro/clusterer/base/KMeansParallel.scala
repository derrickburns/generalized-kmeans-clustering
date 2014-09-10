package com.rincaro.clusterer.base

import com.rincaro.clusterer.util.XORShiftRandom
import org.apache.spark.Logging
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


/**
 * This is a re-write of the KMeans Parallel implementation from Spark Mllib.
 *
 * @param k  number of clusters
 * @param runs  number of runs
 * @param initializationSteps  how many iterations of the initialization step to run
 * @param numPartitions  number of data partitions
 * @param pointOps  distance function
 * @tparam T  user data type
 * @tparam P  point type
 * @tparam C  center type
 */
class KMeansParallel[T, P <: FP[T] : ClassTag, C <: FP[T] : ClassTag](k: Int, runs: Int, initializationSteps: Int, numPartitions: Int, pointOps: PointOps[P,C,T]) extends KMeansInitializer[T,P,C] with Logging {

  val kmeans = new GeneralizedKMeans[T,P, C](pointOps)

  /**
   * Initialize `runs` sets of cluster centers using the k-means|| algorithm by Bahmani et al.
   * (Bahmani et al., Scalable K-Means++, VLDB 2012). This is a variant of k-means++ that tries
   * to find  dissimilar cluster centers by starting with a random center and then doing
   * passes where more centers are chosen with probability proportional to their squared distance
   * to the current cluster set. It results in a provable approximation to an optimal clustering.
   *
   * The original paper can be found at http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf.
   *
   * @param data the RDD of points
   * @param seed the random number generator seed
   * @return
   */
  def init(data: RDD[P], seed: Int): Array[Array[C]] = {
    log.debug("kmeans parallel on {} points" + data.count())

    // randomly select one center per run, putting each into a separate array buffer
    val sample = data.takeSample(withReplacement=true, runs, seed).toSeq.map(pointOps.pointToCenter)
    val centers: Array[ArrayBuffer[C]] = Array.tabulate(runs)(r => ArrayBuffer(sample(r)))

    // add at most 2k points per step
    for (step <- 0 until initializationSteps) {
      if (log.isInfoEnabled) showCenters(centers, step)
      val centerArrays = centers.map{ x: ArrayBuffer[C] => x.toArray}
      val bcCenters = data.sparkContext.broadcast(centerArrays)
      for ((r, p) <- choose(data, seed, step, bcCenters)) {
        centers(r) += pointOps.pointToCenter(p)
      }
      bcCenters.unpersist()
    }

    val bcCenters = data.sparkContext.broadcast(centers.map(_.toArray))
    val result = finalCenters(data, bcCenters, seed)
    bcCenters.unpersist()
    result
  }

  private def showCenters(centers: Array[ArrayBuffer[C]], step: Int) {
    log.info("step {}", step)
    for (run <- 0 until runs) {
      log.info("final: run {} has {} centers", run, centers.length)
    }
  }

  /**
   * Randomly choose at most 2 * k  additional cluster centers by weighting them by their distance to the closest cluster.
   * Note, unlike the Spark version, this one uses the weights of the points in addition to the distances from the closest
   * cluster center.
   *
   * @param data  the RDD of points
   * @param seed  random generator seed
   * @param step  which step of the selection process
   * @return  array of (run, point)
   */
  private def choose(data: RDD[P], seed: Int, step: Int, bcCenters: Broadcast[Array[Array[C]]]): Array[(Int, P)] = {
    // compute the weighted distortion for each run
    val sumCosts = data.flatMap {
      point =>
        val centers = bcCenters.value
        for (r <- 0 until runs) yield (r, point.weight * kmeans.pointCost(centers(r), point))
    }.reduceByKey(_ + _).collectAsMap()

    // choose points in proportion to ratio of weighted cost to weighted distortion
    data.mapPartitionsWithIndex {
      (index, points: Iterator[P]) =>
        val centers = bcCenters.value
        val range = 0 until runs
        val rand = new XORShiftRandom(seed ^ (step << 16) ^ index)
        points.flatMap { p =>
          range.filter { r =>
            rand.nextDouble() < 2.0 * p.weight * kmeans.pointCost(centers(r), p) * k / sumCosts(r)
          }.map((_, p))
        }
    }.collect()
  }

  /**
   * Reduce sets of candidate cluster centers to at most k points per set using KMeansPlusPlus. Weight the points
   * by the distance to the closest cluster center.
   *
   * @param data  original points
   * @param bcCenters  array of sets of candidate centers
   * @param seed  random number seed
   * @return  array of sets of cluster centers
   */
  private def finalCenters(data: RDD[P], bcCenters: Broadcast[Array[Array[C]]], seed: Int): Array[Array[C]] = {
    // for each (run, cluster) compute the sum of the weights of the points in the cluster
    val weightMap = data.flatMap {
      point =>
        val centers = bcCenters.value
        for (r <- 0 until runs) yield ((r, kmeans.findClosest(centers(r), point)._1), point.weight)
    }.reduceByKey(_ + _).collectAsMap()

    val centers = bcCenters.value
    val kmeansPlusPlus = new KMeansPlusPlus(pointOps)
    val finalCenters = (0 until runs).map {
      r =>
        val myCenters = centers(r).toArray
        log.info("run {} has {} centers", r, myCenters.length)
        val myWeights = (0 until myCenters.length).map(i => weightMap.getOrElse((r, i), 0.0)).toArray
        val initialCenters = kmeansPlusPlus.getCenters(data.sparkContext, seed, myCenters, myWeights, if (k > myCenters.length) myCenters.length else k, 30)

        kmeans.cluster(
          data=data.sparkContext.parallelize(myCenters.map(pointOps.centerToPoint)),
          maxIterations = 30,
          centers = Array(initialCenters)
        )._1(0)
    }
    finalCenters.toArray
  }
}
