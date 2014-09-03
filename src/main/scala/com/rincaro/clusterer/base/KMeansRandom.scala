package com.rincaro.clusterer.base

import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 *
 * A re-write of the KMeans Random implementation of the Spark Mllib clusterer. This one uses a general distance function.
 *
 * @param pointOps  the distance function
 * @param k  the number of cluster centers to start with
 * @param runs the number of runs
 * @tparam T  user data type
 * @tparam P  point type
 * @tparam C  center type
 */
class KMeansRandom[T, P <: FP[T] : ClassTag, C <: FP[T] : ClassTag](pointOps: PointOps[P,C, T], k: Int, runs: Int) extends KMeansInitializer[T, P,C] {

  def init(data: RDD[P], seed: Int): Array[Array[C]] = {
    // Sample all the cluster centers in one pass to avoid repeated scans
    val sample = data.takeSample(withReplacement=true, runs * k, seed).map(pointOps.pointToCenter).toSeq
    Array.tabulate(runs)(r => sample.slice(r * k, (r + 1) * k).toArray)
  }
}

