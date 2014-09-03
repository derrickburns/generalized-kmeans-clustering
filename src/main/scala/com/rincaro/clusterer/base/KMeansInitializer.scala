package com.rincaro.clusterer.base

import org.apache.spark.rdd.RDD

/**
 * A trait that produces a set of initial cluster centers for a K-Means clustering.
 *
 * @tparam T  user data type
 * @tparam P  point type
 * @tparam C  center type
 */
trait KMeansInitializer[T, P <: FP[T], C <: FP[T]] extends Serializable {
  def init(data: RDD[P], seed: Int): Array[Array[C]]
}
