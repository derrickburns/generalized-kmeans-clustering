package com.rincaro.clusterer.base

import org.apache.spark.rdd.RDD

trait KMeansInitializer[P <: FP, C <: FP] extends Serializable {
  def init(data: RDD[P], seed: Int): Array[Array[C]]
}
