package com.massivedatascience.clusterer

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

trait SingleKMeansClusterer extends Serializable with Logging {
  def cluster(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centers: Array[BregmanCenter]): (Double, Array[BregmanCenter], Option[RDD[(Int, Double)]])
}

