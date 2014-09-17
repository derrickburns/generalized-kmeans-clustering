package com.massivedatascience.clusterer.base

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD


trait MultiKMeansClusterer[P <: FP, C <: FP] extends Serializable with Logging {
  def cluster(data: RDD[P], centers: Array[Array[C]]): (Double, GeneralizedKMeansModel[P, C])

}
