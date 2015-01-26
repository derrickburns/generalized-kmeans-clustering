package com.massivedatascience.clusterer

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD


object RecursiveKMeans {

  import KMeans._

  def sparseTrain(raw: RDD[Vector], k: Int): KMeansModel = {
    KMeans.train(raw, k, 
      embeddingNames = List(LOW_DIMENSIONAL_RI, MEDIUM_DIMENSIONAL_RI, HIGH_DIMENSIONAL_RI))
  }

  def timeSeriesTrain(raw: RDD[Vector], k: Int): KMeansModel = {
    val dim = raw.first().toArray.length
    require(dim > 0)
    val maxDepth = Math.floor(Math.log(dim) / Math.log(2.0)).toInt
    val target = Math.max(maxDepth - 4, 0)
    KMeans.trainViaSubsampling(raw, k, depth = target)
  }
}
