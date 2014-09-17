package com.rincaro.clusterer.base

import com.rincaro.clusterer.util.XORShiftRandom
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class KMeansRandom[P <: FP : ClassTag, C <: FP : ClassTag](
                                                            pointOps: PointOps[P, C], k: Int, runs: Int) extends KMeansInitializer[P, C] {

  def init(data: RDD[P], seed: Int): Array[Array[C]] = {
    // Sample all the cluster centers in one pass to avoid repeated scans
    val sample = data.takeSample(true, runs * k, new XORShiftRandom().nextInt()).withFilter(x => x.weight > 0).map(pointOps.pointToCenter).toSeq
    Array.tabulate(runs)(r => sample.slice(r * k, (r + 1) * k).toArray)
  }
}

