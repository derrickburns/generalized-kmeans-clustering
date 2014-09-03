package com.rincaro.clusterer.base

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 *
 * A re-write of the  KMeansModel of the Spark Mllib clusterer. This version allows for a general distance function.
 *
 */
class GeneralizedKMeansModel[T, P<:FP[T]:ClassTag, C<:FP[T]:ClassTag](val pointOps: PointOps[P, C, T], val clusterCenters: Array[C]) extends Serializable {

  val kmeans = new GeneralizedKMeans[T,P,C](pointOps)
  /** Total number of clusters. */
  def k: Int = clusterCenters.length

  def findClosest(point: P): (Int, Double) = kmeans.findClosest(clusterCenters, point)


  /** Return the cluster index that a given point belongs to. */
  def predict(raw: Array[Double]) : Int = {
    kmeans.findClosest(clusterCenters, pointOps.userToPoint(raw,None))._1
  }

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  def computeCost(raw: RDD[(Option[T],Array[Double])]): Double = {
    raw.map(p => kmeans.pointCost(clusterCenters, pointOps.userToPoint(p._2,p._1))).sum()
  }
}
