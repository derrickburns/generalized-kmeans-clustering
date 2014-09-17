package com.massivedatascience.clusterer.base

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD


class KMeansModel(specific: GeneralizedKMeansModel[_, _]) {

  val k: Int = specific.k

  /** Returns the cluster index that a given point belongs to. */
  def predict(point: Vector): Int = specific.predict(point)


  /** Maps given points to their cluster indices. */
  def predict(points: RDD[Vector]): RDD[Int] = specific.predict(points)


  /** Maps given points to their cluster indices. */
  def predict(points: JavaRDD[Vector]): JavaRDD[java.lang.Integer] = specific.predict(points)

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  def computeCost(data: RDD[Vector]): Double = specific.computeCost(data)

  def clusterCenters: Array[Vector] = specific.clusterCenters
}
