package com.massivedatascience.clusterer.base

import com.massivedatascience.clusterer.metrics.FastEuclideanOps
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging

import scala.reflect.ClassTag

import org.apache.spark.mllib.linalg.Vector


/**
 * This object offers a Java callable interface that this API compatible
 * with org.apache.spark.mllib.clustering.KMeans.
 *
 */
object KMeans extends Logging {
  // Initialization mode names
  val RANDOM = "random"
  val K_MEANS_PARALLEL = "k-means||"

  def train(data: RDD[Vector], k: Int, maxIterations: Int, runs: Int, mode: String): KMeansModel =
    new KMeansModel(doTrain(new FastEuclideanOps)(data, k, maxIterations, runs, mode)._2)

  /**
   * Trains a k-means model using specified parameters and the default values for unspecified.
   */
  def train(data: RDD[Vector], k: Int, maxIterations: Int): KMeansModel =
    new KMeansModel(doTrain(new FastEuclideanOps)(data, k, maxIterations)._2)


  /**
   * Trains a k-means model using specified parameters and the default values for unspecified.
   */
  def train(data: RDD[Vector], k: Int, maxIterations: Int, runs: Int): KMeansModel =
    new KMeansModel(doTrain(new FastEuclideanOps)(data, k, maxIterations, runs)._2)


  def doTrain[P <: FP, C <: FP](pointOps: PointOps[P, C])(
    raw: RDD[Vector],
    k: Int = 2,
    maxIterations: Int = 20,
    runs: Int = 1,
    initializationMode: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    epsilon: Double = 1e-4)(implicit ctag: ClassTag[C], ptag: ClassTag[P])
  : (Double, GeneralizedKMeansModel[P, C]) = {

    val initializer = if (initializationMode == RANDOM) {
      new KMeansRandom(pointOps, k, runs)
    } else {
      new KMeansParallel(pointOps, k, runs, initializationSteps, 1)
    }
    val data = (raw map { vals => pointOps.vectorToPoint(vals)}).cache()
    val centers = initializer.init(data, 0)
    new GeneralizedKMeans(pointOps, maxIterations).cluster(data, centers)
  }
}