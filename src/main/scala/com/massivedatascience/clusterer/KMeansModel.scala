/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This code is a modified version of the original Spark 1.0.2 K-Means implementation.
 */

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.KMeansInitializer
import com.massivedatascience.clusterer.util.XORShiftRandom
import org.apache.spark.SparkContext._
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.rdd.RDD

trait KMeansPredictor {

  val pointOps: BregmanPointOps

  def centers: IndexedSeq[BregmanCenter]

  lazy val clusterCenters: IndexedSeq[Vector] = centers.map(pointOps.toPoint).map(_.inhomogeneous)

  def predictWeighted(point: WeightedVector): Int = pointOps.findClosestCluster(centers, pointOps.toPoint(point))

  def predictClusterAndDistanceWeighted(point: WeightedVector): (Int, Double) =
    pointOps.findClosest(centers, pointOps.toPoint(point))

  /** Maps given points to their cluster indices. */
  def predictWeighted(points: RDD[WeightedVector]): RDD[Int] = {
    points.map(p => pointOps.findClosestCluster(centers, pointOps.toPoint(p)))
  }

  def computeCostWeighted(data: RDD[WeightedVector]): Double =
    data.map(p => pointOps.findClosest(centers, pointOps.toPoint(p))._2).sum()

  /** Returns the cluster index that a given point belongs to. */
  def predict(point: Vector): Int = predictWeighted(WeightedVector(point))

  /** Returns the cluster index that a given point belongs to. */
  def predictClusterAndDistance(point: Vector): (Int, Double) =
    predictClusterAndDistanceWeighted(WeightedVector(point))

  /** Maps given points to their cluster indices. */
  def predict(points: RDD[Vector]): RDD[Int] =
    predictWeighted(points.map(WeightedVector.apply))

  /** Maps given points to their cluster indices. */
  def predict(points: JavaRDD[Vector]): JavaRDD[java.lang.Integer] =
    predict(points.rdd).toJavaRDD().asInstanceOf[JavaRDD[java.lang.Integer]]

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  def computeCost(data: RDD[Vector]): Double =
    computeCostWeighted(data.map(point => WeightedVector(point)))
}

/**
 * An immutable K-Means model
 *
 * @param pointOps  distance function
 * @param centers cluster centers
 */
case class KMeansModel(pointOps: BregmanPointOps, centers: IndexedSeq[BregmanCenter])
  extends KMeansPredictor with Serializable {

  lazy val k: Int = centers.length


  /** Returns the cluster centers.  N.B. These are in the embedded space where the clustering
    * takes place, which may be different from the space of the input vectors!
    */
  lazy val weightedClusterCenters: IndexedSeq[WeightedVector] = centers.map(pointOps.toPoint)
}


object KMeansModel {

  /**
   * Create an initial model from given cluster centers and weights
   *
   * @param ops distance function
   * @param centers initial cluster centers in homogeneous coordinates
   * @param weights initial cluster weights
   * @return  k-means model
   */
  def fromVectorsAndWeights(ops: BregmanPointOps, centers: Array[Vector], weights: Array[Double]) = {
    val bregmanCenters = centers.zip(weights).map { case (c, w) => ops.toCenter(WeightedVector(c, w))}
    new KMeansModel(ops, bregmanCenters)
  }

  /**
   * Create an initial model from given cluster centers and weights
   *
   * @param ops distance function
   * @param centers initial cluster centers as weighted vectors
   * @return  k-means model
   */
  def fromWeightedVectors(ops: BregmanPointOps, centers: Array[WeightedVector]) = {
    new KMeansModel(ops, centers.map(ops.toCenter))
  }

  /**
   * Create a K-means model by selecting a set of k points at random
   *
   * @param ops distance function
   * @param k number of centers desired
   * @param dim dimension of space
   * @param weight initial weight of points
   * @param seed random number seed
   * @return  k-means model
   */
  def usingRandomGenerator(ops: BregmanPointOps,
    k: Int,
    dim: Int,
    weight: Double,
    seed: Long = XORShiftRandom.random.nextLong()) = {

    val random = new XORShiftRandom(seed)
    val centers = Array.fill(k)(Vectors.dense(Array.fill(dim)(random.nextGaussian())))
    val bregmanCenters = centers.map { c => ops.toCenter(WeightedVector(c, weight))}
    new KMeansModel(ops, bregmanCenters)
  }

  /**
   * Create a K-Means model using the KMeans++ algorithm on an initial set of candidate centers
   *
   * @param ops distance function
   * @param candidates initial candidate centers
   * @param weights initial weights
   * @param k number of clusters desired
   * @param perRound number of candidates to add per round
   * @param numPreselected initial sub-sequence of candidates to always select
   * @param seed random number seed
   * @return  k-means model
   */
  def fromCenters(
    ops: BregmanPointOps,
    candidates: Array[BregmanCenter],
    weights: Array[Double],
    k: Int,
    perRound: Int,
    numPreselected: Int,
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = {

    val bregmanCenters = new KMeansPlusPlus(ops).getCenters(seed, candidates, weights,
      k, perRound, numPreselected)
    new KMeansModel(ops, bregmanCenters)
  }

  /**
   * Create a K-Means Model from a streaming k-means model.
   *
   * @param streamingKMeansModel mutable streaming model
   * @return immutable k-means model
   */
  def fromStreamingModel(streamingKMeansModel: StreamingKMeansModel): KMeansModel = {
    val currentCenters = streamingKMeansModel.centers
    new KMeansModel(streamingKMeansModel.pointOps, currentCenters)
  }

  /**
   * Create a K-Means Model from a set of assignments of points to clusters
   *
   * @param ops distance function
   * @param points initial bregman points
   * @param assignments assignments of points to clusters
   * @return
   */
  def fromAssignments(
    ops: BregmanPointOps,
    points: RDD[BregmanPoint],
    assignments: RDD[Int]): KMeansModel = {

    val centroids = assignments.zip(points).filter(_._1 >= 0).aggregateByKey(ops.getCentroid)(
      (centroid, pt) => centroid.add(pt),
      (c1, c2) => c1.add(c2)
    )

    val bregmanCenters = centroids.map(p => ops.toCenter(p._2.asImmutable))
    new KMeansModel(ops, bregmanCenters.collect().toIndexedSeq)
  }


  /**
   * Create a K-Means Model using K-Means || algorithm from an RDD of Bregman points.
   *
   * @param ops distance function
   * @param points initial bregman points
   * @param k  number of cluster centers desired
   * @param numSteps number of iterations of k-Means ||
   * @param sampleRate fractions of points to use in weighting clusters
   * @param seed random number seed
   * @return  k-means model
   */
  def usingKMeansParallel(
    ops: BregmanPointOps,
    points: RDD[BregmanPoint],
    k: Int,
    numSteps: Int = 2,
    sampleRate: Double = 1.0,
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = {

    val centers = new KMeansParallel(numSteps, sampleRate).init(ops, points, k, None, 1, seed)(0)
    new KMeansModel(ops, centers)
  }

  /**
   * Construct a K-Means model using the K-Means algorithm given a set of initial
   * K-Means models.
   *
   * @param ops distance function
   * @param points points to fit
   * @param initialModels  initial k-means models
   * @param clusterer k-means clusterer to use
   * @param seed random number seed
   * @return  k-means model
   */
  def usingLloyds(
    ops: BregmanPointOps,
    points: RDD[BregmanPoint],
    initialModels: Seq[KMeansModel],
    clusterer: MultiKMeansClusterer = new ColumnTrackingKMeans(),
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = {

    val initialCenters = initialModels.map(_.centers).toArray
    val results = clusterer.cluster(ops, points, initialCenters)
    val (_, finalCenters, _) = MultiKMeansClusterer.bestOf(results)
    results.foreach(_._3.map(_.unpersist()))
    new KMeansModel(ops, finalCenters)
  }
}
