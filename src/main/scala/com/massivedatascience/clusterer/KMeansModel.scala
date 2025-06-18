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

import com.massivedatascience.clusterer.MultiKMeansClusterer.ClusteringWithDistortion
import com.massivedatascience.linalg.{ MutableWeightedVector, WeightedVector }
import com.massivedatascience.util.XORShiftRandom
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

trait KMeansPredictor {

  val pointOps: BregmanPointOps

  def centers: IndexedSeq[BregmanCenter]

  def clusterCenters: IndexedSeq[Vector] = centers.map(pointOps.toPoint).map(_.inhomogeneous)

  // operations on Vectors

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  def computeCost(data: RDD[Vector]): Double =
    computeCostWeighted(data.map(WeightedVector.apply))

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

  // operations on WeightedVectors

  def predictWeighted(point: WeightedVector): Int = predictWeighted(pointOps.toPoint(point))

  /** Maps given points to their cluster indices. */
  def predictWeighted(points: RDD[WeightedVector]): RDD[Int] =
    predictBregman(points.map(p => pointOps.toPoint(p)))

  def computeCostWeighted(data: RDD[WeightedVector]): Double =
    computeCostBregman(data.map(p => pointOps.toPoint(p)))

  def predictClusterAndDistanceWeighted(point: WeightedVector): (Int, Double) =
    predictClusterAndDistanceBregman(pointOps.toPoint(point))

  // operations on BregmanPoints

  def predictBregman(point: BregmanPoint): Int = pointOps.findClosestCluster(centers, point)

  def predictBregman(points: RDD[BregmanPoint]): RDD[Int] =
    points.map(p => pointOps.findClosestCluster(centers, p))

  def predictClusterAndDistanceBregman(point: BregmanPoint): (Int, Double) =
    pointOps.findClosest(centers, point)

  def computeCostBregman(data: RDD[BregmanPoint]): Double =
    data.map(p => pointOps.findClosestDistance(centers, pointOps.toPoint(p))).sum()

}

/**
 * An immutable K-Means model
 *
 * @param pointOps  distance function
 * @param centers cluster centers
 */
case class KMeansModel(pointOps: BregmanPointOps, centers: IndexedSeq[BregmanCenter])
    extends KMeansPredictor with Serializable {
  
  // Validate that centers are not empty
  require(centers.nonEmpty, "KMeansModel requires non-empty centers")

  lazy val k: Int = centers.length

  /**
   * Returns the cluster centers.  N.B. These are in the embedded space where the clustering
   * takes place, which may be different from the space of the input vectors!
   */
  lazy val weightedClusterCenters: IndexedSeq[WeightedVector] = centers.map(pointOps.toPoint)
}

object KMeansModel {

  /**
   * Create a K-means model from given cluster centers and weights
   *
   * @param ops distance function
   * @param centers initial cluster centers in homogeneous coordinates
   * @param weights initial cluster weights
   * @return  k-means model
   */
  def fromVectorsAndWeights(
    ops: BregmanPointOps,
    centers: IndexedSeq[Vector],
    weights: IndexedSeq[Double]): KMeansModel = {
    val bregmanCenters = centers.zip(weights).map { case (c, w) => ops.toCenter(WeightedVector(c, w)) }
    new KMeansModel(ops, bregmanCenters)
  }

  /**
   * Create a K-means model from given weighted vectors
   *
   * @param ops distance function
   * @param centers initial cluster centers as weighted vectors
   * @return  k-means model
   */
  def fromWeightedVectors[T <: WeightedVector](ops: BregmanPointOps, centers: IndexedSeq[T]): KMeansModel = {
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
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = {

    val random = new XORShiftRandom(seed)
    val centers = IndexedSeq.fill(k)(Vectors.dense(Array.fill(dim)(random.nextGaussian())))
    val bregmanCenters = centers.map(c => ops.toCenter(WeightedVector(c, weight)))
    new KMeansModel(ops, bregmanCenters)
  }

  /**
   * Create a K-Means model using the KMeans++ algorithm on an initial set of candidate centers
   *
   * @param ops distance function
   * @param data initial candidate centers
   * @param weights initial weights
   * @param k number of clusters desired
   * @param perRound number of candidates to add per round
   * @param numPreselected initial sub-sequence of candidates to always select
   * @param seed random number seed
   * @return  k-means model
   */
  def fromCenters[T <: WeightedVector](
    ops: BregmanPointOps,
    data: IndexedSeq[T],
    weights: IndexedSeq[Double],
    k: Int,
    perRound: Int,
    numPreselected: Int,
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = {

    val candidates = data.map(ops.toCenter)
    val bregmanCenters = new KMeansPlusPlus(ops).goodCenters(seed, candidates, weights,
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
    new KMeansModel(streamingKMeansModel.pointOps, streamingKMeansModel.centers)
  }

  /**
   * Create a K-Means Model from a set of assignments of points to clusters.
   *
   * @param ops distance function
   * @param points RDD of weighted vectors representing the data points
   * @param assignments RDD of cluster indices (0-based) corresponding to each point
   * @return A KMeansModel with the computed cluster centers
   * @throws IllegalArgumentException if inputs are invalid or misaligned
   * @throws org.apache.spark.SparkException if RDD operations fail
   */
  def fromAssignments[T <: WeightedVector: ClassTag](
    ops: BregmanPointOps,
    points: RDD[T],
    assignments: RDD[Int]): KMeansModel = {

    // Input validation
    require(points != null, "Points RDD must not be null")
    require(assignments != null, "Assignments RDD must not be null")
    require(!points.partitioner.exists(_.numPartitions != assignments.partitions.length),
      "Points and assignments must have the same number of partitions")

    // Cache RDDs to avoid recomputation
    val cachedPoints = points.cache()
    val cachedAssignments = assignments.cache()

    try {
      // Check if RDDs are non-empty
      if (cachedPoints.isEmpty()) {
        throw new IllegalArgumentException("Points RDD must not be empty")
      }
      if (cachedAssignments.isEmpty()) {
        throw new IllegalArgumentException("Assignments RDD must not be empty")
      }

      // Verify RDDs have the same number of elements
      val pointsCount = cachedPoints.count()
      val assignmentsCount = cachedAssignments.count()
      if (pointsCount != assignmentsCount) {
        throw new IllegalArgumentException(
          s"Points (size=$pointsCount) and assignments (size=$assignmentsCount) must have the same number of elements")
      }

      // Process assignments and compute centroids
      val centroids: RDD[(Int, MutableWeightedVector)] = {
        // Zip with index to preserve ordering and detect misaligned data
        val zipped = cachedAssignments.zipWithIndex().map(_.swap)
          .join(cachedPoints.zipWithIndex().map(_.swap))
          .map { case (_, (clusterIdx, point)) => (clusterIdx, point) }

        // Filter out invalid cluster indices and compute centroids
        zipped.filter { case (index, _) => index >= 0 }
          .aggregateByKey(ops.make())(
            (centroid, pt) => centroid.add(pt),
            (c1, c2) => c1.add(c2)
          )
      }

      // Collect and validate centroids
      val bregmanCenters = centroids.map { case (clusterIdx, vector) =>
        if (vector.count == 0) {
          throw new IllegalStateException(
            s"Empty cluster detected at index $clusterIdx. This can happen if all points are assigned to a single cluster.")
        }
        ops.toCenter(vector.asImmutable)
      }

      val centers = bregmanCenters.collect().toIndexedSeq
      if (centers.isEmpty) {
        throw new IllegalStateException("No valid clusters found. All points might be filtered out.")
      }

      new KMeansModel(ops, centers)
    } finally {
      // Clean up cached RDDs
      cachedPoints.unpersist(blocking = false)
      cachedAssignments.unpersist(blocking = false)
    }
  }

  /**
   * Create a K-Means Model using K-Means || algorithm from an RDD of WeightedVectors
   *
   * @param ops distance function
   * @param data initial points
   * @param k  number of cluster centers desired
   * @param numSteps number of iterations of k-Means ||
   * @param sampleRate fractions of points to use in weighting clusters
   * @param seed random number seed
   * @return  k-means model
   */
  def usingKMeansParallel[T <: WeightedVector](
    ops: BregmanPointOps,
    data: RDD[T],
    k: Int,
    numSteps: Int = 2,
    sampleRate: Double = 1.0,
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = {

    val points = data.map(ops.toPoint)
    val centers = new KMeansParallel(numSteps, sampleRate).init(ops, points, k, None, 1, seed).head
    new KMeansModel(ops, centers)
  }

  /**
   * Construct a K-Means model of a set of points using Lloyd's algorithm given a set of initial
   * K-Means models.
   *
   * @param ops distance function
   * @param data points to fit
   * @param initialModels  initial k-means models
   * @param clusterer k-means clusterer to use
   * @param seed random number seed
   * @return  the best K-means model found
   */
  def usingClusterer[T <: WeightedVector: ClassTag](
    ops: BregmanPointOps,
    data: RDD[T],
    initialModels: Seq[KMeansModel],
    maxIterations: Int,
    clusterer: MultiKMeansClusterer = new ColumnTrackingKMeans(),
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = {

    val initialCenters = initialModels.map { model =>
      if (model.pointOps == ops)
        model.centers
      else
        fromAssignments(ops, data, data.map(model.predictWeighted)).centers
    }
    val points = data.map(ops.toPoint)
    val ClusteringWithDistortion(_, centers) = clusterer.best(maxIterations, ops, points, initialCenters)
    new KMeansModel(ops, centers)
  }
}
