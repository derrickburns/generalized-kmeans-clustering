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
import com.massivedatascience.linalg.WeightedVector
import com.massivedatascience.util.XORShiftRandom
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

trait KMeansPredictor {

  val pointOps: BregmanPointOps

  def centers: IndexedSeq[BregmanCenter]

  def validCenters: IndexedSeq[BregmanCenter] = centers

  def clusterCenters: IndexedSeq[Vector] = validCenters.map(pointOps.toPoint).map(_.inhomogeneous)

  // operations on Vectors

  /** Return the K-means cost (sum of squared distances of points to their nearest center) for this
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

  private def validateDimension(point: WeightedVector): Unit = {
    val expectedDim = centers.head.gradient.size
    val actualDim   = point.inhomogeneous.size
    require(
      actualDim == expectedDim,
      s"Point dimension $actualDim does not match model dimension $expectedDim"
    )
  }

  def predictWeighted(point: WeightedVector): Int = {
    validateDimension(point)
    predictBregman(pointOps.toPoint(point))
  }

  /** Maps given points to their cluster indices. */
  def predictWeighted(points: RDD[WeightedVector]): RDD[Int] =
    predictBregman(points.map(pointOps.toPoint))

  def computeCostWeighted(data: RDD[WeightedVector]): Double =
    computeCostBregman(data.map(pointOps.toPoint))

  def predictClusterAndDistanceWeighted(point: WeightedVector): (Int, Double) = {
    validateDimension(point)
    predictClusterAndDistanceBregman(pointOps.toPoint(point))
  }

  // operations on BregmanPoints

  def predictBregman(point: BregmanPoint): Int = {
    val prediction = pointOps.findClosestCluster(validCenters, point)
    if (prediction >= 0 && prediction < validCenters.length) prediction
    else 0 // Fallback to first cluster if invalid
  }

  def predictBregman(points: RDD[BregmanPoint]): RDD[Int] =
    points.map { p =>
      val prediction = pointOps.findClosestCluster(validCenters, p)
      if (prediction >= 0 && prediction < validCenters.length) prediction
      else 0
    }

  def predictClusterAndDistanceBregman(point: BregmanPoint): (Int, Double) =
    pointOps.findClosest(validCenters, point)

  def computeCostBregman(data: RDD[BregmanPoint]): Double =
    data.map(p => pointOps.findClosestDistance(validCenters, p)).sum()

}

/** An immutable K-Means model
  *
  * @param pointOps
  *   distance function
  * @param centers
  *   cluster centers
  */
case class KMeansModel(pointOps: BregmanPointOps, centers: IndexedSeq[BregmanCenter])
    extends KMeansPredictor
    with Serializable
    with Logging {

  // Filter out invalid centers (zero/near-zero weight or non-finite coordinates)
  override lazy val validCenters: IndexedSeq[BregmanCenter] = {
    val valid = centers.filter { c =>
      c.weight > WeightSanitizer.MIN_VALID_WEIGHT &&
      c.gradient.toArray.forall(x => java.lang.Double.isFinite(x))
    }

    if (valid.length < centers.length) {
      logger.warn(
        s"Filtered out ${centers.length - valid.length} invalid centers " +
          s"(zero weight or non-finite coordinates). Using ${valid.length} valid centers."
      )
    }

    valid
  }

  // Validate that we have at least one valid center
  require(
    validCenters.nonEmpty,
    s"KMeansModel requires at least one valid center, got ${centers.length} total, ${validCenters.length} valid"
  )

  lazy val k: Int = validCenters.length

  /** Returns the cluster centers. N.B. These are in the embedded space where the clustering takes
    * place, which may be different from the space of the input vectors!
    */
  lazy val weightedClusterCenters: IndexedSeq[WeightedVector] = centers.map(pointOps.toPoint)
}

object KMeansModel extends Logging {

  /** Create a K-means model from given cluster centers and weights
    *
    * @param ops
    *   distance function
    * @param centers
    *   initial cluster centers in homogeneous coordinates
    * @param weights
    *   initial cluster weights
    * @return
    *   k-means model
    */
  def fromVectorsAndWeights(
    ops: BregmanPointOps,
    centers: IndexedSeq[Vector],
    weights: IndexedSeq[Double]
  ): KMeansModel = {
    val bregmanCenters =
      centers.zip(weights).map { case (c, w) => ops.toCenter(WeightedVector(c, w)) }
    new KMeansModel(ops, bregmanCenters)
  }

  /** Create a K-means model from given weighted vectors
    *
    * @param ops
    *   distance function
    * @param centers
    *   initial cluster centers as weighted vectors
    * @return
    *   k-means model
    */
  def fromWeightedVectors[T <: WeightedVector](
    ops: BregmanPointOps,
    centers: IndexedSeq[T]
  ): KMeansModel = {
    new KMeansModel(ops, centers.map(ops.toCenter))
  }

  /** Create a K-means model by selecting a set of k points at random
    *
    * @param ops
    *   distance function
    * @param k
    *   number of centers desired
    * @param dim
    *   dimension of space
    * @param weight
    *   initial weight of points
    * @param seed
    *   random number seed
    * @return
    *   k-means model
    */
  def usingRandomGenerator(
    ops: BregmanPointOps,
    k: Int,
    dim: Int,
    weight: Double,
    seed: Long = XORShiftRandom.random.nextLong()
  ): KMeansModel = {

    val random         = new XORShiftRandom(seed)
    val centers        = IndexedSeq.fill(k)(Vectors.dense(Array.fill(dim)(random.nextGaussian())))
    val bregmanCenters = centers.map(c => ops.toCenter(WeightedVector(c, weight)))
    new KMeansModel(ops, bregmanCenters)
  }

  /** Create a K-Means model using the KMeans++ algorithm on an initial set of candidate centers
    *
    * @param ops
    *   distance function
    * @param data
    *   initial candidate centers
    * @param weights
    *   initial weights
    * @param k
    *   number of clusters desired
    * @param perRound
    *   number of candidates to add per round
    * @param numPreselected
    *   initial sub-sequence of candidates to always select
    * @param seed
    *   random number seed
    * @return
    *   k-means model
    */
  def fromCenters[T <: WeightedVector](
    ops: BregmanPointOps,
    data: IndexedSeq[T],
    weights: IndexedSeq[Double],
    k: Int,
    perRound: Int,
    numPreselected: Int,
    seed: Long = XORShiftRandom.random.nextLong()
  ): KMeansModel = {

    val candidates = data.map(ops.toCenter)
    val bregmanCenters =
      new KMeansPlusPlus(ops).goodCenters(seed, candidates, weights, k, perRound, numPreselected)
    new KMeansModel(ops, bregmanCenters)
  }

  /** Create a K-Means Model from a streaming k-means model.
    *
    * @param streamingKMeansModel
    *   mutable streaming model
    * @return
    *   immutable k-means model
    */
  def fromStreamingModel(streamingKMeansModel: StreamingKMeansModel): KMeansModel = {
    new KMeansModel(streamingKMeansModel.pointOps, streamingKMeansModel.centers)
  }

  /** Create a K-Means Model from a set of assignments of points to clusters.
    *
    * @param ops
    *   distance function
    * @param points
    *   RDD of weighted vectors representing the data points
    * @param assignments
    *   RDD of cluster indices (0-based) corresponding to each point
    * @param expectedNumClusters
    *   Optional expected number of clusters. If provided and greater than 0, the method will ensure
    *   exactly this many clusters are returned.
    * @return
    *   A KMeansModel with the computed cluster centers
    * @throws IllegalArgumentException
    *   if inputs are invalid or misaligned
    * @throws org.apache.spark.SparkException
    *   if RDD operations fail
    * @throws IllegalStateException
    *   if no valid clusters are found or expected number of clusters cannot be satisfied
    */
  def fromAssignments[T <: WeightedVector: ClassTag](
    ops: BregmanPointOps,
    points: RDD[T],
    assignments: RDD[Int],
    expectedNumClusters: Int = -1
  ): KMeansModel = {

    // Input validation
    require(points != null, "Points RDD must not be null")
    require(assignments != null, "Assignments RDD must not be null")
    require(
      expectedNumClusters == -1 || expectedNumClusters > 0,
      "Expected number of clusters must be either -1 (unspecified) or positive"
    )
    require(
      !points.partitioner.exists(_.numPartitions != assignments.partitions.length),
      "Points and assignments must have the same number of partitions"
    )

    // Cache RDDs to avoid recomputation
    val cachedPoints      = points.cache()
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
      val pointsCount      = cachedPoints.count()
      val assignmentsCount = cachedAssignments.count()
      if (pointsCount != assignmentsCount) {
        throw new IllegalArgumentException(
          s"Points (size=$pointsCount) and assignments (size=$assignmentsCount) must have the same number of elements"
        )
      }

      // Process assignments and compute centroids
      val (centroids, clusterSizes, hasInvalidAssignments) = {
        // Zip with index to preserve ordering and detect misaligned data
        val zipped = cachedAssignments
          .zipWithIndex()
          .map(_.swap)
          .join(cachedPoints.zipWithIndex().map(_.swap))
          .map { case (_, (clusterIdx, point)) => (clusterIdx, point) }

        // Check for invalid cluster indices
        val allIndices     = cachedAssignments.collect()
        val hasNegative    = allIndices.exists(_ < 0)
        val maxIndex       = if (allIndices.nonEmpty) allIndices.max else -1
        val hasOutOfBounds = expectedNumClusters > 0 && maxIndex >= expectedNumClusters

        // Throw exception if there are invalid assignments
        if (hasNegative) {
          throw new IllegalArgumentException(
            s"Invalid cluster assignment: found negative cluster index in assignments"
          )
        }
        if (hasOutOfBounds) {
          throw new IllegalArgumentException(
            s"Invalid cluster assignment: found cluster index $maxIndex but expected at most ${expectedNumClusters - 1}"
          )
        }

        // Filter out invalid cluster indices, count points per cluster, and compute centroids
        val validAssignments = zipped.filter { case (index, _) => index >= 0 }

        // Count points per cluster
        val sizes = validAssignments
          .map { case (clusterIdx, _) => (clusterIdx, 1L) }
          .reduceByKey(_ + _)
          .collectAsMap()
          .toMap
          .withDefaultValue(0L)

        // Compute centroids
        val centers = validAssignments
          .aggregateByKey(ops.make())(
            (centroid, pt) => centroid.add(pt),
            (c1, c2) => c1.add(c2)
          )

        (centers, sizes, hasNegative || hasOutOfBounds)
      }

      // Process centroids and handle empty clusters
      val (bregmanCenters, emptyClusters) = {
        val centersWithIndices = centroids
          .map { case (clusterIdx, vector) =>
            val size = clusterSizes.getOrElse(clusterIdx, 0L)
            if (size == 0) {
              (clusterIdx, None)
            } else {
              (clusterIdx, Some(ops.toCenter(vector.asImmutable)))
            }
          }
          .collect()
          .toMap

        val validCenters = centersWithIndices.collect { case (_, Some(center)) => center }
        val emptyIndices = centersWithIndices.collect { case (idx, None) => idx }.toSet

        (validCenters, emptyIndices)
      }

      // Handle empty clusters if expected number of clusters is specified
      val finalCenters = if (expectedNumClusters > 0) {
        val actualNumClusters = bregmanCenters.size

        if (actualNumClusters == 0) {
          throw new IllegalStateException(
            s"No valid clusters found. All points might be filtered out. " +
              s"Expected $expectedNumClusters clusters but found 0."
          )
        }

        if (actualNumClusters > expectedNumClusters) {
          throw new IllegalStateException(
            s"Found $actualNumClusters clusters but expected at most $expectedNumClusters. " +
              "This indicates invalid cluster indices in the assignments."
          )
        }

        if (actualNumClusters < expectedNumClusters) {
          // Add empty clusters to match the expected number
          val existingIndices = centroids.keys.collect().toSet
          val newClusters = (0 until expectedNumClusters)
            .filterNot(existingIndices.contains)
            .map { idx =>
              // For empty clusters, we could initialize with a zero vector or use another strategy
              // Here we use the first valid center as a fallback, or a zero vector if no centers exist
              if (bregmanCenters.nonEmpty) {
                // Clone the first center but with zero weight as a fallback
                val firstCenter = bregmanCenters.head
                ops.toCenter(WeightedVector(firstCenter.inhomogeneous, 0.0))
              } else {
                // This should not happen due to previous checks, but added for safety
                // Create a zero vector with a reasonable default dimension
                ops.toCenter(WeightedVector(Vectors.dense(Array.fill(10)(0.0)), 0.0))
              }
            }

          (bregmanCenters ++ newClusters).toIndexedSeq
        } else {
          bregmanCenters.toIndexedSeq
        }
      } else {
        // If no expected number of clusters is specified, just use what we have
        if (bregmanCenters.isEmpty) {
          throw new IllegalStateException(
            "No valid clusters found. All points might be filtered out."
          )
        }
        bregmanCenters.toIndexedSeq
      }

      // Log warnings about empty clusters if any
      if (emptyClusters.nonEmpty) {
        val emptyIndicesStr = emptyClusters.toSeq.sorted.mkString(", ")
        val warningMsg =
          s"Found ${emptyClusters.size} empty cluster(s) with indices: $emptyIndicesStr. " +
            "These clusters will be initialized with zero-weighted centers."

        // Log the warning using SLF4J logger
        logger.warn(warningMsg)
      }

      new KMeansModel(ops, finalCenters)
    } finally {
      // Clean up cached RDDs
      cachedPoints.unpersist(blocking = false)
      cachedAssignments.unpersist(blocking = false)
    }
  }

  /** Create a K-Means Model using K-Means || algorithm from an RDD of WeightedVectors
    *
    * @param ops
    *   distance function
    * @param data
    *   initial points
    * @param k
    *   number of cluster centers desired
    * @param numSteps
    *   number of iterations of k-Means ||
    * @param sampleRate
    *   fractions of points to use in weighting clusters
    * @param seed
    *   random number seed
    * @return
    *   k-means model
    */
  def usingKMeansParallel[T <: WeightedVector](
    ops: BregmanPointOps,
    data: RDD[T],
    k: Int,
    numSteps: Int = 2,
    sampleRate: Double = 1.0,
    seed: Long = XORShiftRandom.random.nextLong()
  ): KMeansModel = {

    val points = data.map(ops.toPoint)
    val initializedCenters =
      new KMeansParallel(numSteps, sampleRate).init(ops, points, k, None, 1, seed)
    require(initializedCenters.nonEmpty, "KMeansParallel.init returned empty centers")
    val centers = initializedCenters.head
    new KMeansModel(ops, centers)
  }

  /** Construct a K-Means model of a set of points using Lloyd's algorithm given a set of initial
    * K-Means models.
    *
    * @param ops
    *   distance function
    * @param data
    *   points to fit
    * @param initialModels
    *   initial k-means models
    * @param clusterer
    *   k-means clusterer to use
    * @param seed
    *   random number seed
    * @return
    *   the best K-means model found
    */
  def usingClusterer[T <: WeightedVector: ClassTag](
    ops: BregmanPointOps,
    data: RDD[T],
    initialModels: Seq[KMeansModel],
    maxIterations: Int,
    clusterer: MultiKMeansClusterer = new ColumnTrackingKMeans(),
    seed: Long = XORShiftRandom.random.nextLong()
  ): KMeansModel = {

    val initialCenters = initialModels.map { model =>
      if (model.pointOps == ops)
        model.centers
      else
        fromAssignments(ops, data, data.map(model.predictWeighted)).centers
    }
    val points = data.map(ops.toPoint)
    val ClusteringWithDistortion(_, centers) =
      clusterer.best(maxIterations, ops, points, initialCenters)
    new KMeansModel(ops, centers)
  }
}
