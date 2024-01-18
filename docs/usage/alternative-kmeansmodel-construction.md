---
description: How to creaate K-Means Models using the KMeansModel Helper Object
---

# Alternative KMeansModel Construction

Training a K-Means model from a set of points using `KMeans.train` is one way to create a `KMeansModel`. However, there are many others that are useful. The `KMeansModel` companion object provides a number of these constructors.

```scala
package com.massivedatascience.clusterer

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
    weights: IndexedSeq[Double]) = ???

  /**
   * Create a K-means model from given weighted vectors
   *
   * @param ops distance function
   * @param centers initial cluster centers as weighted vectors
   * @return  k-means model
   */
  def fromWeightedVectors[T <: WeightedVector : ClassTag](
    ops: BregmanPointOps,
    centers: IndexedSeq[T]) = ???

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
    seed: Long = XORShiftRandom.random.nextLong()) = ???

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
  def fromCenters[T <: WeightedVector : ClassTag](
    ops: BregmanPointOps,
    data: IndexedSeq[T],
    weights: IndexedSeq[Double],
    k: Int,
    perRound: Int,
    numPreselected: Int,
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = ???

  /**
   * Create a K-Means Model from a streaming k-means model.
   *
   * @param streamingKMeansModel mutable streaming model
   * @return immutable k-means model
   */
  def fromStreamingModel(streamingKMeansModel: StreamingKMeansModel): KMeansModel = ???

  /**
   * Create a K-Means Model from a set of assignments of points to clusters
   *
   * @param ops distance function
   * @param points initial bregman points
   * @param assignments assignments of points to clusters
   * @return
   */
  def fromAssignments[T <: WeightedVector : ClassTag](
    ops: BregmanPointOps,
    points: RDD[T],
    assignments: RDD[Int]): KMeansModel = ???

  /**
   * Create a K-Means Model using K-Means || algorithm from an RDD of Bregman points.
   *
   * @param ops distance function
   * @param data initial points
   * @param k  number of cluster centers desired
   * @param numSteps number of iterations of k-Means ||
   * @param sampleRate fractions of points to use in weighting clusters
   * @param seed random number seed
   * @return  k-means model
   */
  def usingKMeansParallel[T <: WeightedVector : ClassTag](
    ops: BregmanPointOps,
    data: RDD[T],
    k: Int,
    numSteps: Int = 2,
    sampleRate: Double = 1.0,
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = ???

  /**
   * Construct a K-Means model using the Lloyd's algorithm given a set of initial
   * K-Means models.
   *
   * @param ops distance function
   * @param data points to fit
   * @param initialModels  initial k-means models
   * @param clusterer k-means clusterer to use
   * @param seed random number seed
   * @return  the best K-means model found
   */
  def usingLloyds[T <: WeightedVector : ClassTag](
    ops: BregmanPointOps,
    data: RDD[T],
    initialModels: Seq[KMeansModel],
    clusterer: MultiKMeansClusterer = new ColumnTrackingKMeans(),
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = ???
}
```
