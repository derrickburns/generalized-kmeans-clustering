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
 * This code is a modified version of the original Spark 1.2 Streaming K-Means implementation.
 */

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.StreamingKMeans.SimpleWeightedVector
import com.massivedatascience.linalg.{ BLAS, WeightedVector }
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.dstream.DStream

import scala.collection.immutable.IndexedSeq
import scala.reflect.ClassTag

import org.slf4j.LoggerFactory

/**
 * :: DeveloperApi ::
 * StreamingKMeansModel extends MLlib's KMeansModel for streaming
 * algorithms, so it can keep track of a continuously updated weight
 * associated with each cluster, and also update the model by
 * doing a single iteration of the standard k-means algorithm.
 *
 * The update algorithm uses the "mini-batch" KMeans rule,
 * generalized to incorporate forgetfulness (i.e. decay).
 * The update rule (for each cluster) is:
 *
 * c_t+1 = [(c_t * n_t * a) + (x_t * m_t)] / [n_t + m_t]
 * n_t+t = n_t * a + m_t
 *
 * Where c_t is the previously estimated centroid for that cluster,
 * n_t is the number of points assigned to it thus far, x_t is the centroid
 * estimated on the current batch, and m_t is the number of points assigned
 * to that centroid in the current batch.
 *
 * The decay factor 'a' scales the contribution of the clusters as estimated thus far,
 * by applying a as a discount weighting on the current point when evaluating
 * new incoming data. If a=1, all batches are weighted equally. If a=0, new centroids
 * are determined entirely by recent data. Lower values correspond to
 * more forgetting.
 *
 * Decay can optionally be specified by a half life and associated
 * time unit. The time unit can either be a batch of data or a single
 * data point. Considering data arrived at time t, the half life h is defined
 * such that at time t + h the discount applied to the data from t is 0.5.
 * The definition remains the same whether the time unit is given
 * as batches or points.
 *
 */

/* TODO - perform updates on centers in homogeneous coordinates
 * TODO - lazily discount clusters
 * TODO - maintain weighted distortion per center.
 */

class StreamingKMeansModel(model: KMeansModel) extends KMeansPredictor {
  val pointOps = model.pointOps
  val centerArrays = model.centers.toArray
  val clusterWeights = centerArrays.map(_.weight)
  val logger = LoggerFactory.getLogger(getClass.getName)

  def centers: IndexedSeq[BregmanCenter] = centerArrays.toIndexedSeq

  /** Perform a k-means update on a batch of data. */
  def update(data: RDD[Vector], decayFactor: Double, timeUnit: String): StreamingKMeansModel = {

    // find nearest cluster to each point
    val closest = data.map(point => (model.predict(point), SimpleWeightedVector(point, 1L)))

    // get sums and counts for updating each cluster
    val mergeContribs: (SimpleWeightedVector, SimpleWeightedVector) => SimpleWeightedVector = (p1, p2) => {
      BLAS.axpy(1.0, p2.vector, p1.vector)
      SimpleWeightedVector(p1.vector, p1.weight + p2.weight)
    }
    val dim = centerArrays(0).size
    val pointStats: Array[(Int, SimpleWeightedVector)] = closest
      .aggregateByKey(SimpleWeightedVector(Vectors.zeros(dim), 0L))(mergeContribs, mergeContribs)
      .collect()

    val discount = timeUnit match {
      case StreamingKMeans.BATCHES => decayFactor
      case StreamingKMeans.POINTS =>
        val numNewPoints = pointStats.view.map {
          case (_, SimpleWeightedVector(_, n)) =>
            n
        }.sum
        math.pow(decayFactor, numNewPoints)
    }

    // apply discount to weights
    BLAS.scal(discount, Vectors.dense(clusterWeights))

    // implement update rule
    pointStats.foreach {
      case (label, SimpleWeightedVector(sum, count)) =>
        val centroid = centerArrays(label).inhomogeneous
        val updatedWeight = clusterWeights(label) + count
        val lambda = count / math.max(updatedWeight, 1e-16)
        clusterWeights(label) = updatedWeight

        BLAS.scal(1.0 - lambda, centroid)
        BLAS.axpy(lambda / count, sum, centroid)
        centerArrays(label) = pointOps.toCenter(WeightedVector.fromInhomogeneousWeighted(centroid, 1.0))

        // display the updated cluster centers
        val display = centerArrays(label).size match {
          case x: Int if x > 100 => centroid.toArray.take(100).mkString("[", ",", "...")
          case _ => centroid.toArray.mkString("[", ",", "]")
        }

        logger.info(s"Cluster $label updated with weight $updatedWeight and centroid: $display")
    }

    // Check whether the smallest cluster is dying. If so, split the largest cluster.
    val weightsWithIndex = clusterWeights.view.zipWithIndex
    val (maxWeight, largest) = weightsWithIndex.maxBy { case (weight, _) => weight }
    val (minWeight, smallest) = weightsWithIndex.minBy { case (weight, _) => weight }
    if (minWeight < 1e-8 * maxWeight) {
      logger.info(s"Cluster $smallest is dying. Split the largest cluster $largest into two.")
      val weight = (maxWeight + minWeight) / 2.0
      clusterWeights(largest) = weight
      clusterWeights(smallest) = weight
      val largestClusterCenter = centerArrays(largest).inhomogeneous.toArray
      val l = largestClusterCenter.map(x => x + 1e-14 * math.max(math.abs(x), 1.0))
      val s = largestClusterCenter.map(x => x - 1e-14 * math.max(math.abs(x), 1.0))
      centerArrays(largest) = pointOps.toCenter(WeightedVector.fromInhomogeneousWeighted(l, 1.0))
      centerArrays(smallest) = pointOps.toCenter(WeightedVector.fromInhomogeneousWeighted(s, 1.0))
    }

    this
  }
}

class StreamingKMeans(
    k: Int = 2,
    decayFactor: Double = 1.0,
    timeUnit: String = StreamingKMeans.BATCHES,
    var model: StreamingKMeansModel) {

  /**
   * Update the clustering model by training on batches of data from a DStream.
   * This operation registers a DStream for training the model,
   * checks whether the cluster centers have been initialized,
   * and updates the model using each batch of data from the stream.
   *
   * @param data DStream containing vector data
   */
  def trainOn(data: DStream[Vector]): Unit = {
    assertInitialized()
    data.foreachRDD { (rdd, time) =>
      model = model.update(rdd, decayFactor, timeUnit)
    }
  }

  /**
   * Use the clustering model to make predictions on batches of data from a DStream.
   *
   * @param data DStream containing vector data
   * @return DStream containing predictions
   */
  def predictOn(data: DStream[Vector]): DStream[Int] = {
    assertInitialized()
    data.map(model.predict)
  }

  /**
   * Use the model to make predictions on the values of a DStream and carry over its keys.
   *
   * @param data DStream containing (key, feature vector) pairs
   * @tparam K key type
   * @return DStream containing the input keys and the predictions as values
   */
  def predictOnValues[K: ClassTag](data: DStream[(K, Vector)]): DStream[(K, Int)] = {
    assertInitialized()
    data.mapValues(model.predict)
  }

  /** Check whether cluster centers have been initialized. */
  private[this] def assertInitialized(): Unit = {
    if (model.centers == null) {
      throw new IllegalStateException(
        "Initial cluster centers must be set before starting predictions")
    }
  }
}

object StreamingKMeans {
  final val BATCHES = "batches"
  final val POINTS = "points"

  case class SimpleWeightedVector(vector: Vector, weight: Long)
}
