/*
 * Licensed to the Massive Data Science and Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Massive Data Science and Derrick R. Burns licenses this file to You under the
 * Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.KMeansSelector.InitialCondition
import com.massivedatascience.linalg.BLAS._
import com.massivedatascience.util.{ SparkHelper, XORShiftRandom }
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

/**
 *
 * Initialize `runs` sets of cluster centers using the
 * <a href="http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf">k-means|| algorithm</a>.
 * This is a variant of k-means++ that tries to find  dissimilar cluster centers by starting with a random center and then doing
 * passes where more centers are chosen with probability proportional to their squared distance
 * to the current cluster set. It results in a provable approximation to an optimal clustering.
 *
 * In this implementation, we allow the client to provide an initial set of cluster centers
 * and closest distance for each point to those cluster centers.  This allows us to
 * use this code to find additional cluster centers at any time.
 */
case class KMeansParallel(numSteps: Int, sampleRate: Double = 1.0) extends KMeansSelector
  with SparkHelper {

  /**
   *
   * @param pointOps distance function
   * @param data data
   * @param targetNumberClusters  number of new clusters desired
   * @param initialState  initial clusters and distance data
   * @param runs  number of runs to perform
   * @param seedx random number seed
   * @return  updated set of cluster centers
   */

  def init(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    targetNumberClusters: Int,
    initialState: Option[InitialCondition] = None,
    runs: Int,
    seedx: Long): Seq[Centers] = {

    implicit val sc = data.sparkContext

    require(data.getStorageLevel.useMemory)
    val rand = new XORShiftRandom(seedx)
    val seed = rand.nextLong()
    val preselectedCenters = initialState.map(_.centers)
    val initialCosts = initialState.map(_.distances)
    val centers = preselectedCenters.getOrElse(randomCenters(pointOps, data, runs, seed))
    val requested = numbersRequested(targetNumberClusters, preselectedCenters, runs)
    val expandedCenters = moreCenters(pointOps, data, runs, initialCosts, numSteps, requested, seed,
      centers)
    val numbersRetainedCenters = preselectedCenters.map(_.map(_.size))
    finalClusterCenters(pointOps, data, targetNumberClusters, rand.nextLong(), expandedCenters,
      numbersRetainedCenters)
  }

  /**
   * Use K-Means++ to whittle the candidate centers to the requested number of centers
   *
   * @param numClusters  number of new clusters desired
   * @param seed random number seed
   * @param centers candidate centers
   * @param numberRetained number of pre-selected candidate centers to keep
   * @return arrays of selected centers
   */
  private[this]
  def finalClusterCenters(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    numClusters: Int,
    seed: Long,
    centers: Seq[Centers],
    numberRetained: Option[Seq[Int]]): Seq[Centers] = {

    val centerArrays = centers.map(_.toIndexedSeq)
    val weightMap = weights(pointOps, data, centerArrays, sampleRate, seed)
    val kMeansPlusPlus = new KMeansPlusPlus(pointOps)

    Seq.tabulate(centerArrays.length) { r =>
      val myCenters = centerArrays(r)
      logInfo(s"run $r has ${myCenters.length} centers")
      val weights = IndexedSeq.tabulate(myCenters.length)(i => weightMap.getOrElse((r, i), 0.0))
      val kx = if (numClusters > myCenters.length) myCenters.length else numClusters
      kMeansPlusPlus.goodCenters(seed, myCenters, weights, kx, 1,
        numberRetained.map(_(r)).getOrElse(0))
    }
  }

  /**
   * Set the costs, given a set of centers.
   *
   * @param centers new centers to consider
   * @return
   */
  private[this]
  def costsFromCenters(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    runs: Int,
    centers: Seq[Centers]): RDD[Vector] = {

    implicit val sc = data.sparkContext
    withBroadcast(centers) { bcNewCenters =>
      data.map { point =>
        Vectors.dense(Array.tabulate(runs) { r =>
          pointOps.pointCost(bcNewCenters.value(r), point)
        })
      }
    }
  }

  /**
   * Update the costs, given a previous set of costs and a new set of centers per run.
   *
   * @param centers new centers to consider
   * @param oldCosts best distance to previously considered centers
   * @return
   */
  private[this]
  def updatedCosts(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    runs: Int,
    centers: Seq[Centers],
    oldCosts: RDD[Vector]): RDD[Vector] = {

    implicit val sc = data.sparkContext
    withBroadcast(centers) { bcNewCenters =>
      data.zip(oldCosts).map { case (point, oldCost) =>
        Vectors.dense(Array.tabulate(runs) { r =>
          math.min(pointOps.pointCost(bcNewCenters.value(r), point), oldCost(r))
        })
      }
    }
  }

  /**
   * Select one point per run
   *
   * @param seed seed for random numbers
   * @return random center per run stored in an array buffer
   */
  private[this]
  def randomCenters(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    runs: Int,
    seed: Long): Seq[Centers] = {

    data
      .takeSample(withReplacement = true, runs, seed)
      .map(pointOps.toCenter)
      .map(IndexedSeq(_))
  }

  /**
   * Compute for each cluster the sum of the weights of the points in the cluster
   *
   * @param centerArrays sequence of arrays of centers
   * @param fraction fraction of points to sample
   * @return  A map from (run, cluster index) to the sum of the weights of its points
   */
  private[this]
  def weights(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centerArrays: Seq[Centers],
    fraction: Double,
    seed: Long): Map[(Int, Int), Double] = {

    implicit val sc = data.sparkContext
    withBroadcast(centerArrays) { bcCenters =>
      // for each (run, cluster) compute the sum of the weights of the points in the cluster
      data.sample(withReplacement = false, fraction, seed).flatMap { point =>
        val centers = bcCenters.value
        Seq.tabulate(centers.length)(r =>
          ((r, pointOps.findClosestCluster(centers(r), point)), point.weight))
      }.reduceByKeyLocally(_ + _)
    }
  }

  /**
   * Convert an sequence of RDDs of Doubles into RDD of vectors of Doubles
   *
   * @param rdds sequence of RDDs of Doubles
   * @return RDD of vectors
   */
  private[this]
  def asVectors(rdds: Seq[RDD[Double]]): RDD[Vector] = {
    rdds.zipWithIndex.foldLeft(rdds.head.map { _ => new Array[Double](rdds.length) }) {
      case (arrayRdd, (doubleRdd, i)) =>
        arrayRdd.zip(doubleRdd).map { case (array, double) => array(i) = double; array }

    }.map(Vectors.dense)
  }

  /**
   * Select approximately k points at random with probability proportional to the weight vectors
   * given.
   *
   * @param k number of points desired
   * @param seed random number seed
   * @param costs costs
   * @return k * runs new points, in an array where each entry is the tuple (run, point)
   */
  private[this]
  def select(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    runs: Int,
    k: Seq[Int],
    seed: Long,
    costs: RDD[Vector]): Array[(Int, BregmanCenter)] = {
    logInfo(s"constructing updated costs per point")
    val sumCosts = costs
      .aggregate(Vectors.zeros(runs))(
        (s, v) => axpy(1.0, v, s),
        (s0, s1) => axpy(1.0, s1, s0)
      )

    require(costs.getStorageLevel.useMemory)

    data.zip(costs).mapPartitionsWithIndex { (index, pointsWithCosts) =>
      val rand = new XORShiftRandom(seed ^ index)
      val range = 0 until runs
      pointsWithCosts.flatMap {
        case (p, c) =>
          val selectedRuns = range.filter { r =>
            val v = rand.nextDouble()
            v < c(r) * k(r) / sumCosts(r)
          }
          val nullCenter = null.asInstanceOf[BregmanCenter]
          val center = if (selectedRuns.nonEmpty) pointOps.toCenter(p) else nullCenter
          selectedRuns.map((_, center))
      }
    }.collect()
  }

  /**
   * Identify the number of additional cluster centers needed per run.
   *
   * @param desired total number of clusters desired (for each run)
   * @param centers initial clusters and distance per point to cluster
   * @param runs number of runs
   * @return number of clusters needed to fulfill gap
   */
  private[this]
  def numbersRequested(desired: Int, centers: Option[Seq[Centers]], runs: Int): Seq[Int] =
    centers.map(_.map(desired - _.length))
      .getOrElse(Seq.fill(runs)(desired))


  /**
   * On each step, preRound(run) points on average for each run with probability proportional
   * to their squared distance from the centers. Note that only distances between points
   * and new centers are computed in each iteration.
   *
   * @param initialCosts initial costs
   * @param requested minimum number of points add
   * @param seed random seed
   * @param centers initial centers
   * @return expanded set of centers, including initial centers
   */
  private[this]
  def moreCenters(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    runs: Int,
    initialCosts: Option[Seq[RDD[Double]]],
    numberSteps: Int,
    requested: Seq[Int],
    seed: Long,
    centers: Seq[Centers]): Seq[Centers] = {

    val addedCenters = centers.map(new ArrayBuffer[BregmanCenter] ++= _)
    val startingCosts = initialCosts.map(asVectors)
      .getOrElse(costsFromCenters(pointOps, data, runs, centers))
    var costs = sync("initial costs", startingCosts)
    val newCenters = Seq.fill(runs)(new ArrayBuffer[BregmanCenter]())
    var step = 0
    while (step < numberSteps) {
      logInfo(s"starting step $step")
      val stepSeed = seed ^ (step << 16)
      for ((index, center) <- select(pointOps, data, runs, requested.map(_ * 2), stepSeed, costs)) {
        newCenters(index) += center
      }
      costs = exchange(s"costs at step $step", costs) { oldCosts =>
        updatedCosts(pointOps, data, runs, newCenters, oldCosts)
      }
      for ((c, n) <- addedCenters.zip(newCenters)) {
        c ++= n
        n.clear()
      }

      step += 1
    }
    costs.unpersist(blocking = false)
    addedCenters.map(_.toIndexedSeq)
  }
}
