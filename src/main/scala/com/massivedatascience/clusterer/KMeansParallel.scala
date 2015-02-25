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

import com.massivedatascience.clusterer.util.BLAS._
import com.massivedatascience.clusterer.util.{SparkHelper, XORShiftRandom}

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

/**
 *
 * Initialize `runs` sets of cluster centers using the k-means|| algorithm by Bahmani et al.
 * (Bahmani et al., Scalable K-Means++, VLDB 2012). This is a variant of k-means++ that tries
 * to find  dissimilar cluster centers by starting with a random center and then doing
 * passes where more centers are chosen with probability proportional to their squared distance
 * to the current cluster set. It results in a provable approximation to an optimal clustering.
 *
 * The original paper can be found at http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf.
 *
 * In this implementation, we allow the client to provide an initial set of cluster centers
 * and closest distance for each point to those cluster centers.  This allows us to
 * use this code to find additional cluster centers at any time.
 */
class KMeansParallel(numSteps: Int) extends KMeansInitializer with SparkHelper {

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
    initialState: Option[(Seq[IndexedSeq[BregmanCenter]], Seq[RDD[Double]])] = None,
    runs: Int,
    seedx: Long): Array[Array[BregmanCenter]] = {

    implicit val sc = data.sparkContext

    /**
     * Use K-Means++ to whittle the candidate centers to the requested number of centers
     *
     * @param numClusters  number of new clusters desired
     * @param seed random number seed
     * @param centers candidate centers
     * @param numberRetained number of pre-selected candidate centers to keep
     * @return arrays of selected centers
     */
    def finalClusterCenters(
      numClusters: Int,
      seed: Long,
      centers: Seq[IndexedSeq[BregmanCenter]],
      numberRetained: Option[Seq[Int]]): Array[Array[BregmanCenter]] = {

      val centerArrays = centers.map(_.toArray)
      val weightMap = weights(centerArrays)
      val kMeansPlusPlus = new KMeansPlusPlus(pointOps)

      Array.tabulate(centerArrays.length) { r =>
        val myCenters = centerArrays(r)
        logInfo(s"run $r has ${myCenters.length} centers")
        val weights = Array.tabulate(myCenters.length)(i => weightMap.getOrElse((r, i), 0.0))
        val kx = if (numClusters > myCenters.length) myCenters.length else numClusters
        kMeansPlusPlus.getCenters(seed, myCenters, weights, kx, 1,
          numberRetained.map(_(r)).getOrElse(0))
      }
    }

    /**
     * Set the costs, given a set of centers.
     *
     * @param centers new centers to consider
     * @return
     */
    def setCosts(centers: Seq[IndexedSeq[BregmanCenter]]): RDD[Vector] = {
      val ops = pointOps
      val numRuns = runs
      withBroadcast(centers) { bcNewCenters =>
        data.map { point =>
          Vectors.dense(Array.tabulate(numRuns) { r =>
            ops.pointCost(bcNewCenters.value(r), point)
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
    def updatedCosts(centers: Seq[IndexedSeq[BregmanCenter]], oldCosts: RDD[Vector]): RDD[Vector] = {
      val ops = pointOps
      val numRuns = runs
      withBroadcast(centers) { bcNewCenters =>
        data.zip(oldCosts).map { case (point, oldCost) =>
          Vectors.dense(Array.tabulate(numRuns) { r =>
            math.min(ops.pointCost(bcNewCenters.value(r), point), oldCost(r))
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
    def randomCenters(seed: Long): Seq[IndexedSeq[BregmanCenter]] = {
      val ops = pointOps
      val numRuns = runs
      data
        .takeSample(withReplacement = true, numRuns, seed)
        .map(ops.toCenter)
        .map(IndexedSeq(_))
    }

    /**
     * Compute for each cluster the sum of the weights of the points in the cluster
     *
     * @param centerArrays sequence of arrays of centers
     * @return  A map from (run, cluster index) to the sum of the weights of its points
     */
    def weights(centerArrays: Seq[Array[BregmanCenter]]): Map[(Int, Int), Double] = {
      val ops = pointOps

      withBroadcast(centerArrays) { bcCenters =>
        // for each (run, cluster) compute the sum of the weights of the points in the cluster
        data.flatMap { point =>
          val centers = bcCenters.value
          Array.tabulate(centers.length)(r =>
            ((r, ops.findClosestCluster(centers(r), point)), point.weight))
        }.reduceByKeyLocally(_ + _)
      }
    }

    /**
     * Convert an sequence of RDDs of Doubles into RDD of vectors of Doubles
     *
     * @param rdds sequence of RDDs of Doubles
     * @return RDD of vectors
     */
    def asVectors(rdds: Seq[RDD[Double]]): RDD[Vector] = {
      rdds.zipWithIndex.foldLeft(rdds.head.map { _ => new Array[Double](rdds.length)}) {
        case (arrayRdd, (doubleRdd, i)) =>
          arrayRdd.zip(doubleRdd).map { case (array, double) => array(i) = double; array}

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
    def select(k: Seq[Int], seed: Long, costs: RDD[Vector]): Array[(Int, BregmanCenter)] = {
      logInfo(s"constructing updated costs per point")
      val numRuns = runs
      val sumCosts = costs
        .aggregate(Vectors.zeros(numRuns))(
          (s, v) => axpy(1.0, v, s),
          (s0, s1) => axpy(1.0, s1, s0)
        )

      require(costs.getStorageLevel.useMemory)

      data.zip(costs).mapPartitionsWithIndex { (index, pointsWithCosts) =>
        val rand = new XORShiftRandom(seed ^ index)
        val range = 0 until numRuns
        val ops = pointOps
        pointsWithCosts.flatMap { case (p, c) =>
          val selectedRuns = range.filter { r =>
            rand.nextDouble() < c(r) * k(r) / sumCosts(r)
          }
          val nullCenter = null.asInstanceOf[BregmanCenter]
          val center = if (selectedRuns.nonEmpty) ops.toCenter(p) else nullCenter
          selectedRuns.map((_, center))
        }
      }.collect()
    }

    /**
     * Identify the number of additional cluster centers needed per run.
     *
     * @param totalNumClusters total number of clusters desired (for each run)
     * @param initialCenters initial clusters and distance per point to cluster
     * @param runs number of runs
     * @return number of clusters needed to fulfill gap
     */
    def numbersRequested(
      totalNumClusters: Int,
      initialCenters: Option[Seq[IndexedSeq[BregmanCenter]]],
      runs: Int): Seq[Int] = {

      initialCenters.map(_.map(totalNumClusters - _.length))
        .getOrElse(Array.fill(runs)(totalNumClusters).toSeq)
    }

    /**
     * On each step, preRound(run) points on average for each run with probability proportional
     * to their squared distance from the centers. Note that only distances between points
     * and new centers are computed in each iteration.
     *
     * @param initialCosts initial costs
     * @param perRound number of points to select per round per run
     * @param seed random seed
     * @param centers initial centers
     * @return expanded set of centers, including initial centers
     */
    def moreCenters(
      initialCosts: Option[Seq[RDD[Double]]],
      numberSteps: Int,
      perRound: Seq[Int],
      seed: Long,
      centers: Seq[IndexedSeq[BregmanCenter]]): Seq[IndexedSeq[BregmanCenter]] = {

      val addedCenters = centers.map(new ArrayBuffer[BregmanCenter] ++= _)
      val startingCosts = initialCosts.map(asVectors).getOrElse(setCosts(centers))
      var costs = sync("initial costs", startingCosts)
      val newCenters = Array.fill(runs)(new ArrayBuffer[BregmanCenter]())

      var step = 0
      while (step < numberSteps) {
        logInfo(s"starting step $step")
        for ((index, center) <- select(perRound, seed ^ (step << 16), costs)) {
          newCenters(index) += center
        }
        costs = exchange(s"costs at step $step", costs) { oldCosts =>
          updatedCosts(newCenters, oldCosts)
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

    require(data.getStorageLevel.useMemory)
    val seed = new XORShiftRandom(seedx).nextLong()
    val preselectedCenters = initialState.map(_._1)
    val initialCosts = initialState.map(_._2)
    val centers = preselectedCenters.getOrElse(randomCenters(seed))
    val requested = numbersRequested(targetNumberClusters, preselectedCenters, runs)
    val expandedCenters = moreCenters(initialCosts, numSteps, requested.map(_ * 2), seed, centers)
    val numbersRetainedCenters = preselectedCenters.map(_.map(_.size))
    finalClusterCenters(targetNumberClusters, seed, expandedCenters, numbersRetainedCenters)
  }
}
