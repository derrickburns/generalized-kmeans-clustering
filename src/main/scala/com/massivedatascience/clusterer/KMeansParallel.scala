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
 * This code is a complete re-write version of the original Spark 1.0.2 implementation by
 * Derrick R. Burns.
 *
 *
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
class KMeansParallel(initializationSteps: Int) extends KMeansInitializer with SparkHelper {

  /**
   *
   * @param pointOps distance function
   * @param data data
   * @param totalNumClusters  number of new clusters desired
   * @param initialInfo  initial clusters and distance data
   * @param runs  number of runs to perform
   * @param seedx random number seed
   * @return  updated set of cluster centers
   */

  def init(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    totalNumClusters: Int,
    initialInfo: Option[(Seq[IndexedSeq[BregmanCenter]], Seq[RDD[Double]])] = None,
    runs: Int,
    seedx: Long): Array[Array[BregmanCenter]] = {

    implicit val sc = data.sparkContext

    def startingCenters(
      initialInfo: Option[(Seq[IndexedSeq[BregmanCenter]], Seq[RDD[Double]])],
      seed: Long): Seq[ArrayBuffer[BregmanCenter]] = {
      initialInfo.map(_._1).map(_.map(new ArrayBuffer[BregmanCenter]() ++ _)).getOrElse(randomCenters(seed))
    }

    /**
     * Use K-Means++ to whittle the candidate centers to the requested number of centers
     *
     * @param numClusters  number of new clusters desired
     * @param seed random number seed
     * @param centers candidate centers
     * @param numPreselected number of pre-selected candidate centers to keep
     * @return arrays of selected centers
     */
    def finalClusterCenters(
      numClusters: Int,
      seed: Long,
      centers: Seq[ArrayBuffer[BregmanCenter]],
      numPreselected: Option[Seq[Int]]): Array[Array[BregmanCenter]] = {

      val centerArrays = centers.map(_.toArray)
      val weightMap = weights(centerArrays)
      val kMeansPlusPlus = new KMeansPlusPlus(pointOps)

      Array.tabulate(centerArrays.length) { r =>
        val myCenters = centerArrays(r)
        logInfo(s"run $r has ${myCenters.length} centers")
        val weights = Array.tabulate(myCenters.length)(i => weightMap.getOrElse((r, i), 0.0))
        val kx = if (numClusters > myCenters.length) myCenters.length else numClusters
        kMeansPlusPlus.getCenters(seed, myCenters, weights, kx, 1, numPreselected.map(_(r)).getOrElse(0))
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
        data.zip(oldCosts).map { case (point, cost) =>
          Vectors.dense(Array.tabulate(numRuns) { r =>
            math.min(ops.pointCost(bcNewCenters.value(r), point), cost(r))
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
    def randomCenters(seed: Long): Seq[ArrayBuffer[BregmanCenter]] = {
      val ops = pointOps
      val numRuns = runs
      data
        .takeSample(withReplacement = true, numRuns, seed)
        .map(ops.toCenter)
        .map(new ArrayBuffer[BregmanCenter] += _)
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
     * Define the starting costs.  If no cost vectors are provided,
     * initialize them from the provided cluster centers.
     *
     * @param initialInfo starting cost info
     * @param centers starting centers
     * @return initial value of costs
     */
    def startingCosts(
      initialInfo: Option[(Seq[IndexedSeq[BregmanCenter]], Seq[RDD[Double]])],
      centers: Seq[ArrayBuffer[BregmanCenter]]): RDD[Vector] = {
      initialInfo.map(_._2).map(asVectors).getOrElse(setCosts(centers))
    }

    val seed = new XORShiftRandom(seedx).nextLong()
    val centers = startingCenters(initialInfo, seed)
    val newCenters = Array.fill(runs)(new ArrayBuffer[BregmanCenter]())
    var costs = sync("initial costs", startingCosts(initialInfo, centers))

    val needed = initialInfo.map(_._1.map(totalNumClusters - _.length))
    val perRound = needed.getOrElse(Array.fill(runs)(totalNumClusters).toSeq).map(_ * 2)

    // On each step, sample 2 * k points on average for each run with probability proportional
    // to their squared distance from that run's centers. Note that only distances between points
    // and new centers are computed in each iteration.
    var step = 0
    while (step < initializationSteps) {
      logInfo(s"starting step $step")
      assert(data.getStorageLevel.useMemory)
      val additionalCenters = select(perRound, seed ^ (step << 16), costs)
      additionalCenters.foreach { case (index, center) =>
        newCenters(index) += center
      }
      costs = exchange(s"costs at step $step", costs) { oldCosts =>
        updatedCosts(newCenters, oldCosts)
      }
      centers.zip(newCenters).foreach { case (c, n) => c ++= n; n.clear()}
      step += 1
    }

    costs.unpersist(blocking = false)
    logInfo("creating final centers")

    val keep = initialInfo.map(_._1).map(_.map(_.size))
    finalClusterCenters(totalNumClusters, seed, centers, keep)
  }
}
