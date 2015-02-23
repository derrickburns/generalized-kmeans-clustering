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
 * This code is a modified version of the original Spark 1.0.2 implementation.
 */

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.util.{SparkHelper, XORShiftRandom}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import com.massivedatascience.clusterer.util.BLAS.axpy


class KMeansParallel(
  numClusters: Int,
  r: Int,
  initializationSteps: Int,
  seedx: Int,
  clusterer: MultiKMeansClusterer)
  extends KMeansInitializer with SparkHelper {

  def init(pointOps: BregmanPointOps, data: RDD[BregmanPoint]): Array[Array[BregmanCenter]] = {

    implicit val sc = data.sparkContext

    /**
     * Initialize `runs` sets of cluster centers using the k-means|| algorithm by Bahmani et al.
     * (Bahmani et al., Scalable K-Means++, VLDB 2012). This is a variant of k-means++ that tries
     * to find  dissimilar cluster centers by starting with a random center and then doing
     * passes where more centers are chosen with probability proportional to their squared distance
     * to the current cluster set. It results in a provable approximation to an optimal clustering.
     *
     * The original paper can be found at http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf.
     *
     * @return
     */

    val runs = r

    // Initialize empty centers and point costs.
    val centers = Array.tabulate(runs)(r => ArrayBuffer.empty[BregmanCenter])
    var costs = sync("pre-costs", data.map(_ => Vectors.dense(Array.fill(runs)(Double.PositiveInfinity))))

    // Initialize each run's first center to a random point.
    val seed = new XORShiftRandom(seedx).nextInt()
    val sample = data.takeSample(withReplacement = true, runs, seed).map(pointOps.toCenter)
    val newCenters = Array.tabulate(runs)(r => ArrayBuffer(sample(r)))

    /** Merges new centers to centers. */
    def mergeNewCenters(): Unit = {
      var r = 0
      while (r < runs) {
        centers(r) ++= newCenters(r)
        newCenters(r).clear()
        r += 1
      }
    }

    // On each step, sample 2 * k points on average for each run with probability proportional
    // to their squared distance from that run's centers. Note that only distances between points
    // and new centers are computed in each iteration.
    var step = 0
    while (step < initializationSteps) {
      logInfo(s"starting step $step")
      assert(data.getStorageLevel.useMemory)

      costs = exchange(s"costs at step $step", costs) { preCosts =>
        withBroadcast(newCenters) { bcNewCenters =>
          logInfo(s"constructing costs")
          data.zip(preCosts).map { case (point, cost) =>
            Vectors.dense(Array.tabulate(runs) { r =>
              math.min(pointOps.pointCost(bcNewCenters.value(r), point), cost(r))
            })
          }
        }
      }

      logInfo(s"summing costs")

      val sumCosts = costs
        .aggregate(Vectors.zeros(runs))(
          seqOp = (s, v) => {
            // s += v
            axpy(1.0, v, s)
          },
          combOp = (s0, s1) => {
            // s0 += s1
            axpy(1.0, s1, s0)
          }
        )

      assert(data.getStorageLevel.useMemory)
      assert(costs.getStorageLevel.useMemory)
      logInfo(s"collecting chosen")

      val k = numClusters
      val chosen = data.zip(costs).mapPartitionsWithIndex { (index, pointsWithCosts) =>
        val rand = new XORShiftRandom(seed ^ (step << 16) ^ index)
        val range = 0 until runs
        pointsWithCosts.flatMap { case (p, c) =>
          val selectedRuns = range.filter { r =>
            rand.nextDouble() < 2.0 * c(r) * k / sumCosts(r)
          }
          val nullCenter = null.asInstanceOf[BregmanCenter]
          val center = if (selectedRuns.nonEmpty) pointOps.toCenter(p) else nullCenter
          selectedRuns.map((_, center))
        }
      }.collect()
      logInfo(s"merging centers")
      mergeNewCenters()
      chosen.foreach { case (index, center) =>
        newCenters(index) += center
      }
      step += 1
    }

    mergeNewCenters()

    // Finally, we might have a set of more than k candidate centers for each run; weigh each
    // candidate by the number of points in the dataset mapping to it and run a local k-means++
    // on the weighted centers to pick just k of them


    costs.unpersist(blocking = false)

    logInfo("creating final centers")
    val centers1 = centers.map(_.toArray)
    val result = new WeighedDistanceClusterer(numClusters, seed, clusterer).cluster(pointOps, data, centers1)
    assert(data.getStorageLevel.useMemory)
    logInfo("returning results")
    result.map(_._2).toArray
  }
}
