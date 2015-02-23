package com.massivedatascience.clusterer


import com.massivedatascience.clusterer.util.{SparkHelper, XORShiftRandom}
import com.massivedatascience.clusterer.util.BLAS.axpy

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD


/**
 * Select k points at random with probability proportional to distance from closest cluster
 * center.
 *
 */
trait WeightedSelector extends SparkHelper {

  def select(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    k: Int,
    seed: Int,
    runs: Int,
    newCenters: Seq[IndexedSeq[BregmanCenter]]): (RDD[Vector], Array[(Int, BregmanCenter)]) = {

    val preCosts = data.map(_ => Vectors.dense(Array.fill(runs)(Double.PositiveInfinity)))
    select(pointOps, data, k, seed, runs, preCosts, newCenters)
  }

  def select(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    k: Int,
    seed: Int,
    runs: Int,
    preCosts: RDD[Vector],
    newCenters: Seq[IndexedSeq[BregmanCenter]]): (RDD[Vector], Array[(Int, BregmanCenter)]) = {

    implicit val sc = preCosts.sparkContext

    val costs = withBroadcast(newCenters) { bcNewCenters =>
      logInfo(s"constructing costs")
      data.zip(preCosts).map { case (point, cost) =>
        Vectors.dense(Array.tabulate(runs) { r =>
          math.min(pointOps.pointCost(bcNewCenters.value(r), point), cost(r))
        })
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

    val chosen = data.zip(costs).mapPartitionsWithIndex { (index, pointsWithCosts) =>
      val rand = new XORShiftRandom(seed ^ index)
      val range = 0 until runs
      pointsWithCosts.flatMap { case (p, c) =>
        val selectedRuns = range.filter { r =>
          rand.nextDouble() < c(r) * k / sumCosts(r)
        }
        val nullCenter = null.asInstanceOf[BregmanCenter]
        val center = if (selectedRuns.nonEmpty) pointOps.toCenter(p) else nullCenter
        selectedRuns.map((_, center))
      }
    }.collect()
    (costs, chosen.take(k))
  }

}
