package com.massivedatascience.clusterer

import org.apache.spark.rdd.RDD
import org.apache.spark.{ Logging, SparkContext }
import org.apache.spark.SparkContext._

import com.massivedatascience.clusterer.ColumnTrackingKMeans._

import scala.collection.Map

private[clusterer] class ConvergenceDetector(sc: SparkContext) extends Serializable with Logging {

  private[this] val stats = new TrackingStats(sc)

  def stable(): Boolean = (stats.numNonEmptyClusters == 0) ||
    (stats.movement.value / stats.numNonEmptyClusters < 1e-05)

  /**
   * Collect the statistics about this round
   *
   * @param round the round
   * @param currentCenters the current cluster centers
   * @param previousCenters the previous cluster centers
   * @param currentAssignments the current assignments
   * @param previousAssignments the previous assignments
   */
  def update(
    pointOps: BregmanPointOps,
    round: Int,
    currentCenters: IndexedSeq[CenterWithHistory],
    previousCenters: IndexedSeq[CenterWithHistory],
    currentAssignments: RDD[Assignment],
    previousAssignments: RDD[Assignment]): Boolean = {

    require(currentAssignments.getStorageLevel.useMemory)
    require(previousAssignments.getStorageLevel.useMemory)

    stats.currentRound.setValue(round)
    updateCenterStats(pointOps, currentCenters, previousCenters)
    updatePointStats(currentAssignments, previousAssignments)
    updateClusterStats(currentCenters, currentAssignments)
    report()
    stable()
  }

  /**
   * Report on the changes during the latest round
   */
  def report(): Unit = {
    logInfo(s"round ${stats.currentRound.value}")
    logInfo(s"       relocated centers      ${stats.relocatedCenters.value}")
    logInfo(s"       lowered distortion     ${stats.improvement.value}")
    logInfo(s"       center movement        ${stats.movement.value}")
    logInfo(s"       reassigned points      ${stats.reassignedPoints.value}")
    logInfo(s"       newly assigned points  ${stats.newlyAssignedPoints.value}")
    logInfo(s"       unassigned points      ${stats.unassignedPoints.value}")
    logInfo(s"       non-empty clusters     ${stats.nonemptyClusters.value}")
    logInfo(s"       largest cluster size   ${stats.largestCluster.value}")
    logInfo(s"       replenished clusters   ${stats.replenishedClusters.value}")
  }

  private[this] def updateClusterStats(
    centers: IndexedSeq[CenterWithHistory],
    assignments: RDD[Assignment]): Unit = {

    val clusterCounts = countByCluster(assignments)
    val biggest: (Int, Long) = clusterCounts.maxBy { case (_, size) => size }
    stats.largestCluster.setValue(biggest._2)
    stats.nonemptyClusters.setValue(clusterCounts.size)
    stats.emptyClusters.setValue(centers.size - clusterCounts.size)
  }

  private[this] def updatePointStats(
    currentAssignments: RDD[Assignment],
    previousAssignments: RDD[Assignment]): Unit = {

    stats.reassignedPoints.setValue(0)
    stats.unassignedPoints.setValue(0)
    stats.improvement.setValue(0)
    stats.newlyAssignedPoints.setValue(0)
    currentAssignments.zip(previousAssignments).foreach {
      case (current, previous) =>
        if (current.isAssigned) {
          if (previous.isAssigned) {
            stats.improvement.add(previous.distance - current.distance)
            if (current.cluster != previous.cluster) stats.reassignedPoints.add(1)
          } else {
            stats.newlyAssignedPoints.add(1)
          }
        } else {
          stats.unassignedPoints.add(1)
        }
    }
  }

  private[this] def updateCenterStats(
    pointOps: BregmanPointOps,
    currentCenters: IndexedSeq[CenterWithHistory],
    previousCenters: IndexedSeq[CenterWithHistory]): Unit = {

    stats.movement.setValue(0.0)
    stats.relocatedCenters.setValue(0)
    stats.replenishedClusters.setValue(0)
    for ((current, previous) <- currentCenters.zip(previousCenters)) {
      if (current.round != previous.round && previous.center.weight > pointOps.weightThreshold &&
        current.center.weight > pointOps.weightThreshold) {
        val delta = pointOps.distance(pointOps.toPoint(previous.center), current.center)
        stats.movement.add(delta)
        stats.relocatedCenters.add(1)
      }
      if (!current.initialized) {
        stats.replenishedClusters.add(1)
      }
    }
  }

  /**
   * count number of points assigned to each cluster
   *
   * @param currentAssignments the assignments
   * @return a map from cluster index to number of points assigned to that cluster
   */
  private[this] def countByCluster(currentAssignments: RDD[Assignment]): Map[Int, Long] =
    currentAssignments.filter(_.isAssigned).map { p => (p.cluster, p) }.countByKey()

}