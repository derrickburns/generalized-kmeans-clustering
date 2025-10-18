package com.massivedatascience.clusterer

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

import com.massivedatascience.clusterer.ColumnTrackingKMeans._

import scala.collection.Map

import org.slf4j.LoggerFactory

private[clusterer] class ConvergenceDetector(sc: SparkContext) extends Serializable {

  private[this] val stats = new TrackingStats(sc)

  val logger = LoggerFactory.getLogger(getClass.getName)

  def stable(): Boolean = (stats.numNonEmptyClusters == 0) ||
    (stats.numNonEmptyClusters > 0 && stats.movement.value / stats.numNonEmptyClusters < 1e-05)

  /** Collect the statistics about this round
    *
    * @param round
    *   the round
    * @param currentCenters
    *   the current cluster centers
    * @param previousCenters
    *   the previous cluster centers
    * @param currentAssignments
    *   the current assignments
    * @param previousAssignments
    *   the previous assignments
    */
  def update(
      pointOps: BregmanPointOps,
      round: Int,
      currentCenters: IndexedSeq[CenterWithHistory],
      previousCenters: IndexedSeq[CenterWithHistory],
      currentAssignments: RDD[Assignment],
      previousAssignments: RDD[Assignment]
  ): Boolean = {

    require(currentAssignments.getStorageLevel.useMemory)
    require(previousAssignments.getStorageLevel.useMemory)
    stats.currentRound.reset()
    stats.currentRound.add(round)
    updateCenterStats(pointOps, currentCenters, previousCenters)
    updatePointStats(currentAssignments, previousAssignments)
    updateClusterStats(currentCenters, currentAssignments)
    report()
    stable()
  }

  /** Report on the changes during the latest round
    */
  def report(): Unit = {
    logger.info(s"round ${stats.currentRound.value}")
    logger.info(s"       relocated centers      ${stats.relocatedCenters.value}")
    logger.info(s"       lowered distortion     ${stats.improvement.value}")
    logger.info(s"       center movement        ${stats.movement.value}")
    logger.info(s"       reassigned points      ${stats.reassignedPoints.value}")
    logger.info(s"       newly assigned points  ${stats.newlyAssignedPoints.value}")
    logger.info(s"       unassigned points      ${stats.unassignedPoints.value}")
    logger.info(s"       non-empty clusters     ${stats.nonemptyClusters.value}")
    logger.info(s"       largest cluster size   ${stats.largestCluster.value}")
    logger.info(s"       re-seeded clusters     ${stats.replenishedClusters.value}")
  }

  private[this] def updateClusterStats(
      centers: IndexedSeq[CenterWithHistory],
      assignments: RDD[Assignment]
  ): Unit = {

    val clusterCounts = countByCluster(assignments)

    // Handle the case where there are no clusters (empty assignments)
    if (clusterCounts.nonEmpty) {
      val biggest: (Int, Long) = clusterCounts.maxBy { case (_, size) => size }
      stats.largestCluster.reset()
      stats.largestCluster.add(biggest._2)
    } else {
      stats.largestCluster.reset()
      stats.largestCluster.add(0L)
    }

    stats.nonemptyClusters.reset()
    stats.nonemptyClusters.add(clusterCounts.size)
    stats.emptyClusters.reset()
    stats.emptyClusters.add(centers.size - clusterCounts.size)
  }

  private[this] def updatePointStats(
      currentAssignments: RDD[Assignment],
      previousAssignments: RDD[Assignment]
  ): Unit = {

    stats.reassignedPoints.reset()
    stats.unassignedPoints.reset()
    stats.improvement.reset()
    stats.newlyAssignedPoints.reset()
    currentAssignments.zip(previousAssignments).foreach { case (current, previous) =>
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
      previousCenters: IndexedSeq[CenterWithHistory]
  ): Unit = {

    stats.movement.reset()
    stats.relocatedCenters.reset()
    stats.replenishedClusters.reset()
    for ((current, previous) <- currentCenters.zip(previousCenters)) {
      if (
        current.round != previous.round && previous.center.weight > pointOps.weightThreshold &&
        current.center.weight > pointOps.weightThreshold
      ) {
        val delta = pointOps.distance(pointOps.toPoint(previous.center), current.center)
        stats.movement.add(delta)
        stats.relocatedCenters.add(1)
      }
      if (!current.initialized) {
        stats.replenishedClusters.add(1)
      }
    }
  }

  /** count number of points assigned to each cluster
    *
    * @param currentAssignments
    *   the assignments
    * @return
    *   a map from cluster index to number of points assigned to that cluster
    */
  private[this] def countByCluster(currentAssignments: RDD[Assignment]): Map[Int, Long] =
    currentAssignments.filter(_.isAssigned).map { p => (p.cluster, p) }.countByKey()

}
