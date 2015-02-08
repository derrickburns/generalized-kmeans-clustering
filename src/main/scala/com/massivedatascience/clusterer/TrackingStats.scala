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

import org.apache.spark.SparkContext._
import org.apache.spark.{Logging, SparkContext}


class TrackingStats(sc: SparkContext, val round: Int) extends BasicStats with Serializable with Logging {
  val currentRound = sc.accumulator[Int](round, s"$round")
  val newlyAssignedPoints = sc.accumulator[Int](0, s"Newly Assigned Points")
  val reassignedPoints = sc.accumulator[Int](0, s"Reassigned Points")
  val unassignedPoints = sc.accumulator[Int](0, s"Unassigned Points")
  val improvement = sc.accumulator[Double](0.0, s"Improvement")
  val relocatedCenters = sc.accumulator[Int](0, s"Relocated Centers")
  val dirtyOther = sc.accumulator[Int](0, s"=> Other Moving")
  val dirtySame = sc.accumulator[Int](0, s"=> Same Moving")
  val stationary = sc.accumulator[Int](0, s"Stationary")
  val closestClean = sc.accumulator[Int](0, s"Moving => Other Stationary")
  val closestDirty = sc.accumulator[Int](0, s"Stationary => Other Moving")
  val movement = sc.accumulator[Double](0.0, s"Center Movement")
  val nonemptyClusters = sc.accumulator[Int](0, s"Non-Empty Clusters")
  val emptyClusters = sc.accumulator[Int](0, s"Empty Clusters")
  val largestCluster = sc.accumulator[Long](0, s"Largest Cluster")

  def getMovement = movement.value

  def getNonEmptyClusters = nonemptyClusters.value

  def getEmptyClusters = emptyClusters.value

  def getRound = round

  def report() = {
    logInfo(s"round ${currentRound.value}")
    logInfo(s"relocated centers = ${relocatedCenters.value}")
    logInfo(s"lowered distortion by ${improvement.value}")
    logInfo(s"center movement by ${movement.value}")
    logInfo(s"reassigned points = ${reassignedPoints.value}")
    logInfo(s"newly assigned points = ${newlyAssignedPoints.value}")
    logInfo(s"unassigned points = ${unassignedPoints.value}")
    logInfo(s"non-empty clusters = ${nonemptyClusters.value}")
    logInfo(s"some other moving cluster is closest ${dirtyOther.value}")
    logInfo(s"my cluster moved closest = ${dirtySame.value}")

    logInfo(s"my stationary cluster is closest = ${stationary.value}")
    logInfo(s"my cluster moved away and a stationary cluster is now closest = ${closestClean.value}")
    logInfo(s"my cluster didn't move, but a moving cluster is closest = ${closestDirty.value}")
    logInfo(s"largest cluster size ${largestCluster.value}")
  }
}
