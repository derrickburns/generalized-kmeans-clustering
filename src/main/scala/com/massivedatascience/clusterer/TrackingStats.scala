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
import org.apache.spark.SparkContext

class TrackingStats(sc: SparkContext) extends BasicStats with Serializable {
  val currentRound = sc.accumulator[Int](-1, "Round")
  val newlyAssignedPoints = sc.accumulator[Int](0, "Newly Assigned Points")
  val reassignedPoints = sc.accumulator[Int](0, "Reassigned Points")
  val unassignedPoints = sc.accumulator[Int](0, "Unassigned Points")
  val improvement = sc.accumulator[Double](0.0, "Improvement")
  val relocatedCenters = sc.accumulator[Int](0, "Relocated Centers")
  val movement = sc.accumulator[Double](0.0, "Center Movement")
  val nonemptyClusters = sc.accumulator[Int](0, "Non-Empty Clusters")
  val emptyClusters = sc.accumulator[Int](0, "Empty Clusters")
  val largestCluster = sc.accumulator[Long](0, "Largest Cluster")
  val replenishedClusters = sc.accumulator[Int](0, "Replenished Centers")

  def centerMovement: Double = movement.value

  def numNonEmptyClusters: Int = nonemptyClusters.value

  def numEmptyClusters: Int = emptyClusters.value

  def getRound: Int = currentRound.value

}
