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

import org.apache.spark.SparkContext

class TrackingStats(sc: SparkContext) extends BasicStats with Serializable {
  val currentRound = sc.longAccumulator("Round")
  currentRound.add(-1)
  val newlyAssignedPoints = sc.longAccumulator("Newly Assigned Points")
  val reassignedPoints    = sc.longAccumulator("Reassigned Points")
  val unassignedPoints    = sc.longAccumulator("Unassigned Points")
  val improvement         = sc.doubleAccumulator("Improvement")
  val relocatedCenters    = sc.longAccumulator("Relocated Centers")
  val movement            = sc.doubleAccumulator("Center Movement")
  val nonemptyClusters    = sc.longAccumulator("Non-Empty Clusters")
  val emptyClusters       = sc.longAccumulator("Empty Clusters")
  val largestCluster      = sc.longAccumulator("Largest Cluster")
  val replenishedClusters = sc.longAccumulator("Replenished Centers")

  def centerMovement: Double = movement.value

  def numNonEmptyClusters: Long = nonemptyClusters.value

  def numEmptyClusters: Long = emptyClusters.value

  def getRound: Long = currentRound.value

}
