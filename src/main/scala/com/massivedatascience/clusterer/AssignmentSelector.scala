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
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkException

/** A KMeansSelector that creates cluster centers from pre-assigned cluster indices.
  *
  * @param assignments
  *   RDD containing the cluster index for each point in the dataset. Cluster indices should be in
  *   the range [0, numClusters).
  * @throws IllegalArgumentException
  *   if the number of unique clusters in assignments doesn't match numClusters
  */
class AssignmentSelector(assignments: RDD[Int]) extends KMeansSelector {

  /** Initialize cluster centers based on the pre-assigned cluster indices.
    *
    * @param pointOps
    *   The Bregman point operations
    * @param data
    *   The input data points
    * @param numClusters
    *   The number of clusters to create
    * @param initialInfo
    *   Optional initial condition (not used in this implementation)
    * @param runs
    *   Number of runs (not used in this implementation)
    * @param seed
    *   Random seed (not used in this implementation)
    * @return
    *   A sequence of cluster centers
    * @throws SparkException
    *   if the number of unique clusters in assignments doesn't match numClusters
    */
  def init(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    numClusters: Int,
    initialInfo: Option[InitialCondition] = None,
    runs: Int,
    seed: Long
  ): Seq[IndexedSeq[BregmanCenter]] = {

    // Count the number of unique clusters in the assignments
    val numUniqueClusters = assignments.distinct().count()

    // Verify that the number of unique clusters matches the requested number
    if (numUniqueClusters != numClusters) {
      throw new SparkException(
        s"Number of unique clusters in assignments ($numUniqueClusters) " +
          s"does not match requested numClusters ($numClusters)"
      )
    }

    // Create the model and get the centers
    val model = KMeansModel.fromAssignments(pointOps, data, assignments)

    // Verify that we got the expected number of centers
    require(
      model.centers.length == numClusters,
      s"Expected $numClusters clusters but got ${model.centers.length} centers"
    )

    Seq(model.centers)
  }
}
