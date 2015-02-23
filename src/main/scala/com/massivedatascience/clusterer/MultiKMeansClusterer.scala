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

import com.massivedatascience.clusterer.ColumnTrackingKMeans.CenterWithHistory
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

trait MultiKMeansClusterer extends Serializable with Logging {
  def cluster(
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centers: Array[Array[BregmanCenter]]): Array[(Double, Array[BregmanCenter], Option[RDD[(Int, Double)]])]
}

object MultiKMeansClusterer {
  private[clusterer] val noCluster = -1
  private[clusterer] val unassigned = Assignment(Infinity, noCluster, -2)

  private[clusterer] case class Assignment(distance: Double, cluster: Int, round: Int) {
    def isAssigned = cluster != noCluster

    def isUnassigned = cluster == noCluster
  }

  private[clusterer] val noCenters = Array[CenterWithHistory]()
  private[clusterer] val noAssignments: Option[RDD[Assignment]] = None
  private[clusterer] val unsolved = (Double.MaxValue, noCenters, noAssignments)

  def bestOf(candidates: Seq[(Double, Array[BregmanCenter], Option[RDD[(Int, Double)]])]) = {
    candidates.minBy(x => x._1)
  }


}
