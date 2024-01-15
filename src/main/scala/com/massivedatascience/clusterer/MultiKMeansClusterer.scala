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

import com.massivedatascience.clusterer.MultiKMeansClusterer.ClusteringWithDistortion
import org.apache.spark.rdd.RDD

trait MultiKMeansClusterer extends Serializable {

  def cluster(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centers: Seq[IndexedSeq[BregmanCenter]]): Seq[ClusteringWithDistortion]

  def best(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centers: Seq[IndexedSeq[BregmanCenter]]): ClusteringWithDistortion = {
    cluster(maxIterations, pointOps, data, centers).minBy(_.distortion)
  }
}

object MultiKMeansClusterer {

  val TRACKING = "TRACKING"
  val SIMPLE = "SIMPLE"
  val COLUMN_TRACKING = "COLUMN_TRACKING"
  val MINI_BATCH_10 = "MINI_BATCH_10"
  val CHANGE_TRACKING = "CHANGE_TRACKING"
  val RESEED = "RESEED"

  case class ClusteringWithDistortion(distortion: Double, centers: IndexedSeq[BregmanCenter])

  def apply(clustererName: String): MultiKMeansClusterer = {
    clustererName match {
      case SIMPLE => new ColumnTrackingKMeans
      case TRACKING => new ColumnTrackingKMeans
      case COLUMN_TRACKING => new ColumnTrackingKMeans
      case CHANGE_TRACKING => new ColumnTrackingKMeans(new SimpleKMeansConfig().copy(addOnly = false))
      case MINI_BATCH_10 => new ColumnTrackingKMeans(new SimpleKMeansConfig().copy(updateRate = 0.10))
      case RESEED => new ColumnTrackingKMeans(new SimpleKMeansConfig().copy(maxRoundsToBackfill = 5))
      case _ => throw new RuntimeException(s"unknown clusterer $clustererName")
    }
  }
}
