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

import org.apache.spark.rdd.RDD

trait KMeansSelector extends Serializable {
  def init(
    ops: BregmanPointOps,
    d: RDD[BregmanPoint],
    numClusters: Int,
    initialInfo: Option[KMeansSelector.InitialCondition] = None,
    runs: Int,
    seed: Long): Seq[IndexedSeq[BregmanCenter]]
}

object KMeansSelector {

  case class InitialCondition(centers: Seq[IndexedSeq[BregmanCenter]], distances: Seq[RDD[Double]])

  val RANDOM = "random"
  val K_MEANS_PARALLEL = "k-means||" // a 5 step K-Means parallel initializer

  def apply(name: String): KMeansSelector = {
    name match {
      case RANDOM => KMeansRandom
      case K_MEANS_PARALLEL => new KMeansParallel(5)
      case _ => throw new RuntimeException(s"unknown initializer $name")
    }
  }
}

