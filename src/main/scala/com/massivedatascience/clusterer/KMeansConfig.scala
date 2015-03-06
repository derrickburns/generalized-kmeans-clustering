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

trait KMeansConfig extends Serializable {
  //for stochastic sampling, the percentage of points to update on each round
  val updateRate: Double

  // maxRoundsToBackfill maximum number of rounds to try to fill empty clusters
  val maxRoundsToBackfill: Int

  // fractionOfPointsToWeigh the fraction of the points to use in the weighting in KMeans++
  val fractionOfPointsToWeigh: Double

  val addOnly: Boolean

}

case class SimpleKMeansConfig(
  updateRate: Double = 1.0,
  maxRoundsToBackfill: Int = 0,
  fractionOfPointsToWeigh: Double = 0.10,
  addOnly: Boolean = true) extends KMeansConfig

object DefaultKMeansConfig extends SimpleKMeansConfig()

