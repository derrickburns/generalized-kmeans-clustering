/*
 * Licensed to the Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Derrick R. Burns licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
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

package com.massivedatascience.clusterer.base

import com.massivedatascience.clusterer.util.XORShiftRandom
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class KMeansRandom[P <: FP : ClassTag, C <: FP : ClassTag](
                                                            pointOps: PointOps[P, C], k: Int, runs: Int) extends KMeansInitializer[P, C] {

  def init(data: RDD[P], seed: Int): Array[Array[C]] = {
    // Sample all the cluster centers in one pass to avoid repeated scans
    val sample = data.takeSample(true, runs * k, new XORShiftRandom().nextInt()).withFilter(x => x.weight > 0).map(pointOps.pointToCenter).toSeq
    Array.tabulate(runs)(r => sample.slice(r * k, (r + 1) * k).toArray)
  }
}
