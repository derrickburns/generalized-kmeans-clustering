/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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
 *
 * This code is a modified version of the original Spark 1.0.2 K-Means implementation.
 */

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.KMeansSelector.InitialCondition
import com.massivedatascience.util.{ SparkHelper, XORShiftRandom }
import org.apache.spark.rdd.RDD

/**
 * Initial a KMeansModel with randomly chosen centers from a given set of points.
 */
case object KMeansRandom extends KMeansSelector with SparkHelper {

  def init(
    ops: BregmanPointOps,
    data: RDD[BregmanPoint],
    k: Int,
    initialInfo: Option[InitialCondition] = None,
    runs: Int,
    seed: Long): Seq[IndexedSeq[BregmanCenter]] = {

    val rand = new XORShiftRandom(seed)

    def select(data: RDD[BregmanPoint], count: Int) = {
      val toCenter = ops.toCenter _
      data.takeSample(withReplacement = false, count, rand.nextInt()).map(toCenter)
    }

    withCached("random initial", data.filter(_.weight > ops.weightThreshold)) { filtered =>
      val count = filtered.count()
      if (runs * k <= count) {
        val centers = select(filtered, runs * k)
        Seq.tabulate(runs)(r => centers.slice(r * k, (r + 1) * k))
      } else if (k < count) {
        Seq.fill(runs)(select(filtered, k))
      } else {
        val all = filtered.collect().map(ops.toCenter)
        Seq.fill(runs)(all)
      }
    }
  }
}

