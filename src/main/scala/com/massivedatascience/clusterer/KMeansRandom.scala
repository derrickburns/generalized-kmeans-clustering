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

import com.massivedatascience.clusterer.util.{SparkHelper, XORShiftRandom}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

class KMeansRandom extends KMeansInitializer with SparkHelper {

  // TODO use seed?
  def init(
    ops: BregmanPointOps,
    data: RDD[BregmanPoint],
    k: Int,
    initial: Option[Seq[IndexedSeq[BregmanCenter]]] = None,
    runs: Int,
    seed: Long): Array[Array[BregmanCenter]] = {

    implicit val sc = data.sparkContext

    def select(data: RDD[BregmanPoint], count: Int) = {
      val toCenter = ops.toCenter _
      data.takeSample(withReplacement = false, count, new XORShiftRandom().nextInt()).map(toCenter)
    }

    withCached("random initial", data.filter(_.weight > ops.weightThreshold)) { filtered =>
      val count = filtered.count()
      if (runs * k <= count) {
        val centers = select(filtered, runs * k)
        Array.tabulate(runs)(r => centers.slice(r * k, (r + 1) * k))
      } else if (k < count) {
        Array.fill(runs)(select(filtered, k))
      } else {
        val all = filtered.collect().map(ops.toCenter)
        Array.fill(runs)(all)
      }
    }
  }
}



