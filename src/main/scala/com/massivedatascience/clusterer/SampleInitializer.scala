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
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

class SampleInitializer(val assignments: RDD[Int]) extends KMeansInitializer {
  def init(
    pointOps: BregmanPointOps,
    d: RDD[Vector]): (RDD[BregmanPoint], Array[Array[BregmanCenter]]) = {

    val data = d.map {pt => pointOps.inhomogeneousToPoint(pt, 1.0)}
    data.setName("input to sample initializer")
    data.persist(StorageLevel)

    val centroids = assignments.zip(data).aggregateByKey(pointOps.getCentroid)(
      (centroid, pt) => centroid.add(pt),
      (c1, c2) => c1.add(c2)
    )

    val bregmanCenters = centroids.map {p => pointOps.toCenter(p._2.asImmutable)}
    (data, Array(bregmanCenters.collect()))
  }
}
