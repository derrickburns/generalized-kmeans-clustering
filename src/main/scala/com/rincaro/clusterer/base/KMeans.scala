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
 */

package com.rincaro.clusterer.base

import com.rincaro.clusterer.metrics.EuOps
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkContext}

import scala.reflect.ClassTag

/**
 * A re-write of the Spark KMeans object.  This one creates a initializer object and then a generalized KMeans clusterer.
 */
object KMeans extends Logging  {

  // Initialization mode names
  val RANDOM = "random"
  val K_MEANS_PARALLEL = "k-means||"

  def train[T, P <: FP[T], C <: FP[T]](pointOps: PointOps[P, C, T])(
    raw: RDD[(Option[T], Array[Double])],
    k: Int = 2,
    maxIterations: Int = 20,
    runs: Int = 1,
    initializationMode: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    epsilon: Double = 1e-4)(implicit ctag: ClassTag[C], ptag: ClassTag[P]): GeneralizedKMeansModel[T, P, C] = {
    implicit val ops = pointOps

    val initializer: KMeansInitializer[T,P,C] = if (initializationMode == RANDOM) new KMeansRandom[T,P,C](pointOps, k, runs) else new KMeansParallel[T, P,C](k, runs, initializationSteps, 1, pointOps)
    val data = (raw map { case (name, vals) => pointOps.userToPoint(vals, name) }).cache()

    new GeneralizedKMeans[T, P, C](pointOps).run(data, initializer, maxIterations)
  }

  def main(args: Array[String]) {
    if (args.length < 5) {
      println("Usage: KMeans <master> <input_file> <k> <max_iterations> [<runs>]")
      System.exit(1)
    }
    val (master, inputFile, k, iterations) = (args(0), args(1), args(2).toInt, args(3).toInt)
    val runs = if (args.length >= 5) args(4).toInt else 1
    val sc = new SparkContext(master, "KMeans")
    val data: RDD[(Option[Long], Array[Double])] = sc.textFile(inputFile).zipWithIndex().map { case (line: String, index: Long) => (Some(index), line.split(' ').map(_.toDouble)) }
    val model = KMeans.train(new EuOps[Long])(raw = data, k = k, maxIterations = iterations, runs = runs)
    val cost = model.computeCost(data)
    println("Cluster centers:")
    for (c <- model.clusterCenters) {
      println("  " + c.inh.mkString(" "))
    }
    println("Cost: " + cost)
    System.exit(0)
  }
}
