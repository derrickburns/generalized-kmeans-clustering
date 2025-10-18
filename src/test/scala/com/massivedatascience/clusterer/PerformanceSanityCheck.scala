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

package com.massivedatascience.clusterer

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

/**
 * Performance sanity check for CI.
 *
 * This test runs a simple K-Means clustering task and verifies that it completes
 * within a reasonable time budget. It's designed to catch major performance regressions
 * without requiring extensive benchmarking infrastructure.
 */
object PerformanceSanityCheck {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("PerformanceSanityCheck")
      .master("local[*]")
      .config("spark.ui.enabled", "false")
      .getOrCreate()

    import spark.implicits._

    try {
      // Test parameters
      val numPoints = 10000
      val numClusters = 10
      val numDimensions = 20
      val maxIterations = 20
      val timeBudgetMs = 30000 // 30 seconds

      System.out.println(s"Starting performance sanity check:")
      System.out.println(s"  Points: $numPoints")
      System.out.println(s"  Clusters: $numClusters")
      System.out.println(s"  Dimensions: $numDimensions")
      System.out.println(s"  Max iterations: $maxIterations")
      System.out.println(s"  Time budget: ${timeBudgetMs}ms")

      // Generate synthetic data
      val data = (0 until numPoints).map { i =>
        val features = Vectors.dense(Array.fill(numDimensions)(scala.util.Random.nextDouble()))
        (i.toLong, features)
      }.toDF("id", "features")

      // Run K-Means clustering and measure time
      val startTime = System.currentTimeMillis()

      val kmeans = new GeneralizedKMeans()
        .setK(numClusters)
        .setMaxIter(maxIterations)
        .setFeaturesCol("features")
        .setPredictionCol("cluster")
        .setSeed(42L)

      val model = kmeans.fit(data)
      val predictions = model.transform(data)
      val clusterCount = predictions.select("cluster").distinct().count()

      val elapsedTime = System.currentTimeMillis() - startTime

      // Verify results
      System.out.println(s"\nResults:")
      System.out.println(s"  Elapsed time: ${elapsedTime}ms")
      System.out.println(s"  Clusters found: $clusterCount")
      System.out.println(s"  Cost: ${model.summary.trainingCost}")

      // Check for major regression
      if (elapsedTime > timeBudgetMs) {
        System.err.println(s"\nWARNING: Performance regression detected!")
        System.err.println(s"  Expected: < ${timeBudgetMs}ms")
        System.err.println(s"  Actual: ${elapsedTime}ms")
        System.err.println(s"  Ratio: ${elapsedTime.toDouble / timeBudgetMs}x")
        System.exit(1)
      } else {
        System.out.println(s"\nPerformance sanity check PASSED")
        System.out.println(s"  Time ratio: ${elapsedTime.toDouble / timeBudgetMs}x of budget")
      }

      // Log to file for artifact upload
      val logFile = new java.io.PrintWriter("target/perf-sanity.log")
      try {
        logFile.println(s"timestamp,${System.currentTimeMillis()}")
        logFile.println(s"elapsed_ms,$elapsedTime")
        logFile.println(s"clusters,$clusterCount")
        logFile.println(s"cost,${model.summary.trainingCost}")
        logFile.println(s"budget_ms,$timeBudgetMs")
        logFile.println(s"passed,${elapsedTime <= timeBudgetMs}")
      } finally {
        logFile.close()
      }

    } finally {
      spark.stop()
    }
  }
}
