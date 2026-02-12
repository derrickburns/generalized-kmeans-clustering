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

package com.massivedatascience.clusterer.ml.df

import com.massivedatascience.clusterer.ml.df.kernels.ClusteringKernel
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.DataFrame

import scala.util.Random

/** Shared center initialization methods for clustering algorithms.
  *
  * Provides random and k-means++ initialization using the actual Bregman (or other) divergence for
  * D^2 weighting.
  *
  * ==Supported Modes==
  *   - '''random''': Sample k random points from the dataset
  *   - '''k-means||''' (alias: '''kmeansparallel'''): K-means++ with divergence-proportional
  *     sampling
  *
  * ==Example Usage==
  * {{{
  * val centers = CenterInitializer.initialize(
  *   df, "features", weightCol = None, k = 10,
  *   initMode = "k-means||", initSteps = 2, seed = 42L, kernel = kernel
  * )
  * }}}
  */
private[ml] object CenterInitializer extends Logging {

  /** Dispatch to the appropriate initialization method.
    *
    * @param df
    *   input DataFrame
    * @param featuresCol
    *   name of features column
    * @param weightCol
    *   optional weight column name
    * @param k
    *   number of clusters
    * @param initMode
    *   initialization mode: "random" or "k-means||"
    * @param initSteps
    *   number of oversampling steps (for k-means||)
    * @param seed
    *   random seed for deterministic results
    * @param kernel
    *   clustering kernel (used for k-means++ D^2 weighting)
    * @return
    *   array of k initial cluster centers
    */
  def initialize(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      k: Int,
      initMode: String,
      initSteps: Int,
      seed: Long,
      kernel: ClusteringKernel
  ): Array[Array[Double]] = {
    initMode.toLowerCase match {
      case "random"                        => initializeRandom(df, featuresCol, k, seed)
      case "k-means||" | "kmeansparallel" =>
        initializeKMeansPlusPlus(df, featuresCol, k, seed, kernel)
      case other                           =>
        throw new IllegalArgumentException(
          s"Unknown init mode: '$other'. Valid options: random, k-means||"
        )
    }
  }

  /** Random initialization: sample k random points.
    *
    * Oversamples by 10x to ensure we get at least k points even with small fractions.
    *
    * @param df
    *   input DataFrame
    * @param featuresCol
    *   name of features column
    * @param k
    *   number of centers to select
    * @param seed
    *   random seed
    * @return
    *   array of k initial cluster centers
    */
  def initializeRandom(
      df: DataFrame,
      featuresCol: String,
      k: Int,
      seed: Long
  ): Array[Array[Double]] = {
    val fraction = math.min(1.0, (k * 10.0) / df.count().toDouble)
    df.select(featuresCol)
      .sample(withReplacement = false, fraction, seed)
      .limit(k)
      .collect()
      .map(_.getAs[Vector](0).toArray)
  }

  /** K-means++ initialization with Bregman-native D^2 weighting.
    *
    * Uses the actual clustering divergence for distance-proportional sampling, ensuring proper
    * initialization quality for any divergence (KL, Itakura-Saito, etc.).
    *
    * Algorithm:
    *   1. Select first center uniformly at random 2. For each subsequent center: - Compute D(x,
    *      nearest_center) for all points x - Select next center with probability proportional to
    *      D(x, nearest_center) 3. Repeat until k centers are selected
    *
    * @note
    *   Collects all points to the driver. For very large datasets, consider using a distributed
    *   variant.
    *
    * @param df
    *   input DataFrame
    * @param featuresCol
    *   name of features column
    * @param k
    *   number of centers to select
    * @param seed
    *   random seed
    * @param kernel
    *   clustering kernel for divergence computation
    * @return
    *   array of k initial cluster centers
    */
  def initializeKMeansPlusPlus(
      df: DataFrame,
      featuresCol: String,
      k: Int,
      seed: Long,
      kernel: ClusteringKernel
  ): Array[Array[Double]] = {

    val rand = new Random(seed)

    // Collect all points for local k-means++ (efficient for moderate dataset sizes)
    val allPoints = df.select(featuresCol).collect().map(_.getAs[Vector](0))
    require(
      allPoints.nonEmpty,
      s"Dataset is empty. Cannot initialize k-means++ with k=$k on an empty dataset."
    )

    val n = allPoints.length

    if (n <= k) {
      logInfo(s"Dataset has only $n points (<= k=$k), using all points as centers")
      return allPoints.map(_.toArray)
    }

    logInfo(s"Running Bregman-native k-means++ on $n points with ${kernel.name} divergence")

    // Step 1: Select first center uniformly at random
    val centers = scala.collection.mutable.ArrayBuffer.empty[Array[Double]]
    centers += allPoints(rand.nextInt(n)).toArray

    // Array to store distance to nearest center for each point
    val minDistances = Array.fill(n)(Double.PositiveInfinity)

    // Steps 2-k: Select centers with probability proportional to divergence
    while (centers.length < k) {
      // Update minimum distances with respect to the most recently added center
      val lastCenter = Vectors.dense(centers.last)
      var totalDist  = 0.0

      var i = 0
      while (i < n) {
        val dist = kernel.divergence(allPoints(i), lastCenter)
        if (dist < minDistances(i)) {
          minDistances(i) = dist
        }
        // Handle potential numerical issues
        if (java.lang.Double.isFinite(minDistances(i))) {
          totalDist += minDistances(i)
        }
        i += 1
      }

      // If all distances are zero or invalid, fall back to random selection
      if (totalDist <= 0.0 || !java.lang.Double.isFinite(totalDist)) {
        centers += allPoints(rand.nextInt(n)).toArray
        logInfo(s"K-means++ step ${centers.length}: fallback to random selection")
      } else {
        // Sample with probability proportional to distance (D^2 weighting)
        val threshold = rand.nextDouble() * totalDist
        var cumSum    = 0.0
        var selected  = -1
        i = 0

        while (i < n && selected < 0) {
          if (java.lang.Double.isFinite(minDistances(i))) {
            cumSum += minDistances(i)
          }
          if (cumSum >= threshold) {
            selected = i
          }
          i += 1
        }

        // Fallback to last point if numerical issues
        if (selected < 0) selected = n - 1

        centers += allPoints(selected).toArray

        if (centers.length % 10 == 0 || centers.length == k) {
          logInfo(s"K-means++ progress: ${centers.length}/$k centers selected")
        }
      }
    }

    logInfo(s"K-means++ initialization complete: selected $k centers using ${kernel.name}")
    centers.toArray
  }
}
