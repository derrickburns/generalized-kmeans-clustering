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

import com.massivedatascience.clusterer.ml.df.BregmanKernel
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/** Helpers for soft (probabilistic) cluster assignment. */
object SoftAssignments {

  /** Compute soft assignment probabilities for each point.
    *
    * Uses a Boltzmann distribution with temperature controlled by `beta`. Probabilities are
    * floor-clamped at `minMembership` and renormalized.
    */
  def withProbabilities(
      df: DataFrame,
      centers: Array[Vector],
      kernel: BregmanKernel,
      beta: Double,
      minMembership: Double,
      featuresCol: String,
      probabilityCol: String
  ): DataFrame = {
    val softAssignUDF = udf { (features: Vector) =>
      val distances         = centers.map(center => kernel.divergence(features, center))
      val minDistance       = distances.min
      val unnormalizedProbs = distances.map(dist => math.exp(-beta * (dist - minDistance)))
      val totalProb         = unnormalizedProbs.sum
      val probabilities     =
        if (totalProb > 1e-100) unnormalizedProbs.map(_ / totalProb).toArray
        else Array.fill(centers.length)(1.0 / centers.length)
      val adjusted          = probabilities.map(p => math.max(p, minMembership))
      val adjustedSum       = adjusted.sum
      Vectors.dense(adjusted.map(_ / adjustedSum))
    }

    df.withColumn(probabilityCol, softAssignUDF(col(featuresCol)))
  }
}
