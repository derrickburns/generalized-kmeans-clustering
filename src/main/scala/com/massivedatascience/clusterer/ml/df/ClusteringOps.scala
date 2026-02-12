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

import com.massivedatascience.clusterer.ml.df.kernels._
import com.massivedatascience.clusterer.ml.df.strategies._
import com.massivedatascience.clusterer.ml.df.strategies.impl._
import org.apache.spark.internal.Logging
import org.apache.spark.sql.DataFrame

/** Centralized factory methods for clustering components.
  *
  * Single source of truth for kernel creation, strategy selection, and domain validation. All
  * estimators should delegate to these methods instead of maintaining private copies.
  *
  * ==Usage==
  * {{{
  * val kernel = ClusteringOps.createKernel("kl", smoothing = 1e-10)
  * val assigner = ClusteringOps.createAssignmentStrategy("auto")
  * val updater = ClusteringOps.createUpdateStrategy("kl")
  * val handler = ClusteringOps.createEmptyClusterHandler("reseedRandom", seed = 42L)
  * ClusteringOps.validateDomain(df, "features", "kl")
  * }}}
  */
private[ml] object ClusteringOps extends Logging {

  /** All supported divergence names. */
  val supportedDivergences: Seq[String] = Seq(
    "squaredEuclidean",
    "kl",
    "itakuraSaito",
    "generalizedI",
    "logistic",
    "l1",
    "spherical"
  )

  /** All supported divergence names and aliases. */
  private val validDivergenceNames: Set[String] = Set(
    "squaredEuclidean",
    "kl",
    "itakuraSaito",
    "generalizedI",
    "logistic",
    "l1",
    "manhattan",
    "spherical",
    "cosine"
  )

  /** Create a clustering kernel for the specified divergence.
    *
    * @param divergence
    *   divergence name (e.g., "squaredEuclidean", "kl", "l1")
    * @param smoothing
    *   smoothing parameter for divergences with domain constraints (KL, IS, etc.)
    * @return
    *   configured ClusteringKernel instance
    * @throws IllegalArgumentException
    *   if divergence name is unknown
    */
  def createKernel(divergence: String, smoothing: Double = 1e-10): ClusteringKernel = {
    divergence match {
      case "squaredEuclidean"     => new SquaredEuclideanKernel()
      case "kl"                   => new KLDivergenceKernel(smoothing)
      case "itakuraSaito"         => new ItakuraSaitoKernel(smoothing)
      case "generalizedI"         => new GeneralizedIDivergenceKernel(smoothing)
      case "logistic"             => new LogisticLossKernel(smoothing)
      case "l1" | "manhattan"     => new L1Kernel()
      case "spherical" | "cosine" => new SphericalKernel()
      case _                      =>
        throw new IllegalArgumentException(
          s"Unknown divergence: '$divergence'. " +
            s"Valid options: ${supportedDivergences.mkString(", ")}"
        )
    }
  }

  /** Create an assignment strategy from name.
    *
    * @param strategy
    *   strategy name: "auto", "broadcast", or "crossJoin"
    * @return
    *   configured AssignmentStrategy
    * @throws IllegalArgumentException
    *   if strategy name is unknown
    */
  def createAssignmentStrategy(strategy: String): AssignmentStrategy = {
    strategy match {
      case "broadcast" => new BroadcastUDFAssignment()
      case "crossJoin" => new SECrossJoinAssignment()
      case "auto"      => new AutoAssignment()
      case _           =>
        throw new IllegalArgumentException(
          s"Unknown assignment strategy: '$strategy'. Valid options: auto, broadcast, crossJoin"
        )
    }
  }

  /** Create an update strategy appropriate for the given divergence.
    *
    * L1/Manhattan uses component-wise median; all Bregman divergences use gradient-based mean.
    *
    * @param divergence
    *   divergence name
    * @return
    *   configured UpdateStrategy
    */
  def createUpdateStrategy(divergence: String): UpdateStrategy = {
    divergence match {
      case "l1" | "manhattan" => new MedianUpdateStrategy()
      case _                  => new GradMeanUDAFUpdate()
    }
  }

  /** Create an empty cluster handler.
    *
    * @param strategy
    *   strategy name: "reseedRandom" or "drop"
    * @param seed
    *   random seed for reseed strategy
    * @return
    *   configured EmptyClusterHandler
    * @throws IllegalArgumentException
    *   if strategy name is unknown
    */
  def createEmptyClusterHandler(strategy: String, seed: Long): EmptyClusterHandler = {
    strategy match {
      case "reseedRandom" => new ReseedRandomHandler(seed)
      case "drop"         => new DropEmptyClustersHandler()
      case _              =>
        throw new IllegalArgumentException(
          s"Unknown empty cluster strategy: '$strategy'. Valid options: reseedRandom, drop"
        )
    }
  }

  /** Validate input data domain requirements for the specified divergence.
    *
    * Checks that feature values are in the valid domain (e.g., positive for KL/IS, bounded for
    * logistic). Samples up to `maxSamples` rows for performance.
    *
    * @param df
    *   input DataFrame
    * @param featuresCol
    *   name of features column
    * @param divergence
    *   divergence name
    * @param maxSamples
    *   maximum number of rows to check (default 1000)
    */
  def validateDomain(
      df: DataFrame,
      featuresCol: String,
      divergence: String,
      maxSamples: Int = 1000
  ): Unit = {
    com.massivedatascience.util.DivergenceDomainValidator.validateDataFrame(
      df,
      featuresCol,
      divergence,
      maxSamples = Some(maxSamples)
    )
  }

  /** Check if a divergence name is valid.
    */
  def isValidDivergence(divergence: String): Boolean =
    validDivergenceNames.contains(divergence)
}
