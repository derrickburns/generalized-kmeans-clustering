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

import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/** Configuration for EM algorithm.
  *
  * @param k
  *   number of components
  * @param maxIter
  *   maximum iterations
  * @param tol
  *   convergence tolerance (on log-likelihood)
  * @param kernel
  *   Bregman divergence kernel
  * @param regularization
  *   Dirichlet prior for component weights (0 = no regularization)
  * @param checkpointInterval
  *   checkpoint interval (0 = disabled)
  * @param checkpointDir
  *   checkpoint directory
  */
case class EMConfig(
    k: Int,
    maxIter: Int,
    tol: Double,
    kernel: BregmanKernel,
    regularization: Double = 0.0,
    checkpointInterval: Int = 10,
    checkpointDir: Option[String] = None
) {

  /** Convert to base IteratorConfig. */
  def toIteratorConfig: IteratorConfig =
    IteratorConfig(maxIter, tol, checkpointInterval, checkpointDir)
}

/** Result from EM algorithm including mixture model specific outputs.
  *
  * @param centers
  *   component means (μ_k)
  * @param weights
  *   component weights (π_k), sum to 1
  * @param iterations
  *   number of iterations
  * @param logLikelihoodHistory
  *   log-likelihood at each iteration
  * @param converged
  *   whether converged
  */
case class EMResult(
    centers: Array[Array[Double]],
    weights: Array[Double],
    iterations: Int,
    logLikelihoodHistory: Array[Double],
    converged: Boolean
) {

  /** Convert to generic ClusteringResult. */
  def toClusteringResult: ClusteringResult = ClusteringResult(
    centers = centers,
    iterations = iterations,
    objectiveHistory = logLikelihoodHistory,
    converged = converged,
    metadata = Map(
      "componentWeights"     -> weights,
      "logLikelihoodHistory" -> logLikelihoodHistory
    )
  )
}

/** EM iterator for Bregman mixture models.
  *
  * Implements the Expectation-Maximization algorithm for clustering:
  *
  * '''E-step:''' Compute responsibilities (posterior probabilities)
  * {{{
  * γ_nk = π_k · exp(-D_φ(x_n, μ_k)) / Σ_j π_j · exp(-D_φ(x_n, μ_j))
  * }}}
  *
  * '''M-step:''' Update parameters
  * {{{
  * N_k = Σ_n γ_nk                    (effective count)
  * π_k = N_k / N                     (component weight)
  * μ_k = Σ_n γ_nk · x_n / N_k        (component mean via gradient space)
  * }}}
  *
  * For Bregman divergences, the mean computation uses the dual gradient space:
  * {{{
  * μ_k = invGrad(Σ_n γ_nk · grad(x_n) / N_k)
  * }}}
  *
  * This ensures the correct mean for exponential family distributions.
  */
class BregmanEMIterator extends AbstractClusteringIterator {

  private var emConfig: EMConfig = _

  /** Run EM algorithm with specific configuration.
    *
    * @param df
    *   input DataFrame
    * @param featuresCol
    *   features column name
    * @param weightCol
    *   optional weight column
    * @param initialCenters
    *   initial component means
    * @param config
    *   EM configuration
    * @return
    *   EM result with mixture model parameters
    */
  def runEM(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      initialCenters: Array[Array[Double]],
      config: EMConfig
  ): EMResult = {
    emConfig = config

    // Initialize with uniform weights
    val initialWeights = Array.fill(config.k)(1.0 / config.k)

    val initialState = IterationState(
      df = df,
      parameters = Map(
        "centers" -> initialCenters,
        "weights" -> initialWeights,
        "kernel"  -> config.kernel
      ),
      iteration = 0,
      objective = Double.NegativeInfinity // Log-likelihood starts at -∞
    )

    val result = run(df, featuresCol, weightCol, initialState, config.toIteratorConfig)

    EMResult(
      centers = result.centers,
      weights = result.componentWeights.getOrElse(initialWeights),
      iterations = result.iterations,
      logLikelihoodHistory = result.objectiveHistory,
      converged = result.converged
    )
  }

  override protected def iterate(
      state: IterationState,
      featuresCol: String,
      weightCol: Option[String]
  ): IterationState = {

    val centers = state.parameters("centers").asInstanceOf[Array[Array[Double]]]
    val weights = state.parameters("weights").asInstanceOf[Array[Double]]
    val kernel  = state.parameters("kernel").asInstanceOf[BregmanKernel]
    val k       = centers.length

    // E-step: compute responsibilities
    val (dfWithResponsibilities, logLikelihood) =
      computeResponsibilities(state.df, featuresCol, weightCol, centers, weights, kernel)

    // M-step: update parameters
    val (newCenters, newWeights) =
      updateParameters(dfWithResponsibilities, featuresCol, weightCol, k, kernel)

    // Apply regularization to weights if configured
    val regularizedWeights = if (emConfig.regularization > 0) {
      val alpha   = emConfig.regularization
      val adjusted = newWeights.map(_ + alpha)
      val sum     = adjusted.sum
      adjusted.map(_ / sum)
    } else {
      newWeights
    }

    IterationState(
      df = dfWithResponsibilities,
      parameters = Map(
        "centers" -> newCenters,
        "weights" -> regularizedWeights,
        "kernel"  -> kernel
      ),
      iteration = state.iteration + 1,
      objective = logLikelihood
    )
  }

  /** E-step: Compute responsibilities (posterior probabilities).
    *
    * @return
    *   (DataFrame with responsibilities column, total log-likelihood)
    */
  private def computeResponsibilities(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      weights: Array[Double],
      kernel: BregmanKernel
  ): (DataFrame, Double) = {
    val spark    = df.sparkSession
    val k        = centers.length
    val bcKernel = spark.sparkContext.broadcast(kernel)
    val centersVec = centers.map(Vectors.dense)

    // UDF to compute responsibilities for each point
    val responsibilitiesUDF = udf { (features: Vector) =>
      val logProbs = (0 until k).map { c =>
        val divergence = bcKernel.value.divergence(features, centersVec(c))
        math.log(weights(c)) - divergence
      }.toArray

      // Log-sum-exp for numerical stability
      val maxLogProb  = logProbs.max
      val expSum      = logProbs.map(lp => math.exp(lp - maxLogProb)).sum
      val logNorm     = maxLogProb + math.log(expSum)
      val responsibilities = logProbs.map(lp => math.exp(lp - logNorm))

      // Return (responsibilities, log-likelihood contribution)
      (Vectors.dense(responsibilities), logNorm)
    }

    val withResp = df.withColumn("_em_result", responsibilitiesUDF(col(featuresCol)))
      .withColumn("_responsibilities", col("_em_result._1"))
      .withColumn("_loglik", col("_em_result._2"))
      .drop("_em_result")

    // Compute total log-likelihood
    val totalLogLik = weightCol match {
      case Some(wc) =>
        withResp.select(sum(col("_loglik") * col(wc))).head().getDouble(0)
      case None =>
        withResp.select(sum(col("_loglik"))).head().getDouble(0)
    }

    (withResp.drop("_loglik"), totalLogLik)
  }

  /** M-step: Update component means and weights.
    *
    * Uses gradient-space averaging for Bregman divergences:
    * μ_k = invGrad(Σ_n γ_nk · grad(x_n) / N_k)
    */
  private def updateParameters(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      k: Int,
      kernel: BregmanKernel
  ): (Array[Array[Double]], Array[Double]) = {
    val spark    = df.sparkSession
    val bcKernel = spark.sparkContext.broadcast(kernel)

    // Aggregate using RDD for efficiency
    val data = weightCol match {
      case Some(wc) =>
        df.select(featuresCol, "_responsibilities", wc).rdd.map { row =>
          val features = row.getAs[Vector](0)
          val resp     = row.getAs[Vector](1)
          val weight   = row.getDouble(2)
          (features, resp, weight)
        }
      case None =>
        df.select(featuresCol, "_responsibilities").rdd.map { row =>
          val features = row.getAs[Vector](0)
          val resp     = row.getAs[Vector](1)
          (features, resp, 1.0)
        }
    }

    // Aggregate gradient sums and effective counts per component
    val dim = data.first()._1.size
    val init = (Array.fill(k)(Array.fill(dim)(0.0)), Array.fill(k)(0.0))

    val (gradSums, effectiveCounts) = data.aggregate(init)(
      seqOp = { case ((gradSums, counts), (features, resp, weight)) =>
        val grad = bcKernel.value.grad(features).toArray
        for (c <- 0 until k) {
          val gamma = resp(c) * weight
          counts(c) += gamma
          for (d <- 0 until dim) {
            gradSums(c)(d) += gamma * grad(d)
          }
        }
        (gradSums, counts)
      },
      combOp = { case ((gs1, c1), (gs2, c2)) =>
        for (c <- 0 until k) {
          c1(c) += c2(c)
          for (d <- 0 until dim) {
            gs1(c)(d) += gs2(c)(d)
          }
        }
        (gs1, c1)
      }
    )

    // Compute new centers via inverse gradient
    val newCenters = (0 until k).map { c =>
      if (effectiveCounts(c) > 1e-10) {
        val meanGrad = gradSums(c).map(_ / effectiveCounts(c))
        kernel.invGrad(Vectors.dense(meanGrad)).toArray
      } else {
        // Empty component - keep small random values
        Array.fill(dim)(1e-10)
      }
    }.toArray

    // Compute new weights
    val totalCount = effectiveCounts.sum
    val newWeights = if (totalCount > 1e-10) {
      effectiveCounts.map(_ / totalCount)
    } else {
      Array.fill(k)(1.0 / k)
    }

    (newCenters, newWeights)
  }

  /** For EM, convergence is based on log-likelihood improvement.
    * We override to handle maximization (log-likelihood increases).
    */
  override protected def hasConverged(
      previousObjective: Double,
      currentObjective: Double,
      tol: Double
  ): Boolean = {
    // For log-likelihood, we want to see if improvement is small
    // Note: log-likelihood should increase (or stay same) each iteration
    if (previousObjective == Double.NegativeInfinity) {
      false // First iteration
    } else {
      val improvement = (currentObjective - previousObjective) /
        math.max(math.abs(previousObjective), 1e-10)
      improvement < tol && improvement >= 0
    }
  }
}

object BregmanEMIterator {
  def apply(): BregmanEMIterator = new BregmanEMIterator()
}
