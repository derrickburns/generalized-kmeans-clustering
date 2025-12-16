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

package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{ Estimator, Model }
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{
  DefaultParamsReadable,
  DefaultParamsWritable,
  Identifiable,
  MLReadable,
  MLReader,
  MLWritable,
  MLWriter
}
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Parameters for Bregman Mixture Models.
  */
trait BregmanMixtureParams extends GeneralizedKMeansParams {

  /** Number of mixture components. Same as k, aliased for clarity.
    */
  def getNumComponents: Int = $(k)

  /** Regularization parameter (Dirichlet prior for component weights). 0 = no regularization (MLE)
    * > 0 = MAP estimation with symmetric Dirichlet prior
    */
  final val regularization: DoubleParam = new DoubleParam(
    this,
    "regularization",
    "Dirichlet prior parameter for component weights",
    ParamValidators.gtEq(0.0)
  )
  def getRegularization: Double         = $(regularization)

  /** Column name for component probabilities output. */
  final val probabilityCol: Param[String] = new Param[String](
    this,
    "probabilityCol",
    "Column name for component probabilities"
  )
  def getProbabilityCol: String           = $(probabilityCol)

  setDefault(
    regularization -> 0.0,
    probabilityCol -> "probability"
  )
}

/** Bregman Mixture Model - probabilistic clustering via EM algorithm.
  *
  * Fits a mixture model where each component is parameterized by an exponential family distribution
  * corresponding to the chosen Bregman divergence:
  *
  *   - Squared Euclidean → Gaussian mixture
  *   - KL divergence → Multinomial mixture
  *   - Itakura-Saito → Gamma mixture
  *
  * ==Algorithm==
  *
  * Uses Expectation-Maximization (EM):
  *
  * '''E-step:''' Compute responsibilities (posterior probabilities)
  * {{{
  * γ_nk = π_k · p(x_n | μ_k) / Σ_j π_j · p(x_n | μ_j)
  *      ∝ π_k · exp(-D_φ(x_n, μ_k))
  * }}}
  *
  * '''M-step:''' Update parameters
  * {{{
  * N_k = Σ_n γ_nk                           (effective count)
  * π_k = (N_k + α - 1) / (N + K(α - 1))    (component weight with prior)
  * μ_k = invGrad(Σ_n γ_nk · grad(x_n) / N_k)  (mean via gradient)
  * }}}
  *
  * ==Example Usage==
  *
  * {{{
  * val bmm = new BregmanMixtureModel()
  *   .setK(3)
  *   .setDivergence("kl")
  *   .setRegularization(1.0)  // Symmetric Dirichlet prior
  *   .setMaxIter(100)
  *
  * val model = bmm.fit(data)
  *
  * // Get component weights
  * println(s"Weights: ${model.weights.mkString(", ")}")
  *
  * // Transform with probabilities
  * val predictions = model.transform(data)
  * predictions.select("features", "prediction", "probability").show()
  * }}}
  *
  * ==Advantages over Hard Clustering==
  *
  *   - Proper probabilistic model (can compute likelihoods)
  *   - Soft assignments reveal cluster overlap
  *   - Natural handling of uncertainty
  *   - Regularization prevents empty components
  *
  * @see
  *   [[SoftKMeans]] for a simpler soft clustering approach
  * @see
  *   Banerjee et al. (2005): "Clustering with Bregman Divergences"
  */
class BregmanMixture(override val uid: String)
    extends Estimator[BregmanMixtureModelInstance]
    with BregmanMixtureParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("bregmanmixture"))

  // Parameter setters
  def setK(value: Int): this.type                 = set(k, value)
  def setNumComponents(value: Int): this.type     = set(k, value)
  def setDivergence(value: String): this.type     = set(divergence, value)
  def setSmoothing(value: Double): this.type      = set(smoothing, value)
  def setRegularization(value: Double): this.type = set(regularization, value)
  def setFeaturesCol(value: String): this.type    = set(featuresCol, value)
  def setPredictionCol(value: String): this.type  = set(predictionCol, value)
  def setProbabilityCol(value: String): this.type = set(probabilityCol, value)
  def setWeightCol(value: String): this.type      = set(weightCol, value)
  def setMaxIter(value: Int): this.type           = set(maxIter, value)
  def setTol(value: Double): this.type            = set(tol, value)
  def setSeed(value: Long): this.type             = set(seed, value)

  override def fit(dataset: Dataset[_]): BregmanMixtureModelInstance = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()

    logInfo(
      s"Starting Bregman Mixture Model: k=${$(k)}, divergence=${$(divergence)}, " +
        s"regularization=${$(regularization)}, maxIter=${$(maxIter)}"
    )

    // Validate input
    com.massivedatascience.util.DivergenceDomainValidator.validateDataFrame(
      df,
      $(featuresCol),
      $(divergence),
      maxSamples = Some(1000)
    )

    df.cache()
    try {
      val kernel    = createKernel()
      val startTime = System.currentTimeMillis()

      // Initialize centers randomly
      val initialCenters = initializeCenters(df)

      // Configure EM
      val emConfig = EMConfig(
        k = $(k),
        maxIter = $(maxIter),
        tol = $(tol),
        kernel = kernel,
        regularization = $(regularization)
      )

      // Run EM
      val emIterator = new BregmanEMIterator()
      val result     = emIterator.runEM(
        df,
        $(featuresCol),
        if (hasWeightCol) Some($(weightCol)) else None,
        initialCenters,
        emConfig
      )

      val elapsed = System.currentTimeMillis() - startTime
      logInfo(
        s"Bregman Mixture Model completed: ${result.iterations} iterations, " +
          s"converged=${result.converged}, finalLogLik=${result.logLikelihoodHistory.lastOption
              .getOrElse(Double.NaN)}"
      )

      // Create model
      val centersAsVectors = result.centers.map(Vectors.dense)
      val model            = new BregmanMixtureModelInstance(
        uid,
        centersAsVectors,
        result.weights,
        kernel
      )
      model.modelDivergence = $(divergence)
      model.modelSmoothing = $(smoothing)
      copyValues(model.setParent(this))

      // Training summary
      val summary = TrainingSummary(
        algorithm = "BregmanMixtureModel",
        k = $(k),
        effectiveK = result.centers.length,
        dim = result.centers.headOption.map(_.length).getOrElse(0),
        numPoints = df.count(),
        iterations = result.iterations,
        converged = result.converged,
        distortionHistory = result.logLikelihoodHistory,
        movementHistory = Array.empty,
        assignmentStrategy = "EM",
        divergence = $(divergence),
        elapsedMillis = elapsed
      )
      model.trainingSummary = Some(summary)

      model
    } finally {
      df.unpersist()
    }
  }

  private def createKernel(): BregmanKernel = {
    BregmanKernel.create($(divergence), $(smoothing))
  }

  private def initializeCenters(df: DataFrame): Array[Array[Double]] = {
    val fraction = math.min(1.0, ($(k) * 10.0) / df.count().toDouble)
    df.select($(featuresCol))
      .sample(withReplacement = false, fraction, $(seed))
      .limit($(k))
      .collect()
      .map(_.getAs[Vector](0).toArray)
  }

  override def copy(extra: ParamMap): BregmanMixture = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object BregmanMixture extends DefaultParamsReadable[BregmanMixture] {
  override def load(path: String): BregmanMixture = super.load(path)
}

/** Model from Bregman Mixture Model fitting.
  *
  * @param uid
  *   unique identifier
  * @param means
  *   component means (μ_k)
  * @param weights
  *   component weights (π_k), sum to 1
  * @param kernel
  *   Bregman divergence kernel
  */
class BregmanMixtureModelInstance(
    override val uid: String,
    val means: Array[Vector],
    val weights: Array[Double],
    val kernel: BregmanKernel
) extends Model[BregmanMixtureModelInstance]
    with BregmanMixtureParams
    with MLWritable
    with Logging
    with HasTrainingSummary
    with CentroidModelHelpers {

  // Mutable fields for persistence
  private[ml] var modelDivergence: String = "squaredEuclidean"
  private[ml] var modelSmoothing: Double  = 1e-10

  /** Optional access to training summary without throwing. */
  def summaryOption: Option[TrainingSummary] = trainingSummary

  /** Number of mixture components. */
  def numComponents: Int = means.length

  override def clusterCentersAsVectors: Array[Vector] = means

  /** Component means as array. */
  def componentMeans: Array[Vector] = means

  /** Component weights (mixing proportions). */
  def componentWeights: Array[Double] = weights

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()

    val bcKernel  = df.sparkSession.sparkContext.broadcast(kernel)
    val bcMeans   = df.sparkSession.sparkContext.broadcast(means)
    val bcWeights = df.sparkSession.sparkContext.broadcast(weights)
    val k         = numComponents

    // Compute probabilities and predictions
    val predictUDF = udf { (features: Vector) =>
      val logProbs = (0 until k).map { c =>
        val divergence = bcKernel.value.divergence(features, bcMeans.value(c))
        math.log(bcWeights.value(c)) - divergence
      }.toArray

      // Log-sum-exp normalization
      val maxLogProb = logProbs.max
      val expSum     = logProbs.map(lp => math.exp(lp - maxLogProb)).sum
      val logNorm    = maxLogProb + math.log(expSum)
      val probs      = logProbs.map(lp => math.exp(lp - logNorm))

      // Best cluster
      val prediction = probs.zipWithIndex.maxBy(_._1)._2

      (prediction, Vectors.dense(probs))
    }

    val result = df
      .withColumn("_bmm_result", predictUDF(col($(featuresCol))))
      .withColumn($(predictionCol), col("_bmm_result._1"))
      .withColumn($(probabilityCol), col("_bmm_result._2"))
      .drop("_bmm_result")

    result
  }

  /** Compute log-likelihood of data under the mixture model. */
  def logLikelihood(dataset: Dataset[_]): Double = {
    val df = dataset.toDF()

    val bcKernel  = df.sparkSession.sparkContext.broadcast(kernel)
    val bcMeans   = df.sparkSession.sparkContext.broadcast(means)
    val bcWeights = df.sparkSession.sparkContext.broadcast(weights)
    val k         = numComponents

    val logLikUDF = udf { (features: Vector) =>
      val logProbs = (0 until k).map { c =>
        val divergence = bcKernel.value.divergence(features, bcMeans.value(c))
        math.log(bcWeights.value(c)) - divergence
      }.toArray

      val maxLogProb = logProbs.max
      val expSum     = logProbs.map(lp => math.exp(lp - maxLogProb)).sum
      maxLogProb + math.log(expSum)
    }

    df.select(logLikUDF(col($(featuresCol))).as("loglik")).agg(sum("loglik")).head().getDouble(0)
  }

  /** Compute BIC (Bayesian Information Criterion). Lower is better. */
  def bic(dataset: Dataset[_]): Double = {
    val n         = dataset.count()
    val logLik    = logLikelihood(dataset)
    val dim       = means.headOption.map(_.size).getOrElse(0)
    val numParams = numComponents * dim + numComponents - 1 // means + weights
    -2 * logLik + numParams * math.log(n.toDouble)
  }

  /** Compute AIC (Akaike Information Criterion). Lower is better. */
  def aic(dataset: Dataset[_]): Double = {
    val logLik    = logLikelihood(dataset)
    val dim       = means.headOption.map(_.size).getOrElse(0)
    val numParams = numComponents * dim + numComponents - 1
    -2 * logLik + 2 * numParams
  }

  override def copy(extra: ParamMap): BregmanMixtureModelInstance = {
    val copied = new BregmanMixtureModelInstance(uid, means, weights, kernel)
    copyValues(copied, extra).setParent(parent)
    copied.modelDivergence = this.modelDivergence
    copied.modelSmoothing = this.modelSmoothing
    copied.trainingSummary = this.trainingSummary
    copied
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new BregmanMixtureModelInstance.BregmanMixtureModelWriter(this)
}

object BregmanMixtureModelInstance extends MLReadable[BregmanMixtureModelInstance] {

  override def read: MLReader[BregmanMixtureModelInstance] = new BregmanMixtureModelReader

  private class BregmanMixtureModelWriter(instance: BregmanMixtureModelInstance)
      extends MLWriter
      with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession
      logInfo(s"Saving BregmanMixtureModelInstance to $path")

      // (center_id, weight, vector) - weight is component weight
      val centersData =
        instance.means.indices.map(i => (i, instance.weights(i), instance.means(i)))

      val centersHash = writeCenters(spark, path, centersData)

      val params: Map[String, Any] = Map(
        "k"              -> instance.numComponents,
        "featuresCol"    -> instance.getOrDefault(instance.featuresCol),
        "predictionCol"  -> instance.getOrDefault(instance.predictionCol),
        "probabilityCol" -> instance.getOrDefault(instance.probabilityCol),
        "divergence"     -> instance.modelDivergence,
        "smoothing"      -> instance.modelSmoothing,
        "regularization" -> instance.getOrDefault(instance.regularization)
      )

      val k   = instance.numComponents
      val dim = instance.means.headOption.map(_.size).getOrElse(0)

      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaObj: Map[String, Any]             = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "BregmanMixtureModelInstance",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "divergence"         -> instance.modelDivergence,
        "k"                  -> k,
        "dim"                -> dim,
        "uid"                -> instance.uid,
        "params"             -> params,
        "centers"            -> Map[String, Any](
          "count"    -> k,
          "ordering" -> "center_id ASC (0..k-1)",
          "storage"  -> "parquet"
        ),
        "checksums"          -> Map[String, String](
          "centersParquetSHA256" -> centersHash
        )
      )

      val json = Serialization.write(metaObj)
      writeMetadata(path, json)
      logInfo(s"BregmanMixtureModelInstance saved to $path")
    }
  }

  private class BregmanMixtureModelReader
      extends MLReader[BregmanMixtureModelInstance]
      with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): BregmanMixtureModelInstance = {
      val spark = sparkSession
      logInfo(s"Loading BregmanMixtureModelInstance from $path")

      val metaStr                               = readMetadata(path)
      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaJ                                 = JsonMethods.parse(metaStr)

      val layoutVersion = (metaJ \ "layoutVersion").extract[Int]
      val k             = (metaJ \ "k").extract[Int]
      val dim           = (metaJ \ "dim").extract[Int]
      val uid           = (metaJ \ "uid").extract[String]
      val divergence    = (metaJ \ "divergence").extract[String]

      val centersDF = readCenters(spark, path)
      val rows      = centersDF.collect()
      validateMetadata(layoutVersion, k, dim, rows.length)

      val sortedRows = rows.sortBy(_.getInt(0))
      val means      = sortedRows.map(_.getAs[Vector]("vector"))
      val weights    = sortedRows.map(_.getDouble(1))

      val paramsJ   = metaJ \ "params"
      val smoothing = (paramsJ \ "smoothing").extract[Double]

      val kernel = BregmanKernel.create(divergence, smoothing)

      val model = new BregmanMixtureModelInstance(uid, means, weights, kernel)
      model.modelDivergence = divergence
      model.modelSmoothing = smoothing

      model.set(model.featuresCol, (paramsJ \ "featuresCol").extract[String])
      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])
      model.set(model.probabilityCol, (paramsJ \ "probabilityCol").extract[String])

      logInfo(s"BregmanMixtureModelInstance loaded from $path")
      model
    }
  }
}
