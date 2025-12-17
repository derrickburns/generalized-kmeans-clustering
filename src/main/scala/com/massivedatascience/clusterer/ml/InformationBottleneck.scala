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

import com.massivedatascience.clusterer.ml.df.MutualInformation
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{ Estimator, Model }
import org.apache.spark.ml.linalg.{ Vector, Vectors, SQLDataTypes }
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/** Parameters for Information Bottleneck clustering. */
trait InformationBottleneckParams
    extends Params
    with HasFeaturesCol
    with HasPredictionCol
    with HasMaxIter
    with HasTol
    with HasSeed {

  /** Number of clusters (compressed representations). Default: 2. */
  final val k   = new IntParam(this, "k", "Number of clusters", ParamValidators.gt(1))
  def getK: Int = $(k)

  /** Relevance column name. If not set, uses self-information mode. */
  final val relevanceCol      = new Param[String](
    this,
    "relevanceCol",
    "Column containing relevance variable Y (discrete labels)"
  )
  def getRelevanceCol: String = $(relevanceCol)

  /** Trade-off parameter β controlling compression vs. relevance.
    *
    * Higher β → preserve more information about Y (less compression) Lower β → more compression
    * (lose information about Y)
    *
    * Default: 1.0
    */
  final val beta      = new DoubleParam(
    this,
    "beta",
    "Trade-off parameter (compression vs. relevance)",
    ParamValidators.gt(0.0)
  )
  def getBeta: Double = $(beta)

  /** Column name for soft membership probabilities. Default: "probabilities" */
  final val probabilityCol      =
    new Param[String](this, "probabilityCol", "Column for membership probabilities")
  def getProbabilityCol: String = $(probabilityCol)

  /** Number of bins for discretizing continuous features. Default: 20 */
  final val numBins   = new IntParam(
    this,
    "numBins",
    "Number of bins for feature discretization",
    ParamValidators.gt(1)
  )
  def getNumBins: Int = $(numBins)

  /** Smoothing parameter for probability estimation. Default: 1e-10 */
  final val smoothing      = new DoubleParam(
    this,
    "smoothing",
    "Smoothing for probability estimation",
    ParamValidators.gt(0.0)
  )
  def getSmoothing: Double = $(smoothing)

  setDefault(
    k              -> 2,
    beta           -> 1.0,
    maxIter        -> 100,
    tol            -> 1e-6,
    probabilityCol -> "probabilities",
    numBins        -> 20,
    smoothing      -> 1e-10,
    seed           -> 42L
  )

  protected def hasRelevanceCol: Boolean = isDefined(relevanceCol) && $(relevanceCol).nonEmpty

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    require(
      schema.fieldNames.contains($(featuresCol)),
      s"Features column '${$(featuresCol)}' not found"
    )
    if (hasRelevanceCol) {
      require(
        schema.fieldNames.contains($(relevanceCol)),
        s"Relevance column '${$(relevanceCol)}' not found"
      )
    }
    schema
      .add($(predictionCol), IntegerType, nullable = false)
      .add($(probabilityCol), SQLDataTypes.VectorType, nullable = false)
  }
}

/** Information Bottleneck clustering using the Blahut-Arimoto algorithm.
  *
  * The Information Bottleneck (IB) method finds a compressed representation T of data X that
  * preserves maximum information about a relevance variable Y. It solves:
  *
  * '''min I(X; T) - β · I(T; Y)'''
  *
  * where:
  *   - I(X; T) is the mutual information between data and clusters (compression)
  *   - I(T; Y) is the mutual information between clusters and relevance (preservation)
  *   - β controls the trade-off (higher β → more preservation, less compression)
  *
  * ==Algorithm (Blahut-Arimoto)==
  *
  * The algorithm iteratively updates:
  *   1. p(t) = Σ_x p(t|x) p(x) — cluster prior
  *   1. p(y|t) = Σ_x p(y|x) p(t|x) p(x) / p(t) — cluster-relevance distribution
  *   1. p(t|x) ∝ p(t) exp(-β · D_KL(p(y|x) || p(y|t))) — soft assignments
  *
  * ==Connection to Bregman Clustering==
  *
  * IB with KL divergence is equivalent to soft clustering with Bregman divergence on probability
  * simplex. This provides theoretical grounding for the algorithm.
  *
  * ==Modes of Operation==
  *
  *   - '''With relevanceCol:''' Standard IB - compress X while preserving Y
  *   - '''Without relevanceCol:''' Self-information mode (rate-distortion like)
  *
  * ==Example Usage==
  *
  * {{{
  * // Standard IB with relevance variable
  * val ib = new InformationBottleneck()
  *   .setK(5)
  *   .setBeta(2.0)              // Trade-off parameter
  *   .setFeaturesCol("features")
  *   .setRelevanceCol("label")  // Relevance variable
  *   .setMaxIter(100)
  *
  * val model = ib.fit(dataset)
  * val predictions = model.transform(dataset)
  *
  * // Access IB curve information
  * println(s"I(X;T) = ${model.compressionInfo}")
  * println(s"I(T;Y) = ${model.relevanceInfo}")
  * }}}
  *
  * ==Use Cases==
  *
  *   - '''Feature compression:''' Reduce dimensionality while preserving class information
  *   - '''Document clustering:''' Cluster documents by topic while preserving word distributions
  *   - '''Word clustering:''' Cluster words with similar contexts
  *   - '''Neural network analysis:''' Analyze information flow through layers
  *
  * @see
  *   Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method.
  * @param uid
  *   Unique identifier
  */
class InformationBottleneck(override val uid: String)
    extends Estimator[InformationBottleneckModel]
    with InformationBottleneckParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("ib"))

  // Parameter setters
  def setK(value: Int): this.type                 = set(k, value)
  def setBeta(value: Double): this.type           = set(beta, value)
  def setRelevanceCol(value: String): this.type   = set(relevanceCol, value)
  def setFeaturesCol(value: String): this.type    = set(featuresCol, value)
  def setPredictionCol(value: String): this.type  = set(predictionCol, value)
  def setProbabilityCol(value: String): this.type = set(probabilityCol, value)
  def setNumBins(value: Int): this.type           = set(numBins, value)
  def setSmoothing(value: Double): this.type      = set(smoothing, value)
  def setMaxIter(value: Int): this.type           = set(maxIter, value)
  def setTol(value: Double): this.type            = set(tol, value)
  def setSeed(value: Long): this.type             = set(seed, value)

  override def fit(dataset: Dataset[_]): InformationBottleneckModel = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()
    logInfo(s"Starting Information Bottleneck: k=${$(k)}, β=${$(beta)}, maxIter=${$(maxIter)}")

    df.cache()
    try {
      // Extract and discretize data
      val (discreteX, discreteY, numX, numY, pX) = prepareData(df)
      logInfo(s"Data: ${discreteX.length} points, $numX X-states, $numY Y-states")

      // Estimate joint distribution p(x, y)
      val jointXY = MutualInformation.estimateJoint(discreteX, discreteY, numX, numY)

      // Compute p(y|x) for the IB iterations
      val pYgivenX = MutualInformation.conditionalYgivenX(jointXY, $(smoothing))

      // Run Blahut-Arimoto algorithm
      val (pTgivenX, pT, pYgivenT, iterations, converged, history) =
        runBlahutArimoto(pX, pYgivenX, numX, numY)

      // Compute information metrics
      val compressionInfo = computeCompressionInfo(pTgivenX, pX, pT)
      val relevanceInfo   = computeRelevanceInfo(pTgivenX, pX, pYgivenX, pT, pYgivenT)

      logInfo(f"IB converged=$converged after $iterations iterations")
      logInfo(f"I(X;T) = $compressionInfo%.4f nats, I(T;Y) = $relevanceInfo%.4f nats")

      // Build model
      val model = new InformationBottleneckModel(uid, pTgivenX, pT, pYgivenT)
      model.setCompressionInfo(compressionInfo)
      model.setRelevanceInfo(relevanceInfo)
      model.setIterations(iterations)
      model.setConverged(converged)
      model.setInfoHistory(history)
      copyValues(model.setParent(this))
    } finally {
      df.unpersist()
    }
  }

  /** Prepare data by discretizing features and relevance variable. */
  private def prepareData(
      df: DataFrame
  ): (Array[Int], Array[Int], Int, Int, Array[Double]) = {
    // Collect features
    val features = df.select($(featuresCol)).collect().map(_.getAs[Vector](0))
    val n        = features.length

    // Discretize features (use hash-based binning for simplicity)
    val numBinsVal = $(numBins)
    val discreteX  = features.map { v =>
      // Simple discretization: hash the vector to a bin
      val hashCode = java.util.Arrays.hashCode(v.toArray)
      math.abs(hashCode % (numBinsVal * numBinsVal))
    }
    val numX       = discreteX.max + 1

    // Discretize or use relevance variable
    val (discreteY, numY) = if (hasRelevanceCol) {
      val relCol   = df.select($(relevanceCol)).collect()
      val relField = df.schema($(relevanceCol))

      relField.dataType match {
        case IntegerType | LongType =>
          val labels = relCol.map(_.getAs[Number](0).intValue())
          val minY   = labels.min
          val maxY   = labels.max
          (labels.map(_ - minY), maxY - minY + 1)

        case DoubleType | FloatType =>
          // Bin continuous relevance
          val values = relCol.map(_.getAs[Number](0).doubleValue())
          val minV   = values.min
          val maxV   = values.max
          val range  = maxV - minV + 1e-10
          val binned = values.map(v => ((v - minV) / range * numBinsVal).toInt.min(numBinsVal - 1))
          (binned, numBinsVal)

        case StringType =>
          val labels   = relCol.map(_.getString(0))
          val labelMap = labels.distinct.sorted.zipWithIndex.toMap
          (labels.map(labelMap), labelMap.size)

        case _ =>
          throw new IllegalArgumentException(
            s"Unsupported relevance type: ${relField.dataType}"
          )
      }
    } else {
      // Self-information mode: Y = X (use same discretization)
      (discreteX.clone(), numX)
    }

    // Compute p(X)
    val counts = new Array[Double](numX)
    var i      = 0
    while (i < n) {
      counts(discreteX(i)) += 1.0
      i += 1
    }
    val pX     = counts.map(_ / n)

    (discreteX, discreteY, numX, numY, pX)
  }

  /** Run Blahut-Arimoto algorithm for IB.
    *
    * @return
    *   (p(t|x), p(t), p(y|t), iterations, converged, info history)
    */
  private def runBlahutArimoto(
      pX: Array[Double],
      pYgivenX: Array[Array[Double]],
      numX: Int,
      numY: Int
  ): (Array[Array[Double]], Array[Double], Array[Array[Double]], Int, Boolean, Array[Double]) = {

    val numT     = $(k)
    val betaVal  = $(beta)
    val maxIt    = $(maxIter)
    val tolVal   = $(tol)
    val smooth   = $(smoothing)
    val rng      = new scala.util.Random($(seed))
    val infoHist = scala.collection.mutable.ArrayBuffer[Double]()

    // Initialize p(t|x) randomly
    var pTgivenX = Array.fill(numX) {
      val row = Array.fill(numT)(rng.nextDouble() + smooth)
      MutualInformation.normalize(row)
    }

    var pT       = computePT(pTgivenX, pX)
    var pYgivenT = computePYgivenT(pTgivenX, pX, pYgivenX, pT, numT, numY, smooth)

    var iteration     = 0
    var converged     = false
    var prevObjective = Double.MaxValue

    while (iteration < maxIt && !converged) {
      iteration += 1

      // Update p(t|x) using IB update rule
      val newPTgivenX = updatePTgivenX(pT, pYgivenX, pYgivenT, betaVal, smooth)

      // Update p(t) and p(y|t)
      pT = computePT(newPTgivenX, pX)
      pYgivenT = computePYgivenT(newPTgivenX, pX, pYgivenX, pT, numT, numY, smooth)

      // Compute objective: I(X;T) - β * I(T;Y)
      val iXT       = computeCompressionInfo(newPTgivenX, pX, pT)
      val iTY       = computeRelevanceInfo(newPTgivenX, pX, pYgivenX, pT, pYgivenT)
      val objective = iXT - betaVal * iTY
      infoHist += objective

      // Check convergence
      val improvement = math.abs(prevObjective - objective)
      if (improvement < tolVal) {
        converged = true
        logDebug(s"Converged at iteration $iteration (improvement=$improvement)")
      }

      prevObjective = objective
      pTgivenX = newPTgivenX
    }

    (pTgivenX, pT, pYgivenT, iteration, converged, infoHist.toArray)
  }

  /** Compute p(t) = Σ_x p(t|x) p(x). */
  private def computePT(pTgivenX: Array[Array[Double]], pX: Array[Double]): Array[Double] = {
    val numX = pTgivenX.length
    val numT = pTgivenX(0).length
    val pT   = new Array[Double](numT)

    var x = 0
    while (x < numX) {
      var t = 0
      while (t < numT) {
        pT(t) += pTgivenX(x)(t) * pX(x)
        t += 1
      }
      x += 1
    }

    MutualInformation.normalize(pT)
  }

  /** Compute p(y|t) = Σ_x p(y|x) p(t|x) p(x) / p(t). */
  private def computePYgivenT(
      pTgivenX: Array[Array[Double]],
      pX: Array[Double],
      pYgivenX: Array[Array[Double]],
      pT: Array[Double],
      numT: Int,
      numY: Int,
      smooth: Double
  ): Array[Array[Double]] = {
    val numX     = pTgivenX.length
    val pYgivenT = Array.fill(numT, numY)(0.0)

    var t = 0
    while (t < numT) {
      var y      = 0
      while (y < numY) {
        var sum = 0.0
        var x   = 0
        while (x < numX) {
          sum += pYgivenX(x)(y) * pTgivenX(x)(t) * pX(x)
          x += 1
        }
        pYgivenT(t)(y) = sum / (pT(t) + smooth)
        y += 1
      }
      // Normalize row
      val rowSum = pYgivenT(t).sum
      if (rowSum > 0) {
        var y2 = 0
        while (y2 < numY) {
          pYgivenT(t)(y2) /= rowSum
          y2 += 1
        }
      }
      t += 1
    }

    pYgivenT
  }

  /** Update p(t|x) ∝ p(t) exp(-β * D_KL(p(y|x) || p(y|t))). */
  private def updatePTgivenX(
      pT: Array[Double],
      pYgivenX: Array[Array[Double]],
      pYgivenT: Array[Array[Double]],
      beta: Double,
      smooth: Double
  ): Array[Array[Double]] = {
    val numX = pYgivenX.length
    val numT = pT.length

    val newPTgivenX = new Array[Array[Double]](numX)

    var x = 0
    while (x < numX) {
      newPTgivenX(x) = new Array[Double](numT)
      var t      = 0
      while (t < numT) {
        val klDiv = MutualInformation.klDivergence(pYgivenX(x), pYgivenT(t), smooth)
        newPTgivenX(x)(t) = (pT(t) + smooth) * math.exp(-beta * klDiv)
        t += 1
      }
      // Normalize
      val rowSum = newPTgivenX(x).sum
      if (rowSum > 0) {
        var t2 = 0
        while (t2 < numT) {
          newPTgivenX(x)(t2) /= rowSum
          t2 += 1
        }
      } else {
        // Fallback to uniform
        java.util.Arrays.fill(newPTgivenX(x), 1.0 / numT)
      }
      x += 1
    }

    newPTgivenX
  }

  /** Compute I(X; T) = Σ_x,t p(x) p(t|x) log(p(t|x) / p(t)). */
  private def computeCompressionInfo(
      pTgivenX: Array[Array[Double]],
      pX: Array[Double],
      pT: Array[Double]
  ): Double = {
    val numX   = pTgivenX.length
    val numT   = pT.length
    val smooth = $(smoothing)
    var mi     = 0.0

    var x = 0
    while (x < numX) {
      var t = 0
      while (t < numT) {
        val pxt = pX(x) * pTgivenX(x)(t)
        if (pxt > smooth) {
          mi += pxt * math.log((pTgivenX(x)(t) + smooth) / (pT(t) + smooth))
        }
        t += 1
      }
      x += 1
    }

    math.max(0.0, mi)
  }

  /** Compute I(T; Y) = Σ_t,y p(t,y) log(p(t,y) / (p(t) p(y))). */
  private def computeRelevanceInfo(
      pTgivenX: Array[Array[Double]],
      pX: Array[Double],
      pYgivenX: Array[Array[Double]],
      pT: Array[Double],
      pYgivenT: Array[Array[Double]]
  ): Double = {
    val numT   = pT.length
    val numY   = pYgivenT(0).length
    val numX   = pX.length
    val smooth = $(smoothing)

    // Compute p(Y)
    val pY = new Array[Double](numY)
    var x  = 0
    while (x < numX) {
      var y = 0
      while (y < numY) {
        pY(y) += pX(x) * pYgivenX(x)(y)
        y += 1
      }
      x += 1
    }

    // Compute I(T;Y) = Σ_t p(t) Σ_y p(y|t) log(p(y|t) / p(y))
    var mi = 0.0
    var t  = 0
    while (t < numT) {
      var y = 0
      while (y < numY) {
        val pyt = pT(t) * pYgivenT(t)(y)
        if (pyt > smooth && pY(y) > smooth) {
          mi += pyt * math.log((pYgivenT(t)(y) + smooth) / (pY(y) + smooth))
        }
        y += 1
      }
      t += 1
    }

    math.max(0.0, mi)
  }

  override def copy(extra: ParamMap): InformationBottleneck = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object InformationBottleneck extends DefaultParamsReadable[InformationBottleneck] {
  override def load(path: String): InformationBottleneck = super.load(path)
}

/** Model produced by Information Bottleneck clustering.
  *
  * Contains the soft assignment mapping p(t|x) and decoder p(y|t).
  */
class InformationBottleneckModel(
    override val uid: String,
    val encoderMatrix: Array[Array[Double]],
    val clusterPrior: Array[Double],
    val decoderMatrix: Array[Array[Double]]
) extends Model[InformationBottleneckModel]
    with InformationBottleneckParams
    with MLWritable
    with Logging {

  // Metrics
  private var _compressionInfo: Double    = 0.0
  private var _relevanceInfo: Double      = 0.0
  private var _iterations: Int            = 0
  private var _converged: Boolean         = false
  private var _infoHistory: Array[Double] = Array.empty

  def setCompressionInfo(v: Double): this.type    = { _compressionInfo = v; this }
  def setRelevanceInfo(v: Double): this.type      = { _relevanceInfo = v; this }
  def setIterations(v: Int): this.type            = { _iterations = v; this }
  def setConverged(v: Boolean): this.type         = { _converged = v; this }
  def setInfoHistory(v: Array[Double]): this.type = { _infoHistory = v; this }

  /** Mutual information I(X; T) - compression. */
  def compressionInfo: Double = _compressionInfo

  /** Mutual information I(T; Y) - relevance preservation. */
  def relevanceInfo: Double = _relevanceInfo

  /** Number of iterations run. */
  def iterations: Int = _iterations

  /** Whether algorithm converged. */
  def converged: Boolean = _converged

  /** History of IB objective values. */
  def infoHistory: Array[Double] = _infoHistory

  /** Number of clusters (|T|). */
  def numClusters: Int = clusterPrior.length

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()

    val numBinsVal = $(numBins)
    val encoder    = encoderMatrix

    // UDF for soft assignment
    val assignUDF = udf { (features: Vector) =>
      // Discretize using same method as training
      val hashCode   = java.util.Arrays.hashCode(features.toArray)
      val discreteX  = math.abs(hashCode % (numBinsVal * numBinsVal))
      val encoderIdx = discreteX.min(encoder.length - 1)

      // Get p(t|x) for this point
      val probs = if (encoderIdx < encoder.length) {
        encoder(encoderIdx)
      } else {
        // Fallback to prior
        clusterPrior
      }

      Vectors.dense(probs)
    }

    // UDF for hard assignment
    val predUDF = udf { (probs: Vector) =>
      var maxProb = -1.0
      var maxIdx  = 0
      val arr     = probs.toArray
      var i       = 0
      while (i < arr.length) {
        if (arr(i) > maxProb) {
          maxProb = arr(i)
          maxIdx = i
        }
        i += 1
      }
      maxIdx
    }

    df.withColumn($(probabilityCol), assignUDF(col($(featuresCol))))
      .withColumn($(predictionCol), predUDF(col($(probabilityCol))))
  }

  /** Get a summary of the IB model. */
  def summary: String = {
    f"""Information Bottleneck Model Summary
       |====================================
       |Clusters (|T|): $numClusters
       |Compression I(X;T): ${_compressionInfo}%.4f nats
       |Relevance I(T;Y): ${_relevanceInfo}%.4f nats
       |Iterations: ${_iterations}
       |Converged: ${_converged}
       |""".stripMargin
  }

  override def copy(extra: ParamMap): InformationBottleneckModel = {
    val copied = new InformationBottleneckModel(uid, encoderMatrix, clusterPrior, decoderMatrix)
    copyValues(copied, extra).setParent(parent)
    copied.setCompressionInfo(_compressionInfo)
    copied.setRelevanceInfo(_relevanceInfo)
    copied.setIterations(_iterations)
    copied.setConverged(_converged)
    copied.setInfoHistory(_infoHistory)
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new InformationBottleneckModel.IBModelWriter(this)
}

object InformationBottleneckModel extends MLReadable[InformationBottleneckModel] {

  override def read: MLReader[InformationBottleneckModel] = new IBModelReader

  override def load(path: String): InformationBottleneckModel = super.load(path)

  private[ml] class IBModelWriter(instance: InformationBottleneckModel)
      extends MLWriter
      with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession
      logInfo(s"Saving InformationBottleneckModel to $path")

      // Save matrices as Parquet
      import spark.implicits._
      val dataPath = new Path(path, "data").toString
      val data     = Seq(
        (
          instance.encoderMatrix.map(_.toSeq).toSeq,
          instance.clusterPrior.toSeq,
          instance.decoderMatrix.map(_.toSeq).toSeq,
          instance._compressionInfo,
          instance._relevanceInfo,
          instance._iterations,
          instance._converged
        )
      )
      spark
        .createDataFrame(data)
        .toDF("encoder", "prior", "decoder", "compression", "relevance", "iterations", "converged")
        .write
        .parquet(dataPath)

      val params: Map[String, Any] = Map(
        "k"              -> instance.getOrDefault(instance.k),
        "beta"           -> instance.getOrDefault(instance.beta),
        "featuresCol"    -> instance.getOrDefault(instance.featuresCol),
        "predictionCol"  -> instance.getOrDefault(instance.predictionCol),
        "probabilityCol" -> instance.getOrDefault(instance.probabilityCol),
        "numBins"        -> instance.getOrDefault(instance.numBins),
        "smoothing"      -> instance.getOrDefault(instance.smoothing)
      )

      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaObj: Map[String, Any]             = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "InformationBottleneckModel",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "k"                  -> instance.numClusters,
        "uid"                -> instance.uid,
        "params"             -> params,
        "metrics"            -> Map(
          "compressionInfo" -> instance._compressionInfo,
          "relevanceInfo"   -> instance._relevanceInfo,
          "iterations"      -> instance._iterations,
          "converged"       -> instance._converged
        )
      )

      val json = Serialization.write(metaObj)
      writeMetadata(path, json)
      logInfo(s"InformationBottleneckModel saved to $path")
    }
  }

  private class IBModelReader extends MLReader[InformationBottleneckModel] with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): InformationBottleneckModel = {
      val spark = sparkSession
      logInfo(s"Loading InformationBottleneckModel from $path")

      val metaStr                               = readMetadata(path)
      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaJ                                 = JsonMethods.parse(metaStr)

      val uid = (metaJ \ "uid").extract[String]
      val k   = (metaJ \ "k").extract[Int]

      val dataPath = new Path(path, "data").toString
      val data     = spark.read.parquet(dataPath).head()

      // Handle Scala collection conversion carefully - Parquet returns mutable ArraySeq
      val encoderSeq  = data.getAs[scala.collection.Seq[scala.collection.Seq[Double]]](0)
      val encoder     = encoderSeq.toArray.map(_.toArray)
      val priorSeq    = data.getAs[scala.collection.Seq[Double]](1)
      val prior       = priorSeq.toArray
      val decoderSeq  = data.getAs[scala.collection.Seq[scala.collection.Seq[Double]]](2)
      val decoder     = decoderSeq.toArray.map(_.toArray)
      val compression = data.getDouble(3)
      val relevance   = data.getDouble(4)
      val iterations  = data.getInt(5)
      val converged   = data.getBoolean(6)

      val paramsJ = metaJ \ "params"

      val model = new InformationBottleneckModel(uid, encoder, prior, decoder)
      model.set(model.k, k)
      model.set(model.beta, (paramsJ \ "beta").extract[Double])
      model.set(model.featuresCol, (paramsJ \ "featuresCol").extract[String])
      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])
      model.set(model.probabilityCol, (paramsJ \ "probabilityCol").extract[String])
      model.set(model.numBins, (paramsJ \ "numBins").extract[Int])
      model.set(model.smoothing, (paramsJ \ "smoothing").extract[Double])
      model.setCompressionInfo(compression)
      model.setRelevanceInfo(relevance)
      model.setIterations(iterations)
      model.setConverged(converged)

      logInfo(s"InformationBottleneckModel loaded from $path")
      model
    }
  }
}
