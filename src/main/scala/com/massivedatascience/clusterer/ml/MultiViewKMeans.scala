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
import com.massivedatascience.clusterer.ml.df.kernels.KernelFactory
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{ Estimator, Model }
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Configuration for a single view in multi-view clustering.
  *
  * @param featuresCol
  *   name of the feature column for this view
  * @param weight
  *   importance weight for this view (default 1.0)
  * @param divergence
  *   Bregman divergence for this view (default "squaredEuclidean")
  */
case class ViewSpec(
    featuresCol: String,
    weight: Double = 1.0,
    divergence: String = "squaredEuclidean"
) {
  require(weight > 0.0, s"View weight must be positive, got $weight")
  require(featuresCol.nonEmpty, "Feature column name cannot be empty")
}

/** Parameters for Multi-View K-Means clustering.
  */
trait MultiViewKMeansParams extends Params with Logging {

  /** Number of clusters. Must be > 1. Default: 2
    */
  final val k: IntParam = new IntParam(
    this,
    "k",
    "Number of clusters",
    ParamValidators.gt(1)
  )
  def getK: Int         = $(k)

  /** Maximum number of iterations. Default: 20
    */
  final val maxIter: IntParam = new IntParam(
    this,
    "maxIter",
    "Maximum number of iterations",
    ParamValidators.gtEq(1)
  )
  def getMaxIter: Int         = $(maxIter)

  /** Convergence tolerance. Default: 1e-4
    */
  final val tol: DoubleParam = new DoubleParam(
    this,
    "tol",
    "Convergence tolerance",
    ParamValidators.gtEq(0.0)
  )
  def getTol: Double         = $(tol)

  /** Random seed. Default: current time
    */
  final val seed: LongParam = new LongParam(this, "seed", "Random seed")
  def getSeed: Long         = $(seed)

  /** Output column for cluster predictions. Default: "prediction"
    */
  final val predictionCol: Param[String] = new Param[String](
    this,
    "predictionCol",
    "Output column for cluster predictions"
  )
  def getPredictionCol: String           = $(predictionCol)

  /** Smoothing parameter for divergences with domain constraints. Default: 1e-10
    */
  final val smoothing: DoubleParam = new DoubleParam(
    this,
    "smoothing",
    "Smoothing parameter",
    ParamValidators.gtEq(0.0)
  )
  def getSmoothing: Double         = $(smoothing)

  /** View combination strategy.
    *   - "weighted": Weighted sum of view distances (default)
    *   - "max": Maximum of view distances
    *   - "min": Minimum of view distances
    */
  final val combineStrategy: Param[String] = new Param[String](
    this,
    "combineStrategy",
    "View combination strategy: weighted, max, min",
    ParamValidators.inArray(Array("weighted", "max", "min"))
  )
  def getCombineStrategy: String           = $(combineStrategy)

  /** Whether to normalize view weights to sum to 1. Default: true
    */
  final val normalizeWeights: BooleanParam = new BooleanParam(
    this,
    "normalizeWeights",
    "Normalize view weights to sum to 1"
  )
  def getNormalizeWeights: Boolean         = $(normalizeWeights)

  setDefault(
    k                -> 2,
    maxIter          -> 20,
    tol              -> 1e-4,
    seed             -> System.currentTimeMillis(),
    predictionCol    -> "prediction",
    smoothing        -> 1e-10,
    combineStrategy  -> "weighted",
    normalizeWeights -> true
  )
}

/** Multi-View K-Means clustering with per-view divergences and weights.
  *
  * This estimator clusters data represented by multiple feature views, where each view can have:
  *   - Its own feature column
  *   - Its own Bregman divergence
  *   - Its own importance weight
  *
  * ==Use Cases==
  *
  *   - '''Document clustering''': content + metadata + citations
  *   - '''Image clustering''': pixel features + text captions + metadata
  *   - '''Web page clustering''': content + link structure + URL features
  *   - '''Multi-modal data''': text + audio + video features
  *
  * ==Algorithm==
  *
  * The algorithm extends standard K-means by: 1. Computing per-view divergences independently 2.
  * Combining view divergences using weighted sum (default) or other strategies 3. Assigning points
  * to clusters based on combined distance 4. Updating per-view centers independently
  *
  * ==Example Usage==
  *
  * {{{
  * // Create view specifications
  * val views = Seq(
  *   ViewSpec("content_features", weight = 2.0, divergence = "kl"),
  *   ViewSpec("metadata_features", weight = 1.0, divergence = "squaredEuclidean"),
  *   ViewSpec("link_features", weight = 0.5, divergence = "spherical")
  * )
  *
  * val mvkm = new MultiViewKMeans()
  *   .setK(10)
  *   .setViews(views)
  *   .setMaxIter(30)
  *
  * val model = mvkm.fit(multiViewData)
  * val predictions = model.transform(multiViewData)
  * }}}
  *
  * @see
  *   [[ViewSpec]] for view configuration
  * @see
  *   [[GeneralizedKMeans]] for single-view clustering
  */
class MultiViewKMeans(override val uid: String)
    extends Estimator[MultiViewKMeansModel]
    with MultiViewKMeansParams
    with DefaultParamsWritable
    with Logging {

  /** View specifications (not a Param to avoid serialization complexity). */
  private var viewSpecs: Seq[ViewSpec] = Seq.empty

  def this() = this(Identifiable.randomUID("mvkmeans"))

  // Parameter setters
  def setK(value: Int): this.type                    = set(k, value)
  def setMaxIter(value: Int): this.type              = set(maxIter, value)
  def setTol(value: Double): this.type               = set(tol, value)
  def setSeed(value: Long): this.type                = set(seed, value)
  def setPredictionCol(value: String): this.type     = set(predictionCol, value)
  def setSmoothing(value: Double): this.type         = set(smoothing, value)
  def setCombineStrategy(value: String): this.type   = set(combineStrategy, value)
  def setNormalizeWeights(value: Boolean): this.type = set(normalizeWeights, value)

  /** Set the view specifications.
    *
    * @param views
    *   sequence of ViewSpec defining each view
    */
  def setViews(views: Seq[ViewSpec]): this.type = {
    require(views.nonEmpty, "At least one view must be specified")
    viewSpecs = views
    this
  }

  /** Get the current view specifications. */
  def getViews: Seq[ViewSpec] = viewSpecs

  /** Get the number of views. */
  def numViews: Int = viewSpecs.length

  override def fit(dataset: Dataset[_]): MultiViewKMeansModel = {
    require(viewSpecs.nonEmpty, "Views must be set before calling fit(). Use setViews().")

    val df = dataset.toDF()

    // Validate schema - ensure all view columns exist
    viewSpecs.foreach { view =>
      require(
        df.schema.fieldNames.contains(view.featuresCol),
        s"View feature column '${view.featuresCol}' not found in DataFrame"
      )
    }

    // Compute normalized weights
    val weights = if ($(normalizeWeights)) {
      val totalWeight = viewSpecs.map(_.weight).sum
      viewSpecs.map(_.weight / totalWeight)
    } else {
      viewSpecs.map(_.weight)
    }

    // Create kernels for each view
    val kernels = viewSpecs.map { view =>
      KernelFactory.create(view.divergence, smoothing = $(smoothing))
    }

    // Get dimensions for each view
    val dims = viewSpecs.map { view =>
      df.select(view.featuresCol).first().getAs[Vector](0).size
    }

    val numViewsVal = viewSpecs.length
    logInfo(
      s"MultiViewKMeans: k=${$(k)}, numViews=$numViewsVal, " +
        s"views=${viewSpecs.map(v => s"${v.featuresCol}(${v.divergence},w=${v.weight})").mkString(", ")}"
    )

    df.cache()
    try {
      val startTime = System.currentTimeMillis()

      // Initialize centers for each view
      val initialCenters = initializeCenters(df, kernels, dims.toArray)
      logInfo(
        s"Initialized ${initialCenters.head.length} centers for ${initialCenters.length} views"
      )

      // Run multi-view Lloyd's algorithm
      val result = runLloyds(df, initialCenters, kernels, weights)

      val elapsed = System.currentTimeMillis() - startTime
      logInfo(
        s"MultiViewKMeans completed: ${result.iterations} iterations, converged=${result.converged}"
      )

      // Create model
      val model = new MultiViewKMeansModel(
        uid,
        result.centers.map(_.map(Vectors.dense)),
        viewSpecs,
        weights
      )
      copyValues(model.setParent(this))

      // Training summary
      model.trainingSummary = Some(
        TrainingSummary(
          algorithm = "MultiViewKMeans",
          k = $(k),
          effectiveK = result.centers.head.length,
          dim = dims.sum, // Total dimensions across all views
          numPoints = df.count(),
          iterations = result.iterations,
          converged = result.converged,
          distortionHistory = result.distortionHistory,
          movementHistory = result.movementHistory,
          assignmentStrategy = s"multiview(${viewSpecs.length} views)",
          divergence = viewSpecs.map(_.divergence).mkString(","),
          elapsedMillis = elapsed
        )
      )

      model
    } finally {
      df.unpersist()
    }
  }

  /** Initialize cluster centers for each view. */
  private def initializeCenters(
      df: DataFrame,
      kernels: Seq[BregmanKernel],
      dims: Array[Int]
  ): Array[Array[Array[Double]]] = {
    // Note: kernels and dims are validated to match viewSpecs
    val _        = (kernels, dims) // Mark as used (for validation purposes)
    val n        = df.count()
    val fraction = math.min(1.0, ($(k) * 10.0) / n)

    // Sample points for initialization
    val sampledRows = df.sample(withReplacement = false, fraction, $(seed)).limit($(k)).collect()

    // Initialize centers for each view from the same sampled points
    viewSpecs.zipWithIndex.map { case (view, viewIdx) =>
      sampledRows.map { row =>
        row.getAs[Vector](view.featuresCol).toArray
      }
    }.toArray
  }

  /** Run multi-view Lloyd's algorithm. */
  private def runLloyds(
      df: DataFrame,
      initialCenters: Array[Array[Array[Double]]],
      kernels: Seq[BregmanKernel],
      weights: Seq[Double]
  ): MultiViewResult = {
    // centers: Array[viewIdx][clusterIdx][dim]
    var centers           = initialCenters
    var iteration         = 0
    var converged         = false
    val distortionHistory = scala.collection.mutable.ArrayBuffer[Double]()
    val movementHistory   = scala.collection.mutable.ArrayBuffer[Double]()

    val combineStrat = $(combineStrategy)

    while (iteration < $(maxIter) && !converged) {
      iteration += 1

      // Broadcast centers and kernels for each view
      val bcCentersList = centers.map { viewCenters =>
        df.sparkSession.sparkContext.broadcast(viewCenters.map(Vectors.dense))
      }
      val bcKernelsList = kernels.map { kernel =>
        df.sparkSession.sparkContext.broadcast(kernel)
      }
      val bcWeights     = df.sparkSession.sparkContext.broadcast(weights.toArray)
      val numViews      = viewSpecs.length

      // Assignment UDF - compute combined distance across views
      val assignUDF = udf { (row: org.apache.spark.sql.Row) =>
        val wts         = bcWeights.value
        var bestCluster = 0
        var bestDist    = Double.MaxValue
        val numClusters = bcCentersList(0).value.length

        var c = 0
        while (c < numClusters) {
          val combinedDist = combineStrat match {
            case "weighted" =>
              var totalDist = 0.0
              var v         = 0
              while (v < numViews) {
                val features = row.getAs[Vector](v)
                val center   = bcCentersList(v).value(c)
                val kernel   = bcKernelsList(v).value
                totalDist += wts(v) * kernel.divergence(features, center)
                v += 1
              }
              totalDist
            case "max"      =>
              var maxDist = Double.MinValue
              var v       = 0
              while (v < numViews) {
                val features = row.getAs[Vector](v)
                val center   = bcCentersList(v).value(c)
                val kernel   = bcKernelsList(v).value
                val dist     = wts(v) * kernel.divergence(features, center)
                if (dist > maxDist) maxDist = dist
                v += 1
              }
              maxDist
            case "min"      =>
              var minDist = Double.MaxValue
              var v       = 0
              while (v < numViews) {
                val features = row.getAs[Vector](v)
                val center   = bcCentersList(v).value(c)
                val kernel   = bcKernelsList(v).value
                val dist     = wts(v) * kernel.divergence(features, center)
                if (dist < minDist) minDist = dist
                v += 1
              }
              minDist
          }

          if (combinedDist < bestDist) {
            bestDist = combinedDist
            bestCluster = c
          }
          c += 1
        }
        bestCluster
      }

      // Create struct of all view features and apply assignment
      val viewColsExpr = viewSpecs.map(v => col(v.featuresCol))
      val assigned     = df.withColumn(
        "_cluster",
        assignUDF(struct(viewColsExpr: _*))
      )

      // Update centers for each view independently
      val newCenters = updateCenters(assigned, kernels, $(k))

      // Compute movement (max across all views)
      val movement = centers
        .zip(newCenters)
        .map { case (oldViewCenters, newViewCenters) =>
          computeMovement(oldViewCenters, newViewCenters)
        }
        .max
      movementHistory += movement

      // Compute distortion
      val distortion = computeDistortion(df, newCenters, kernels, weights, combineStrat)
      distortionHistory += distortion

      converged = movement < $(tol)
      centers = newCenters

      // Clean up broadcasts
      bcCentersList.foreach(_.destroy())
      bcKernelsList.foreach(_.destroy())
      bcWeights.destroy()

      logDebug(f"Iteration $iteration: distortion=$distortion%.4f, movement=$movement%.6f")
    }

    MultiViewResult(
      centers = centers,
      iterations = iteration,
      converged = converged,
      distortionHistory = distortionHistory.toArray,
      movementHistory = movementHistory.toArray
    )
  }

  /** Update centers for each view using gradient mean. */
  private def updateCenters(
      assigned: DataFrame,
      kernels: Seq[BregmanKernel],
      numClusters: Int
  ): Array[Array[Array[Double]]] = {

    viewSpecs.zipWithIndex.map { case (view, viewIdx) =>
      val kernel   = kernels(viewIdx)
      val bcKernel = assigned.sparkSession.sparkContext.broadcast(kernel)

      val gradUDF = udf { (features: Vector) =>
        bcKernel.value.grad(features).toArray
      }

      val dim = assigned.select(view.featuresCol).head().getAs[Vector](0).size

      val aggregated = assigned
        .withColumn("_grad", gradUDF(col(view.featuresCol)))
        .groupBy("_cluster")
        .agg(
          count("*").as("count"),
          array((0 until dim).map(i => sum(element_at(col("_grad"), i + 1))): _*).as("grad_sum")
        )
        .collect()

      val centers = Array.fill(numClusters)(Array.fill(dim)(0.0))
      aggregated.foreach { row =>
        val clusterId = row.getInt(0)
        if (clusterId >= 0 && clusterId < numClusters) {
          val count   = row.getLong(1)
          val gradSum = row.getSeq[Double](2).toArray
          if (count > 0) {
            val avgGrad = Vectors.dense(gradSum.map(_ / count))
            centers(clusterId) = bcKernel.value.invGrad(avgGrad).toArray
          }
        }
      }

      centers
    }.toArray
  }

  private def computeMovement(
      oldCenters: Array[Array[Double]],
      newCenters: Array[Array[Double]]
  ): Double = {
    oldCenters
      .zip(newCenters)
      .map { case (old, newC) =>
        math.sqrt(old.zip(newC).map { case (a, b) => val d = a - b; d * d }.sum)
      }
      .max
  }

  private def computeDistortion(
      df: DataFrame,
      centers: Array[Array[Array[Double]]],
      kernels: Seq[BregmanKernel],
      weights: Seq[Double],
      combineStrat: String
  ): Double = {
    val bcCentersList = centers.map { viewCenters =>
      df.sparkSession.sparkContext.broadcast(viewCenters.map(Vectors.dense))
    }
    val bcKernelsList = kernels.map { kernel =>
      df.sparkSession.sparkContext.broadcast(kernel)
    }
    val bcWeights     = df.sparkSession.sparkContext.broadcast(weights.toArray)
    val numViews      = viewSpecs.length

    val distUDF = udf { (row: org.apache.spark.sql.Row) =>
      val wts         = bcWeights.value
      var minDist     = Double.MaxValue
      val numClusters = bcCentersList(0).value.length

      var c = 0
      while (c < numClusters) {
        val combinedDist = combineStrat match {
          case "weighted" =>
            var totalDist = 0.0
            var v         = 0
            while (v < numViews) {
              val features = row.getAs[Vector](v)
              val center   = bcCentersList(v).value(c)
              val kernel   = bcKernelsList(v).value
              totalDist += wts(v) * kernel.divergence(features, center)
              v += 1
            }
            totalDist
          case "max"      =>
            var maxD = Double.MinValue
            var v    = 0
            while (v < numViews) {
              val features = row.getAs[Vector](v)
              val center   = bcCentersList(v).value(c)
              val kernel   = bcKernelsList(v).value
              val dist     = wts(v) * kernel.divergence(features, center)
              if (dist > maxD) maxD = dist
              v += 1
            }
            maxD
          case "min"      =>
            var minD = Double.MaxValue
            var v    = 0
            while (v < numViews) {
              val features = row.getAs[Vector](v)
              val center   = bcCentersList(v).value(c)
              val kernel   = bcKernelsList(v).value
              val dist     = wts(v) * kernel.divergence(features, center)
              if (dist < minD) minD = dist
              v += 1
            }
            minD
        }

        if (combinedDist < minDist) minDist = combinedDist
        c += 1
      }
      minDist
    }

    val viewColsExpr = viewSpecs.map(v => col(v.featuresCol))
    val result       =
      df.select(distUDF(struct(viewColsExpr: _*)).as("dist")).agg(sum("dist")).first().getDouble(0)

    // Clean up
    bcCentersList.foreach(_.destroy())
    bcKernelsList.foreach(_.destroy())
    bcWeights.destroy()

    result
  }

  override def copy(extra: ParamMap): MultiViewKMeans = {
    val copied = defaultCopy(extra).asInstanceOf[MultiViewKMeans]
    copied.viewSpecs = this.viewSpecs
    copied
  }

  override def transformSchema(schema: StructType): StructType = {
    // Validate view columns exist
    viewSpecs.foreach { view =>
      require(
        schema.fieldNames.contains(view.featuresCol),
        s"View feature column '${view.featuresCol}' not found in schema"
      )
    }
    // Add prediction column
    require(
      !schema.fieldNames.contains($(predictionCol)),
      s"Prediction column '${$(predictionCol)}' already exists"
    )
    schema.add($(predictionCol), org.apache.spark.sql.types.IntegerType, nullable = false)
  }

  override def write: MLWriter = new MultiViewKMeans.MultiViewKMeansWriter(this)
}

object MultiViewKMeans extends MLReadable[MultiViewKMeans] {
  override def read: MLReader[MultiViewKMeans] = new MultiViewKMeansReader

  override def load(path: String): MultiViewKMeans = super.load(path)

  private[MultiViewKMeans] class MultiViewKMeansWriter(instance: MultiViewKMeans)
      extends MLWriter
      with Logging {
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      implicit val formats: DefaultFormats.type = DefaultFormats

      val viewSpecsData: Seq[Map[String, Any]] = instance.viewSpecs.map { v =>
        Map[String, Any](
          "featuresCol" -> v.featuresCol,
          "weight"      -> v.weight,
          "divergence"  -> v.divergence
        )
      }

      val paramMap: Map[String, Any] = Map[String, Any](
        "k"                -> instance.$(instance.k),
        "maxIter"          -> instance.$(instance.maxIter),
        "tol"              -> instance.$(instance.tol),
        "seed"             -> instance.$(instance.seed),
        "predictionCol"    -> instance.$(instance.predictionCol),
        "smoothing"        -> instance.$(instance.smoothing),
        "combineStrategy"  -> instance.$(instance.combineStrategy),
        "normalizeWeights" -> instance.$(instance.normalizeWeights)
      )

      val metadata: Map[String, Any] = Map[String, Any](
        "class"     -> classOf[MultiViewKMeans].getName,
        "uid"       -> instance.uid,
        "paramMap"  -> paramMap,
        "viewSpecs" -> viewSpecsData
      )

      val json = Serialization.write(metadata)
      sc.parallelize(Seq(json), 1).saveAsTextFile(s"$path/metadata")
    }
  }

  private class MultiViewKMeansReader extends MLReader[MultiViewKMeans] with Logging {
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): MultiViewKMeans = {
      implicit val formats: DefaultFormats.type = DefaultFormats

      val json  = sparkSession.sparkContext.textFile(s"$path/metadata").collect().mkString("\n")
      val metaJ = JsonMethods.parse(json)

      val uid       = (metaJ \ "uid").extract[String]
      val paramMap  = (metaJ \ "paramMap").values.asInstanceOf[Map[String, Any]]
      val viewSpecs = (metaJ \ "viewSpecs").extract[List[Map[String, Any]]].map { v =>
        ViewSpec(
          featuresCol = v("featuresCol").asInstanceOf[String],
          weight = v("weight") match {
            case d: Double  => d
            case bi: BigInt => bi.toDouble
            case other      => other.toString.toDouble
          },
          divergence = v("divergence").asInstanceOf[String]
        )
      }

      val instance = new MultiViewKMeans(uid)
      instance.set(instance.k, paramMap("k").asInstanceOf[BigInt].toInt)
      instance.set(instance.maxIter, paramMap("maxIter").asInstanceOf[BigInt].toInt)
      instance.set(instance.tol, paramMap("tol").asInstanceOf[Double])
      instance.set(instance.seed, paramMap("seed").asInstanceOf[BigInt].toLong)
      instance.set(instance.predictionCol, paramMap("predictionCol").asInstanceOf[String])
      instance.set(instance.smoothing, paramMap("smoothing").asInstanceOf[Double])
      instance.set(instance.combineStrategy, paramMap("combineStrategy").asInstanceOf[String])
      instance.set(instance.normalizeWeights, paramMap("normalizeWeights").asInstanceOf[Boolean])
      instance.setViews(viewSpecs)

      instance
    }
  }
}

/** Model from Multi-View K-Means clustering.
  *
  * @param uid
  *   unique identifier
  * @param clusterCenters
  *   cluster centers for each view: Array[viewIdx][clusterIdx]
  * @param viewSpecs
  *   view specifications used during training
  * @param weights
  *   normalized weights for each view
  */
class MultiViewKMeansModel(
    override val uid: String,
    val clusterCenters: Array[Array[Vector]],
    val viewSpecs: Seq[ViewSpec],
    val weights: Seq[Double]
) extends Model[MultiViewKMeansModel]
    with MultiViewKMeansParams
    with MLWritable
    with HasTrainingSummary
    with Logging {

  /** Number of clusters. */
  def numClusters: Int = clusterCenters.headOption.map(_.length).getOrElse(0)

  /** Number of views. */
  def numViews: Int = viewSpecs.length

  /** Get cluster centers for a specific view. */
  def getCentersForView(viewIdx: Int): Array[Vector] = {
    require(viewIdx >= 0 && viewIdx < numViews, s"View index must be in [0, $numViews)")
    clusterCenters(viewIdx)
  }

  /** Get cluster centers for a view by column name. */
  def getCentersForView(featuresCol: String): Array[Vector] = {
    val idx = viewSpecs.indexWhere(_.featuresCol == featuresCol)
    require(idx >= 0, s"View with features column '$featuresCol' not found")
    clusterCenters(idx)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()

    // Create kernels for each view
    val kernels = viewSpecs.map { view =>
      KernelFactory.create(view.divergence, smoothing = $(smoothing))
    }

    val bcCentersList = clusterCenters.map { viewCenters =>
      df.sparkSession.sparkContext.broadcast(viewCenters)
    }
    val bcKernelsList = kernels.map { kernel =>
      df.sparkSession.sparkContext.broadcast(kernel)
    }
    val bcWeights     = df.sparkSession.sparkContext.broadcast(weights.toArray)
    val combineStrat  = $(combineStrategy)
    val numViewsVal   = viewSpecs.length

    val assignUDF = udf { (row: org.apache.spark.sql.Row) =>
      val wts            = bcWeights.value
      var bestCluster    = 0
      var bestDist       = Double.MaxValue
      val numClustersVal = bcCentersList(0).value.length

      var c = 0
      while (c < numClustersVal) {
        val combinedDist = combineStrat match {
          case "weighted" =>
            var totalDist = 0.0
            var v         = 0
            while (v < numViewsVal) {
              val features = row.getAs[Vector](v)
              val center   = bcCentersList(v).value(c)
              val kernel   = bcKernelsList(v).value
              totalDist += wts(v) * kernel.divergence(features, center)
              v += 1
            }
            totalDist
          case "max"      =>
            var maxDist = Double.MinValue
            var v       = 0
            while (v < numViewsVal) {
              val features = row.getAs[Vector](v)
              val center   = bcCentersList(v).value(c)
              val kernel   = bcKernelsList(v).value
              val dist     = wts(v) * kernel.divergence(features, center)
              if (dist > maxDist) maxDist = dist
              v += 1
            }
            maxDist
          case "min"      =>
            var minDist = Double.MaxValue
            var v       = 0
            while (v < numViewsVal) {
              val features = row.getAs[Vector](v)
              val center   = bcCentersList(v).value(c)
              val kernel   = bcKernelsList(v).value
              val dist     = wts(v) * kernel.divergence(features, center)
              if (dist < minDist) minDist = dist
              v += 1
            }
            minDist
        }

        if (combinedDist < bestDist) {
          bestDist = combinedDist
          bestCluster = c
        }
        c += 1
      }
      bestCluster
    }

    val viewColsExpr = viewSpecs.map(v => col(v.featuresCol))
    df.withColumn($(predictionCol), assignUDF(struct(viewColsExpr: _*)))
  }

  override def copy(extra: ParamMap): MultiViewKMeansModel = {
    val copied = new MultiViewKMeansModel(uid, clusterCenters, viewSpecs, weights)
    copyValues(copied, extra).setParent(parent)
    copied.trainingSummary = this.trainingSummary
    copied
  }

  override def transformSchema(schema: StructType): StructType = {
    viewSpecs.foreach { view =>
      require(
        schema.fieldNames.contains(view.featuresCol),
        s"View feature column '${view.featuresCol}' not found in schema"
      )
    }
    schema.add($(predictionCol), org.apache.spark.sql.types.IntegerType, nullable = false)
  }

  override def write: MLWriter = new MultiViewKMeansModel.MultiViewKMeansModelWriter(this)

  override def toString: String = {
    s"MultiViewKMeansModel: uid=$uid, k=$numClusters, views=$numViews, " +
      s"viewSpecs=${viewSpecs.map(v => s"${v.featuresCol}(${v.divergence})").mkString(", ")}"
  }
}

object MultiViewKMeansModel extends MLReadable[MultiViewKMeansModel] {

  override def read: MLReader[MultiViewKMeansModel] = new MultiViewKMeansModelReader

  override def load(path: String): MultiViewKMeansModel = super.load(path)

  private[MultiViewKMeansModel] class MultiViewKMeansModelWriter(instance: MultiViewKMeansModel)
      extends MLWriter
      with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      implicit val formats: DefaultFormats.type = DefaultFormats
      val spark                                 = sparkSession
      logInfo(s"Saving MultiViewKMeansModel to $path")

      // Save centers for each view in separate directories
      val centerHashes = instance.clusterCenters.zipWithIndex.map { case (viewCenters, viewIdx) =>
        val centersData = viewCenters.indices.map { i =>
          (i, 1.0, viewCenters(i))
        }
        val viewPath    = s"$path/view_$viewIdx"
        writeCenters(spark, viewPath, centersData)
      }

      val k                                = instance.numClusters
      val viewsData: Seq[Map[String, Any]] = instance.viewSpecs.map { v =>
        Map[String, Any](
          "featuresCol" -> v.featuresCol,
          "weight"      -> v.weight,
          "divergence"  -> v.divergence
        )
      }

      val params: Map[String, Any] = Map(
        "predictionCol"    -> instance.getOrDefault(instance.predictionCol),
        "smoothing"        -> instance.getOrDefault(instance.smoothing),
        "combineStrategy"  -> instance.getOrDefault(instance.combineStrategy),
        "normalizeWeights" -> instance.getOrDefault(instance.normalizeWeights),
        "weights"          -> instance.weights.toList
      )

      val metaObj: Map[String, Any] = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "MultiViewKMeansModel",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "k"                  -> k,
        "numViews"           -> instance.numViews,
        "uid"                -> instance.uid,
        "viewSpecs"          -> viewsData,
        "params"             -> params,
        "centerHashes"       -> centerHashes.toList
      )

      val json = Serialization.write(metaObj)
      writeMetadata(path, json)
      logInfo(s"MultiViewKMeansModel saved to $path")
    }
  }

  private class MultiViewKMeansModelReader extends MLReader[MultiViewKMeansModel] with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): MultiViewKMeansModel = {
      implicit val formats: DefaultFormats.type = DefaultFormats
      val spark                                 = sparkSession
      logInfo(s"Loading MultiViewKMeansModel from $path")

      val metaStr = readMetadata(path)
      val metaJ   = JsonMethods.parse(metaStr)

      val numViews = (metaJ \ "numViews").extract[Int]
      val uid      = (metaJ \ "uid").extract[String]

      val viewSpecsData = (metaJ \ "viewSpecs").extract[List[Map[String, Any]]]
      val viewSpecs     = viewSpecsData.map { v =>
        ViewSpec(
          featuresCol = v("featuresCol").asInstanceOf[String],
          weight = v("weight") match {
            case d: Double  => d
            case bi: BigInt => bi.toDouble
            case other      => other.toString.toDouble
          },
          divergence = v("divergence").asInstanceOf[String]
        )
      }

      // Load centers for each view
      val centers = (0 until numViews).map { viewIdx =>
        val viewPath  = s"$path/view_$viewIdx"
        val centersDF = readCenters(spark, viewPath)
        centersDF.collect().sortBy(_.getInt(0)).map(_.getAs[Vector]("vector"))
      }.toArray

      val paramsJ = metaJ \ "params"
      val weights = (paramsJ \ "weights").extract[List[Double]]

      val model = new MultiViewKMeansModel(uid, centers, viewSpecs, weights)

      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])
      model.set(model.smoothing, (paramsJ \ "smoothing").extract[Double])
      model.set(model.combineStrategy, (paramsJ \ "combineStrategy").extract[String])
      model.set(model.normalizeWeights, (paramsJ \ "normalizeWeights").extract[Boolean])

      logInfo(s"MultiViewKMeansModel loaded from $path")
      model
    }
  }
}

/** Internal result class for multi-view Lloyd's algorithm. */
private[ml] case class MultiViewResult(
    centers: Array[Array[Array[Double]]],
    iterations: Int,
    converged: Boolean,
    distortionHistory: Array[Double],
    movementHistory: Array[Double]
)
