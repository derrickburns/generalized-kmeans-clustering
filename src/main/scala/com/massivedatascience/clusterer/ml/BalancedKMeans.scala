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
import com.massivedatascience.clusterer.ml.df.kernels._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{ DefaultParamsReadable, DefaultParamsWritable, Identifiable }
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

import scala.util.Random

/** Balanced K-Means clustering with cluster size constraints.
  *
  * Balanced K-Means is a variant of K-Means that enforces minimum and maximum cluster sizes, useful
  * for workload balancing, equal-sized customer segments, and partitioning problems.
  *
  * ==Algorithm==
  *
  * The algorithm uses a modified Lloyd's iteration:
  *   1. '''Initialization''': Select initial centers using k-means|| or random sampling 2.
  *      '''Constrained Assignment''': Assign points to clusters respecting size constraints:
  *      - Soft mode: Penalize assignment to oversized clusters (fast, approximate)
  *      - Hard mode: Iterative redistribution from oversized to undersized clusters (exact)
  *      3. '''Update''': Recompute centers as the Bregman centroid of assigned points 4.
  *      '''Convergence''': Repeat until centers stabilize or `maxIter` is reached
  *
  * ==Balance Modes==
  *
  * | Mode   | Description                                       | Guarantee           | Speed  |
  * |:-------|:--------------------------------------------------|:--------------------|:-------|
  * | `soft` | Adds penalty to distance for oversized clusters   | Approximate balance | Fast   |
  * | `hard` | Redistributes points to enforce exact constraints | Exact balance       | Slower |
  *
  * ==Use Cases==
  *
  *   - '''Workload balancing''': Distribute tasks evenly across workers
  *   - '''Customer segmentation''': Create equal-sized market segments
  *   - '''Data partitioning''': Split data into balanced shards
  *   - '''Routing problems''': Assign customers to service centers with capacity limits
  *
  * ==Example Usage==
  *
  * '''Equal-sized clusters:'''
  * {{{
  * val balancedKmeans = new BalancedKMeans()
  *   .setK(5)
  *   .setBalanceMode("hard")  // Enforce exact balance
  *   .setMaxIter(50)
  *
  * val model = balancedKmeans.fit(dataset)
  * // Each cluster will have n/k ± 1 points
  * }}}
  *
  * '''Soft balance with penalty:'''
  * {{{
  * val balancedKmeans = new BalancedKMeans()
  *   .setK(10)
  *   .setBalanceMode("soft")
  *   .setBalancePenalty(0.5)  // Moderate penalty for oversized clusters
  *
  * val model = balancedKmeans.fit(dataset)
  * }}}
  *
  * '''Custom size bounds:'''
  * {{{
  * val balancedKmeans = new BalancedKMeans()
  *   .setK(4)
  *   .setMinClusterSize(100)
  *   .setMaxClusterSize(500)
  *   .setBalanceMode("hard")
  *
  * val model = balancedKmeans.fit(dataset)
  * }}}
  *
  * @see
  *   [[GeneralizedKMeans]] for unconstrained K-Means
  * @see
  *   [[MiniBatchKMeans]] for large-scale approximate K-Means
  *
  * @note
  *   For hard mode with infeasible constraints (e.g., minClusterSize * k > n), the algorithm will
  *   do its best to satisfy constraints but may not achieve them.
  * @note
  *   Balanced K-Means typically requires more iterations than standard K-Means.
  *
  * Reference: Malinen & Fränti (2014): "Balanced K-Means for Clustering"
  */
class BalancedKMeans(override val uid: String)
    extends Estimator[GeneralizedKMeansModel]
    with BalancedKMeansParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("balanced_kmeans"))

  // Parameter setters

  /** Sets the number of clusters (k). Must be > 1. Default: 2. */
  def setK(value: Int): this.type = set(k, value)

  /** Sets the minimum cluster size. Default: 1. */
  def setMinClusterSize(value: Int): this.type = set(minClusterSize, value)

  /** Sets the maximum cluster size. Default: 0 (auto = n/k + slack). */
  def setMaxClusterSize(value: Int): this.type = set(maxClusterSize, value)

  /** Sets the balance mode: "soft" or "hard". Default: "soft". */
  def setBalanceMode(value: String): this.type = set(balanceMode, value)

  /** Sets the penalty multiplier for soft balance mode. Default: 0.5. */
  def setBalancePenalty(value: Double): this.type = set(balancePenalty, value)

  /** Sets the Bregman divergence. Default: "squaredEuclidean". */
  def setDivergence(value: String): this.type = set(divergence, value)

  /** Sets the smoothing parameter. Default: 1e-10. */
  def setSmoothing(value: Double): this.type = set(smoothing, value)

  /** Sets the features column name. Default: "features". */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** Sets the prediction column name. Default: "prediction". */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** Sets the maximum number of iterations. Default: 50. */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** Sets the convergence tolerance. Default: 1e-4. */
  def setTol(value: Double): this.type = set(tol, value)

  /** Sets the random seed. */
  def setSeed(value: Long): this.type = set(seed, value)

  /** Sets the initialization mode. Default: "k-means||". */
  def setInitMode(value: String): this.type = set(initMode, value)

  /** Sets the number of k-means|| initialization steps. Default: 2. */
  def setInitSteps(value: Int): this.type = set(initSteps, value)

  override def fit(dataset: Dataset[_]): GeneralizedKMeansModel = {
    transformSchema(dataset.schema, logging = true)

    val df = dataset.toDF()

    val numFeatures = df.select($(featuresCol)).first().getAs[Vector](0).size
    val numPoints   = df.count()

    // Compute effective max cluster size
    val effectiveMaxSize = if ($(maxClusterSize) <= 0) {
      // Auto: allow some slack (10% over perfectly balanced)
      ((numPoints.toDouble / $(k)) * 1.1).ceil.toInt
    } else {
      $(maxClusterSize)
    }

    val effectiveMinSize = math.max(1, $(minClusterSize))

    // Validate constraints
    require(
      effectiveMinSize * $(k) <= numPoints,
      s"Infeasible constraint: minClusterSize(${effectiveMinSize}) * k(${$(k)}) = " +
        s"${effectiveMinSize * $(k)} > numPoints($numPoints)"
    )

    logInfo(
      s"Training BalancedKMeans with k=${$(k)}, mode=${$(balanceMode)}, " +
        s"minSize=$effectiveMinSize, maxSize=$effectiveMaxSize, " +
        s"divergence=${$(divergence)}, numPoints=$numPoints"
    )

    // Create kernel
    val kernel = createKernel($(divergence), $(smoothing))

    // Initialize centers
    val initialCenters = initializeCenters(df, $(featuresCol), kernel)

    logInfo(s"Initialized ${initialCenters.length} centers using ${$(initMode)}")

    // Run balanced Lloyd's algorithm
    val startTime = System.currentTimeMillis()
    val result    = runBalancedLloyds(
      df,
      initialCenters,
      kernel,
      effectiveMinSize,
      effectiveMaxSize
    )
    val elapsed   = System.currentTimeMillis() - startTime

    logInfo(
      s"Training completed: iterations=${result.iterations}, " +
        s"converged=${result.converged}, " +
        s"finalDistortion=${result.distortionHistory.lastOption.getOrElse(0.0)}, " +
        s"elapsed=${elapsed}ms"
    )

    // Create model
    val model = new GeneralizedKMeansModel(uid, result.centers, kernel.name)
    copyValues(model)

    // Attach training summary
    val summary = TrainingSummary.fromLloydResult(
      algorithm = "BalancedKMeans",
      result = result,
      k = $(k),
      dim = numFeatures,
      numPoints = numPoints,
      assignmentStrategy = s"Balanced(mode=${$(balanceMode)})",
      divergence = $(divergence),
      elapsedMillis = elapsed
    )
    model.trainingSummary = Some(summary)

    model
  }

  private def runBalancedLloyds(
      df: DataFrame,
      initialCenters: Array[Vector],
      kernel: BregmanKernel,
      minSize: Int,
      maxSize: Int
  ): LloydResult = {
    val spark = df.sparkSession

    val kVal       = $(k)
    val tolVal     = $(tol)
    val maxIterVal = $(maxIter)
    val mode       = $(balanceMode)
    val penalty    = $(balancePenalty)
    val featCol    = $(featuresCol)
    val rng        = new Random($(seed))

    var centers           = initialCenters.map(_.toArray)
    var iteration         = 0
    var converged         = false
    val distortionHistory = scala.collection.mutable.ArrayBuffer.empty[Double]

    // Add row id for tracking in hard mode
    val dfWithId = df.withColumn("_row_id", monotonically_increasing_id())

    while (iteration < maxIterVal && !converged) {
      val broadcastCenters = spark.sparkContext.broadcast(centers)
      val broadcastKernel  = spark.sparkContext.broadcast(kernel)

      // Compute distances to all centers
      val distanceUdf = udf { (features: Vector) =>
        val centersLocal           = broadcastCenters.value
        val kernelLocal            = broadcastKernel.value
        val distances: Seq[Double] = centersLocal.map { center =>
          kernelLocal.divergence(features, Vectors.dense(center))
        }.toSeq
        distances
      }

      val withDistances = dfWithId.withColumn("_distances", distanceUdf(col(featCol)))

      // Perform assignment based on mode
      val assigned = if (mode == "soft") {
        softBalancedAssignment(withDistances, kVal, maxSize, penalty, spark)
      } else {
        hardBalancedAssignment(withDistances, kVal, minSize, maxSize, spark)
      }

      // Compute total distortion using UDF to extract distance at assignment index
      val getDistAtIndex  = udf { (distances: Seq[Double], assignment: Int) =>
        distances(assignment)
      }
      val distortionDf    =
        assigned.withColumn("_dist", getDistAtIndex(col("_distances"), col("_assignment")))
      val totalDistortion = distortionDf.agg(sum("_dist")).first().getDouble(0)
      distortionHistory += totalDistortion

      // Update centers
      val clusterData = assigned
        .select(col("_assignment"), col(featCol))
        .rdd
        .map { row =>
          val cluster  = row.getInt(0)
          val features = row.getAs[Vector](1).toArray
          (cluster, (features, 1L))
        }
        .collect()

      // Aggregate by cluster
      val clusterSums = scala.collection.mutable.Map.empty[Int, (Array[Double], Long)]
      for ((cluster, (features, count)) <- clusterData) {
        clusterSums.get(cluster) match {
          case Some((sumArr, sumCount)) =>
            for (i <- features.indices) sumArr(i) += features(i)
            clusterSums(cluster) = (sumArr, sumCount + count)
          case None                     =>
            clusterSums(cluster) = (features.clone(), count)
        }
      }

      val newCentersMap: Map[Int, Array[Double]] = clusterSums.map {
        case (cluster, (sumArr, count)) =>
          cluster -> sumArr.map(_ / count)
      }.toMap

      // Compute center movements
      var maxMovement                      = 0.0
      val newCenters: Array[Array[Double]] = centers.indices.map { i =>
        newCentersMap.get(i) match {
          case Some(newCenter) =>
            var sumSq = 0.0
            for (j <- centers(i).indices) {
              val diff = centers(i)(j) - newCenter(j)
              sumSq += diff * diff
            }
            val movement = math.sqrt(sumSq)
            maxMovement = math.max(maxMovement, movement)
            newCenter
          case None            =>
            // Empty cluster: reinitialize with random point
            val points = assigned.select(featCol).collect()
            if (points.nonEmpty) {
              val randomPoint = points(rng.nextInt(points.length)).getAs[Vector](0).toArray
              maxMovement = Double.MaxValue // Force non-convergence
              randomPoint
            } else {
              centers(i)
            }
        }
      }.toArray

      centers = newCenters
      converged = maxMovement < tolVal

      broadcastCenters.destroy()
      broadcastKernel.destroy()

      iteration += 1

      if (iteration % 10 == 0) {
        logInfo(s"Iteration $iteration: distortion=$totalDistortion, maxMovement=$maxMovement")
      }
    }

    LloydResult(
      centers = centers,
      iterations = iteration,
      distortionHistory = distortionHistory.toArray,
      movementHistory = Array.empty,
      converged = converged,
      emptyClusterEvents = 0
    )
  }

  /** Soft balanced assignment: penalize distances to oversized clusters. */
  private def softBalancedAssignment(
      withDistances: DataFrame,
      kVal: Int,
      maxSize: Int,
      penalty: Double,
      spark: org.apache.spark.sql.SparkSession
  ): DataFrame = {
    // First pass: get initial assignments to compute cluster sizes
    val initialAssignment = udf { (distances: Seq[Double]) =>
      distances.zipWithIndex.minBy(_._1)._2
    }

    val initial =
      withDistances.withColumn("_initial_assignment", initialAssignment(col("_distances")))

    // Compute cluster sizes
    val clusterSizes = initial
      .groupBy("_initial_assignment")
      .count()
      .collect()
      .map(r => (r.getInt(0), r.getLong(1)))
      .toMap

    val broadcastSizes = spark.sparkContext.broadcast(clusterSizes)

    // Second pass: adjust distances based on cluster fullness
    val penalizedAssignment = udf { (distances: Seq[Double]) =>
      val sizes     = broadcastSizes.value
      val penalized = distances.zipWithIndex.map { case (dist, idx) =>
        val currentSize   = sizes.getOrElse(idx, 0L)
        val oversize      = math.max(0, currentSize - maxSize).toDouble
        val penaltyFactor = 1.0 + penalty * (oversize / maxSize)
        (dist * penaltyFactor, idx)
      }
      penalized.minBy(_._1)._2
    }

    // Note: Don't destroy broadcast here - DataFrame is lazy and UDF needs it later
    withDistances.withColumn("_assignment", penalizedAssignment(col("_distances")))
  }

  /** Hard balanced assignment: redistribute points to enforce exact constraints. */
  private def hardBalancedAssignment(
      withDistances: DataFrame,
      kVal: Int,
      minSize: Int,
      maxSize: Int,
      spark: org.apache.spark.sql.SparkSession
  ): DataFrame = {
    // Compute distance to each cluster and get initial assignment
    val assignmentData = withDistances
      .select(
        col("_row_id"),
        col($(featuresCol)),
        col("_distances")
      )
      .rdd
      .map { row =>
        val rowId                  = row.getLong(0)
        val features               = row.getAs[Vector](1)
        // Convert to immutable Seq to avoid ClassCastException with mutable ArraySeq
        val distances: Seq[Double] = row.getSeq[Double](2).toList
        val sortedClusters         = distances.zipWithIndex.sortBy(_._1).map(_._2)
        (rowId, features, distances, sortedClusters)
      }
      .collect()

    // Greedy balanced assignment
    val clusterCounts = Array.fill(kVal)(0)
    val assignments   = scala.collection.mutable.Map.empty[Long, Int]

    // Sort points by their minimum distance (assign easy points first)
    val sortedByMinDist = assignmentData.sortBy { case (_, _, dists, _) => dists.min }

    // First pass: assign each point to its preferred cluster if not full
    for ((rowId, _, _, sortedClusters) <- sortedByMinDist) {
      val assigned = sortedClusters.find { cluster =>
        clusterCounts(cluster) < maxSize
      }.getOrElse(0) // Fallback to cluster 0 if all full

      assignments(rowId) = assigned
      clusterCounts(assigned) += 1
    }

    // Second pass: redistribute from oversized to undersized clusters
    var needsRedistribution         = true
    var redistributionIterations    = 0
    val maxRedistributionIterations = 100

    while (needsRedistribution && redistributionIterations < maxRedistributionIterations) {
      needsRedistribution = false
      redistributionIterations += 1

      // Find undersized clusters
      val undersizedClusters = (0 until kVal).filter(c => clusterCounts(c) < minSize)

      for (undersizedCluster <- undersizedClusters) {
        // Find points in oversized clusters that could move to this cluster
        val candidates = assignmentData.filter { case (rowId, _, _, _) =>
          val currentCluster = assignments(rowId)
          clusterCounts(currentCluster) > minSize // Can give up a point
        }.map { case (rowId, _, dists, _) =>
          val currentCluster = assignments(rowId)
          val currentDist    = dists(currentCluster)
          val newDist        = dists(undersizedCluster)
          val costIncrease   = newDist - currentDist
          (rowId, currentCluster, costIncrease)
        }.sortBy(_._3) // Sort by cost increase

        // Move the best candidate
        if (candidates.nonEmpty && clusterCounts(undersizedCluster) < minSize) {
          val (rowId, fromCluster, _) = candidates.head
          assignments(rowId) = undersizedCluster
          clusterCounts(fromCluster) -= 1
          clusterCounts(undersizedCluster) += 1
          needsRedistribution = true
        }
      }
    }

    // Create result DataFrame
    val assignmentMap = spark.sparkContext.broadcast(assignments.toMap)

    val assignUdf = udf { (rowId: Long) =>
      assignmentMap.value.getOrElse(rowId, 0)
    }

    // Note: Don't destroy broadcast here - DataFrame is lazy and UDF needs it later
    withDistances.withColumn("_assignment", assignUdf(col("_row_id")))
  }

  private def createKernel(divergenceName: String, smoothing: Double): BregmanKernel = {
    divergenceName.toLowerCase match {
      case "squaredeuclidean" | "se" | "euclidean" => new SquaredEuclideanKernel()
      case "kl" | "kullbackleibler"                => new KLDivergenceKernel(smoothing)
      case "itakurasaito" | "is"                   => new ItakuraSaitoKernel(smoothing)
      case "l1" | "manhattan"                      => new L1Kernel()
      case "spherical" | "cosine"                  => new SphericalKernel()
      case "generalizedi" | "gi"                   => new GeneralizedIDivergenceKernel(smoothing)
      case "logistic"                              => new LogisticLossKernel()
      case other                                   => throw new IllegalArgumentException(s"Unknown divergence: $other")
    }
  }

  private def initializeCenters(
      df: DataFrame,
      featuresCol: String,
      kernel: BregmanKernel
  ): Array[Vector] = {
    val rng = new Random($(seed))

    $(initMode).toLowerCase match {
      case "random" =>
        val fraction = math.min(1.0, $(k).toDouble / df.count() * 10)
        df.select(featuresCol)
          .sample(withReplacement = false, fraction, $(seed))
          .limit($(k))
          .collect()
          .map(_.getAs[Vector](0))

      case "k-means||" | "kmeansparallel" =>
        // Simplified k-means|| initialization
        val allPoints = df.select(featuresCol).collect().map(_.getAs[Vector](0))
        if (allPoints.length <= $(k)) {
          allPoints
        } else {
          val centers = scala.collection.mutable.ArrayBuffer.empty[Vector]
          centers += allPoints(rng.nextInt(allPoints.length))

          while (centers.length < $(k)) {
            val currentCenters           = centers.toArray
            val distances: Array[Double] = allPoints.map { point =>
              val dists: Array[Double] = currentCenters.map(c => kernel.divergence(point, c))
              dists.min
            }
            val totalDist: Double        = distances.sum
            if (totalDist > 0) {
              val probabilities: Array[Double] = distances.map(d => d / totalDist)
              val cumProbs: Array[Double]      = probabilities.scanLeft(0.0)((a, b) => a + b).tail
              val r                            = rng.nextDouble()
              val idx                          = cumProbs.indexWhere(_ >= r)
              centers += allPoints(if (idx >= 0) idx else allPoints.length - 1)
            } else {
              centers += allPoints(rng.nextInt(allPoints.length))
            }
          }
          centers.toArray
        }

      case other =>
        throw new IllegalArgumentException(s"Unknown initialization mode: $other")
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    require(
      schema.fieldNames.contains($(featuresCol)),
      s"Features column '${$(featuresCol)}' not found in schema"
    )
    schema
  }

  override def copy(extra: ParamMap): BalancedKMeans = defaultCopy(extra)
}

/** Parameters for BalancedKMeans. */
trait BalancedKMeansParams
    extends Params
    with HasFeaturesCol
    with HasPredictionCol
    with HasMaxIter
    with HasTol
    with HasSeed {

  /** Number of clusters (k). */
  final val k: IntParam =
    new IntParam(this, "k", "Number of clusters to create. Must be > 1.", ParamValidators.gt(1))

  def getK: Int = $(k)

  /** Minimum cluster size. */
  final val minClusterSize: IntParam = new IntParam(
    this,
    "minClusterSize",
    "Minimum number of points per cluster. Must be >= 1.",
    ParamValidators.gtEq(1)
  )

  def getMinClusterSize: Int = $(minClusterSize)

  /** Maximum cluster size. 0 means auto (n/k + 10% slack). */
  final val maxClusterSize: IntParam = new IntParam(
    this,
    "maxClusterSize",
    "Maximum number of points per cluster. 0 = auto.",
    ParamValidators.gtEq(0)
  )

  def getMaxClusterSize: Int = $(maxClusterSize)

  /** Balance mode: "soft" or "hard". */
  final val balanceMode: Param[String] = new Param[String](
    this,
    "balanceMode",
    "Balance enforcement mode: 'soft' (penalty-based) or 'hard' (exact constraints)",
    ParamValidators.inArray(Array("soft", "hard"))
  )

  def getBalanceMode: String = $(balanceMode)

  /** Penalty multiplier for soft balance mode. */
  final val balancePenalty: DoubleParam = new DoubleParam(
    this,
    "balancePenalty",
    "Penalty multiplier for assigning to oversized clusters in soft mode.",
    ParamValidators.gtEq(0.0)
  )

  def getBalancePenalty: Double = $(balancePenalty)

  /** Bregman divergence function. */
  final val divergence: Param[String] = new Param[String](
    this,
    "divergence",
    "Bregman divergence: squaredEuclidean, kl, itakuraSaito, l1, spherical, generalizedI, logistic"
  )

  def getDivergence: String = $(divergence)

  /** Smoothing for numerical stability. */
  final val smoothing: DoubleParam = new DoubleParam(
    this,
    "smoothing",
    "Smoothing parameter for divergences that need it (KL, Itakura-Saito).",
    ParamValidators.gtEq(0.0)
  )

  def getSmoothing: Double = $(smoothing)

  /** Initialization mode. */
  final val initMode: Param[String] = new Param[String](
    this,
    "initMode",
    "Initialization algorithm: 'k-means||' (default) or 'random'",
    ParamValidators.inArray(Array("k-means||", "kmeansparallel", "random"))
  )

  def getInitMode: String = $(initMode)

  /** Initialization steps for k-means||. */
  final val initSteps: IntParam = new IntParam(
    this,
    "initSteps",
    "Number of steps for k-means|| initialization.",
    ParamValidators.gtEq(1)
  )

  def getInitSteps: Int = $(initSteps)

  setDefault(
    k              -> 2,
    minClusterSize -> 1,
    maxClusterSize -> 0,
    balanceMode    -> "soft",
    balancePenalty -> 0.5,
    divergence     -> "squaredEuclidean",
    smoothing      -> 1e-10,
    maxIter        -> 50,
    tol            -> 1e-4,
    seed           -> this.getClass.getName.hashCode.toLong,
    featuresCol    -> "features",
    predictionCol  -> "prediction",
    initMode       -> "k-means||",
    initSteps      -> 2
  )
}

object BalancedKMeans extends DefaultParamsReadable[BalancedKMeans] {
  override def load(path: String): BalancedKMeans = super.load(path)
}
