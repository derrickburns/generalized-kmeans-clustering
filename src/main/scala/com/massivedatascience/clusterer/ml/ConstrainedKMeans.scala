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
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Parameters for Constrained K-Means clustering (COP-KMeans).
  */
trait ConstrainedKMeansParams extends GeneralizedKMeansParams {

  /** Column name for point IDs (Long). Required for constraint matching.
    */
  final val idCol: Param[String] = new Param[String](
    this,
    "idCol",
    "Column name for point IDs (Long)"
  )
  def getIdCol: String           = $(idCol)

  /** Constraint enforcement mode: "soft" or "hard".
    *   - soft: Add penalty to objective (may violate constraints)
    *   - hard: Strictly enforce (may fail to find valid assignment)
    */
  final val constraintMode: Param[String] = new Param[String](
    this,
    "constraintMode",
    "Constraint mode: soft or hard",
    ParamValidators.inArray(Array("soft", "hard"))
  )
  def getConstraintMode: String           = $(constraintMode)

  /** Weight for constraint violations in soft mode. Higher = stricter enforcement.
    */
  final val constraintWeight: DoubleParam = new DoubleParam(
    this,
    "constraintWeight",
    "Weight for constraint violations",
    ParamValidators.gtEq(0.0)
  )
  def getConstraintWeight: Double         = $(constraintWeight)

  setDefault(
    idCol            -> "id",
    constraintMode   -> "soft",
    constraintWeight -> 1.0
  )
}

/** Constrained K-Means clustering (COP-KMeans) with Bregman divergences.
  *
  * Incorporates domain knowledge via pairwise constraints:
  *   - '''Must-link:''' Points should be in the same cluster
  *   - '''Cannot-link:''' Points should be in different clusters
  *
  * ==Algorithm==
  *
  * '''Soft mode''' (default): Adds penalty terms to the objective
  * {{{
  * minimize Σ_x D(x, μ_c(x)) + λ Σ_{(i,j) ∈ ML} w_ij · 1[c(i) ≠ c(j)]
  *                          + λ Σ_{(i,j) ∈ CL} w_ij · 1[c(i) = c(j)]
  * }}}
  *
  * '''Hard mode:''' Strictly enforces constraints during assignment
  *   - If no valid assignment exists, falls back to nearest center
  *
  * ==Constraint Input==
  *
  * Constraints are provided via `setConstraints()`:
  * {{{
  * val constraints = ConstraintSet.fromPairs(
  *   mustLinks = Seq((1L, 2L), (3L, 4L)),
  *   cannotLinks = Seq((1L, 5L), (2L, 6L))
  * )
  *
  * val ckm = new ConstrainedKMeans()
  *   .setK(3)
  *   .setConstraints(constraints)
  *   .setConstraintMode("soft")
  *   .setConstraintWeight(0.5)
  *
  * val model = ckm.fit(data)
  * }}}
  *
  * ==Use Cases==
  *
  *   - '''Background knowledge:''' Some points known to be same/different class
  *   - '''Active learning:''' Incorporate user feedback
  *   - '''Hierarchical structure:''' Encode partial hierarchy
  *   - '''Data augmentation:''' Use similarity from external source
  *
  * @see
  *   [[ConstraintSet]] for constraint specification
  * @see
  *   Wagstaff et al. (2001): "Constrained K-means Clustering with Background Knowledge"
  */
class ConstrainedKMeans(override val uid: String)
    extends Estimator[GeneralizedKMeansModel]
    with ConstrainedKMeansParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("constrainedkmeans"))

  // Constraint set (not a Param since it's complex object)
  private var constraintSet: ConstraintSet = ConstraintSet.empty

  /** Set pairwise constraints.
    *
    * @param constraints
    *   constraint set with must-link and cannot-link pairs
    */
  def setConstraints(constraints: ConstraintSet): this.type = {
    this.constraintSet = constraints
    this
  }

  /** Get current constraint set. */
  def getConstraints: ConstraintSet = constraintSet

  // Parameter setters
  def setIdCol(value: String): this.type            = set(idCol, value)
  def setConstraintMode(value: String): this.type   = set(constraintMode, value)
  def setConstraintWeight(value: Double): this.type = set(constraintWeight, value)
  def setK(value: Int): this.type                   = set(k, value)
  def setDivergence(value: String): this.type       = set(divergence, value)
  def setSmoothing(value: Double): this.type        = set(smoothing, value)
  def setFeaturesCol(value: String): this.type      = set(featuresCol, value)
  def setPredictionCol(value: String): this.type    = set(predictionCol, value)
  def setWeightCol(value: String): this.type        = set(weightCol, value)
  def setMaxIter(value: Int): this.type             = set(maxIter, value)
  def setTol(value: Double): this.type              = set(tol, value)
  def setSeed(value: Long): this.type               = set(seed, value)

  override def fit(dataset: Dataset[_]): GeneralizedKMeansModel = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()

    logInfo(
      s"Starting Constrained K-Means: k=${$(k)}, mode=${$(constraintMode)}, " +
        s"constraints=${constraintSet.size} (${constraintSet.numMustLink} ML, ${constraintSet.numCannotLink} CL)"
    )

    // Validate constraints are satisfiable
    if (!constraintSet.isEmpty && !constraintSet.isSatisfiable) {
      logWarning("Constraints may not be satisfiable (conflicting must-link and cannot-link)")
    }

    df.cache()
    try {
      val kernel    = createKernel()
      val startTime = System.currentTimeMillis()

      // Initialize centers
      val initialCenters = initializeCenters(df, kernel)
      logInfo(s"Initialized ${initialCenters.length} centers")

      // Run constrained Lloyd's iterations
      val result = runConstrainedLloyds(df, initialCenters, kernel)

      val elapsed = System.currentTimeMillis() - startTime
      logInfo(
        s"Constrained K-Means completed: ${result.iterations} iterations, " +
          s"converged=${result.converged}, violations=${result.constraintViolations}"
      )

      // Create model
      val model = new GeneralizedKMeansModel(uid, result.centers, kernel.name)
      copyValues(model.setParent(this))

      // Training summary
      val summary = TrainingSummary(
        algorithm = "ConstrainedKMeans",
        k = $(k),
        effectiveK = result.centers.length,
        dim = result.centers.headOption.map(_.length).getOrElse(0),
        numPoints = df.count(),
        iterations = result.iterations,
        converged = result.converged,
        distortionHistory = result.distortionHistory,
        movementHistory = result.movementHistory,
        assignmentStrategy = $(constraintMode),
        divergence = $(divergence),
        elapsedMillis = elapsed
      )
      model.trainingSummary = Some(summary)

      model
    } finally {
      df.unpersist()
    }
  }

  /** Run constrained Lloyd's algorithm. */
  private def runConstrainedLloyds(
      df: DataFrame,
      initialCenters: Array[Array[Double]],
      kernel: BregmanKernel
  ): ConstrainedResult = {
    val isSoftMode = $(constraintMode) == "soft"
    val lambda     = $(constraintWeight)

    var centers           = initialCenters
    var iteration         = 0
    var converged         = false
    val distortionHistory = scala.collection.mutable.ArrayBuffer[Double]()
    val movementHistory   = scala.collection.mutable.ArrayBuffer[Double]()
    var totalViolations   = 0

    while (iteration < $(maxIter) && !converged) {
      iteration += 1

      // Assignment step (with constraints)
      val (assignments, violations) = if (isSoftMode) {
        assignSoftConstrained(df, centers, kernel, lambda)
      } else {
        assignHardConstrained(df, centers, kernel)
      }
      totalViolations = violations

      // Update step
      val newCenters = updateCenters(df, assignments, kernel)

      // Check convergence
      val movement = computeMovement(centers, newCenters)
      movementHistory += movement

      val distortion = computeDistortion(df, assignments, newCenters, kernel)
      distortionHistory += distortion

      converged = movement < $(tol)
      centers = newCenters

      logDebug(
        f"Iteration $iteration: distortion=$distortion%.4f, " +
          f"movement=$movement%.6f, violations=$violations"
      )
    }

    ConstrainedResult(
      centers = centers,
      iterations = iteration,
      converged = converged,
      distortionHistory = distortionHistory.toArray,
      movementHistory = movementHistory.toArray,
      constraintViolations = totalViolations
    )
  }

  /** Soft constrained assignment: minimize divergence + constraint penalty. */
  private def assignSoftConstrained(
      df: DataFrame,
      centers: Array[Array[Double]],
      kernel: BregmanKernel,
      lambda: Double
  ): (Map[Long, Int], Int) = {
    val k          = centers.length
    val centersVec = centers.map(Vectors.dense)
    val penalty    = new LinearConstraintPenalty(lambda, lambda)

    // Collect data with IDs
    val points = df.select($(idCol), $(featuresCol)).collect()

    // Build assignments incrementally to enable constraint checking
    val assignments = scala.collection.mutable.Map.empty[Long, Int]
    var violations  = 0

    for (row <- points) {
      val pointId  = row.getLong(0)
      val features = row.getAs[Vector](1)

      // Compute divergence + penalty for each cluster
      var bestCluster = 0
      var bestCost    = Double.MaxValue

      for (c <- 0 until k) {
        val divergence  = kernel.divergence(features, centersVec(c))
        val penaltyCost = penalty.computeAssignmentPenalty(
          pointId,
          c,
          assignments.toMap,
          constraintSet
        )
        val totalCost   = divergence + penaltyCost

        if (totalCost < bestCost) {
          bestCost = totalCost
          bestCluster = c
        }
      }

      // Check for violations after assignment
      if (!constraintSet.isValidAssignment(pointId, bestCluster, assignments.toMap)) {
        violations += 1
      }

      assignments(pointId) = bestCluster
    }

    (assignments.toMap, violations)
  }

  /** Hard constrained assignment: enforce constraints strictly. */
  private def assignHardConstrained(
      df: DataFrame,
      centers: Array[Array[Double]],
      kernel: BregmanKernel
  ): (Map[Long, Int], Int) = {
    val k          = centers.length
    val centersVec = centers.map(Vectors.dense)

    val points      = df.select($(idCol), $(featuresCol)).collect()
    val assignments = scala.collection.mutable.Map.empty[Long, Int]
    var violations  = 0

    // First, handle must-link components (reserved for future use)
    val _ = constraintSet.mustLinkComponents()

    // Sort points to process constrained ones first
    val sortedPoints = points.sortBy { row =>
      val pointId = row.getLong(0)
      -constraintSet.mustLinkPartners(pointId).size - constraintSet.cannotLinkPartners(pointId).size
    }

    for (row <- sortedPoints) {
      val pointId  = row.getLong(0)
      val features = row.getAs[Vector](1)

      // Find valid clusters (respecting constraints)
      val validClusters = (0 until k).filter { c =>
        constraintSet.isValidAssignment(pointId, c, assignments.toMap)
      }

      val bestCluster = if (validClusters.nonEmpty) {
        // Choose best valid cluster
        validClusters.minBy(c => kernel.divergence(features, centersVec(c)))
      } else {
        // No valid cluster - fall back to nearest (constraint violation)
        violations += 1
        (0 until k).minBy(c => kernel.divergence(features, centersVec(c)))
      }

      assignments(pointId) = bestCluster
    }

    (assignments.toMap, violations)
  }

  /** Update centers based on assignments. */
  private def updateCenters(
      df: DataFrame,
      assignments: Map[Long, Int],
      kernel: BregmanKernel
  ): Array[Array[Double]] = {
    val k             = $(this.k)
    val bcAssignments = df.sparkSession.sparkContext.broadcast(assignments)
    val bcKernel      = df.sparkSession.sparkContext.broadcast(kernel)

    // Add cluster column based on assignments
    val assignUDF = udf { (id: Long) =>
      bcAssignments.value.getOrElse(id, -1)
    }

    val assigned = df.withColumn("cluster", assignUDF(col($(idCol))))

    // Compute centers via gradient mean
    val numFeatures = df.select($(featuresCol)).first().getAs[Vector](0).size

    val centers = (0 until k).map { clusterId =>
      val clusterPoints = assigned.filter(col("cluster") === clusterId)
      val count         = clusterPoints.count()

      if (count == 0) {
        Array.fill(numFeatures)(0.0)
      } else {
        // Gradient mean for Bregman divergence
        val gradSum = clusterPoints
          .select($(featuresCol))
          .rdd
          .map { row =>
            val features = row.getAs[Vector](0)
            bcKernel.value.grad(features).toArray
          }
          .reduce((a, b) => a.zip(b).map { case (x, y) => x + y })

        val meanGrad = gradSum.map(_ / count)
        kernel.invGrad(Vectors.dense(meanGrad)).toArray
      }
    }.toArray

    bcAssignments.destroy()
    centers
  }

  /** Compute max center movement. */
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

  /** Compute total distortion. */
  private def computeDistortion(
      df: DataFrame,
      assignments: Map[Long, Int],
      centers: Array[Array[Double]],
      kernel: BregmanKernel
  ): Double = {
    val centersVec = centers.map(Vectors.dense)
    df.select($(idCol), $(featuresCol))
      .collect()
      .map { row =>
        val pointId  = row.getLong(0)
        val features = row.getAs[Vector](1)
        val cluster  = assignments.getOrElse(pointId, 0)
        kernel.divergence(features, centersVec(cluster))
      }
      .sum
  }

  private def createKernel(): BregmanKernel = {
    BregmanKernel.create($(divergence), $(smoothing))
  }

  private def initializeCenters(df: DataFrame, kernel: BregmanKernel): Array[Array[Double]] = {
    val fraction = math.min(1.0, ($(k) * 10.0) / df.count().toDouble)
    df.select($(featuresCol))
      .sample(withReplacement = false, fraction, $(seed))
      .limit($(k))
      .collect()
      .map(_.getAs[Vector](0).toArray)
  }

  override def copy(extra: ParamMap): ConstrainedKMeans = {
    val copied = defaultCopy(extra).asInstanceOf[ConstrainedKMeans]
    copied.constraintSet = this.constraintSet
    copied
  }

  override def transformSchema(schema: StructType): StructType = {
    require(
      schema.fieldNames.contains($(idCol)),
      s"ID column '${$(idCol)}' not found in schema"
    )
    validateAndTransformSchema(schema)
  }
}

object ConstrainedKMeans extends DefaultParamsReadable[ConstrainedKMeans] {
  override def load(path: String): ConstrainedKMeans = super.load(path)
}

/** Internal result class for constrained Lloyd's algorithm. */
private[ml] case class ConstrainedResult(
    centers: Array[Array[Double]],
    iterations: Int,
    converged: Boolean,
    distortionHistory: Array[Double],
    movementHistory: Array[Double],
    constraintViolations: Int
)
