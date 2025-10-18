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

import org.apache.spark.sql.DataFrame

/** Algebraic Data Type representing cluster assignment strategies.
  *
  * This ADT defines a composable, testable algebra for expressing how points are assigned to
  * clusters. Instead of implicit string-based dispatch in AssignmentStrategy, this provides:
  *   - Explicit, type-safe representation of assignment strategies
  *   - Composability through case class constructors
  *   - Easy testing through pattern matching
  *   - Clear separation between strategy definition and execution
  *
  * Design principles:
  *   - Each case class represents a distinct assignment strategy
  *   - Strategies are data (no behavior), execution is in interpreter
  *   - Support for both RDD and DataFrame-based assignments
  *   - Enable optimization and analysis of assignment plans
  *
  * Example usage:
  * {{{
  *   val plan = CrossJoinAssignmentPlan(
  *     divergence = "squaredEuclidean",
  *     rowIdProvider = RowIdProvider.monotonic(),
  *     featuresCol = "features",
  *     predictionCol = "prediction"
  *   )
  *
  *   val interpreter = new AssignmentPlanInterpreter(ops, centers)
  *   val assignments = interpreter.execute(plan, dataFrame)
  * }}}
  */
sealed trait AssignmentPlan extends Serializable {

  /** Column name for input features */
  def featuresCol: String

  /** Column name for output cluster predictions */
  def predictionCol: String
}

/** Cross-join based assignment for squared Euclidean distance.
  *
  * This strategy performs a cross join between points and centers, computes distances using
  * broadcast centers, and selects the closest center for each point. Optimized for squared
  * Euclidean distance.
  *
  * Performance characteristics:
  *   - Efficient for moderate k (< 1000)
  *   - Uses broadcast join when centers fit in memory
  *   - Leverages Spark's shuffle/broadcast optimizations
  *
  * @param divergence
  *   divergence function name (typically "squaredEuclidean")
  * @param rowIdProvider
  *   provider for stable row IDs (needed for join)
  * @param featuresCol
  *   input features column name
  * @param predictionCol
  *   output prediction column name
  */
case class CrossJoinAssignmentPlan(
    divergence: String,
    rowIdProvider: RowIdProvider,
    featuresCol: String = "features",
    predictionCol: String = "prediction"
) extends AssignmentPlan

/** RDD-based assignment using map-side distance computation.
  *
  * This strategy converts the DataFrame to RDD, broadcasts centers, and computes distances in a map
  * operation. This is the traditional Spark MLlib approach.
  *
  * Performance characteristics:
  *   - Efficient for large k or custom divergences
  *   - No shuffle required (map-only operation)
  *   - Requires serialization/deserialization overhead
  *
  * @param divergence
  *   divergence function name
  * @param featuresCol
  *   input features column name
  * @param predictionCol
  *   output prediction column name
  */
case class RDDMapAssignmentPlan(
    divergence: String,
    featuresCol: String = "features",
    predictionCol: String = "prediction"
) extends AssignmentPlan

/** DataFrame-based assignment using UDF for distance computation.
  *
  * This strategy uses a UDF to compute distances to all centers for each point, avoiding RDD
  * conversion. Useful for integration with DataFrame-based pipelines.
  *
  * Performance characteristics:
  *   - Stays in DataFrame API (no RDD conversion)
  *   - UDF overhead for distance computation
  *   - Good for integration with ML pipelines
  *
  * @param divergence
  *   divergence function name
  * @param featuresCol
  *   input features column name
  * @param predictionCol
  *   output prediction column name
  */
case class UDFAssignmentPlan(
    divergence: String,
    featuresCol: String = "features",
    predictionCol: String = "prediction"
) extends AssignmentPlan

/** Conditional assignment plan that selects strategy based on data characteristics.
  *
  * This meta-strategy analyzes the data (e.g., number of centers, divergence type) and selects the
  * most appropriate assignment plan. This enables automatic optimization.
  *
  * @param defaultPlan
  *   fallback plan if no condition matches
  * @param featuresCol
  *   input features column name
  * @param predictionCol
  *   output prediction column name
  */
case class ConditionalAssignmentPlan(
    defaultPlan: AssignmentPlan,
    featuresCol: String = "features",
    predictionCol: String = "prediction"
) extends AssignmentPlan

/** Interpreter for executing AssignmentPlan strategies.
  *
  * This class implements the interpreter pattern, translating AssignmentPlan ADT values into
  * concrete DataFrame operations. It separates strategy definition from execution, enabling:
  *   - Testing strategies without executing them
  *   - Optimizing/analyzing plans before execution
  *   - Switching implementations without changing client code
  *
  * @param ops
  *   Bregman point operations for distance calculations
  * @param centers
  *   cluster centers to assign points to
  */
class AssignmentPlanInterpreter(ops: BregmanPointOps, centers: CenterStore)
    extends Serializable
    with Logging {

  /** Execute an assignment plan to assign points to clusters.
    *
    * @param plan
    *   the assignment plan to execute
    * @param df
    *   input DataFrame with points to assign
    * @return
    *   DataFrame with prediction column added
    */
  def execute(plan: AssignmentPlan, df: DataFrame): DataFrame = {
    plan match {
      case p: CrossJoinAssignmentPlan =>
        executeCrossJoin(p, df)

      case p: RDDMapAssignmentPlan =>
        executeRDDMap(p, df)

      case p: UDFAssignmentPlan =>
        executeUDF(p, df)

      case p: ConditionalAssignmentPlan =>
        executeConditional(p, df)
    }
  }

  /** Execute cross-join based assignment. */
  private def executeCrossJoin(plan: CrossJoinAssignmentPlan, df: DataFrame): DataFrame = {
    import org.apache.spark.sql.functions._

    logger.info(s"Executing CrossJoinAssignment with ${centers.count} centers")

    // Add row IDs to enable efficient join
    val dfWithIds = plan.rowIdProvider.addRowId(df, "row_id")

    // Convert centers to DataFrame
    val centersDF = centers.toDataFrame(df.sparkSession, "cluster_id", "center")

    // Broadcast centers for efficient join
    val broadcastCenters = broadcast(centersDF)

    // Cross join and compute distances
    // For squared Euclidean: distance = ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x'c
    val distanceUDF = udf(
      (features: org.apache.spark.ml.linalg.Vector, center: org.apache.spark.ml.linalg.Vector) => {
        val featureArr = features.toArray
        val centerArr  = center.toArray
        var sum        = 0.0
        var i          = 0
        while (i < featureArr.length) {
          val diff = featureArr(i) - centerArr(i)
          sum += diff * diff
          i += 1
        }
        sum
      }
    )

    val joined = dfWithIds
      .crossJoin(broadcastCenters)
      .withColumn("distance", distanceUDF(col(plan.featuresCol), col("center")))

    // Select minimum distance for each point
    val window =
      org.apache.spark.sql.expressions.Window.partitionBy("row_id").orderBy(col("distance").asc)

    val withPredictions = joined
      .withColumn("rank", row_number().over(window))
      .filter(col("rank") === 1)
      .select(
        dfWithIds.columns.map(c => col(c)) :+ col("cluster_id").as(plan.predictionCol): _*
      )

    // Remove row_id if it was added
    plan.rowIdProvider.removeRowId(withPredictions, "row_id")
  }

  /** Execute RDD map-based assignment. */
  private def executeRDDMap(plan: RDDMapAssignmentPlan, df: DataFrame): DataFrame = {
    logger.info(s"Executing RDDMapAssignment with ${centers.count} centers")

    val spark       = df.sparkSession
    val schema      = df.schema
    val featuresIdx = schema.fieldIndex(plan.featuresCol)

    // Broadcast centers
    val bcCenters = spark.sparkContext.broadcast(centers.toArray)

    // Convert to RDD, compute assignments, convert back
    val assignmentsRDD = df.rdd.map { row =>
      val features      = row.getAs[org.apache.spark.ml.linalg.Vector](featuresIdx)
      val point         = ops.toPoint(
        com.massivedatascience.linalg.WeightedVector.fromInhomogeneousWeighted(features, 1.0)
      )
      val assignedIndex = ops.findClosestCluster(bcCenters.value, point)

      org.apache.spark.sql.Row.fromSeq(row.toSeq :+ assignedIndex)
    }

    // Create new schema with prediction column
    val newSchema = org.apache.spark.sql.types.StructType(
      schema.fields :+ org.apache.spark.sql.types
        .StructField(plan.predictionCol, org.apache.spark.sql.types.IntegerType, nullable = false)
    )

    spark.createDataFrame(assignmentsRDD, newSchema)
  }

  /** Execute UDF-based assignment. */
  private def executeUDF(plan: UDFAssignmentPlan, df: DataFrame): DataFrame = {
    import org.apache.spark.sql.functions._

    logger.info(s"Executing UDFAssignment with ${centers.count} centers")

    // Broadcast centers
    val bcCenters = df.sparkSession.sparkContext.broadcast(centers.toArray)

    // Create UDF for assignment
    val assignmentUDF = udf((features: org.apache.spark.ml.linalg.Vector) => {
      val point = ops.toPoint(
        com.massivedatascience.linalg.WeightedVector.fromInhomogeneousWeighted(features, 1.0)
      )
      ops.findClosestCluster(bcCenters.value, point)
    })

    df.withColumn(plan.predictionCol, assignmentUDF(col(plan.featuresCol)))
  }

  /** Execute conditional assignment by selecting best strategy. */
  private def executeConditional(plan: ConditionalAssignmentPlan, df: DataFrame): DataFrame = {
    logger.info("Executing ConditionalAssignment: selecting strategy")

    // Simple heuristic: use CrossJoin for squared Euclidean with small k, otherwise RDD map
    val selectedPlan = if (centers.count < 100) {
      logger.info("Selected CrossJoinAssignment (k < 100)")
      CrossJoinAssignmentPlan(
        divergence = "squaredEuclidean",
        rowIdProvider = RowIdProvider.monotonic(),
        featuresCol = plan.featuresCol,
        predictionCol = plan.predictionCol
      )
    } else {
      logger.info("Selected RDDMapAssignment (k >= 100)")
      RDDMapAssignmentPlan(
        divergence = "general",
        featuresCol = plan.featuresCol,
        predictionCol = plan.predictionCol
      )
    }

    execute(selectedPlan, df)
  }
}

/** Factory methods for creating assignment plans. */
object AssignmentPlan {

  /** Create a cross-join assignment plan for squared Euclidean distance.
    *
    * @param featuresCol
    *   input features column
    * @param predictionCol
    *   output prediction column
    * @param rowIdProvider
    *   row ID provider (default: monotonic)
    * @return
    *   cross-join assignment plan
    */
  def crossJoin(
      featuresCol: String = "features",
      predictionCol: String = "prediction",
      rowIdProvider: RowIdProvider = RowIdProvider.monotonic()
  ): AssignmentPlan = {
    CrossJoinAssignmentPlan("squaredEuclidean", rowIdProvider, featuresCol, predictionCol)
  }

  /** Create an RDD map assignment plan.
    *
    * @param divergence
    *   divergence function name
    * @param featuresCol
    *   input features column
    * @param predictionCol
    *   output prediction column
    * @return
    *   RDD map assignment plan
    */
  def rddMap(
      divergence: String = "squaredEuclidean",
      featuresCol: String = "features",
      predictionCol: String = "prediction"
  ): AssignmentPlan = {
    RDDMapAssignmentPlan(divergence, featuresCol, predictionCol)
  }

  /** Create a UDF-based assignment plan.
    *
    * @param divergence
    *   divergence function name
    * @param featuresCol
    *   input features column
    * @param predictionCol
    *   output prediction column
    * @return
    *   UDF assignment plan
    */
  def udf(
      divergence: String = "squaredEuclidean",
      featuresCol: String = "features",
      predictionCol: String = "prediction"
  ): AssignmentPlan = {
    UDFAssignmentPlan(divergence, featuresCol, predictionCol)
  }

  /** Create a conditional assignment plan that auto-selects strategy.
    *
    * @param featuresCol
    *   input features column
    * @param predictionCol
    *   output prediction column
    * @return
    *   conditional assignment plan
    */
  def auto(
      featuresCol: String = "features",
      predictionCol: String = "prediction"
  ): AssignmentPlan = {
    ConditionalAssignmentPlan(
      defaultPlan = rddMap(featuresCol = featuresCol, predictionCol = predictionCol),
      featuresCol = featuresCol,
      predictionCol = predictionCol
    )
  }
}
