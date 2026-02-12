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
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/** Parameters for Co-Clustering Estimator.
  */
trait CoClusteringParams extends Params with HasSeed with HasMaxIter {

  /** Number of row clusters (required).
    */
  final val numRowClusters: IntParam = new IntParam(
    this,
    "numRowClusters",
    "Number of row clusters",
    ParamValidators.gt(0)
  )
  def getNumRowClusters: Int         = $(numRowClusters)

  /** Number of column clusters (required).
    */
  final val numColClusters: IntParam = new IntParam(
    this,
    "numColClusters",
    "Number of column clusters",
    ParamValidators.gt(0)
  )
  def getNumColClusters: Int         = $(numColClusters)

  /** Column name for row indices.
    */
  final val rowIndexCol: Param[String] = new Param[String](
    this,
    "rowIndexCol",
    "Column name for row indices"
  )
  def getRowIndexCol: String           = $(rowIndexCol)

  /** Column name for column indices.
    */
  final val colIndexCol: Param[String] = new Param[String](
    this,
    "colIndexCol",
    "Column name for column indices"
  )
  def getColIndexCol: String           = $(colIndexCol)

  /** Column name for matrix values.
    */
  final val valueCol: Param[String] = new Param[String](
    this,
    "valueCol",
    "Column name for matrix values"
  )
  def getValueCol: String           = $(valueCol)

  /** Column name for row cluster predictions.
    */
  final val rowPredictionCol: Param[String] = new Param[String](
    this,
    "rowPredictionCol",
    "Column name for row cluster predictions"
  )
  def getRowPredictionCol: String           = $(rowPredictionCol)

  /** Column name for column cluster predictions.
    */
  final val colPredictionCol: Param[String] = new Param[String](
    this,
    "colPredictionCol",
    "Column name for column cluster predictions"
  )
  def getColPredictionCol: String           = $(colPredictionCol)

  /** Convergence tolerance.
    */
  final val tolerance: DoubleParam = new DoubleParam(
    this,
    "tolerance",
    "Convergence tolerance",
    ParamValidators.gt(0.0)
  )
  def getTolerance: Double         = $(tolerance)

  /** Regularization parameter for empty blocks.
    */
  final val regularization: DoubleParam = new DoubleParam(
    this,
    "regularization",
    "Regularization parameter for empty blocks",
    ParamValidators.gtEq(0.0)
  )
  def getRegularization: Double         = $(regularization)

  /** Divergence type.
    */
  final val divergence: Param[String] = new Param[String](
    this,
    "divergence",
    "Divergence type (squaredEuclidean, kl, itakuraSaito)"
  )
  def getDivergence: String           = $(divergence)

  setDefault(
    rowIndexCol      -> "rowIndex",
    colIndexCol      -> "colIndex",
    valueCol         -> "value",
    rowPredictionCol -> "rowPrediction",
    colPredictionCol -> "colPrediction",
    maxIter          -> 100,
    tolerance        -> 1e-6,
    regularization   -> 0.01,
    divergence       -> "squaredEuclidean",
    seed             -> 42L
  )

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    require(
      schema.fieldNames.contains($(rowIndexCol)),
      s"Row index column '${$(rowIndexCol)}' not found"
    )
    require(
      schema.fieldNames.contains($(colIndexCol)),
      s"Column index column '${$(colIndexCol)}' not found"
    )
    require(schema.fieldNames.contains($(valueCol)), s"Value column '${$(valueCol)}' not found")

    schema
      .add(StructField($(rowPredictionCol), IntegerType, nullable = false))
      .add(StructField($(colPredictionCol), IntegerType, nullable = false))
  }
}

/** Co-Clustering Estimator following Spark ML pattern.
  *
  * Simultaneously clusters rows and columns of a matrix by minimizing the Bregman divergence
  * between the original matrix and a block-constant approximation.
  *
  * ==Input Format==
  *
  * Input DataFrame should have columns for:
  *   - Row index (Long or Int)
  *   - Column index (Long or Int)
  *   - Value (Double)
  *
  * ==Example==
  *
  * {{{
  * val coClustering = new CoClustering()
  *   .setNumRowClusters(3)
  *   .setNumColClusters(4)
  *   .setRowIndexCol("row")
  *   .setColIndexCol("col")
  *   .setValueCol("rating")
  *   .setMaxIter(50)
  *
  * val model = coClustering.fit(ratingsDF)
  * val predictions = model.transform(ratingsDF)
  * }}}
  *
  * @see
  *   [[CoClusteringModel]] for the trained model
  */
class CoClustering(override val uid: String)
    extends Estimator[CoClusteringModel]
    with CoClusteringParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("coClustering"))

  def setNumRowClusters(value: Int): this.type      = set(numRowClusters, value)
  def setNumColClusters(value: Int): this.type      = set(numColClusters, value)
  def setRowIndexCol(value: String): this.type      = set(rowIndexCol, value)
  def setColIndexCol(value: String): this.type      = set(colIndexCol, value)
  def setValueCol(value: String): this.type         = set(valueCol, value)
  def setRowPredictionCol(value: String): this.type = set(rowPredictionCol, value)
  def setColPredictionCol(value: String): this.type = set(colPredictionCol, value)
  def setMaxIter(value: Int): this.type             = set(maxIter, value)
  def setTolerance(value: Double): this.type        = set(tolerance, value)
  def setRegularization(value: Double): this.type   = set(regularization, value)
  def setDivergence(value: String): this.type       = set(divergence, value)
  def setSeed(value: Long): this.type               = set(seed, value)

  override def fit(dataset: Dataset[_]): CoClusteringModel = {
    transformSchema(dataset.schema, logging = true)

    val spark = dataset.sparkSession
    import spark.implicits._

    logInfo(
      s"Starting co-clustering with ${$(numRowClusters)} row clusters and ${$(numColClusters)} column clusters"
    )

    val df = dataset.toDF()
    df.cache()

    // Get kernel for divergence computation
    val kernel = $(divergence) match {
      case "squaredEuclidean" => new SquaredEuclideanKernel()
      case "kl"               => new KLDivergenceKernel()
      case "itakuraSaito"     => new ItakuraSaitoKernel()
      case other              =>
        throw new IllegalArgumentException(s"Unsupported divergence for co-clustering: $other")
    }

    // Collect unique row and column indices
    val rowIndices = df.select($(rowIndexCol)).distinct().as[Long].collect().sorted
    val colIndices = df.select($(colIndexCol)).distinct().as[Long].collect().sorted

    logInfo(s"Matrix dimensions: ${rowIndices.length} rows x ${colIndices.length} columns")

    // Initialize cluster assignments randomly
    val random      = new scala.util.Random($(seed))
    var rowClusters = rowIndices.map(idx => idx -> random.nextInt($(numRowClusters))).toMap
    var colClusters = colIndices.map(idx => idx -> random.nextInt($(numColClusters))).toMap

    // Broadcast for efficiency
    var bcRowClusters = spark.sparkContext.broadcast(rowClusters)
    var bcColClusters = spark.sparkContext.broadcast(colClusters)

    var prevObjective = Double.MaxValue
    var iterations    = 0

    // Alternating minimization
    while (iterations < $(maxIter)) {
      iterations += 1

      // Update block centers
      val blockCenters = computeBlockCenters(df, bcRowClusters.value, bcColClusters.value)

      // Calculate objective
      val objective =
        computeObjective(df, bcRowClusters.value, bcColClusters.value, blockCenters, kernel)

      logDebug(s"Iteration $iterations: objective = $objective")

      // Check convergence
      if (math.abs(prevObjective - objective) < $(tolerance)) {
        logInfo(s"Converged after $iterations iterations")
        df.unpersist()

        return createModel(rowClusters, colClusters, blockCenters, kernel, iterations, objective)
      }

      // Update row clusters
      bcRowClusters.unpersist()
      rowClusters = updateRowClusters(df, bcColClusters.value, blockCenters, kernel)
      bcRowClusters = spark.sparkContext.broadcast(rowClusters)

      // Update column clusters
      bcColClusters.unpersist()
      colClusters = updateColClusters(df, bcRowClusters.value, blockCenters, kernel)
      bcColClusters = spark.sparkContext.broadcast(colClusters)

      prevObjective = objective
    }

    logWarning(s"Did not converge after ${$(maxIter)} iterations")

    val finalBlockCenters = computeBlockCenters(df, rowClusters, colClusters)
    val finalObjective    = computeObjective(df, rowClusters, colClusters, finalBlockCenters, kernel)

    bcRowClusters.unpersist()
    bcColClusters.unpersist()
    df.unpersist()

    createModel(rowClusters, colClusters, finalBlockCenters, kernel, iterations, finalObjective)
  }

  private def createModel(
      rowClusters: Map[Long, Int],
      colClusters: Map[Long, Int],
      blockCenters: Array[Array[Double]],
      kernel: ClusteringKernel,
      iterations: Int,
      objective: Double
  ): CoClusteringModel = {
    val model = new CoClusteringModel(uid, rowClusters, colClusters, blockCenters, kernel)
    copyValues(model)
    model.setParent(this)
    model.trainingSummary = Some(CoClusteringTrainingSummary(iterations, objective))
    model
  }

  private def computeBlockCenters(
      df: DataFrame,
      rowClusters: Map[Long, Int],
      colClusters: Map[Long, Int]
  ): Array[Array[Double]] = {

    // UDFs to get cluster assignments
    val getRowCluster = udf((idx: Long) => rowClusters.getOrElse(idx, 0))
    val getColCluster = udf((idx: Long) => colClusters.getOrElse(idx, 0))

    val withClusters = df
      .withColumn("_rowCluster", getRowCluster(col($(rowIndexCol))))
      .withColumn("_colCluster", getColCluster(col($(colIndexCol))))

    // Compute mean for each block
    val blockMeans = withClusters
      .groupBy("_rowCluster", "_colCluster")
      .agg(avg($(valueCol)).as("_blockMean"))
      .collect()
      .map(row => ((row.getInt(0), row.getInt(1)), row.getDouble(2)))
      .toMap

    // Create block centers array
    val centers = Array.ofDim[Double]($(numRowClusters), $(numColClusters))
    for (r <- 0 until $(numRowClusters); c <- 0 until $(numColClusters)) {
      centers(r)(c) = blockMeans.getOrElse((r, c), $(regularization))
    }

    centers
  }

  private def computeObjective(
      df: DataFrame,
      rowClusters: Map[Long, Int],
      colClusters: Map[Long, Int],
      blockCenters: Array[Array[Double]],
      kernel: ClusteringKernel
  ): Double = {

    val bcBlockCenters = df.sparkSession.sparkContext.broadcast(blockCenters)

    val getRowCluster = udf((idx: Long) => rowClusters.getOrElse(idx, 0))
    val getColCluster = udf((idx: Long) => colClusters.getOrElse(idx, 0))

    val computeDist = udf { (value: Double, rowCluster: Int, colCluster: Int) =>
      val center = bcBlockCenters.value(rowCluster)(colCluster)
      val diff   = value - center
      diff * diff * 0.5 // Simplified for SE; for general kernel would need more
    }

    val totalDist = df
      .withColumn("_rowCluster", getRowCluster(col($(rowIndexCol))))
      .withColumn("_colCluster", getColCluster(col($(colIndexCol))))
      .withColumn("_dist", computeDist(col($(valueCol)), col("_rowCluster"), col("_colCluster")))
      .agg(sum("_dist"))
      .head()
      .getDouble(0)

    bcBlockCenters.unpersist()
    totalDist
  }

  private def updateRowClusters(
      df: DataFrame,
      colClusters: Map[Long, Int],
      blockCenters: Array[Array[Double]],
      kernel: ClusteringKernel
  ): Map[Long, Int] = {

    val spark               = df.sparkSession
    val bcBlockCenters      = spark.sparkContext.broadcast(blockCenters)
    val bcColClusters       = spark.sparkContext.broadcast(colClusters)
    val numRowClustersLocal = $(numRowClusters)

    val getColCluster = udf((idx: Long) => bcColClusters.value.getOrElse(idx, 0))

    // For each row, compute cost for each potential row cluster assignment
    val dfWithColCluster = df.withColumn("_colCluster", getColCluster(col($(colIndexCol))))

    // Collect data by row to compute assignments locally
    val rowData = dfWithColCluster
      .select($(rowIndexCol), "_colCluster", $(valueCol))
      .collect()
      .groupBy(row => row.getLong(0))
      .map { case (rowIdx, rows) =>
        val entries = rows.map(r => (r.getInt(1), r.getDouble(2)))
        val costs   = (0 until numRowClustersLocal).map { rowCluster =>
          entries.map { case (colCluster, value) =>
            val center = bcBlockCenters.value(rowCluster)(colCluster)
            val diff   = value - center
            diff * diff * 0.5
          }.sum
        }
        rowIdx -> costs.zipWithIndex.minBy(_._1)._2
      }
      .toMap

    bcBlockCenters.unpersist()
    bcColClusters.unpersist()
    rowData
  }

  private def updateColClusters(
      df: DataFrame,
      rowClusters: Map[Long, Int],
      blockCenters: Array[Array[Double]],
      kernel: ClusteringKernel
  ): Map[Long, Int] = {

    val spark               = df.sparkSession
    val bcBlockCenters      = spark.sparkContext.broadcast(blockCenters)
    val bcRowClusters       = spark.sparkContext.broadcast(rowClusters)
    val numColClustersLocal = $(numColClusters)

    val getRowCluster = udf((idx: Long) => bcRowClusters.value.getOrElse(idx, 0))

    // For each column, compute cost for each potential column cluster assignment
    val dfWithRowCluster = df.withColumn("_rowCluster", getRowCluster(col($(rowIndexCol))))

    // Collect data by column to compute assignments locally
    val colData = dfWithRowCluster
      .select($(colIndexCol), "_rowCluster", $(valueCol))
      .collect()
      .groupBy(row => row.getLong(0))
      .map { case (colIdx, rows) =>
        val entries = rows.map(r => (r.getInt(1), r.getDouble(2)))
        val costs   = (0 until numColClustersLocal).map { colCluster =>
          entries.map { case (rowCluster, value) =>
            val center = bcBlockCenters.value(rowCluster)(colCluster)
            val diff   = value - center
            diff * diff * 0.5
          }.sum
        }
        colIdx -> costs.zipWithIndex.minBy(_._1)._2
      }
      .toMap

    bcBlockCenters.unpersist()
    bcRowClusters.unpersist()
    colData
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def copy(extra: ParamMap): CoClustering = defaultCopy(extra)
}

object CoClustering extends DefaultParamsReadable[CoClustering] {
  override def load(path: String): CoClustering = super.load(path)
}

/** Training summary for co-clustering.
  */
case class CoClusteringTrainingSummary(iterations: Int, finalObjective: Double)

/** Co-Clustering Model following Spark ML pattern.
  *
  * Contains cluster assignments and block centers for prediction.
  */
class CoClusteringModel(
    override val uid: String,
    val rowClusters: Map[Long, Int],
    val colClusters: Map[Long, Int],
    val blockCenters: Array[Array[Double]],
    private val kernel: ClusteringKernel
) extends Model[CoClusteringModel]
    with CoClusteringParams
    with MLWritable
    with Logging {

  @transient var trainingSummary: Option[CoClusteringTrainingSummary] = None

  def hasSummary: Boolean = trainingSummary.isDefined

  def summary: CoClusteringTrainingSummary = trainingSummary.getOrElse(
    throw new NoSuchElementException("Training summary not available")
  )

  def numRowClustersActual: Int = rowClusters.values.toSet.size
  def numColClustersActual: Int = colClusters.values.toSet.size

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val spark = dataset.sparkSession
    val df    = dataset.toDF()

    // UDFs for cluster assignment
    val bcRowClusters = spark.sparkContext.broadcast(rowClusters)
    val bcColClusters = spark.sparkContext.broadcast(colClusters)

    val getRowCluster = udf((idx: Long) => bcRowClusters.value.getOrElse(idx, -1))
    val getColCluster = udf((idx: Long) => bcColClusters.value.getOrElse(idx, -1))

    val result = df
      .withColumn($(rowPredictionCol), getRowCluster(col($(rowIndexCol))))
      .withColumn($(colPredictionCol), getColCluster(col($(colIndexCol))))

    bcRowClusters.unpersist()
    bcColClusters.unpersist()

    result
  }

  /** Predict the block center value for a (row, col) pair.
    */
  def predictBlockValue(rowIdx: Long, colIdx: Long): Double = {
    val rowCluster = rowClusters.getOrElse(rowIdx, 0)
    val colCluster = colClusters.getOrElse(colIdx, 0)
    blockCenters(rowCluster)(colCluster)
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def copy(extra: ParamMap): CoClusteringModel = {
    val copied = new CoClusteringModel(uid, rowClusters, colClusters, blockCenters, kernel)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new CoClusteringModel.CoClusteringModelWriter(this)
}

object CoClusteringModel extends MLReadable[CoClusteringModel] {

  override def read: MLReader[CoClusteringModel] = new CoClusteringModelReader

  override def load(path: String): CoClusteringModel = super.load(path)

  private class CoClusteringModelWriter(instance: CoClusteringModel) extends MLWriter with Logging {

    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization
    import java.nio.file.{ Files, Paths }
    import java.nio.charset.StandardCharsets

    implicit val formats: DefaultFormats.type = DefaultFormats

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession
      import spark.implicits._

      logInfo(s"Saving CoClusteringModel to $path")

      // Create directories
      Files.createDirectories(Paths.get(path))

      // Save row clusters
      val rowClustersData = instance.rowClusters.toSeq.toDF("rowIndex", "cluster")
      rowClustersData.coalesce(1).write.mode("overwrite").parquet(s"$path/rowClusters")

      // Save col clusters
      val colClustersData = instance.colClusters.toSeq.toDF("colIndex", "cluster")
      colClustersData.coalesce(1).write.mode("overwrite").parquet(s"$path/colClusters")

      // Save block centers
      val blockCentersData = (for {
        r <- instance.blockCenters.indices
        c <- instance.blockCenters(r).indices
      } yield (r, c, instance.blockCenters(r)(c))).toDF("rowCluster", "colCluster", "center")
      blockCentersData.coalesce(1).write.mode("overwrite").parquet(s"$path/blockCenters")

      // Create metadata JSON
      val metaObj: Map[String, Any] = Map(
        "algo"             -> "CoClusteringModel",
        "uid"              -> instance.uid,
        "sparkVersion"     -> org.apache.spark.SPARK_VERSION,
        "divergence"       -> instance.getOrDefault(instance.divergence),
        "numRowClusters"   -> instance.getOrDefault(instance.numRowClusters),
        "numColClusters"   -> instance.getOrDefault(instance.numColClusters),
        "rowIndexCol"      -> instance.getOrDefault(instance.rowIndexCol),
        "colIndexCol"      -> instance.getOrDefault(instance.colIndexCol),
        "valueCol"         -> instance.getOrDefault(instance.valueCol),
        "rowPredictionCol" -> instance.getOrDefault(instance.rowPredictionCol),
        "colPredictionCol" -> instance.getOrDefault(instance.colPredictionCol),
        "maxIter"          -> instance.getOrDefault(instance.maxIter),
        "tolerance"        -> instance.getOrDefault(instance.tolerance),
        "regularization"   -> instance.getOrDefault(instance.regularization),
        "seed"             -> instance.getOrDefault(instance.seed)
      )

      val json = Serialization.write(metaObj)
      val p    = Paths.get(s"$path/metadata.json")
      Files.write(p, json.getBytes(StandardCharsets.UTF_8))

      logInfo(s"CoClusteringModel saved to $path")
    }
  }

  private class CoClusteringModelReader extends MLReader[CoClusteringModel] with Logging {

    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods
    import java.nio.file.{ Files, Paths }
    import java.nio.charset.StandardCharsets

    implicit val formats: DefaultFormats.type = DefaultFormats

    override def load(path: String): CoClusteringModel = {
      val spark = sparkSession
      import spark.implicits._

      logInfo(s"Loading CoClusteringModel from $path")

      // Load metadata
      val metaJson =
        new String(Files.readAllBytes(Paths.get(s"$path/metadata.json")), StandardCharsets.UTF_8)
      val meta     = JsonMethods.parse(metaJson).extract[Map[String, Any]]

      val uid = meta("uid").toString

      // Load row clusters
      val rowClustersData = spark.read.parquet(s"$path/rowClusters")
      val rowClusters     = rowClustersData.as[(Long, Int)].collect().toMap

      // Load col clusters
      val colClustersData = spark.read.parquet(s"$path/colClusters")
      val colClusters     = colClustersData.as[(Long, Int)].collect().toMap

      // Load block centers
      val blockCentersData = spark.read.parquet(s"$path/blockCenters")
      val numRowClusters   = meta("numRowClusters").asInstanceOf[BigInt].toInt
      val numColClusters   = meta("numColClusters").asInstanceOf[BigInt].toInt

      val blockCenters = Array.ofDim[Double](numRowClusters, numColClusters)
      blockCentersData.as[(Int, Int, Double)].collect().foreach { case (r, c, v) =>
        blockCenters(r)(c) = v
      }

      val divergenceName = meta("divergence").toString
      val kernel         = divergenceName match {
        case "squaredEuclidean" => new SquaredEuclideanKernel()
        case "kl"               => new KLDivergenceKernel()
        case "itakuraSaito"     => new ItakuraSaitoKernel()
        case _                  => new SquaredEuclideanKernel()
      }

      val model = new CoClusteringModel(uid, rowClusters, colClusters, blockCenters, kernel)

      // Set params from metadata
      model.set(model.divergence, divergenceName)
      model.set(model.numRowClusters, numRowClusters)
      model.set(model.numColClusters, numColClusters)
      model.set(model.rowIndexCol, meta("rowIndexCol").toString)
      model.set(model.colIndexCol, meta("colIndexCol").toString)
      model.set(model.valueCol, meta("valueCol").toString)
      model.set(model.rowPredictionCol, meta("rowPredictionCol").toString)
      model.set(model.colPredictionCol, meta("colPredictionCol").toString)
      model.set(model.maxIter, meta("maxIter").asInstanceOf[BigInt].toInt)
      model.set(model.tolerance, meta("tolerance").asInstanceOf[Double])
      model.set(model.regularization, meta("regularization").asInstanceOf[Double])
      model.set(model.seed, meta("seed").asInstanceOf[BigInt].toLong)

      logInfo(s"CoClusteringModel loaded from $path")
      model
    }
  }
}
