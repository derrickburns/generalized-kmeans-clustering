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

import com.massivedatascience.linalg.WeightedVector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.util.Random

/** Configuration for Bregman co-clustering algorithm.
  *
  * @param numRowClusters
  *   Number of row clusters
  * @param numColClusters
  *   Number of column clusters
  * @param maxIterations
  *   Maximum number of iterations
  * @param tolerance
  *   Convergence tolerance for objective function
  * @param initStrategy
  *   Initialization strategy ("random", "kmeans++")
  * @param regularization
  *   Regularization parameter for smoothing
  * @param seed
  *   Random seed for reproducibility
  */
case class BregmanCoClusteringConfig(
    numRowClusters: Int,
    numColClusters: Int,
    maxIterations: Int = 100,
    tolerance: Double = 1e-6,
    initStrategy: String = "random",
    regularization: Double = 0.01,
    seed: Long = System.currentTimeMillis()
) {

  require(numRowClusters > 0, s"Number of row clusters must be positive, got: $numRowClusters")
  require(numColClusters > 0, s"Number of column clusters must be positive, got: $numColClusters")
  require(maxIterations > 0, s"Max iterations must be positive, got: $maxIterations")
  require(tolerance > 0.0, s"Tolerance must be positive, got: $tolerance")
  require(regularization >= 0.0, s"Regularization must be non-negative, got: $regularization")
}

/** Represents a data matrix entry with row and column indices.
  *
  * @param rowIndex
  *   Row index in the original matrix
  * @param colIndex
  *   Column index in the original matrix
  * @param value
  *   The value at this matrix position
  * @param weight
  *   Weight for this entry (default 1.0)
  */
case class MatrixEntry(rowIndex: Long, colIndex: Long, value: Double, weight: Double = 1.0) {

  require(weight >= 0.0, s"Weight must be non-negative, got: $weight")
}

/** Result of co-clustering with cluster assignments and statistics.
  *
  * @param rowClusters
  *   Assignment of rows to clusters
  * @param colClusters
  *   Assignment of columns to clusters
  * @param rowCenters
  *   Centroids for row clusters
  * @param colCenters
  *   Centroids for column clusters
  * @param blockCenters
  *   Centroids for each (row_cluster, col_cluster) block
  * @param iterations
  *   Number of iterations performed
  * @param objective
  *   Final objective function value
  * @param convergenceHistory
  *   History of objective values
  */
case class BregmanCoClusteringResult(
    rowClusters: Map[Long, Int],
    colClusters: Map[Long, Int],
    rowCenters: IndexedSeq[BregmanCenter],
    colCenters: IndexedSeq[BregmanCenter],
    blockCenters: Array[Array[BregmanCenter]],
    iterations: Int,
    objective: Double,
    convergenceHistory: Seq[Double]
) {

  /** Get the number of row clusters.
    */
  def numRowClusters: Int = rowCenters.length

  /** Get the number of column clusters.
    */
  def numColClusters: Int = colCenters.length

  /** Predict the cluster assignment for a new matrix entry.
    */
  def predict(entry: MatrixEntry, pointOps: BregmanPointOps): (Int, Int) = {
    val point = BregmanPoint(WeightedVector(Vectors.dense(entry.value), entry.weight), entry.value)

    // Find closest row cluster
    val (rowCluster, _) = pointOps.findClosest(rowCenters, point)

    // Find closest column cluster
    val (colCluster, _) = pointOps.findClosest(colCenters, point)

    (rowCluster, colCluster)
  }

  /** Get comprehensive statistics about the co-clustering result.
    */
  def getStats: Map[String, Double] = {
    Map(
      "numRowClusters"  -> numRowClusters.toDouble,
      "numColClusters"  -> numColClusters.toDouble,
      "totalBlocks"     -> (numRowClusters * numColClusters).toDouble,
      "iterations"      -> iterations.toDouble,
      "finalObjective"  -> objective,
      "convergenceRate" -> {
        if (convergenceHistory.length > 1) {
          val initial    = convergenceHistory.head
          val finalValue = convergenceHistory.last
          if (initial != 0.0) (initial - finalValue) / initial else 0.0
        } else 0.0
      }
    )
  }
}

/** Bregman co-clustering algorithm implementation.
  *
  * This class implements the alternating minimization algorithm for Bregman co-clustering, which
  * simultaneously clusters rows and columns of a data matrix by minimizing the Bregman divergence
  * between the original matrix and a block-constant approximation.
  */
class BregmanCoClustering(config: BregmanCoClusteringConfig, pointOps: BregmanPointOps)
    extends Serializable {

  @transient private lazy val logger = LoggerFactory.getLogger(getClass.getName)

  /** Perform co-clustering on a matrix represented as RDD of MatrixEntry objects.
    *
    * @param data
    *   RDD of matrix entries
    * @return
    *   Co-clustering result with cluster assignments and statistics
    */
  def fit(data: RDD[MatrixEntry]): BregmanCoClusteringResult = {
    logger.info(
      s"Starting Bregman co-clustering with ${config.numRowClusters} row clusters and ${config.numColClusters} column clusters"
    )

    // Cache the data for multiple iterations
    data.cache()

    // Collect row and column indices
    val rowIndices = data.map(_.rowIndex).distinct().collect().sorted
    val colIndices = data.map(_.colIndex).distinct().collect().sorted

    logger.info(s"Matrix dimensions: ${rowIndices.length} rows x ${colIndices.length} columns")

    // Initialize cluster assignments
    val (initialRowClusters, initialColClusters) = initializeClusters(rowIndices, colIndices)

    // Perform alternating minimization
    val result =
      alternatingMinimization(data, rowIndices, colIndices, initialRowClusters, initialColClusters)

    data.unpersist()

    logger.info(
      s"Co-clustering completed in ${result.iterations} iterations with objective ${result.objective}"
    )
    result
  }

  /** Initialize cluster assignments for rows and columns.
    */
  private def initializeClusters(
      rowIndices: Array[Long],
      colIndices: Array[Long]
  ): (Map[Long, Int], Map[Long, Int]) = {

    val random = new Random(config.seed)

    config.initStrategy match {
      case "random" =>
        val rowClusters = rowIndices.map(idx => idx -> random.nextInt(config.numRowClusters)).toMap
        val colClusters = colIndices.map(idx => idx -> random.nextInt(config.numColClusters)).toMap
        (rowClusters, colClusters)

      case "balanced" =>
        // Ensure roughly equal cluster sizes
        val rowClusters = rowIndices.zipWithIndex.map { case (idx, i) =>
          idx -> (i % config.numRowClusters)
        }.toMap
        val colClusters = colIndices.zipWithIndex.map { case (idx, i) =>
          idx -> (i % config.numColClusters)
        }.toMap
        (rowClusters, colClusters)

      case _ =>
        throw new IllegalArgumentException(
          s"Unknown initialization strategy: ${config.initStrategy}"
        )
    }
  }

  /** Perform alternating minimization to optimize co-clustering.
    */
  private def alternatingMinimization(
      data: RDD[MatrixEntry],
      rowIndices: Array[Long],
      colIndices: Array[Long],
      initialRowClusters: Map[Long, Int],
      initialColClusters: Map[Long, Int]
  ): BregmanCoClusteringResult = {

    var rowClusters        = initialRowClusters
    var colClusters        = initialColClusters
    var previousObjective  = Double.MaxValue
    val convergenceHistory = mutable.ListBuffer[Double]()

    for (iteration <- 1 to config.maxIterations) {
      logger.debug(s"Co-clustering iteration $iteration")

      // Update block centers based on current assignments
      val blockCenters = updateBlockCenters(data, rowClusters, colClusters)

      // Calculate current objective
      val currentObjective = calculateObjective(data, rowClusters, colClusters, blockCenters)
      convergenceHistory += currentObjective

      logger.debug(s"Iteration $iteration: objective = $currentObjective")

      // Check convergence
      if (math.abs(previousObjective - currentObjective) < config.tolerance) {
        logger.info(s"Converged after $iteration iterations")

        val rowCenters = computeRowCenters(data, rowClusters)
        val colCenters = computeColCenters(data, colClusters)

        return BregmanCoClusteringResult(
          rowClusters = rowClusters,
          colClusters = colClusters,
          rowCenters = rowCenters,
          colCenters = colCenters,
          blockCenters = blockCenters,
          iterations = iteration,
          objective = currentObjective,
          convergenceHistory = convergenceHistory.toSeq
        )
      }

      // Update row clusters (fix column clusters)
      rowClusters = updateRowClusters(data, colClusters, blockCenters)

      // Update column clusters (fix row clusters)
      colClusters = updateColClusters(data, rowClusters, blockCenters)

      previousObjective = currentObjective
    }

    // Return result even if not converged
    val finalBlockCenters = updateBlockCenters(data, rowClusters, colClusters)
    val finalObjective    = calculateObjective(data, rowClusters, colClusters, finalBlockCenters)
    convergenceHistory += finalObjective

    val rowCenters = computeRowCenters(data, rowClusters)
    val colCenters = computeColCenters(data, colClusters)

    logger.warn(s"Did not converge after ${config.maxIterations} iterations")

    BregmanCoClusteringResult(
      rowClusters = rowClusters,
      colClusters = colClusters,
      rowCenters = rowCenters,
      colCenters = colCenters,
      blockCenters = finalBlockCenters,
      iterations = config.maxIterations,
      objective = finalObjective,
      convergenceHistory = convergenceHistory.toSeq
    )
  }

  /** Update block centers based on current cluster assignments.
    */
  private def updateBlockCenters(
      data: RDD[MatrixEntry],
      rowClusters: Map[Long, Int],
      colClusters: Map[Long, Int]
  ): Array[Array[BregmanCenter]] = {

    // Group data by block assignments
    val blockData = data.groupBy { entry =>
      val rowCluster = rowClusters(entry.rowIndex)
      val colCluster = colClusters(entry.colIndex)
      (rowCluster, colCluster)
    }

    // Compute center for each block
    val blockCenters = Array.ofDim[BregmanCenter](config.numRowClusters, config.numColClusters)

    blockData.collect().foreach { case ((rowCluster, colCluster), entries) =>
      val points = entries.map { entry =>
        BregmanPoint(WeightedVector(Vectors.dense(entry.value), entry.weight), entry.value)
      }

      if (points.nonEmpty) {
        val accumulator = pointOps.make()
        points.foreach(accumulator.add)
        blockCenters(rowCluster)(colCluster) = pointOps.toCenter(accumulator.asImmutable)
      } else {
        // Empty block - use regularized center
        val regularizedPoint = BregmanPoint(
          WeightedVector(Vectors.dense(config.regularization), config.regularization),
          config.regularization
        )
        blockCenters(rowCluster)(colCluster) = pointOps.toCenter(regularizedPoint)
      }
    }

    // Fill in any missing blocks with regularized centers
    for (r <- 0 until config.numRowClusters; c <- 0 until config.numColClusters) {
      if (blockCenters(r)(c) == null) {
        val regularizedPoint = BregmanPoint(
          WeightedVector(Vectors.dense(config.regularization), config.regularization),
          config.regularization
        )
        blockCenters(r)(c) = pointOps.toCenter(regularizedPoint)
      }
    }

    blockCenters
  }

  /** Update row cluster assignments given fixed column clusters and block centers.
    */
  private def updateRowClusters(
      data: RDD[MatrixEntry],
      colClusters: Map[Long, Int],
      blockCenters: Array[Array[BregmanCenter]]
  ): Map[Long, Int] = {

    // For each row, find the best row cluster assignment
    val rowAssignments = data.groupBy(_.rowIndex).map { case (rowIndex, entries) =>
      val entryList = entries.toList

      // Calculate cost for assigning this row to each row cluster
      val costs = (0 until config.numRowClusters).map { rowCluster =>
        entryList.map { entry =>
          val colCluster  = colClusters(entry.colIndex)
          val blockCenter = blockCenters(rowCluster)(colCluster)
          val point       =
            BregmanPoint(WeightedVector(Vectors.dense(entry.value), entry.weight), entry.value)
          pointOps.distance(point, blockCenter) * entry.weight
        }.sum
      }

      // Assign to cluster with minimum cost
      val bestCluster = costs.zipWithIndex.minBy(_._1)._2
      rowIndex -> bestCluster
    }

    rowAssignments.collect().toMap
  }

  /** Update column cluster assignments given fixed row clusters and block centers.
    */
  private def updateColClusters(
      data: RDD[MatrixEntry],
      rowClusters: Map[Long, Int],
      blockCenters: Array[Array[BregmanCenter]]
  ): Map[Long, Int] = {

    // For each column, find the best column cluster assignment
    val colAssignments = data.groupBy(_.colIndex).map { case (colIndex, entries) =>
      val entryList = entries.toList

      // Calculate cost for assigning this column to each column cluster
      val costs = (0 until config.numColClusters).map { colCluster =>
        entryList.map { entry =>
          val rowCluster  = rowClusters(entry.rowIndex)
          val blockCenter = blockCenters(rowCluster)(colCluster)
          val point       =
            BregmanPoint(WeightedVector(Vectors.dense(entry.value), entry.weight), entry.value)
          pointOps.distance(point, blockCenter) * entry.weight
        }.sum
      }

      // Assign to cluster with minimum cost
      val bestCluster = costs.zipWithIndex.minBy(_._1)._2
      colIndex -> bestCluster
    }

    colAssignments.collect().toMap
  }

  /** Calculate the total objective function value.
    */
  private def calculateObjective(
      data: RDD[MatrixEntry],
      rowClusters: Map[Long, Int],
      colClusters: Map[Long, Int],
      blockCenters: Array[Array[BregmanCenter]]
  ): Double = {

    val totalDistance = data.map { entry =>
      val rowCluster  = rowClusters(entry.rowIndex)
      val colCluster  = colClusters(entry.colIndex)
      val blockCenter = blockCenters(rowCluster)(colCluster)
      val point       =
        BregmanPoint(WeightedVector(Vectors.dense(entry.value), entry.weight), entry.value)
      pointOps.distance(point, blockCenter) * entry.weight
    }.sum()

    totalDistance
  }

  /** Compute row cluster centers.
    */
  private def computeRowCenters(
      data: RDD[MatrixEntry],
      rowClusters: Map[Long, Int]
  ): IndexedSeq[BregmanCenter] = {

    val rowData = data.groupBy(entry => rowClusters(entry.rowIndex))

    val centers = (0 until config.numRowClusters).map { cluster =>
      rowData.filter(_._1 == cluster).values.collect().flatten.toList match {
        case Nil     =>
          // Empty cluster
          val regularizedPoint = BregmanPoint(
            WeightedVector(Vectors.dense(config.regularization), config.regularization),
            config.regularization
          )
          pointOps.toCenter(regularizedPoint)
        case entries =>
          val accumulator = pointOps.make()
          entries.foreach { entry =>
            val point =
              BregmanPoint(WeightedVector(Vectors.dense(entry.value), entry.weight), entry.value)
            accumulator.add(point)
          }
          pointOps.toCenter(accumulator.asImmutable)
      }
    }

    centers.toIndexedSeq
  }

  /** Compute column cluster centers.
    */
  private def computeColCenters(
      data: RDD[MatrixEntry],
      colClusters: Map[Long, Int]
  ): IndexedSeq[BregmanCenter] = {

    val colData = data.groupBy(entry => colClusters(entry.colIndex))

    val centers = (0 until config.numColClusters).map { cluster =>
      colData.filter(_._1 == cluster).values.collect().flatten.toList match {
        case Nil     =>
          // Empty cluster
          val regularizedPoint = BregmanPoint(
            WeightedVector(Vectors.dense(config.regularization), config.regularization),
            config.regularization
          )
          pointOps.toCenter(regularizedPoint)
        case entries =>
          val accumulator = pointOps.make()
          entries.foreach { entry =>
            val point =
              BregmanPoint(WeightedVector(Vectors.dense(entry.value), entry.weight), entry.value)
            accumulator.add(point)
          }
          pointOps.toCenter(accumulator.asImmutable)
      }
    }

    centers.toIndexedSeq
  }
}

object BregmanCoClustering {

  /** Default configuration for Bregman co-clustering.
    */
  def defaultConfig(numRowClusters: Int, numColClusters: Int): BregmanCoClusteringConfig = {
    BregmanCoClusteringConfig(
      numRowClusters = numRowClusters,
      numColClusters = numColClusters,
      maxIterations = 100,
      tolerance = 1e-6,
      initStrategy = "random",
      regularization = 0.01
    )
  }

  /** Create co-clustering with specified parameters.
    */
  def apply(
      numRowClusters: Int,
      numColClusters: Int,
      pointOps: BregmanPointOps,
      config: BregmanCoClusteringConfig = null
  ): BregmanCoClustering = {

    val finalConfig = if (config != null) config else defaultConfig(numRowClusters, numColClusters)
    new BregmanCoClustering(finalConfig, pointOps)
  }

  /** Create co-clustering optimized for sparse data.
    */
  def forSparseData(
      numRowClusters: Int,
      numColClusters: Int,
      pointOps: BregmanPointOps
  ): BregmanCoClustering = {

    val config = BregmanCoClusteringConfig(
      numRowClusters = numRowClusters,
      numColClusters = numColClusters,
      maxIterations = 200,
      tolerance = 1e-8,
      initStrategy = "balanced",
      regularization = 0.001
    )

    new BregmanCoClustering(config, pointOps)
  }

  /** Create co-clustering optimized for dense data.
    */
  def forDenseData(
      numRowClusters: Int,
      numColClusters: Int,
      pointOps: BregmanPointOps
  ): BregmanCoClustering = {

    val config = BregmanCoClusteringConfig(
      numRowClusters = numRowClusters,
      numColClusters = numColClusters,
      maxIterations = 50,
      tolerance = 1e-4,
      initStrategy = "random",
      regularization = 0.05
    )

    new BregmanCoClustering(config, pointOps)
  }

  /** Convert a dense matrix to MatrixEntry format.
    */
  def matrixToEntries(matrix: Array[Array[Double]]): Seq[MatrixEntry] = {
    val entries = mutable.ListBuffer[MatrixEntry]()

    for (i <- matrix.indices; j <- matrix(i).indices) {
      entries += MatrixEntry(i.toLong, j.toLong, matrix(i)(j))
    }

    entries.toSeq
  }

  /** Convert sparse matrix entries back to dense matrix format.
    */
  def entriesToMatrix(
      entries: Seq[MatrixEntry],
      numRows: Int,
      numCols: Int,
      defaultValue: Double = 0.0
  ): Array[Array[Double]] = {

    val matrix = Array.ofDim[Double](numRows, numCols)

    // Initialize with default values
    for (i <- 0 until numRows; j <- 0 until numCols) {
      matrix(i)(j) = defaultValue
    }

    // Fill in the entries
    entries.foreach { entry =>
      if (entry.rowIndex < numRows && entry.colIndex < numCols) {
        matrix(entry.rowIndex.toInt)(entry.colIndex.toInt) = entry.value
      }
    }

    matrix
  }
}
