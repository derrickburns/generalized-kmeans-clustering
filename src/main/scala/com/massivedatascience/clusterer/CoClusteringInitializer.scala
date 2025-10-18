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

/** Advanced initialization strategies for Bregman co-clustering.
  *
  * This class provides various initialization methods that can significantly impact the quality and
  * convergence speed of co-clustering algorithms.
  */
class CoClusteringInitializer(
    numRowClusters: Int,
    numColClusters: Int,
    pointOps: BregmanPointOps,
    seed: Long = System.currentTimeMillis()
) extends Serializable {

  @transient private lazy val logger = LoggerFactory.getLogger(getClass.getName)
  private val random                 = new Random(seed)

  /** Initialize using random assignment.
    */
  def randomInit(
      rowIndices: Array[Long],
      colIndices: Array[Long]
  ): (Map[Long, Int], Map[Long, Int]) = {

    logger.debug("Initializing with random assignment")

    val rowClusters = rowIndices.map(idx => idx -> random.nextInt(numRowClusters)).toMap
    val colClusters = colIndices.map(idx => idx -> random.nextInt(numColClusters)).toMap

    (rowClusters, colClusters)
  }

  /** Initialize with balanced cluster sizes.
    */
  def balancedInit(
      rowIndices: Array[Long],
      colIndices: Array[Long]
  ): (Map[Long, Int], Map[Long, Int]) = {

    logger.debug("Initializing with balanced assignment")

    val shuffledRows = random.shuffle(rowIndices.toList).toArray
    val shuffledCols = random.shuffle(colIndices.toList).toArray

    val rowClusters = shuffledRows.zipWithIndex.map { case (idx, i) =>
      idx -> (i % numRowClusters)
    }.toMap

    val colClusters = shuffledCols.zipWithIndex.map { case (idx, i) =>
      idx -> (i % numColClusters)
    }.toMap

    (rowClusters, colClusters)
  }

  /** Initialize using K-means++ style selection on marginal distributions.
    */
  def kmeansPlusPlusInit(
      data: RDD[MatrixEntry],
      rowIndices: Array[Long],
      colIndices: Array[Long]
  ): (Map[Long, Int], Map[Long, Int]) = {

    logger.debug("Initializing with K-means++ style selection")

    // Compute row and column marginals
    val rowMarginals = computeRowMarginals(data, rowIndices)
    val colMarginals = computeColMarginals(data, colIndices)

    // Use K-means++ to select row cluster centers
    val rowClusterCenters = kmeansPlusPlusSelection(rowMarginals, numRowClusters)
    val rowClusters       = assignToNearestCenters(rowMarginals, rowClusterCenters)

    // Use K-means++ to select column cluster centers
    val colClusterCenters = kmeansPlusPlusSelection(colMarginals, numColClusters)
    val colClusters       = assignToNearestCenters(colMarginals, colClusterCenters)

    (rowClusters, colClusters)
  }

  /** Initialize using spectral co-clustering approach.
    */
  def spectralInit(
      data: RDD[MatrixEntry],
      rowIndices: Array[Long],
      colIndices: Array[Long]
  ): (Map[Long, Int], Map[Long, Int]) = {

    logger.debug("Initializing with spectral approach")

    // For simplicity, we'll use a heuristic based on data distribution
    // In a full implementation, this would involve SVD/eigendecomposition

    val rowSums = data
      .groupBy(_.rowIndex)
      .map { case (rowIdx, entries) =>
        rowIdx -> entries.map(_.value).sum
      }
      .collect()
      .toMap

    val colSums = data
      .groupBy(_.colIndex)
      .map { case (colIdx, entries) =>
        colIdx -> entries.map(_.value).sum
      }
      .collect()
      .toMap

    // Sort rows and columns by their sums and assign clusters cyclically
    val sortedRows = rowIndices.sortBy(idx => rowSums.getOrElse(idx, 0.0))
    val sortedCols = colIndices.sortBy(idx => colSums.getOrElse(idx, 0.0))

    val rowClusters = sortedRows.zipWithIndex.map { case (idx, i) =>
      idx -> (i % numRowClusters)
    }.toMap

    val colClusters = sortedCols.zipWithIndex.map { case (idx, i) =>
      idx -> (i % numColClusters)
    }.toMap

    (rowClusters, colClusters)
  }

  /** Initialize using density-based clustering on marginals.
    */
  def densityBasedInit(
      data: RDD[MatrixEntry],
      rowIndices: Array[Long],
      colIndices: Array[Long]
  ): (Map[Long, Int], Map[Long, Int]) = {

    logger.debug("Initializing with density-based approach")

    // Compute density scores for rows and columns
    val rowDensities = computeRowDensities(data, rowIndices)
    val colDensities = computeColDensities(data, colIndices)

    // Cluster based on density patterns
    val rowClusters = clusterByDensity(rowDensities, numRowClusters)
    val colClusters = clusterByDensity(colDensities, numColClusters)

    (rowClusters, colClusters)
  }

  /** Initialize using multiple strategies and select the best one.
    */
  def multiStrategyInit(
      data: RDD[MatrixEntry],
      rowIndices: Array[Long],
      colIndices: Array[Long],
      strategies: Seq[String] = Seq("random", "balanced", "kmeans++")
  ): (Map[Long, Int], Map[Long, Int]) = {

    logger.debug(s"Initializing with multiple strategies: ${strategies.mkString(", ")}")

    val candidates = strategies.map { strategy =>
      val (rowClusters, colClusters) = strategy match {
        case "random"   => randomInit(rowIndices, colIndices)
        case "balanced" => balancedInit(rowIndices, colIndices)
        case "kmeans++" => kmeansPlusPlusInit(data, rowIndices, colIndices)
        case "spectral" => spectralInit(data, rowIndices, colIndices)
        case "density"  => densityBasedInit(data, rowIndices, colIndices)
        case _          => randomInit(rowIndices, colIndices)
      }

      // Evaluate quality of this initialization
      val quality = evaluateInitialization(data, rowClusters, colClusters)
      (strategy, rowClusters, colClusters, quality)
    }

    // Select the best initialization
    val best = candidates.minBy(_._4)
    logger.debug(s"Selected ${best._1} initialization with quality ${best._4}")

    (best._2, best._3)
  }

  /** Compute row marginal distributions.
    */
  private def computeRowMarginals(
      data: RDD[MatrixEntry],
      rowIndices: Array[Long]
  ): Map[Long, BregmanPoint] = {

    val rowData = data.groupBy(_.rowIndex)

    rowIndices.map { rowIdx =>
      rowData.filter(_._1 == rowIdx).values.collect().flatten.toList match {
        case Nil     =>
          // Empty row
          val emptyPoint = BregmanPoint(WeightedVector(Vectors.dense(0.0), 0.0), 0.0)
          rowIdx -> emptyPoint
        case entries =>
          // Compute marginal statistics
          val values      = entries.map(_.value)
          val weights     = entries.map(_.weight)
          val mean        = values.zip(weights).map { case (v, w) => v * w }.sum / weights.sum
          val totalWeight = weights.sum

          val marginalPoint = BregmanPoint(WeightedVector(Vectors.dense(mean), totalWeight), mean)
          rowIdx -> marginalPoint
      }
    }.toMap
  }

  /** Compute column marginal distributions.
    */
  private def computeColMarginals(
      data: RDD[MatrixEntry],
      colIndices: Array[Long]
  ): Map[Long, BregmanPoint] = {

    val colData = data.groupBy(_.colIndex)

    colIndices.map { colIdx =>
      colData.filter(_._1 == colIdx).values.collect().flatten.toList match {
        case Nil     =>
          // Empty column
          val emptyPoint = BregmanPoint(WeightedVector(Vectors.dense(0.0), 0.0), 0.0)
          colIdx -> emptyPoint
        case entries =>
          // Compute marginal statistics
          val values      = entries.map(_.value)
          val weights     = entries.map(_.weight)
          val mean        = values.zip(weights).map { case (v, w) => v * w }.sum / weights.sum
          val totalWeight = weights.sum

          val marginalPoint = BregmanPoint(WeightedVector(Vectors.dense(mean), totalWeight), mean)
          colIdx -> marginalPoint
      }
    }.toMap
  }

  /** K-means++ style center selection.
    */
  private def kmeansPlusPlusSelection(
      marginals: Map[Long, BregmanPoint],
      numClusters: Int
  ): IndexedSeq[BregmanPoint] = {

    val points = marginals.values.toSeq
    if (points.isEmpty) return IndexedSeq.empty

    val centers = mutable.ListBuffer[BregmanPoint]()

    // Select first center randomly
    centers += points(random.nextInt(points.length))

    // Select remaining centers using D^2 weighting
    for (_ <- 1 until numClusters) {
      val centerSeq = centers.toIndexedSeq.map(c => pointOps.toCenter(c))
      val weights   = points.map { point =>
        val minDistance = centerSeq.map(pointOps.distance(point, _)).min
        minDistance * minDistance
      }

      val totalWeight = weights.sum
      if (totalWeight > 0) {
        val threshold        = random.nextDouble() * totalWeight
        var cumulativeWeight = 0.0
        var selectedIndex    = 0

        for (i <- weights.indices) {
          cumulativeWeight += weights(i)
          if (cumulativeWeight >= threshold) {
            selectedIndex = i
          }
        }

        centers += points(selectedIndex)
      } else {
        // Fallback to random selection
        centers += points(random.nextInt(points.length))
      }
    }

    centers.toIndexedSeq
  }

  /** Assign marginals to nearest centers.
    */
  private def assignToNearestCenters(
      marginals: Map[Long, BregmanPoint],
      centers: IndexedSeq[BregmanPoint]
  ): Map[Long, Int] = {

    val centerSeq = centers.map(c => pointOps.toCenter(c))
    marginals.map { case (idx, point) =>
      val (closestIndex, _) = pointOps.findClosest(centerSeq, point)
      idx -> closestIndex
    }
  }

  /** Compute density scores for rows.
    */
  private def computeRowDensities(
      data: RDD[MatrixEntry],
      rowIndices: Array[Long]
  ): Map[Long, Double] = {

    val rowCounts = data
      .groupBy(_.rowIndex)
      .map { case (rowIdx, entries) =>
        rowIdx -> entries.size
      }
      .collect()
      .toMap

    rowIndices.map { idx =>
      idx -> rowCounts.getOrElse(idx, 0).toDouble
    }.toMap
  }

  /** Compute density scores for columns.
    */
  private def computeColDensities(
      data: RDD[MatrixEntry],
      colIndices: Array[Long]
  ): Map[Long, Double] = {

    val colCounts = data
      .groupBy(_.colIndex)
      .map { case (colIdx, entries) =>
        colIdx -> entries.size
      }
      .collect()
      .toMap

    colIndices.map { idx =>
      idx -> colCounts.getOrElse(idx, 0).toDouble
    }.toMap
  }

  /** Cluster indices based on density scores.
    */
  private def clusterByDensity(densities: Map[Long, Double], numClusters: Int): Map[Long, Int] = {

    val sortedByDensity = densities.toSeq.sortBy(_._2)
    val clusterSize     = math.ceil(sortedByDensity.length.toDouble / numClusters).toInt

    sortedByDensity.zipWithIndex.map { case ((idx, _), i) =>
      val cluster = math.min(i / clusterSize, numClusters - 1)
      idx -> cluster
    }.toMap
  }

  /** Evaluate the quality of an initialization.
    */
  private def evaluateInitialization(
      data: RDD[MatrixEntry],
      rowClusters: Map[Long, Int],
      colClusters: Map[Long, Int]
  ): Double = {

    // Compute initial block dispersion as quality metric
    val blockData = data.groupBy { entry =>
      val rowCluster = rowClusters(entry.rowIndex)
      val colCluster = colClusters(entry.colIndex)
      (rowCluster, colCluster)
    }

    val totalDispersion = blockData.map { case (_, entries) =>
      val values = entries.map(_.value).toSeq
      if (values.length > 1) {
        val mean = values.sum / values.length
        values.map(v => math.pow(v - mean, 2)).sum
      } else 0.0
    }.sum()

    totalDispersion
  }
}

object CoClusteringInitializer {

  /** Create initializer with default settings.
    */
  def apply(
      numRowClusters: Int,
      numColClusters: Int,
      pointOps: BregmanPointOps
  ): CoClusteringInitializer = {

    new CoClusteringInitializer(numRowClusters, numColClusters, pointOps)
  }

  /** Create initializer with specific seed.
    */
  def withSeed(
      numRowClusters: Int,
      numColClusters: Int,
      pointOps: BregmanPointOps,
      seed: Long
  ): CoClusteringInitializer = {

    new CoClusteringInitializer(numRowClusters, numColClusters, pointOps, seed)
  }

  /** Quick initialization with specified strategy.
    */
  def initialize(
      data: RDD[MatrixEntry],
      rowIndices: Array[Long],
      colIndices: Array[Long],
      numRowClusters: Int,
      numColClusters: Int,
      pointOps: BregmanPointOps,
      strategy: String = "random"
  ): (Map[Long, Int], Map[Long, Int]) = {

    val initializer = new CoClusteringInitializer(numRowClusters, numColClusters, pointOps)

    strategy match {
      case "random"   => initializer.randomInit(rowIndices, colIndices)
      case "balanced" => initializer.balancedInit(rowIndices, colIndices)
      case "kmeans++" => initializer.kmeansPlusPlusInit(data, rowIndices, colIndices)
      case "spectral" => initializer.spectralInit(data, rowIndices, colIndices)
      case "density"  => initializer.densityBasedInit(data, rowIndices, colIndices)
      case "multi"    => initializer.multiStrategyInit(data, rowIndices, colIndices)
      case _          => initializer.randomInit(rowIndices, colIndices)
    }
  }
}
