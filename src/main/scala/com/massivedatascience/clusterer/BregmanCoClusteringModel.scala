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

/**
 * Trained Bregman co-clustering model with prediction and analysis capabilities.
 * 
 * This class encapsulates the result of co-clustering training and provides
 * methods for making predictions, reconstructing the matrix, and analyzing
 * the clustering structure.
 * 
 * @param result The co-clustering result from training
 * @param pointOps Bregman point operations used for distance calculations
 */
class BregmanCoClusteringModel(
    val result: BregmanCoClusteringResult,
    pointOps: BregmanPointOps) extends Serializable {
  
  
  /**
   * Predict cluster assignments for new matrix entries.
   * 
   * @param entries RDD of new matrix entries
   * @return RDD of (entry, (rowCluster, colCluster)) pairs
   */
  def predict(entries: RDD[MatrixEntry]): RDD[(MatrixEntry, (Int, Int))] = {
    entries.map { entry =>
      val prediction = result.predict(entry, pointOps)
      (entry, prediction)
    }
  }
  
  /**
   * Predict cluster assignment for a single matrix entry.
   */
  def predict(entry: MatrixEntry): (Int, Int) = {
    result.predict(entry, pointOps)
  }
  
  /**
   * Predict cluster assignments for new rows given their column patterns.
   * 
   * @param rowEntries RDD of (rowIndex, Seq[MatrixEntry]) representing new rows
   * @return RDD of (rowIndex, rowCluster) pairs
   */
  def predictRows(rowEntries: RDD[(Long, Seq[MatrixEntry])]): RDD[(Long, Int)] = {
    rowEntries.map { case (rowIndex, entries) =>
      val rowCluster = findBestRowCluster(entries)
      (rowIndex, rowCluster)
    }
  }
  
  /**
   * Predict cluster assignments for new columns given their row patterns.
   * 
   * @param colEntries RDD of (colIndex, Seq[MatrixEntry]) representing new columns
   * @return RDD of (colIndex, colCluster) pairs
   */
  def predictColumns(colEntries: RDD[(Long, Seq[MatrixEntry])]): RDD[(Long, Int)] = {
    colEntries.map { case (colIndex, entries) =>
      val colCluster = findBestColCluster(entries)
      (colIndex, colCluster)
    }
  }
  
  /**
   * Reconstruct the original matrix using the block structure.
   * 
   * @param entries Original matrix entries
   * @return RDD of reconstructed matrix entries
   */
  def reconstruct(entries: RDD[MatrixEntry]): RDD[MatrixEntry] = {
    entries.map { entry =>
      val (rowCluster, colCluster) = result.predict(entry, pointOps)
      val blockCenter = result.blockCenters(rowCluster)(colCluster)
      
      // Extract the reconstructed value from the block center
      val reconstructedValue = extractValue(blockCenter)
      
      entry.copy(value = reconstructedValue)
    }
  }
  
  /**
   * Compute reconstruction error for the co-clustering.
   * 
   * @param entries Original matrix entries
   * @return Total reconstruction error
   */
  def computeReconstructionError(entries: RDD[MatrixEntry]): Double = {
    val totalError = entries.map { entry =>
      val (rowCluster, colCluster) = result.predict(entry, pointOps)
      val blockCenter = result.blockCenters(rowCluster)(colCluster)
      val point = BregmanPoint(WeightedVector(Vectors.dense(entry.value), entry.weight), entry.value)
      pointOps.distance(point, blockCenter) * entry.weight
    }.sum()
    
    totalError
  }
  
  /**
   * Get block statistics for analysis.
   * 
   * @param entries Original matrix entries
   * @return Statistics for each block
   */
  def getBlockStats(entries: RDD[MatrixEntry]): Array[Array[BlockStats]] = {
    val blockData = entries.groupBy { entry =>
      val (rowCluster, colCluster) = result.predict(entry, pointOps)
      (rowCluster, colCluster)
    }
    
    val stats = Array.ofDim[BlockStats](result.numRowClusters, result.numColClusters)
    
    // Initialize with empty stats
    for (r <- 0 until result.numRowClusters; c <- 0 until result.numColClusters) {
      stats(r)(c) = BlockStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
    
    // Compute stats for each block
    blockData.collect().foreach { case ((rowCluster, colCluster), blockEntries) =>
      val values = blockEntries.map(_.value).toSeq
      val weights = blockEntries.map(_.weight).toSeq
      
      if (values.nonEmpty) {
        val count = values.length
        val weightedSum = values.zip(weights).map { case (v, w) => v * w }.sum
        val totalWeight = weights.sum
        val mean = if (totalWeight > 0) weightedSum / totalWeight else 0.0
        val variance = if (totalWeight > 0) {
          values.zip(weights).map { case (v, w) => w * math.pow(v - mean, 2) }.sum / totalWeight
        } else 0.0
        val min = values.min
        val max = values.max
        
        stats(rowCluster)(colCluster) = BlockStats(count, mean, variance, min, max, totalWeight)
      }
    }
    
    stats
  }
  
  /**
   * Get row cluster characteristics.
   * 
   * @param entries Original matrix entries
   * @return Statistics for each row cluster
   */
  def getRowClusterStats(entries: RDD[MatrixEntry]): IndexedSeq[ClusterStats] = {
    val rowData = entries.groupBy { entry =>
      val (rowCluster, _) = result.predict(entry, pointOps)
      rowCluster
    }
    
    (0 until result.numRowClusters).map { cluster =>
      rowData.filter(_._1 == cluster).values.collect().flatten.toList match {
        case Nil =>
          ClusterStats(0, 0, 0.0, 0.0, 0.0, 0.0)
        case clusterEntries =>
          val uniqueRows = clusterEntries.map(_.rowIndex).distinct.length
          val totalEntries = clusterEntries.length
          val values = clusterEntries.map(_.value)
          val weights = clusterEntries.map(_.weight)
          
          val weightedSum = values.zip(weights).map { case (v, w) => v * w }.sum
          val totalWeight = weights.sum
          val mean = if (totalWeight > 0) weightedSum / totalWeight else 0.0
          val variance = if (totalWeight > 0) {
            values.zip(weights).map { case (v, w) => w * math.pow(v - mean, 2) }.sum / totalWeight
          } else 0.0
          
          ClusterStats(uniqueRows, totalEntries, mean, variance, values.min, values.max)
      }
    }
  }
  
  /**
   * Get column cluster characteristics.
   * 
   * @param entries Original matrix entries
   * @return Statistics for each column cluster
   */
  def getColClusterStats(entries: RDD[MatrixEntry]): IndexedSeq[ClusterStats] = {
    val colData = entries.groupBy { entry =>
      val (_, colCluster) = result.predict(entry, pointOps)
      colCluster
    }
    
    (0 until result.numColClusters).map { cluster =>
      colData.filter(_._1 == cluster).values.collect().flatten.toList match {
        case Nil =>
          ClusterStats(0, 0, 0.0, 0.0, 0.0, 0.0)
        case clusterEntries =>
          val uniqueCols = clusterEntries.map(_.colIndex).distinct.length
          val totalEntries = clusterEntries.length
          val values = clusterEntries.map(_.value)
          val weights = clusterEntries.map(_.weight)
          
          val weightedSum = values.zip(weights).map { case (v, w) => v * w }.sum
          val totalWeight = weights.sum
          val mean = if (totalWeight > 0) weightedSum / totalWeight else 0.0
          val variance = if (totalWeight > 0) {
            values.zip(weights).map { case (v, w) => w * math.pow(v - mean, 2) }.sum / totalWeight
          } else 0.0
          
          ClusterStats(uniqueCols, totalEntries, mean, variance, values.min, values.max)
      }
    }
  }
  
  /**
   * Generate summary report of the co-clustering model.
   * 
   * @param entries Original matrix entries
   * @return Comprehensive model summary
   */
  def summary(entries: RDD[MatrixEntry]): CoClusteringModelSummary = {
    val reconstructionError = computeReconstructionError(entries)
    val blockStats = getBlockStats(entries)
    val rowStats = getRowClusterStats(entries)
    val colStats = getColClusterStats(entries)
    
    // Compute additional metrics
    val totalEntries = entries.count()
    val nonEmptyBlocks = blockStats.flatten.count(_.count > 0)
    val sparsity = nonEmptyBlocks.toDouble / (result.numRowClusters * result.numColClusters)
    
    CoClusteringModelSummary(
      numRowClusters = result.numRowClusters,
      numColClusters = result.numColClusters,
      totalEntries = totalEntries,
      reconstructionError = reconstructionError,
      sparsity = sparsity,
      iterations = result.iterations,
      finalObjective = result.objective,
      convergenceHistory = result.convergenceHistory,
      blockStats = blockStats,
      rowStats = rowStats,
      colStats = colStats
    )
  }
  
  /**
   * Find the best row cluster for a set of entries.
   */
  private def findBestRowCluster(entries: Seq[MatrixEntry]): Int = {
    if (entries.isEmpty) return 0
    
    val costs = (0 until result.numRowClusters).map { rowCluster =>
      entries.map { entry =>
        // Predict column cluster for this entry
        val (_, colCluster) = result.predict(entry, pointOps)
        val blockCenter = result.blockCenters(rowCluster)(colCluster)
        val point = BregmanPoint(WeightedVector(Vectors.dense(entry.value), entry.weight), entry.value)
        pointOps.distance(point, blockCenter) * entry.weight
      }.sum
    }
    
    costs.zipWithIndex.minBy(_._1)._2
  }
  
  /**
   * Find the best column cluster for a set of entries.
   */
  private def findBestColCluster(entries: Seq[MatrixEntry]): Int = {
    if (entries.isEmpty) return 0
    
    val costs = (0 until result.numColClusters).map { colCluster =>
      entries.map { entry =>
        // Predict row cluster for this entry
        val (rowCluster, _) = result.predict(entry, pointOps)
        val blockCenter = result.blockCenters(rowCluster)(colCluster)
        val point = BregmanPoint(WeightedVector(Vectors.dense(entry.value), entry.weight), entry.value)
        pointOps.distance(point, blockCenter) * entry.weight
      }.sum
    }
    
    costs.zipWithIndex.minBy(_._1)._2
  }
  
  /**
   * Extract a scalar value from a Bregman center (for reconstruction).
   */
  private def extractValue(center: BregmanCenter): Double = {
    // This is a simplified extraction - in practice, you might want to
    // extract the most representative value from the center
    center.homogeneous.apply(0)
  }
}

/**
 * Statistics for a single block in the co-clustering.
 */
case class BlockStats(
    count: Int,
    mean: Double,
    variance: Double,
    min: Double,
    max: Double,
    totalWeight: Double)

/**
 * Statistics for a cluster (row or column).
 */
case class ClusterStats(
    uniqueIndices: Int,
    totalEntries: Int,
    mean: Double,
    variance: Double,
    min: Double,
    max: Double)

/**
 * Comprehensive summary of a co-clustering model.
 */
case class CoClusteringModelSummary(
    numRowClusters: Int,
    numColClusters: Int,
    totalEntries: Long,
    reconstructionError: Double,
    sparsity: Double,
    iterations: Int,
    finalObjective: Double,
    convergenceHistory: Seq[Double],
    blockStats: Array[Array[BlockStats]],
    rowStats: IndexedSeq[ClusterStats],
    colStats: IndexedSeq[ClusterStats]) {
  
  /**
   * Get a compact text summary.
   */
  def toText: String = {
    val sb = new mutable.StringBuilder()
    
    sb.append(s"Bregman Co-clustering Model Summary\n")
    sb.append(s"===================================\n")
    sb.append(s"Row clusters: $numRowClusters\n")
    sb.append(s"Column clusters: $numColClusters\n")
    sb.append(s"Total blocks: ${numRowClusters * numColClusters}\n")
    sb.append(s"Total entries: $totalEntries\n")
    sb.append(s"Block sparsity: ${(sparsity * 100).formatted("%.2f")}%\n")
    sb.append(s"Reconstruction error: ${reconstructionError.formatted("%.6f")}\n")
    sb.append(s"Final objective: ${finalObjective.formatted("%.6f")}\n")
    sb.append(s"Iterations: $iterations\n")
    sb.append(s"Convergence rate: ${getConvergenceRate.formatted("%.6f")}\n")
    
    sb.append(s"\nRow Cluster Statistics:\n")
    rowStats.zipWithIndex.foreach { case (stats, idx) =>
      sb.append(s"  Cluster $idx: ${stats.uniqueIndices} rows, ${stats.totalEntries} entries, mean=${stats.mean.formatted("%.3f")}\n")
    }
    
    sb.append(s"\nColumn Cluster Statistics:\n")
    colStats.zipWithIndex.foreach { case (stats, idx) =>
      sb.append(s"  Cluster $idx: ${stats.uniqueIndices} columns, ${stats.totalEntries} entries, mean=${stats.mean.formatted("%.3f")}\n")
    }
    
    sb.toString()
  }
  
  /**
   * Get convergence rate from history.
   */
  def getConvergenceRate: Double = {
    if (convergenceHistory.length > 1) {
      val initial = convergenceHistory.head
      val finalValue = convergenceHistory.last
      if (initial != 0.0) (initial - finalValue) / initial else 0.0
    } else 0.0
  }
  
  /**
   * Get average block size.
   */
  def getAverageBlockSize: Double = {
    val nonEmptyBlocks = blockStats.flatten.filter(_.count > 0)
    if (nonEmptyBlocks.nonEmpty) {
      nonEmptyBlocks.map(_.count).sum.toDouble / nonEmptyBlocks.length
    } else 0.0
  }
  
  /**
   * Get block size distribution.
   */
  def getBlockSizeDistribution: Map[String, Int] = {
    val sizes = blockStats.flatten.map(_.count)
    val empty = sizes.count(_ == 0)
    val small = sizes.count(s => s > 0 && s <= 10)
    val medium = sizes.count(s => s > 10 && s <= 100)
    val large = sizes.count(s => s > 100)
    
    Map(
      "empty" -> empty,
      "small (1-10)" -> small,
      "medium (11-100)" -> medium,
      "large (>100)" -> large
    )
  }
}

object BregmanCoClusteringModel {
  
  /**
   * Create a model from training result.
   */
  def apply(result: BregmanCoClusteringResult, pointOps: BregmanPointOps): BregmanCoClusteringModel = {
    new BregmanCoClusteringModel(result, pointOps)
  }
  
  /**
   * Train a model and return the fitted model.
   */
  def fit(
      data: RDD[MatrixEntry],
      config: BregmanCoClusteringConfig,
      pointOps: BregmanPointOps): BregmanCoClusteringModel = {
    
    val coClustering = new BregmanCoClustering(config, pointOps)
    val result = coClustering.fit(data)
    new BregmanCoClusteringModel(result, pointOps)
  }
  
  /**
   * Train a model with default configuration.
   */
  def fit(
      data: RDD[MatrixEntry],
      numRowClusters: Int,
      numColClusters: Int,
      pointOps: BregmanPointOps): BregmanCoClusteringModel = {
    
    val config = BregmanCoClustering.defaultConfig(numRowClusters, numColClusters)
    fit(data, config, pointOps)
  }
}