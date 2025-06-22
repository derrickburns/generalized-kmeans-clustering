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

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.LoggerFactory

import scala.collection.mutable

/**
 * Advanced analysis and visualization utilities for Bregman co-clustering results.
 * 
 * This class provides comprehensive analysis capabilities including:
 * - Block pattern detection and characterization
 * - Cluster quality metrics and validation
 * - Data export for external visualization tools
 * - Statistical significance testing
 * - Comparative analysis between different clustering results
 */
class CoClusteringAnalysis(model: BregmanCoClusteringModel) extends Serializable {
  
  @transient private lazy val logger = LoggerFactory.getLogger(getClass.getName)
  
  /**
   * Analyze block patterns and identify characteristic structures.
   * 
   * @param entries Original matrix entries
   * @return Pattern analysis results
   */
  def analyzeBlockPatterns(entries: RDD[MatrixEntry]): BlockPatternAnalysis = {
    val blockStats = model.getBlockStats(entries)
    val patterns = mutable.Map[String, Int]()
    val dominantBlocks = mutable.ArrayBuffer[(Int, Int, Double)]()
    
    // Analyze block patterns
    for (r <- blockStats.indices; c <- blockStats(r).indices) {
      val stats = blockStats(r)(c)
      
      if (stats.count > 0) {
        // Classify block patterns
        val pattern = classifyBlockPattern(stats)
        patterns(pattern) = patterns.getOrElse(pattern, 0) + 1
        
        // Identify dominant blocks (high weight, low variance)
        if (stats.totalWeight > 0 && stats.variance > 0) {
          val significance = stats.totalWeight / stats.variance
          dominantBlocks += ((r, c, significance))
        }
      }
    }
    
    val topDominantBlocks = dominantBlocks.sortBy(-_._3).take(10).toSeq
    val blockDensity = computeBlockDensity(blockStats)
    val connectivity = computeBlockConnectivity(blockStats)
    
    BlockPatternAnalysis(
      patterns = patterns.toMap,
      dominantBlocks = topDominantBlocks,
      blockDensity = blockDensity,
      connectivity = connectivity,
      totalNonEmptyBlocks = blockStats.flatten.count(_.count > 0)
    )
  }
  
  /**
   * Compute cluster quality metrics including silhouette scores and cohesion.
   * 
   * @param entries Original matrix entries
   * @return Quality metrics for the clustering
   */
  def computeQualityMetrics(entries: RDD[MatrixEntry]): ClusterQualityMetrics = {
    logger.info("Computing cluster quality metrics...")
    
    // Sample entries for efficiency with large datasets
    val sampleEntries = sampleForQualityAnalysis(entries, 10000)
    
    val rowSilhouette = computeRowSilhouetteScore(sampleEntries)
    val colSilhouette = computeColSilhouetteScore(sampleEntries)
    val blockCohesion = computeBlockCohesion(sampleEntries)
    val separationIndex = computeBlockSeparation(sampleEntries)
    val modularity = computeModularity(sampleEntries)
    
    ClusterQualityMetrics(
      rowSilhouetteScore = rowSilhouette,
      colSilhouetteScore = colSilhouette,
      blockCohesion = blockCohesion,
      separationIndex = separationIndex,
      modularity = modularity,
      overallQuality = (rowSilhouette + colSilhouette + blockCohesion + separationIndex + modularity) / 5.0
    )
  }
  
  /**
   * Export clustering results in various formats for external visualization.
   * 
   * @param entries Original matrix entries
   * @param spark SparkSession for DataFrame operations
   * @return Export data in multiple formats
   */
  def exportForVisualization(entries: RDD[MatrixEntry], spark: SparkSession): VisualizationExport = {
    import spark.implicits._
    
    logger.info("Preparing data for visualization export...")
    
    // Block structure matrix
    val blockMatrix = createBlockMatrix(entries)
    val blockMatrixDF = blockMatrix.toDF("rowCluster", "colCluster", "meanValue", "entryCount", "weight")
    
    // Cluster assignments
    val assignments = entries.map { entry =>
      val (rowCluster, colCluster) = model.predict(entry)
      (entry.rowIndex, entry.colIndex, entry.value, entry.weight, rowCluster, colCluster)
    }.toDF("rowIndex", "colIndex", "value", "weight", "rowCluster", "colCluster")
    
    // Row cluster profiles
    val rowProfiles = createRowClusterProfiles(entries, spark)
    
    // Column cluster profiles
    val colProfiles = createColClusterProfiles(entries, spark)
    
    // Convergence data
    val convergenceDF = model.result.convergenceHistory.zipWithIndex
      .map { case (objective, iteration) => (iteration, objective) }
      .toDF("iteration", "objective")
    
    VisualizationExport(
      blockMatrix = blockMatrixDF,
      assignments = assignments,
      rowProfiles = rowProfiles,
      colProfiles = colProfiles,
      convergenceHistory = convergenceDF
    )
  }
  
  /**
   * Perform stability analysis by comparing results across multiple runs.
   * 
   * @param entries Original matrix entries
   * @param config Clustering configuration
   * @param pointOps Bregman point operations
   * @param numRuns Number of runs for stability analysis
   * @return Stability analysis results
   */
  def analyzeStability(
      entries: RDD[MatrixEntry],
      config: BregmanCoClusteringConfig,
      pointOps: BregmanPointOps,
      numRuns: Int = 10): StabilityAnalysis = {
    
    logger.info(s"Performing stability analysis with $numRuns runs...")
    
    val results = (1 to numRuns).map { run =>
      logger.info(s"Stability run $run/$numRuns")
      
      // Use different random seeds for each run
      val seedConfig = config.copy(seed = config.seed + run)
      val runModel = BregmanCoClusteringModel.fit(entries, seedConfig, pointOps)
      
      (run, runModel.summary(entries))
    }
    
    val objectives = results.map(_._2.finalObjective)
    val reconstructionErrors = results.map(_._2.reconstructionError)
    val iterations = results.map(_._2.iterations)
    
    val objectiveStability = computeStabilityMeasure(objectives)
    val errorStability = computeStabilityMeasure(reconstructionErrors)
    val iterationStability = computeStabilityMeasure(iterations.map(_.toDouble))
    
    // Compute average metrics
    val avgObjective = objectives.sum / objectives.length
    val avgError = reconstructionErrors.sum / reconstructionErrors.length
    val avgIterations = iterations.sum / iterations.length
    
    StabilityAnalysis(
      numRuns = numRuns,
      objectiveStability = objectiveStability,
      errorStability = errorStability,
      iterationStability = iterationStability,
      averageObjective = avgObjective,
      averageReconstructionError = avgError,
      averageIterations = avgIterations,
      objectiveRange = (objectives.min, objectives.max),
      errorRange = (reconstructionErrors.min, reconstructionErrors.max)
    )
  }
  
  /**
   * Generate a comprehensive HTML report of the analysis.
   * 
   * @param entries Original matrix entries
   * @param spark SparkSession for DataFrame operations
   * @return HTML report as a string
   */
  def generateHTMLReport(entries: RDD[MatrixEntry], spark: SparkSession): String = {
    val summary = model.summary(entries)
    val patterns = analyzeBlockPatterns(entries)
    val quality = computeQualityMetrics(entries)
    
    val html = new mutable.StringBuilder()
    
    html.append("""
      |<!DOCTYPE html>
      |<html>
      |<head>
      |  <title>Co-clustering Analysis Report</title>
      |  <style>
      |    body { font-family: Arial, sans-serif; margin: 40px; }
      |    .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
      |    .section { margin: 20px 0; }
      |    .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e6f3ff; border-radius: 3px; }
      |    .pattern-table { border-collapse: collapse; width: 100%; }
      |    .pattern-table th, .pattern-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
      |    .pattern-table th { background-color: #f2f2f2; }
      |    .quality-good { color: green; font-weight: bold; }
      |    .quality-medium { color: orange; font-weight: bold; }
      |    .quality-poor { color: red; font-weight: bold; }
      |  </style>
      |</head>
      |<body>
      |""".stripMargin)
    
    // Header
    html.append(s"""
      |<div class="header">
      |  <h1>Bregman Co-clustering Analysis Report</h1>
      |  <p>Generated on ${java.time.LocalDateTime.now()}</p>
      |</div>
      |""".stripMargin)
    
    // Model summary
    html.append(s"""
      |<div class="section">
      |  <h2>Model Summary</h2>
      |  <div class="metric">Row Clusters: ${summary.numRowClusters}</div>
      |  <div class="metric">Column Clusters: ${summary.numColClusters}</div>
      |  <div class="metric">Total Entries: ${summary.totalEntries}</div>
      |  <div class="metric">Reconstruction Error: ${summary.reconstructionError.formatted("%.6f")}</div>
      |  <div class="metric">Block Sparsity: ${(summary.sparsity * 100).formatted("%.2f")}%</div>
      |  <div class="metric">Iterations: ${summary.iterations}</div>
      |</div>
      |""".stripMargin)
    
    // Quality metrics
    val qualityClass = if (quality.overallQuality > 0.7) "quality-good" 
                      else if (quality.overallQuality > 0.4) "quality-medium" 
                      else "quality-poor"
    
    html.append(s"""
      |<div class="section">
      |  <h2>Quality Metrics</h2>
      |  <div class="metric">Overall Quality: <span class="$qualityClass">${quality.overallQuality.formatted("%.3f")}</span></div>
      |  <div class="metric">Row Silhouette: ${quality.rowSilhouetteScore.formatted("%.3f")}</div>
      |  <div class="metric">Column Silhouette: ${quality.colSilhouetteScore.formatted("%.3f")}</div>
      |  <div class="metric">Block Cohesion: ${quality.blockCohesion.formatted("%.3f")}</div>
      |  <div class="metric">Separation Index: ${quality.separationIndex.formatted("%.3f")}</div>
      |  <div class="metric">Modularity: ${quality.modularity.formatted("%.3f")}</div>
      |</div>
      |""".stripMargin)
    
    // Block patterns
    html.append(s"""
      |<div class="section">
      |  <h2>Block Pattern Analysis</h2>
      |  <div class="metric">Non-empty Blocks: ${patterns.totalNonEmptyBlocks}</div>
      |  <div class="metric">Block Density: ${patterns.blockDensity.formatted("%.3f")}</div>
      |  <div class="metric">Connectivity: ${patterns.connectivity.formatted("%.3f")}</div>
      |  
      |  <h3>Pattern Distribution</h3>
      |  <table class="pattern-table">
      |    <tr><th>Pattern Type</th><th>Count</th></tr>
      |""".stripMargin)
    
    patterns.patterns.foreach { case (pattern, count) =>
      html.append(s"    <tr><td>$pattern</td><td>$count</td></tr>\n")
    }
    
    html.append("""
      |  </table>
      |  
      |  <h3>Top Dominant Blocks</h3>
      |  <table class="pattern-table">
      |    <tr><th>Row Cluster</th><th>Column Cluster</th><th>Significance Score</th></tr>
      |""".stripMargin)
    
    patterns.dominantBlocks.take(5).foreach { case (row, col, score) =>
      html.append(s"    <tr><td>$row</td><td>$col</td><td>${score.formatted("%.3f")}</td></tr>\n")
    }
    
    html.append("""
      |  </table>
      |</div>
      |</body>
      |</html>
      |""".stripMargin)
    
    html.toString()
  }
  
  // Private helper methods
  
  private def classifyBlockPattern(stats: BlockStats): String = {
    val cv = if (stats.mean != 0) stats.variance / (stats.mean * stats.mean) else Double.MaxValue
    
    if (stats.count == 0) "empty"
    else if (cv < 0.1) "uniform"
    else if (cv < 0.5) "low-variance"
    else if (cv < 1.0) "medium-variance"
    else "high-variance"
  }
  
  private def computeBlockDensity(blockStats: Array[Array[BlockStats]]): Double = {
    val totalBlocks = blockStats.length * blockStats.head.length
    val nonEmptyBlocks = blockStats.flatten.count(_.count > 0)
    nonEmptyBlocks.toDouble / totalBlocks
  }
  
  private def computeBlockConnectivity(blockStats: Array[Array[BlockStats]]): Double = {
    val nonEmptyBlocks = blockStats.flatten.filter(_.count > 0)
    if (nonEmptyBlocks.isEmpty) 0.0
    else {
      val totalWeight = nonEmptyBlocks.map(_.totalWeight).sum
      val weightedVariance = nonEmptyBlocks.map(b => b.totalWeight * b.variance).sum
      if (totalWeight > 0) 1.0 - (weightedVariance / (totalWeight * totalWeight)) else 0.0
    }
  }
  
  private def sampleForQualityAnalysis(entries: RDD[MatrixEntry], maxSamples: Int): Seq[MatrixEntry] = {
    val totalCount = entries.count()
    if (totalCount <= maxSamples) {
      entries.collect().toSeq
    } else {
      val fraction = maxSamples.toDouble / totalCount
      entries.sample(withReplacement = false, fraction, seed = 42L).collect().toSeq
    }
  }
  
  private def computeRowSilhouetteScore(entries: Seq[MatrixEntry]): Double = {
    // Simplified silhouette computation for co-clustering
    val groupedByRow = entries.groupBy(_.rowIndex)
    var totalScore = 0.0
    var count = 0
    
    groupedByRow.foreach { case (rowIndex, rowEntries) =>
      if (rowEntries.size > 1) {
        val (rowCluster, _) = model.predict(rowEntries.head)
        val sameClusterEntries = rowEntries.filter(e => model.predict(e)._1 == rowCluster)
        val diffClusterEntries = rowEntries.filter(e => model.predict(e)._1 != rowCluster)
        
        if (sameClusterEntries.nonEmpty && diffClusterEntries.nonEmpty) {
          val a = sameClusterEntries.map(_.value).sum / sameClusterEntries.length
          val b = diffClusterEntries.map(_.value).sum / diffClusterEntries.length
          val silhouette = (b - a) / math.max(a, b)
          totalScore += silhouette
          count += 1
        }
      }
    }
    
    if (count > 0) totalScore / count else 0.0
  }
  
  private def computeColSilhouetteScore(entries: Seq[MatrixEntry]): Double = {
    // Similar to row silhouette but for columns
    val groupedByCol = entries.groupBy(_.colIndex)
    var totalScore = 0.0
    var count = 0
    
    groupedByCol.foreach { case (colIndex, colEntries) =>
      if (colEntries.size > 1) {
        val (_, colCluster) = model.predict(colEntries.head)
        val sameClusterEntries = colEntries.filter(e => model.predict(e)._2 == colCluster)
        val diffClusterEntries = colEntries.filter(e => model.predict(e)._2 != colCluster)
        
        if (sameClusterEntries.nonEmpty && diffClusterEntries.nonEmpty) {
          val a = sameClusterEntries.map(_.value).sum / sameClusterEntries.length
          val b = diffClusterEntries.map(_.value).sum / diffClusterEntries.length
          val silhouette = (b - a) / math.max(a, b)
          totalScore += silhouette
          count += 1
        }
      }
    }
    
    if (count > 0) totalScore / count else 0.0
  }
  
  private def computeBlockCohesion(entries: Seq[MatrixEntry]): Double = {
    val blockEntries = entries.groupBy(e => model.predict(e))
    val cohesionScores = blockEntries.map { case (_, blockData) =>
      if (blockData.size > 1) {
        val mean = blockData.map(_.value).sum / blockData.length
        val variance = blockData.map(e => math.pow(e.value - mean, 2)).sum / blockData.length
        if (mean != 0) 1.0 / (1.0 + variance / (mean * mean)) else 0.0
      } else 1.0
    }
    
    if (cohesionScores.nonEmpty) cohesionScores.sum / cohesionScores.size else 0.0
  }
  
  private def computeBlockSeparation(entries: Seq[MatrixEntry]): Double = {
    val blockMeans = entries.groupBy(e => model.predict(e))
      .mapValues(blockData => blockData.map(_.value).sum / blockData.length)
    
    if (blockMeans.size < 2) return 0.0
    
    val meanPairs = blockMeans.values.toSeq.combinations(2).map { case Seq(m1, m2) =>
      math.abs(m1 - m2)
    }.toSeq
    
    if (meanPairs.nonEmpty) meanPairs.sum / meanPairs.length else 0.0
  }
  
  private def computeModularity(entries: Seq[MatrixEntry]): Double = {
    // Simplified modularity computation for bipartite graphs
    val totalWeight = entries.map(_.weight).sum
    if (totalWeight == 0) return 0.0
    
    val blockEntries = entries.groupBy(e => model.predict(e))
    val internalWeight = blockEntries.map { case (_, blockData) =>
      blockData.map(_.weight).sum
    }.sum
    
    val expectedInternal = entries.map(_.weight).sum / blockEntries.size
    (internalWeight - expectedInternal) / totalWeight
  }
  
  private def createBlockMatrix(entries: RDD[MatrixEntry]): RDD[(Int, Int, Double, Int, Double)] = {
    entries.map { entry =>
      val (rowCluster, colCluster) = model.predict(entry)
      ((rowCluster, colCluster), (entry.value, 1, entry.weight))
    }.reduceByKey { case ((v1, c1, w1), (v2, c2, w2)) =>
      (v1 + v2, c1 + c2, w1 + w2)
    }.map { case ((rowCluster, colCluster), (totalValue, count, totalWeight)) =>
      val meanValue = if (totalWeight > 0) totalValue * totalWeight / totalWeight else 0.0
      (rowCluster, colCluster, meanValue, count, totalWeight)
    }
  }
  
  private def createRowClusterProfiles(entries: RDD[MatrixEntry], spark: SparkSession): DataFrame = {
    import spark.implicits._
    
    entries.map { entry =>
      val (rowCluster, colCluster) = model.predict(entry)
      (rowCluster, colCluster, entry.value, entry.weight)
    }.toDF("rowCluster", "colCluster", "value", "weight")
      .groupBy("rowCluster", "colCluster")
      .agg(
        org.apache.spark.sql.functions.avg("value").alias("avgValue"),
        org.apache.spark.sql.functions.sum("weight").alias("totalWeight"),
        org.apache.spark.sql.functions.count("*").alias("entryCount")
      )
  }
  
  private def createColClusterProfiles(entries: RDD[MatrixEntry], spark: SparkSession): DataFrame = {
    import spark.implicits._
    
    entries.map { entry =>
      val (rowCluster, colCluster) = model.predict(entry)
      (colCluster, rowCluster, entry.value, entry.weight)
    }.toDF("colCluster", "rowCluster", "value", "weight")
      .groupBy("colCluster", "rowCluster")
      .agg(
        org.apache.spark.sql.functions.avg("value").alias("avgValue"),
        org.apache.spark.sql.functions.sum("weight").alias("totalWeight"),
        org.apache.spark.sql.functions.count("*").alias("entryCount")
      )
  }
  
  private def computeStabilityMeasure(values: Seq[Double]): Double = {
    if (values.length < 2) return 1.0
    
    val mean = values.sum / values.length
    val variance = values.map(v => math.pow(v - mean, 2)).sum / values.length
    val cv = if (mean != 0) math.sqrt(variance) / math.abs(mean) else 0.0
    
    // Stability is inverse of coefficient of variation
    1.0 / (1.0 + cv)
  }
}

/**
 * Results of block pattern analysis.
 */
case class BlockPatternAnalysis(
    patterns: Map[String, Int],
    dominantBlocks: Seq[(Int, Int, Double)],
    blockDensity: Double,
    connectivity: Double,
    totalNonEmptyBlocks: Int)

/**
 * Cluster quality metrics.
 */
case class ClusterQualityMetrics(
    rowSilhouetteScore: Double,
    colSilhouetteScore: Double,
    blockCohesion: Double,
    separationIndex: Double,
    modularity: Double,
    overallQuality: Double)

/**
 * Data export for visualization tools.
 */
case class VisualizationExport(
    blockMatrix: DataFrame,
    assignments: DataFrame,
    rowProfiles: DataFrame,
    colProfiles: DataFrame,
    convergenceHistory: DataFrame)

/**
 * Results of stability analysis.
 */
case class StabilityAnalysis(
    numRuns: Int,
    objectiveStability: Double,
    errorStability: Double,
    iterationStability: Double,
    averageObjective: Double,
    averageReconstructionError: Double,
    averageIterations: Double,
    objectiveRange: (Double, Double),
    errorRange: (Double, Double))

object CoClusteringAnalysis {
  
  /**
   * Create an analysis instance for a trained model.
   */
  def apply(model: BregmanCoClusteringModel): CoClusteringAnalysis = {
    new CoClusteringAnalysis(model)
  }
  
  /**
   * Export cluster assignments to CSV format.
   */
  def exportAssignmentsCSV(
      model: BregmanCoClusteringModel,
      entries: RDD[MatrixEntry],
      outputPath: String): Unit = {
    
    val assignments = entries.map { entry =>
      val (rowCluster, colCluster) = model.predict(entry)
      s"${entry.rowIndex},${entry.colIndex},${entry.value},${entry.weight},$rowCluster,$colCluster"
    }
    
    // Header: "rowIndex,colIndex,value,weight,rowCluster,colCluster"
    assignments.coalesce(1).saveAsTextFile(outputPath)
  }
  
  /**
   * Export block statistics to JSON format.
   */
  def exportBlockStatsJSON(
      model: BregmanCoClusteringModel,
      entries: RDD[MatrixEntry]): String = {
    
    val blockStats = model.getBlockStats(entries)
    val jsonBuilder = new mutable.StringBuilder()
    
    jsonBuilder.append("{\n")
    jsonBuilder.append("  \"blockStats\": [\n")
    
    for (r <- blockStats.indices) {
      for (c <- blockStats(r).indices) {
        val stats = blockStats(r)(c)
        jsonBuilder.append(s"""    {
          |      "rowCluster": $r,
          |      "colCluster": $c,
          |      "count": ${stats.count},
          |      "mean": ${stats.mean},
          |      "variance": ${stats.variance},
          |      "min": ${stats.min},
          |      "max": ${stats.max},
          |      "totalWeight": ${stats.totalWeight}
          |    }""".stripMargin)
        
        if (r < blockStats.length - 1 || c < blockStats(r).length - 1) {
          jsonBuilder.append(",")
        }
        jsonBuilder.append("\n")
      }
    }
    
    jsonBuilder.append("  ]\n")
    jsonBuilder.append("}")
    
    jsonBuilder.toString()
  }
}