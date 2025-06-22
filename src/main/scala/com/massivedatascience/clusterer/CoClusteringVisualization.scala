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
import scala.util.{Random, Try}

/**
 * Specialized visualization utilities for co-clustering results.
 * 
 * This object provides methods to generate data in formats suitable for
 * popular visualization libraries and tools including:
 * - Heatmaps for block structure visualization
 * - Network graphs for cluster relationships
 * - Statistical plots for convergence and quality metrics
 * - Interactive web-based visualizations
 */
object CoClusteringVisualization {
  
  @transient private lazy val logger = LoggerFactory.getLogger(getClass.getName)
  
  /**
   * Generate heatmap data for block structure visualization.
   * 
   * This creates a matrix representation suitable for heatmap visualization
   * tools like seaborn, matplotlib, or D3.js.
   * 
   * @param model Trained co-clustering model
   * @param entries Original matrix entries
   * @param normalizeValues Whether to normalize values for better visualization
   * @return Heatmap data with coordinates and values
   */
  def generateHeatmapData(
      model: BregmanCoClusteringModel,
      entries: RDD[MatrixEntry],
      normalizeValues: Boolean = true): HeatmapData = {
    
    logger.info("Generating heatmap data for block structure visualization")
    
    val blockStats = model.getBlockStats(entries)
    val numRowClusters = model.result.numRowClusters
    val numColClusters = model.result.numColClusters
    
    val values = mutable.ArrayBuffer[(Int, Int, Double)]()
    val labels = mutable.ArrayBuffer[(Int, Int, String)]()
    
    var globalMin = Double.MaxValue
    var globalMax = Double.MinValue
    
    // Extract values and find global min/max for normalization
    for (r <- 0 until numRowClusters; c <- 0 until numColClusters) {
      val stats = blockStats(r)(c)
      val value = if (stats.count > 0) stats.mean else 0.0
      
      values += ((r, c, value))
      labels += ((r, c, s"Block($r,$c): ${stats.count} entries, mean=${value.formatted("%.3f")}"))
      
      if (value != 0.0) {
        globalMin = math.min(globalMin, value)
        globalMax = math.max(globalMax, value)
      }
    }
    
    // Normalize values if requested
    val normalizedValues = if (normalizeValues && globalMax != globalMin) {
      values.map { case (r, c, value) =>
        val normalized = if (value != 0.0) (value - globalMin) / (globalMax - globalMin) else 0.0
        (r, c, normalized)
      }
    } else {
      values
    }
    
    HeatmapData(
      values = normalizedValues.toSeq,
      labels = labels.toSeq,
      rowClusterNames = (0 until numRowClusters).map(i => s"RC$i"),
      colClusterNames = (0 until numColClusters).map(i => s"CC$i"),
      valueRange = (globalMin, globalMax)
    )
  }
  
  /**
   * Generate network graph data showing relationships between clusters.
   * 
   * Creates nodes and edges for visualization in network graph libraries
   * like NetworkX, vis.js, or D3.js force-directed layouts.
   * 
   * @param model Trained co-clustering model
   * @param entries Original matrix entries
   * @param minWeight Minimum edge weight to include in the graph
   * @return Network graph data with nodes and edges
   */
  def generateNetworkGraphData(
      model: BregmanCoClusteringModel,
      entries: RDD[MatrixEntry],
      minWeight: Double = 0.01): NetworkGraphData = {
    
    logger.info("Generating network graph data for cluster relationships")
    
    val blockStats = model.getBlockStats(entries)
    val nodes = mutable.ArrayBuffer[GraphNode]()
    val edges = mutable.ArrayBuffer[GraphEdge]()
    
    // Create row cluster nodes
    val rowStats = model.getRowClusterStats(entries)
    rowStats.zipWithIndex.foreach { case (stats, idx) =>
      nodes += GraphNode(
        id = s"row_$idx",
        label = s"Row Cluster $idx",
        nodeType = "row",
        size = math.sqrt(stats.totalEntries).toInt,
        weight = stats.totalEntries,
        metadata = Map(
          "uniqueRows" -> stats.uniqueIndices.toString,
          "totalEntries" -> stats.totalEntries.toString,
          "mean" -> stats.mean.formatted("%.3f")
        )
      )
    }
    
    // Create column cluster nodes
    val colStats = model.getColClusterStats(entries)
    colStats.zipWithIndex.foreach { case (stats, idx) =>
      nodes += GraphNode(
        id = s"col_$idx",
        label = s"Column Cluster $idx",
        nodeType = "column",
        size = math.sqrt(stats.totalEntries).toInt,
        weight = stats.totalEntries,
        metadata = Map(
          "uniqueCols" -> stats.uniqueIndices.toString,
          "totalEntries" -> stats.totalEntries.toString,
          "mean" -> stats.mean.formatted("%.3f")
        )
      )
    }
    
    // Create edges between row and column clusters based on block strength
    for (r <- blockStats.indices; c <- blockStats(r).indices) {
      val stats = blockStats(r)(c)
      if (stats.totalWeight >= minWeight && stats.count > 0) {
        val strength = stats.totalWeight / stats.variance.max(1.0)
        
        edges += GraphEdge(
          source = s"row_$r",
          target = s"col_$c",
          weight = strength,
          label = s"Block($r,$c)",
          metadata = Map(
            "blockCount" -> stats.count.toString,
            "blockMean" -> stats.mean.formatted("%.3f"),
            "blockWeight" -> stats.totalWeight.formatted("%.3f")
          )
        )
      }
    }
    
    NetworkGraphData(nodes.toSeq, edges.toSeq)
  }
  
  /**
   * Generate convergence plot data for training visualization.
   * 
   * @param model Trained co-clustering model
   * @return Data for plotting convergence curves
   */
  def generateConvergencePlotData(model: BregmanCoClusteringModel): ConvergencePlotData = {
    val history = model.result.convergenceHistory
    val iterations = history.indices
    val objectives = history
    
    // Compute convergence rate
    val rates = if (history.length > 1) {
      history.zip(history.tail).map { case (prev, current) =>
        if (prev != 0.0) math.abs(current - prev) / math.abs(prev) else 0.0
      }
    } else Seq.empty
    
    ConvergencePlotData(
      iterations = iterations,
      objectives = objectives,
      convergenceRates = rates,
      finalObjective = history.lastOption.getOrElse(0.0),
      totalIterations = history.length
    )
  }
  
  /**
   * Generate cluster distribution data for statistical visualization.
   * 
   * @param model Trained co-clustering model
   * @param entries Original matrix entries
   * @return Distribution data for various statistical plots
   */
  def generateDistributionData(
      model: BregmanCoClusteringModel,
      entries: RDD[MatrixEntry]): DistributionData = {
    
    logger.info("Generating distribution data for statistical visualization")
    
    val rowStats = model.getRowClusterStats(entries)
    val colStats = model.getColClusterStats(entries)
    val blockStats = model.getBlockStats(entries)
    
    // Row cluster size distribution
    val rowSizes = rowStats.map(_.totalEntries)
    val rowSizeHist = createHistogram(rowSizes, 10)
    
    // Column cluster size distribution
    val colSizes = colStats.map(_.totalEntries)
    val colSizeHist = createHistogram(colSizes, 10)
    
    // Block density distribution
    val blockDensities = blockStats.flatten.filter(_.count > 0).map(_.totalWeight)
    val densityHist = createHistogram(blockDensities, 15)
    
    // Value distribution by cluster
    val valuesByRowCluster = rowStats.zipWithIndex.map { case (stats, idx) =>
      (s"RC$idx", Seq(stats.mean, stats.variance, stats.min, stats.max))
    }.toMap
    
    val valuesByColCluster = colStats.zipWithIndex.map { case (stats, idx) =>
      (s"CC$idx", Seq(stats.mean, stats.variance, stats.min, stats.max))
    }.toMap
    
    DistributionData(
      rowSizeDistribution = rowSizeHist,
      colSizeDistribution = colSizeHist,
      blockDensityDistribution = densityHist,
      valuesByRowCluster = valuesByRowCluster,
      valuesByColCluster = valuesByColCluster,
      overallStats = Map(
        "totalRowClusters" -> rowStats.length,
        "totalColClusters" -> colStats.length,
        "nonEmptyBlocks" -> blockStats.flatten.count(_.count > 0),
        "averageBlockSize" -> blockStats.flatten.filter(_.count > 0).map(_.count).sum.toDouble / blockStats.flatten.count(_.count > 0)
      )
    )
  }
  
  /**
   * Export data in D3.js compatible JSON format.
   * 
   * @param model Trained co-clustering model
   * @param entries Original matrix entries
   * @return JSON string for D3.js visualizations
   */
  def exportD3JSON(
      model: BregmanCoClusteringModel,
      entries: RDD[MatrixEntry]): String = {
    
    val heatmap = generateHeatmapData(model, entries)
    val network = generateNetworkGraphData(model, entries)
    val convergence = generateConvergencePlotData(model)
    val distribution = generateDistributionData(model, entries)
    
    val json = new mutable.StringBuilder()
    json.append("{\n")
    
    // Heatmap data
    json.append("""  "heatmap": {""").append("\n")
    json.append(s"""    "numRows": ${heatmap.rowClusterNames.length},""").append("\n")
    json.append(s"""    "numCols": ${heatmap.colClusterNames.length},""").append("\n")
    json.append("""    "data": [""").append("\n")
    heatmap.values.zipWithIndex.foreach { case ((r, c, value), idx) =>
      json.append(s"""      {"row": $r, "col": $c, "value": $value}""")
      if (idx < heatmap.values.length - 1) json.append(",")
      json.append("\n")
    }
    json.append("""    ],""").append("\n")
    json.append(s"""    "valueRange": [${heatmap.valueRange._1}, ${heatmap.valueRange._2}]""").append("\n")
    json.append("""  },""").append("\n")
    
    // Network data
    json.append("""  "network": {""").append("\n")
    json.append("""    "nodes": [""").append("\n")
    network.nodes.zipWithIndex.foreach { case (node, idx) =>
      json.append(s"""      {"id": "${node.id}", "label": "${node.label}", "type": "${node.nodeType}", "size": ${node.size}}""")
      if (idx < network.nodes.length - 1) json.append(",")
      json.append("\n")
    }
    json.append("""    ],""").append("\n")
    json.append("""    "edges": [""").append("\n")
    network.edges.zipWithIndex.foreach { case (edge, idx) =>
      json.append(s"""      {"source": "${edge.source}", "target": "${edge.target}", "weight": ${edge.weight}}""")
      if (idx < network.edges.length - 1) json.append(",")
      json.append("\n")
    }
    json.append("""    ]""").append("\n")
    json.append("""  },""").append("\n")
    
    // Convergence data
    json.append("""  "convergence": {""").append("\n")
    json.append("""    "data": [""").append("\n")
    convergence.iterations.zip(convergence.objectives).zipWithIndex.foreach { case ((iter, obj), idx) =>
      json.append(s"""      {"iteration": $iter, "objective": $obj}""")
      if (idx < convergence.iterations.length - 1) json.append(",")
      json.append("\n")
    }
    json.append("""    ]""").append("\n")
    json.append("""  }""").append("\n")
    
    json.append("}")
    json.toString()
  }
  
  /**
   * Generate interactive HTML visualization with embedded D3.js.
   * 
   * @param model Trained co-clustering model
   * @param entries Original matrix entries
   * @param title Title for the visualization
   * @return Complete HTML page with interactive visualizations
   */
  def generateInteractiveHTML(
      model: BregmanCoClusteringModel,
      entries: RDD[MatrixEntry],
      title: String = "Co-clustering Results"): String = {
    
    val jsonData = exportD3JSON(model, entries)
    val summary = model.summary(entries)
    
    s"""<!DOCTYPE html>
       |<html>
       |<head>
       |  <title>$title</title>
       |  <script src="https://d3js.org/d3.v7.min.js"></script>
       |  <style>
       |    body { font-family: Arial, sans-serif; margin: 20px; }
       |    .container { display: flex; flex-wrap: wrap; gap: 20px; }
       |    .panel { border: 1px solid #ccc; padding: 15px; border-radius: 5px; min-width: 400px; }
       |    .heatmap-cell { stroke: #fff; stroke-width: 1px; }
       |    .tooltip { position: absolute; background: rgba(0,0,0,0.8); color: white; padding: 8px; border-radius: 4px; pointer-events: none; font-size: 12px; }
       |    .summary { background-color: #f9f9f9; }
       |    .metric { display: inline-block; margin: 10px; padding: 8px; background-color: #e6f3ff; border-radius: 3px; }
       |  </style>
       |</head>
       |<body>
       |  <h1>$title</h1>
       |  
       |  <div class="panel summary">
       |    <h2>Model Summary</h2>
       |    <div class="metric">Row Clusters: ${summary.numRowClusters}</div>
       |    <div class="metric">Column Clusters: ${summary.numColClusters}</div>
       |    <div class="metric">Total Entries: ${summary.totalEntries}</div>
       |    <div class="metric">Reconstruction Error: ${summary.reconstructionError.formatted("%.6f")}</div>
       |    <div class="metric">Iterations: ${summary.iterations}</div>
       |  </div>
       |  
       |  <div class="container">
       |    <div class="panel">
       |      <h2>Block Structure Heatmap</h2>
       |      <div id="heatmap"></div>
       |    </div>
       |    
       |    <div class="panel">
       |      <h2>Convergence Plot</h2>
       |      <div id="convergence"></div>
       |    </div>
       |    
       |    <div class="panel">
       |      <h2>Cluster Network</h2>
       |      <div id="network"></div>
       |    </div>
       |  </div>
       |  
       |  <script>
       |    const data = $jsonData;
       |    
       |    // Heatmap visualization
       |    function createHeatmap() {
       |      const margin = {top: 30, right: 30, bottom: 30, left: 50};
       |      const width = 400 - margin.left - margin.right;
       |      const height = 300 - margin.bottom - margin.top;
       |      
       |      const svg = d3.select("#heatmap")
       |        .append("svg")
       |        .attr("width", width + margin.left + margin.right)
       |        .attr("height", height + margin.top + margin.bottom)
       |        .append("g")
       |        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
       |      
       |      const xScale = d3.scaleBand()
       |        .domain(d3.range(data.heatmap.numCols))
       |        .range([0, width])
       |        .padding(0.1);
       |      
       |      const yScale = d3.scaleBand()
       |        .domain(d3.range(data.heatmap.numRows))
       |        .range([0, height])
       |        .padding(0.1);
       |      
       |      const colorScale = d3.scaleSequential(d3.interpolateBlues)
       |        .domain(data.heatmap.valueRange);
       |      
       |      svg.selectAll(".heatmap-cell")
       |        .data(data.heatmap.data)
       |        .enter()
       |        .append("rect")
       |        .attr("class", "heatmap-cell")
       |        .attr("x", d => xScale(d.col))
       |        .attr("y", d => yScale(d.row))
       |        .attr("width", xScale.bandwidth())
       |        .attr("height", yScale.bandwidth())
       |        .attr("fill", d => colorScale(d.value))
       |        .append("title")
       |        .text(d => "Block (" + d.row + ", " + d.col + "): " + d.value.toFixed(3));
       |      
       |      // Add axes
       |      svg.append("g")
       |        .attr("transform", "translate(0," + height + ")")
       |        .call(d3.axisBottom(xScale));
       |      
       |      svg.append("g")
       |        .call(d3.axisLeft(yScale));
       |    }
       |    
       |    // Convergence plot
       |    function createConvergencePlot() {
       |      const margin = {top: 20, right: 30, bottom: 40, left: 60};
       |      const width = 400 - margin.left - margin.right;
       |      const height = 300 - margin.top - margin.bottom;
       |      
       |      const svg = d3.select("#convergence")
       |        .append("svg")
       |        .attr("width", width + margin.left + margin.right)
       |        .attr("height", height + margin.top + margin.bottom)
       |        .append("g")
       |        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
       |      
       |      const xScale = d3.scaleLinear()
       |        .domain(d3.extent(data.convergence.data, d => d.iteration))
       |        .range([0, width]);
       |      
       |      const yScale = d3.scaleLinear()
       |        .domain(d3.extent(data.convergence.data, d => d.objective))
       |        .range([height, 0]);
       |      
       |      const line = d3.line()
       |        .x(d => xScale(d.iteration))
       |        .y(d => yScale(d.objective));
       |      
       |      svg.append("path")
       |        .datum(data.convergence.data)
       |        .attr("fill", "none")
       |        .attr("stroke", "steelblue")
       |        .attr("stroke-width", 2)
       |        .attr("d", line);
       |      
       |      // Add axes
       |      svg.append("g")
       |        .attr("transform", "translate(0," + height + ")")
       |        .call(d3.axisBottom(xScale));
       |      
       |      svg.append("g")
       |        .call(d3.axisLeft(yScale));
       |      
       |      // Add labels
       |      svg.append("text")
       |        .attr("transform", "rotate(-90)")
       |        .attr("y", 0 - margin.left)
       |        .attr("x", 0 - (height / 2))
       |        .attr("dy", "1em")
       |        .style("text-anchor", "middle")
       |        .text("Objective Value");
       |      
       |      svg.append("text")
       |        .attr("transform", "translate(" + (width / 2) + " ," + (height + margin.bottom) + ")")
       |        .style("text-anchor", "middle")
       |        .text("Iteration");
       |    }
       |    
       |    // Network visualization (simplified)
       |    function createNetwork() {
       |      const width = 400;
       |      const height = 300;
       |      
       |      const svg = d3.select("#network")
       |        .append("svg")
       |        .attr("width", width)
       |        .attr("height", height);
       |      
       |      const simulation = d3.forceSimulation(data.network.nodes)
       |        .force("link", d3.forceLink(data.network.edges).id(d => d.id).distance(50))
       |        .force("charge", d3.forceManyBody().strength(-100))
       |        .force("center", d3.forceCenter(width / 2, height / 2));
       |      
       |      const link = svg.append("g")
       |        .selectAll("line")
       |        .data(data.network.edges)
       |        .join("line")
       |        .attr("stroke", "#999")
       |        .attr("stroke-opacity", 0.6)
       |        .attr("stroke-width", d => Math.sqrt(d.weight));
       |      
       |      const node = svg.append("g")
       |        .selectAll("circle")
       |        .data(data.network.nodes)
       |        .join("circle")
       |        .attr("r", d => d.size)
       |        .attr("fill", d => d.type === "row" ? "#ff7f0e" : "#1f77b4")
       |        .call(d3.drag()
       |          .on("start", dragstarted)
       |          .on("drag", dragged)
       |          .on("end", dragended));
       |      
       |      node.append("title")
       |        .text(d => d.label);
       |      
       |      simulation.on("tick", () => {
       |        link
       |          .attr("x1", d => d.source.x)
       |          .attr("y1", d => d.source.y)
       |          .attr("x2", d => d.target.x)
       |          .attr("y2", d => d.target.y);
       |        
       |        node
       |          .attr("cx", d => d.x)
       |          .attr("cy", d => d.y);
       |      });
       |      
       |      function dragstarted(event) {
       |        if (!event.active) simulation.alphaTarget(0.3).restart();
       |        event.subject.fx = event.subject.x;
       |        event.subject.fy = event.subject.y;
       |      }
       |      
       |      function dragged(event) {
       |        event.subject.fx = event.x;
       |        event.subject.fy = event.y;
       |      }
       |      
       |      function dragended(event) {
       |        if (!event.active) simulation.alphaTarget(0);
       |        event.subject.fx = null;
       |        event.subject.fy = null;
       |      }
       |    }
       |    
       |    // Initialize visualizations
       |    createHeatmap();
       |    createConvergencePlot();
       |    createNetwork();
       |  </script>
       |</body>
       |</html>""".stripMargin
  }
  
  // Helper methods
  
  private def createHistogram(values: Seq[Int], numBins: Int): Seq[(Double, Int)] = {
    if (values.isEmpty) return Seq.empty
    
    val min = values.min.toDouble
    val max = values.max.toDouble
    val binWidth = (max - min) / numBins
    
    val bins = Array.fill(numBins)(0)
    
    values.foreach { value =>
      val binIndex = if (value == max) numBins - 1 else ((value - min) / binWidth).toInt
      bins(binIndex) += 1
    }
    
    bins.zipWithIndex.map { case (count, idx) =>
      val binCenter = min + (idx + 0.5) * binWidth
      (binCenter, count)
    }.toSeq
  }
  
  private def createHistogram(values: Seq[Double], numBins: Int): Seq[(Double, Int)] = {
    if (values.isEmpty) return Seq.empty
    
    val min = values.min
    val max = values.max
    val binWidth = (max - min) / numBins
    
    val bins = Array.fill(numBins)(0)
    
    values.foreach { value =>
      val binIndex = if (value == max) numBins - 1 else ((value - min) / binWidth).toInt
      bins(binIndex) += 1
    }
    
    bins.zipWithIndex.map { case (count, idx) =>
      val binCenter = min + (idx + 0.5) * binWidth
      (binCenter, count)
    }.toSeq
  }
}

/**
 * Data structure for heatmap visualization.
 */
case class HeatmapData(
    values: Seq[(Int, Int, Double)],
    labels: Seq[(Int, Int, String)],
    rowClusterNames: Seq[String],
    colClusterNames: Seq[String],
    valueRange: (Double, Double))

/**
 * Data structure for network graph visualization.
 */
case class NetworkGraphData(
    nodes: Seq[GraphNode],
    edges: Seq[GraphEdge])

case class GraphNode(
    id: String,
    label: String,
    nodeType: String,
    size: Int,
    weight: Int,
    metadata: Map[String, String])

case class GraphEdge(
    source: String,
    target: String,
    weight: Double,
    label: String,
    metadata: Map[String, String])

/**
 * Data structure for convergence plot visualization.
 */
case class ConvergencePlotData(
    iterations: Seq[Int],
    objectives: Seq[Double],
    convergenceRates: Seq[Double],
    finalObjective: Double,
    totalIterations: Int)

/**
 * Data structure for distribution analysis.
 */
case class DistributionData(
    rowSizeDistribution: Seq[(Double, Int)],
    colSizeDistribution: Seq[(Double, Int)],
    blockDensityDistribution: Seq[(Double, Int)],
    valuesByRowCluster: Map[String, Seq[Double]],
    valuesByColCluster: Map[String, Seq[Double]],
    overallStats: Map[String, Any])