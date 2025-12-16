package com.massivedatascience.clusterer.ml.df.strategies.impl

import com.massivedatascience.clusterer.ml.df.BregmanKernel
import com.massivedatascience.clusterer.ml.df.strategies.AssignmentStrategy
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/** Chunked broadcast assignment strategy.
  *
  * Processes centers in chunks to avoid OOM when k×dim is large. Trades off multiple scans for
  * reduced memory footprint.
  *
  * Algorithm:
  *   1. Split centers into chunks of size chunkSize 2. For each chunk: broadcast chunk, compute
  *      local min distance & cluster ID 3. Reduce across chunks to find global minimum
  *
  * Memory: O(chunkSize × dim) broadcast per executor Scans: ceil(k / chunkSize) passes over the
  * data
  */
private[df] class ChunkedBroadcastAssignment(chunkSize: Int = 100)
    extends AssignmentStrategy
    with Logging {

  override def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: BregmanKernel
  ): DataFrame = {

    val k = centers.length
    logInfo(s"ChunkedBroadcastAssignment: assigning $k clusters using chunkSize=$chunkSize")

    if (k <= chunkSize) {
      // Small enough to use regular broadcast
      logInfo(s"ChunkedBroadcastAssignment: k=$k <= chunkSize=$chunkSize, using BroadcastUDF")
      return new BroadcastUDFAssignment().assign(df, featuresCol, weightCol, centers, kernel)
    }

    val spark = df.sparkSession

    // Add row ID for tracking minimum across chunks
    val withId = df.withColumn("_row_id", monotonically_increasing_id())

    // Process centers in chunks
    val chunks = centers.grouped(chunkSize).zipWithIndex.toSeq

    var result = withId

    chunks.foreach { case (chunk, chunkIdx) =>
      val chunkStart   = chunkIdx * chunkSize
      val bcChunk      = spark.sparkContext.broadcast(chunk)
      val bcKernel     = spark.sparkContext.broadcast(kernel)
      val bcChunkStart = spark.sparkContext.broadcast(chunkStart)

      // Compute local minimum for this chunk
      val chunkMinUDF = udf { (features: Vector) =>
        val ctrs    = bcChunk.value
        val kern    = bcKernel.value
        val offset  = bcChunkStart.value
        var minDist = Double.PositiveInfinity
        var minIdx  = 0
        var i       = 0
        while (i < ctrs.length) {
          val center = Vectors.dense(ctrs(i))
          val dist   = kern.divergence(features, center)
          if (dist < minDist) {
            minDist = dist
            minIdx = offset + i
          }
          i += 1
        }
        (minIdx, minDist)
      }

      val chunkCol = s"_chunk_${chunkIdx}"
      result = result.withColumn(chunkCol, chunkMinUDF(col(featuresCol)))

      // Cleanup broadcast
      bcChunk.unpersist()
      bcKernel.unpersist()
      bcChunkStart.unpersist()
    }

    // Find global minimum across all chunks
    val numChunks    = chunks.size
    val globalMinUDF = udf { (row: org.apache.spark.sql.Row) =>
      var minDist    = Double.PositiveInfinity
      var minCluster = 0
      var i          = 0
      while (i < numChunks) {
        val chunkResult = row.getStruct(row.fieldIndex(s"_chunk_$i"))
        val clusterId   = chunkResult.getInt(0)
        val dist        = chunkResult.getDouble(1)
        if (dist < minDist) {
          minDist = dist
          minCluster = clusterId
        }
        i += 1
      }
      minCluster
    }

    // Collect all chunk columns into a struct for the UDF
    val chunkCols  = (0 until numChunks).map(i => col(s"_chunk_$i"))
    val withStruct = result.withColumn("_all_chunks", struct(chunkCols: _*))

    // Apply global minimum UDF
    val assigned = withStruct
      .withColumn("cluster", globalMinUDF(col("_all_chunks")))
      .drop("_row_id", "_all_chunks")
      .drop((0 until numChunks).map(i => s"_chunk_$i"): _*)

    logInfo(s"ChunkedBroadcastAssignment: completed in $numChunks passes")
    assigned
  }
}
