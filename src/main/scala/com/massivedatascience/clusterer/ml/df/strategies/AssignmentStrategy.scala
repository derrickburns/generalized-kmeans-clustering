package com.massivedatascience.clusterer.ml.df.strategies

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import com.massivedatascience.clusterer.ml.df.BregmanKernel

/** Strategy for assigning points to clusters.
  *
  * Takes a DataFrame with features and produces a DataFrame with cluster assignments.
  */
trait AssignmentStrategy extends Serializable {

  /** Assign each point to the nearest cluster center.
    *
    * @param df
    *   input DataFrame
    * @param featuresCol
    *   name of features column
    * @param weightCol
    *   optional weight column
    * @param centers
    *   current cluster centers
    * @param kernel
    *   Bregman kernel
    * @return
    *   DataFrame with additional "cluster" column (Int)
    */
  def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: BregmanKernel
  ): DataFrame
}

/** Broadcast UDF assignment strategy.
  *
  * Broadcasts centers to executors and uses UDF to compute assignments. Works with any Bregman
  * divergence but may be slower than expression-based approaches.
  */
private[df] class BroadcastUDFAssignment extends AssignmentStrategy with Logging {

  /** Format broadcast size with human-readable units. */
  private def formatBroadcastSize(elements: Long): String = {
    val bytes = elements * 8 // doubles are 8 bytes
    if (bytes < 1024) {
      f"${bytes}B"
    } else if (bytes < 1024 * 1024) {
      f"${bytes / 1024.0}%.1fKB"
    } else if (bytes < 1024 * 1024 * 1024) {
      f"${bytes / (1024.0 * 1024.0)}%.1fMB"
    } else {
      f"${bytes / (1024.0 * 1024.0 * 1024.0)}%.1fGB"
    }
  }

  override def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: BregmanKernel
  ): DataFrame = {

    val k         = centers.length
    val dim       = centers.headOption.map(_.length).getOrElse(0)
    val kTimesDim = k * dim
    val sizeStr   = formatBroadcastSize(kTimesDim)

    logDebug(
      s"BroadcastUDFAssignment: broadcasting k=$k clusters × dim=$dim = $kTimesDim elements ≈ $sizeStr"
    )

    // Warn if broadcast size is very large (>100MB)
    val warningThreshold = 12500000 // ~100MB
    if (kTimesDim > warningThreshold) {
      val warningStr = formatBroadcastSize(warningThreshold)
      logWarning(
        s"""BroadcastUDFAssignment: Large broadcast detected
           |  Size: k=$k × dim=$dim = $kTimesDim elements ≈ $sizeStr
           |  This exceeds the recommended size for broadcasting ($warningStr).
           |
           |  Potential issues:
           |    - Executor OOM errors during broadcast
           |    - Slow broadcast distribution across cluster
           |    - Driver memory pressure
           |
           |  Consider:
           |    1. Using ChunkedBroadcastAssignment for large k×dim
           |    2. Reducing k or dimensionality
           |    3. Increasing executor memory
           |    4. Using AutoAssignment strategy (automatically selects best approach)""".stripMargin
      )
    }

    val spark     = df.sparkSession
    val bcCenters = spark.sparkContext.broadcast(centers)
    val bcKernel  = spark.sparkContext.broadcast(kernel)

    val assignUDF = udf { (features: Vector) =>
      val ctrs    = bcCenters.value
      val kern    = bcKernel.value
      var minDist = Double.PositiveInfinity
      var minIdx  = 0
      var i       = 0
      while (i < ctrs.length) {
        val center = Vectors.dense(ctrs(i))
        val dist   = kern.divergence(features, center)
        if (dist < minDist) {
          minDist = dist
          minIdx = i
        }
        i += 1
      }
      minIdx
    }

    df.withColumn("cluster", assignUDF(col(featuresCol)))
  }
}

/** Squared Euclidean cross-join assignment strategy.
  *
  * Uses DataFrame cross-join with expression-based distance computation. Much faster than UDF for
  * Squared Euclidean, but only works with SE kernel.
  */
private[df] class SECrossJoinAssignment extends AssignmentStrategy with Logging {

  override def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: BregmanKernel
  ): DataFrame = {

    require(
      kernel.supportsExpressionOptimization,
      s"SECrossJoinAssignment only works with Squared Euclidean kernel, got ${kernel.name}"
    )

    logDebug(s"SECrossJoinAssignment: assigning ${centers.length} clusters")

    val spark = df.sparkSession

    // Create centers DataFrame using RDD
    val centersRDD = spark.sparkContext.parallelize(
      centers.toIndexedSeq.zipWithIndex.map { case (center, idx) =>
        (idx, Vectors.dense(center))
      }
    )
    val centersDF  = spark.createDataFrame(centersRDD).toDF("clusterId", "center")

    // Cross-join and compute squared distances using expressions
    val joined = df.crossJoin(centersDF)

    // Compute squared Euclidean distance using array operations
    val distanceUDF = udf { (features: Vector, center: Vector) =>
      val fArr = features.toArray
      val cArr = center.toArray
      var sum  = 0.0
      var i    = 0
      while (i < fArr.length) {
        val diff = fArr(i) - cArr(i)
        sum += diff * diff
        i += 1
      }
      sum * 0.5
    }

    val withDistances = joined.withColumn("distance", distanceUDF(col(featuresCol), col("center")))

    // Find minimum distance cluster for each point
    // Use window function for efficiency
    import org.apache.spark.sql.expressions.Window
    import scala.collection.immutable.ArraySeq
    val windowSpec = Window.partitionBy(ArraySeq.unsafeWrapArray(df.columns.map(col)): _*)

    val withRank = withDistances
      .withColumn("rank", row_number().over(windowSpec.orderBy("distance")))
      .filter(col("rank") === 1)
      .drop("center", "distance", "rank")
      .withColumnRenamed("clusterId", "cluster")

    withRank
  }
}

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
private[df] class ChunkedBroadcastAssignment(chunkSize: Int = 100) extends AssignmentStrategy with Logging {

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

/** Adaptive broadcast assignment strategy.
  *
  * Automatically determines optimal chunk size and broadcast threshold based on
  * executor memory configuration, rather than using fixed defaults.
  *
  * ==Memory Calculation==
  *
  * The strategy queries Spark configuration for:
  *   - `spark.executor.memory`: Total executor memory (default: 1g)
  *   - `spark.memory.fraction`: Fraction for execution/storage (default: 0.6)
  *
  * From this, it calculates:
  *   - Available broadcast memory = executor_memory × memory_fraction × broadcast_fraction
  *   - Optimal chunk size = available_memory / (dim × 8 bytes × safety_factor)
  *   - Broadcast threshold = chunk_size × dim (elements that fit in one broadcast)
  *
  * ==Parameters==
  *
  *   - '''broadcastFraction''': Fraction of available memory to use for broadcast (default: 0.1 = 10%)
  *   - '''safetyFactor''': Additional margin for serialization overhead (default: 2.0)
  *   - '''minChunkSize''': Minimum chunk size to use (default: 10)
  *   - '''maxChunkSize''': Maximum chunk size to use (default: 1000)
  *
  * ==Example==
  *
  * With 4GB executor memory, 0.6 memory fraction, 0.1 broadcast fraction:
  *   - Available for broadcast: 4GB × 0.6 × 0.1 = 245MB
  *   - For dim=100: chunk_size ≈ 245MB / (100 × 8 × 2) ≈ 153,600 centers per chunk
  *
  * @param broadcastFraction fraction of available memory for broadcast (default 0.1)
  * @param safetyFactor overhead multiplier for serialization (default 2.0)
  * @param minChunkSize minimum chunk size (default 10)
  * @param maxChunkSize maximum chunk size (default 1000)
  */
private[df] class AdaptiveBroadcastAssignment(
    broadcastFraction: Double = 0.1,
    safetyFactor: Double = 2.0,
    minChunkSize: Int = 10,
    maxChunkSize: Int = 1000
) extends AssignmentStrategy
    with Logging {

  require(broadcastFraction > 0 && broadcastFraction <= 1.0,
    s"broadcastFraction must be in (0, 1], got $broadcastFraction")
  require(safetyFactor >= 1.0, s"safetyFactor must be >= 1.0, got $safetyFactor")
  require(minChunkSize > 0, s"minChunkSize must be > 0, got $minChunkSize")
  require(maxChunkSize >= minChunkSize, s"maxChunkSize must be >= minChunkSize")

  /** Parse memory string (e.g., "4g", "512m", "1024k") to bytes. */
  private def parseMemoryString(mem: String): Long = {
    val trimmed = mem.trim.toLowerCase
    val numPart = trimmed.dropRight(1)
    val unit = trimmed.last

    val multiplier = unit match {
      case 'k' => 1024L
      case 'm' => 1024L * 1024L
      case 'g' => 1024L * 1024L * 1024L
      case 't' => 1024L * 1024L * 1024L * 1024L
      case c if c.isDigit => return trimmed.toLong // No suffix, assume bytes
      case _ => throw new IllegalArgumentException(s"Unknown memory unit: $unit in $mem")
    }

    (numPart.toDouble * multiplier).toLong
  }

  /** Format bytes as human-readable string. */
  private def formatBytes(bytes: Long): String = {
    if (bytes < 1024) {
      f"${bytes}B"
    } else if (bytes < 1024 * 1024) {
      f"${bytes / 1024.0}%.1fKB"
    } else if (bytes < 1024L * 1024L * 1024L) {
      f"${bytes / (1024.0 * 1024.0)}%.1fMB"
    } else {
      f"${bytes / (1024.0 * 1024.0 * 1024.0)}%.1fGB"
    }
  }

  /** Calculate optimal chunk size based on Spark configuration. */
  private def calculateOptimalChunkSize(spark: org.apache.spark.sql.SparkSession, dim: Int): Int = {
    val conf = spark.sparkContext.getConf

    // Get executor memory (default 1g)
    val executorMemoryStr = conf.get("spark.executor.memory", "1g")
    val executorMemory = parseMemoryString(executorMemoryStr)

    // Get memory fraction (default 0.6)
    val memoryFraction = conf.getDouble("spark.memory.fraction", 0.6)

    // Calculate available memory for broadcast
    val availableMemory = (executorMemory * memoryFraction * broadcastFraction).toLong

    // Calculate chunk size: available_memory / (dim × 8 bytes × safety_factor)
    val bytesPerCenter = dim * 8L // 8 bytes per double
    val effectiveBytesPerCenter = (bytesPerCenter * safetyFactor).toLong

    val optimalChunkSize = if (effectiveBytesPerCenter > 0) {
      (availableMemory / effectiveBytesPerCenter).toInt
    } else {
      maxChunkSize
    }

    // Clamp to [minChunkSize, maxChunkSize]
    val clampedChunkSize = math.max(minChunkSize, math.min(maxChunkSize, optimalChunkSize))

    logInfo(
      s"""AdaptiveBroadcastAssignment: Memory configuration
         |  Executor memory: $executorMemoryStr (${formatBytes(executorMemory)})
         |  Memory fraction: $memoryFraction
         |  Broadcast fraction: $broadcastFraction
         |  Available for broadcast: ${formatBytes(availableMemory)}
         |  Dimension: $dim
         |  Bytes per center: $bytesPerCenter (with safety: ${formatBytes(effectiveBytesPerCenter)})
         |  Optimal chunk size: $optimalChunkSize (clamped to $clampedChunkSize)""".stripMargin
    )

    clampedChunkSize
  }

  override def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: BregmanKernel
  ): DataFrame = {

    val k = centers.length
    val dim = centers.headOption.map(_.length).getOrElse(0)

    if (k == 0 || dim == 0) {
      logWarning("AdaptiveBroadcastAssignment: No centers provided, returning empty assignment")
      return df.withColumn("cluster", lit(0))
    }

    val spark = df.sparkSession

    // Calculate optimal chunk size based on memory configuration
    val chunkSize = calculateOptimalChunkSize(spark, dim)
    val broadcastThreshold = chunkSize * dim

    val kTimesDim = k * dim

    logInfo(
      s"""AdaptiveBroadcastAssignment: Strategy selection
         |  k=$k, dim=$dim, k×dim=$kTimesDim elements (${formatBytes(kTimesDim * 8L)})
         |  Adaptive chunk size: $chunkSize
         |  Broadcast threshold: $broadcastThreshold elements (${formatBytes(broadcastThreshold * 8L)})""".stripMargin
    )

    // Use SE fast path if available
    if (kernel.supportsExpressionOptimization) {
      logInfo(s"AdaptiveBroadcastAssignment: Using SECrossJoin (kernel=${kernel.name})")
      return new SECrossJoinAssignment().assign(df, featuresCol, weightCol, centers, kernel)
    }

    // Choose between direct broadcast and chunked based on adaptive threshold
    if (k <= chunkSize) {
      logInfo(s"AdaptiveBroadcastAssignment: Using BroadcastUDF (k=$k <= chunkSize=$chunkSize)")
      new BroadcastUDFAssignment().assign(df, featuresCol, weightCol, centers, kernel)
    } else {
      val numChunks = math.ceil(k.toDouble / chunkSize).toInt
      logInfo(
        s"""AdaptiveBroadcastAssignment: Using ChunkedBroadcast
           |  k=$k > chunkSize=$chunkSize
           |  Will process in $numChunks chunks
           |  Trade-off: ${numChunks}x data scans for memory safety""".stripMargin
      )
      new ChunkedBroadcastAssignment(chunkSize).assign(df, featuresCol, weightCol, centers, kernel)
    }
  }
}

/** Auto-selecting assignment strategy.
  *
  * Chooses between BroadcastUDF, SECrossJoin, and Chunked based on kernel, data size, and k×dim.
  *
  * Strategy selection:
  *   - SE kernel → SECrossJoin (most efficient for Squared Euclidean)
  *   - Non-SE, k×dim < threshold → BroadcastUDF (fast, low memory overhead)
  *   - Non-SE, k×dim >= threshold → ChunkedBroadcast (multiple scans, avoids OOM)
  *
  * Default threshold: 200,000 elements (~1.5MB of doubles)
  */
private[df] class AutoAssignment(broadcastThresholdElems: Int = 200000, chunkSize: Int = 100)
    extends AssignmentStrategy
    with Logging {

  private val broadcastStrategy = new BroadcastUDFAssignment()
  private val seStrategy        = new SECrossJoinAssignment()
  private val chunkedStrategy   = new ChunkedBroadcastAssignment(chunkSize)

  /** Format broadcast size with human-readable units. */
  private def formatBroadcastSize(elements: Long): String = {
    val bytes = elements * 8 // doubles are 8 bytes
    if (bytes < 1024) {
      f"${bytes}B"
    } else if (bytes < 1024 * 1024) {
      f"${bytes / 1024.0}%.1fKB"
    } else if (bytes < 1024 * 1024 * 1024) {
      f"${bytes / (1024.0 * 1024.0)}%.1fMB"
    } else {
      f"${bytes / (1024.0 * 1024.0 * 1024.0)}%.1fGB"
    }
  }

  override def assign(
      df: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      centers: Array[Array[Double]],
      kernel: BregmanKernel
  ): DataFrame = {

    val k         = centers.length
    val dim       = centers.headOption.map(_.length).getOrElse(0)
    val kTimesDim = k * dim

    if (kernel.supportsExpressionOptimization) {
      logInfo(s"AutoAssignment: strategy=SECrossJoin (kernel=${kernel.name})")
      seStrategy.assign(df, featuresCol, weightCol, centers, kernel)
    } else if (kTimesDim < broadcastThresholdElems) {
      val sizeStr = formatBroadcastSize(kTimesDim)
      logInfo(
        s"AutoAssignment: strategy=BroadcastUDF (kernel=${kernel.name}, k=$k, dim=$dim, " +
          s"broadcast_size=$kTimesDim elements ≈ $sizeStr < threshold=$broadcastThresholdElems)"
      )
      broadcastStrategy.assign(df, featuresCol, weightCol, centers, kernel)
    } else {
      val sizeStr          = formatBroadcastSize(kTimesDim)
      val thresholdStr     = formatBroadcastSize(broadcastThresholdElems)
      val overagePercent   = ((kTimesDim.toDouble / broadcastThresholdElems - 1.0) * 100).toInt
      val suggestedChunkK  = math.max(1, broadcastThresholdElems / dim)

      logWarning(
        s"""AutoAssignment: Broadcast size exceeds threshold
           |  Current: k=$k × dim=$dim = $kTimesDim elements ≈ $sizeStr
           |  Threshold: $broadcastThresholdElems elements ≈ $thresholdStr
           |  Overage: +$overagePercent%
           |
           |  Using ChunkedBroadcast (chunkSize=$chunkSize) to avoid OOM.
           |  This will scan the data ${math.ceil(k.toDouble / chunkSize).toInt} times.
           |
           |  To avoid chunking overhead, consider:
           |    1. Reduce k (number of clusters)
           |    2. Reduce dimensionality (current: $dim dimensions)
           |    3. Increase broadcastThreshold (suggested: k=$k would need ~${kTimesDim} elements)
           |    4. Use Squared Euclidean divergence if appropriate (enables fast SE path)
           |
           |  Current configuration can broadcast up to k≈$suggestedChunkK clusters of $dim dimensions.""".stripMargin
      )
      chunkedStrategy.assign(df, featuresCol, weightCol, centers, kernel)
    }
  }
}
