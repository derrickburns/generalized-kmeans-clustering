package com.massivedatascience.clusterer.ml.df.strategies.impl

import com.massivedatascience.clusterer.ml.df.BregmanKernel
import com.massivedatascience.clusterer.ml.df.strategies.AssignmentStrategy
import org.apache.spark.internal.Logging
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

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
