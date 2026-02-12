package com.massivedatascience.clusterer.ml.df.strategies.impl

import com.massivedatascience.clusterer.ml.df.kernels.ClusteringKernel
import com.massivedatascience.clusterer.ml.df.strategies.AssignmentStrategy
import org.apache.spark.internal.Logging
import org.apache.spark.sql.DataFrame

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
      kernel: ClusteringKernel
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
      val sizeStr         = formatBroadcastSize(kTimesDim)
      val thresholdStr    = formatBroadcastSize(broadcastThresholdElems)
      val overagePercent  = ((kTimesDim.toDouble / broadcastThresholdElems - 1.0) * 100).toInt
      val suggestedChunkK = math.max(1, broadcastThresholdElems / dim)

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
