package com.massivedatascience.clusterer.ml.df.strategies.impl

import com.massivedatascience.clusterer.ml.df.BregmanKernel
import com.massivedatascience.clusterer.ml.df.strategies.AssignmentStrategy
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

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
