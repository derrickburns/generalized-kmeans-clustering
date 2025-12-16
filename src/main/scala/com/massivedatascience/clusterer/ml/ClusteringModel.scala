package com.massivedatascience.clusterer.ml

import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.linalg.Vector

/** Shared training summary handling for clustering models.
  *
  * Models mix this in to get consistent summary/hasSummary behavior while
  * keeping the summary payload optionally available for persisted models.
  */
trait HasTrainingSummary extends Params { self: Logging =>

  @transient private[ml] var trainingSummary: Option[TrainingSummary] = None

  /** True if the model was trained in the current session and captured a summary. */
  def hasSummary: Boolean = trainingSummary.isDefined

  /** Retrieve the training summary or throw if absent (e.g., after load). */
  def summary: TrainingSummary =
    trainingSummary.getOrElse(
      throw new NoSuchElementException(
        "Training summary not available. Summary is only available for models trained " +
          "in the current session, not for models loaded from disk."
      )
    )

  /** Attach a training summary in a fluent style. */
  private[ml] def withSummary(summary: TrainingSummary): this.type = {
    trainingSummary = Some(summary)
    this
  }
}

/** Base helpers for centroid-based clustering models. */
trait CentroidModelHelpers { self: Params =>

  /** Concrete models must expose their centers as vectors. */
  def clusterCentersAsVectors: Array[Vector]

  /** Number of clusters (k). */
  def numClusters: Int = clusterCentersAsVectors.length

  /** Dimensionality of feature vectors. */
  def numFeatures: Int = clusterCentersAsVectors.headOption.map(_.size).getOrElse(0)
}
