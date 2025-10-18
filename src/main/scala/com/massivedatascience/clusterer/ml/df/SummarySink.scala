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

package com.massivedatascience.clusterer.ml.df

import scala.collection.mutable.ArrayBuffer

/** Events that occur during k-means clustering.
  *
  * These events provide observability into the clustering process, enabling monitoring, debugging,
  * and analysis of algorithm behavior.
  */
sealed trait ClusteringEvent {
  def timestamp: Long
  def eventType: String
}

/** Iteration started event.
  *
  * @param iteration
  *   iteration number (0-based)
  * @param timestamp
  *   event timestamp in milliseconds
  */
case class IterationStarted(iteration: Int, timestamp: Long = System.currentTimeMillis())
    extends ClusteringEvent {
  override def eventType: String = "iteration_started"
}

/** Iteration completed event.
  *
  * @param iteration
  *   iteration number
  * @param cost
  *   total cost (sum of squared distances)
  * @param centerMovement
  *   average distance centers moved
  * @param assignmentChanges
  *   number of points that changed cluster assignment
  * @param duration
  *   iteration duration in milliseconds
  * @param timestamp
  *   event timestamp in milliseconds
  */
case class IterationCompleted(
    iteration: Int,
    cost: Double,
    centerMovement: Double,
    assignmentChanges: Long,
    duration: Long,
    timestamp: Long = System.currentTimeMillis()
) extends ClusteringEvent {
  override def eventType: String = "iteration_completed"
}

/** Convergence detected event.
  *
  * @param iteration
  *   iteration at which convergence occurred
  * @param reason
  *   reason for convergence (e.g., "cost_delta_below_threshold", "max_iterations")
  * @param timestamp
  *   event timestamp in milliseconds
  */
case class ConvergenceDetected(
    iteration: Int,
    reason: String,
    timestamp: Long = System.currentTimeMillis()
) extends ClusteringEvent {
  override def eventType: String = "convergence_detected"
}

/** Empty clusters detected event.
  *
  * @param iteration
  *   iteration where empty clusters were found
  * @param clusterIds
  *   IDs of empty clusters
  * @param action
  *   action taken (e.g., "reseeded", "ignored")
  * @param timestamp
  *   event timestamp in milliseconds
  */
case class EmptyClustersDetected(
    iteration: Int,
    clusterIds: Set[Int],
    action: String,
    timestamp: Long = System.currentTimeMillis()
) extends ClusteringEvent {
  override def eventType: String = "empty_clusters_detected"
}

/** Warning event.
  *
  * @param iteration
  *   iteration where warning occurred
  * @param message
  *   warning message
  * @param severity
  *   warning severity (e.g., "low", "medium", "high")
  * @param timestamp
  *   event timestamp in milliseconds
  */
case class WarningEvent(
    iteration: Int,
    message: String,
    severity: String = "medium",
    timestamp: Long = System.currentTimeMillis()
) extends ClusteringEvent {
  override def eventType: String = "warning"
}

/** Initialization completed event.
  *
  * @param method
  *   initialization method used (e.g., "random", "kmeans++")
  * @param duration
  *   initialization duration in milliseconds
  * @param timestamp
  *   event timestamp in milliseconds
  */
case class InitializationCompleted(
    method: String,
    duration: Long,
    timestamp: Long = System.currentTimeMillis()
) extends ClusteringEvent {
  override def eventType: String = "initialization_completed"
}

/** Training completed event.
  *
  * @param totalIterations
  *   total number of iterations
  * @param finalCost
  *   final clustering cost
  * @param totalDuration
  *   total training duration in milliseconds
  * @param timestamp
  *   event timestamp in milliseconds
  */
case class TrainingCompleted(
    totalIterations: Int,
    finalCost: Double,
    totalDuration: Long,
    timestamp: Long = System.currentTimeMillis()
) extends ClusteringEvent {
  override def eventType: String = "training_completed"
}

/** Summary of clustering run with all collected events and computed metrics.
  *
  * This is the primary interface for accessing telemetry data after training completes. It provides
  * both raw events and derived metrics for analysis.
  */
case class ClusteringSummary(
    events: Seq[ClusteringEvent],
    startTime: Long,
    endTime: Long
) {

  /** Get all events of a specific type. */
  def eventsOfType[T <: ClusteringEvent: Manifest]: Seq[T] = {
    events.collect { case e: T => e }
  }

  /** Get all iteration events. */
  def iterations: Seq[IterationCompleted] = eventsOfType[IterationCompleted]

  /** Get all warnings. */
  def warnings: Seq[WarningEvent] = eventsOfType[WarningEvent]

  /** Get convergence event if present. */
  def convergence: Option[ConvergenceDetected] = eventsOfType[ConvergenceDetected].headOption

  /** Get initialization event if present. */
  def initialization: Option[InitializationCompleted] =
    eventsOfType[InitializationCompleted].headOption

  /** Get training completion event if present. */
  def completion: Option[TrainingCompleted] = eventsOfType[TrainingCompleted].headOption

  /** Total number of iterations. */
  def numIterations: Int = iterations.size

  /** Total training duration in milliseconds. */
  def totalDuration: Long = endTime - startTime

  /** Average iteration duration in milliseconds. */
  def avgIterationDuration: Double = {
    if (iterations.isEmpty) 0.0
    else iterations.map(_.duration).sum.toDouble / iterations.size
  }

  /** Final clustering cost. */
  def finalCost: Option[Double] = iterations.lastOption.map(_.cost)

  /** Cost improvement from first to last iteration. */
  def costImprovement: Option[Double] = {
    for {
      first <- iterations.headOption
      last  <- iterations.lastOption
    } yield first.cost - last.cost
  }

  /** Percentage cost improvement. */
  def costImprovementPercent: Option[Double] = {
    for {
      first       <- iterations.headOption
      improvement <- costImprovement
    } yield (improvement / first.cost) * 100.0
  }

  /** Average center movement per iteration. */
  def avgCenterMovement: Double = {
    if (iterations.isEmpty) 0.0
    else iterations.map(_.centerMovement).sum / iterations.size
  }

  /** Total assignment changes across all iterations. */
  def totalAssignmentChanges: Long = iterations.map(_.assignmentChanges).sum

  /** Number of empty cluster events. */
  def numEmptyClusterEvents: Int = eventsOfType[EmptyClustersDetected].size

  /** Whether convergence was achieved. */
  def converged: Boolean = convergence.isDefined

  /** Convergence reason if available. */
  def convergenceReason: Option[String] = convergence.map(_.reason)

  /** Generate human-readable summary report. */
  def report: String = {
    val sb = new StringBuilder

    sb.append("=== Clustering Summary ===\n")
    sb.append(s"Duration: ${totalDuration}ms\n")
    sb.append(s"Iterations: $numIterations\n")

    finalCost.foreach(cost => sb.append(f"Final Cost: $cost%.4f\n"))

    costImprovementPercent.foreach(pct => sb.append(f"Cost Improvement: $pct%.2f%%\n"))

    convergence.foreach { conv =>
      sb.append(s"Converged: Yes (${conv.reason})\n")
    }

    if (warnings.nonEmpty) {
      sb.append(s"\nWarnings (${warnings.size}):\n")
      warnings.foreach { w =>
        sb.append(s"  [Iter ${w.iteration}] ${w.message}\n")
      }
    }

    if (numEmptyClusterEvents > 0) {
      sb.append(s"\nEmpty Clusters: $numEmptyClusterEvents events\n")
    }

    sb.append("\nPer-Iteration Metrics:\n")
    sb.append("Iter\tCost\t\tMovement\tChanges\tDuration\n")
    iterations.foreach { iter =>
      sb.append(
        f"${iter.iteration}%4d\t${iter.cost}%.4f\t${iter.centerMovement}%.4f\t" +
          f"${iter.assignmentChanges}%6d\t${iter.duration}%4dms\n"
      )
    }

    sb.toString()
  }
}

/** Sink for collecting clustering events during training.
  *
  * This is a mutable collector that accumulates events as they occur. After training completes,
  * call `summary()` to get an immutable summary.
  *
  * Example usage:
  * {{{
  *   val sink = SummarySink()
  *   sink.record(IterationStarted(0))
  *   sink.record(IterationCompleted(0, cost = 100.0, ...))
  *   val summary = sink.summary()
  *   println(summary.report)
  * }}}
  */
class SummarySink extends Serializable {
  private val events    = ArrayBuffer[ClusteringEvent]()
  private val startTime = System.currentTimeMillis()

  /** Record an event. */
  def record(event: ClusteringEvent): Unit = synchronized {
    events += event
  }

  /** Record multiple events. */
  def recordAll(newEvents: Seq[ClusteringEvent]): Unit = synchronized {
    events ++= newEvents
  }

  /** Get current event count. */
  def size: Int = synchronized { events.size }

  /** Get all recorded events (immutable copy). */
  def getEvents: Seq[ClusteringEvent] = synchronized { events.toSeq }

  /** Generate summary from collected events. */
  def summary(): ClusteringSummary = synchronized {
    ClusteringSummary(
      events = events.toSeq,
      startTime = startTime,
      endTime = System.currentTimeMillis()
    )
  }

  /** Clear all recorded events. */
  def clear(): Unit = synchronized {
    events.clear()
  }
}

object SummarySink {

  /** Create a new summary sink. */
  def apply(): SummarySink = new SummarySink()

  /** Create a no-op sink that discards all events (for performance). */
  def noop(): SummarySink = new SummarySink() {
    override def record(event: ClusteringEvent): Unit             = {} // Discard
    override def recordAll(newEvents: Seq[ClusteringEvent]): Unit = {} // Discard
  }
}

/** Helper for tracking iteration metrics.
  *
  * This provides a convenient way to track iteration start/end and automatically compute duration.
  */
class IterationTracker(sink: SummarySink, iteration: Int) {
  private val startTime = System.currentTimeMillis()

  sink.record(IterationStarted(iteration, startTime))

  /** Complete the iteration with metrics. */
  def complete(cost: Double, centerMovement: Double, assignmentChanges: Long): Unit = {
    val duration = System.currentTimeMillis() - startTime
    sink.record(
      IterationCompleted(
        iteration = iteration,
        cost = cost,
        centerMovement = centerMovement,
        assignmentChanges = assignmentChanges,
        duration = duration,
        timestamp = System.currentTimeMillis()
      )
    )
  }
}

object IterationTracker {

  /** Start tracking an iteration. */
  def apply(sink: SummarySink, iteration: Int): IterationTracker = {
    new IterationTracker(sink, iteration)
  }
}
