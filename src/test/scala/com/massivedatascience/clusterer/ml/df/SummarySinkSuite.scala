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

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class SummarySinkSuite extends AnyFunSuite with Matchers {

  test("SummarySink should record events") {
    val sink = SummarySink()

    sink.record(IterationStarted(0))
    sink.record(IterationCompleted(0, 100.0, 1.5, 50, 100))

    assert(sink.size == 2)
    assert(sink.getEvents.size == 2)
  }

  test("SummarySink should record multiple events") {
    val sink = SummarySink()

    val events = Seq(
      IterationStarted(0),
      IterationCompleted(0, 100.0, 1.5, 50, 100),
      IterationStarted(1),
      IterationCompleted(1, 90.0, 0.8, 20, 80)
    )

    sink.recordAll(events)

    assert(sink.size == 4)
  }

  test("SummarySink should generate summary") {
    val sink = SummarySink()

    sink.record(IterationStarted(0))
    sink.record(IterationCompleted(0, 100.0, 1.5, 50, 100))
    sink.record(ConvergenceDetected(1, "cost_delta_below_threshold"))

    val summary = sink.summary()

    assert(summary.events.size == 3)
    assert(summary.numIterations == 1)
    assert(summary.converged)
  }

  test("SummarySink should clear events") {
    val sink = SummarySink()

    sink.record(IterationStarted(0))
    sink.record(IterationCompleted(0, 100.0, 1.5, 50, 100))

    assert(sink.size == 2)

    sink.clear()

    assert(sink.size == 0)
    assert(sink.getEvents.isEmpty)
  }

  test("SummarySink.noop should discard events") {
    val sink = SummarySink.noop()

    sink.record(IterationStarted(0))
    sink.record(IterationCompleted(0, 100.0, 1.5, 50, 100))

    // Noop sink discards events
    assert(sink.size == 0)
  }

  test("ClusteringSummary should filter events by type") {
    val events = Seq(
      IterationStarted(0),
      IterationCompleted(0, 100.0, 1.5, 50, 100),
      IterationStarted(1),
      IterationCompleted(1, 90.0, 0.8, 20, 80),
      WarningEvent(1, "Test warning"),
      ConvergenceDetected(2, "max_iterations")
    )

    val summary = ClusteringSummary(events, 0, 1000)

    assert(summary.iterations.size == 2)
    assert(summary.warnings.size == 1)
    assert(summary.convergence.isDefined)
    assert(summary.eventsOfType[IterationStarted].size == 2)
  }

  test("ClusteringSummary should compute iteration metrics") {
    val events = Seq(
      IterationCompleted(0, 100.0, 1.5, 50, 100),
      IterationCompleted(1, 90.0, 0.8, 20, 80),
      IterationCompleted(2, 85.0, 0.3, 5, 60)
    )

    val summary = ClusteringSummary(events, 0, 1000)

    assert(summary.numIterations == 3)
    assert(summary.finalCost.contains(85.0))
    assert(summary.costImprovement.contains(15.0))
    assert(math.abs(summary.costImprovementPercent.get - 15.0) < 0.01)
    assert(math.abs(summary.avgIterationDuration - 80.0) < 0.01)
  }

  test("ClusteringSummary should compute average center movement") {
    val events = Seq(
      IterationCompleted(0, 100.0, 2.0, 50, 100),
      IterationCompleted(1, 90.0, 1.0, 20, 80),
      IterationCompleted(2, 85.0, 0.5, 5, 60)
    )

    val summary = ClusteringSummary(events, 0, 1000)

    assert(math.abs(summary.avgCenterMovement - 1.166) < 0.01)
  }

  test("ClusteringSummary should compute total assignment changes") {
    val events = Seq(
      IterationCompleted(0, 100.0, 2.0, 50, 100),
      IterationCompleted(1, 90.0, 1.0, 20, 80),
      IterationCompleted(2, 85.0, 0.5, 5, 60)
    )

    val summary = ClusteringSummary(events, 0, 1000)

    assert(summary.totalAssignmentChanges == 75)
  }

  test("ClusteringSummary should detect convergence") {
    val eventsWithConv = Seq(
      IterationCompleted(0, 100.0, 1.5, 50, 100),
      ConvergenceDetected(1, "cost_delta_below_threshold")
    )

    val summaryWithConv = ClusteringSummary(eventsWithConv, 0, 1000)
    assert(summaryWithConv.converged)
    assert(summaryWithConv.convergenceReason.contains("cost_delta_below_threshold"))

    val eventsWithoutConv = Seq(
      IterationCompleted(0, 100.0, 1.5, 50, 100)
    )

    val summaryWithoutConv = ClusteringSummary(eventsWithoutConv, 0, 1000)
    assert(!summaryWithoutConv.converged)
    assert(summaryWithoutConv.convergenceReason.isEmpty)
  }

  test("ClusteringSummary should track empty cluster events") {
    val events = Seq(
      IterationCompleted(0, 100.0, 1.5, 50, 100),
      EmptyClustersDetected(1, Set(2, 5), "reseeded"),
      IterationCompleted(1, 90.0, 0.8, 20, 80),
      EmptyClustersDetected(2, Set(3), "ignored")
    )

    val summary = ClusteringSummary(events, 0, 1000)

    assert(summary.numEmptyClusterEvents == 2)
  }

  test("ClusteringSummary should track warnings") {
    val events = Seq(
      IterationCompleted(0, 100.0, 1.5, 50, 100),
      WarningEvent(0, "Cluster 2 is very small", "low"),
      IterationCompleted(1, 90.0, 0.8, 20, 80),
      WarningEvent(1, "High variance in cluster sizes", "medium")
    )

    val summary = ClusteringSummary(events, 0, 1000)

    assert(summary.warnings.size == 2)
    assert(summary.warnings.head.message == "Cluster 2 is very small")
  }

  test("ClusteringSummary should track initialization") {
    val events = Seq(
      InitializationCompleted("kmeans++", 500),
      IterationCompleted(0, 100.0, 1.5, 50, 100)
    )

    val summary = ClusteringSummary(events, 0, 1000)

    assert(summary.initialization.isDefined)
    assert(summary.initialization.get.method == "kmeans++")
    assert(summary.initialization.get.duration == 500)
  }

  test("ClusteringSummary should track training completion") {
    val events = Seq(
      IterationCompleted(0, 100.0, 1.5, 50, 100),
      IterationCompleted(1, 90.0, 0.8, 20, 80),
      TrainingCompleted(2, 85.0, 1000)
    )

    val summary = ClusteringSummary(events, 0, 1000)

    assert(summary.completion.isDefined)
    assert(summary.completion.get.totalIterations == 2)
    assert(summary.completion.get.finalCost == 85.0)
  }

  test("ClusteringSummary.report should generate readable output") {
    val events = Seq(
      InitializationCompleted("kmeans++", 500),
      IterationCompleted(0, 100.0, 2.0, 50, 100),
      IterationCompleted(1, 90.0, 1.0, 20, 80),
      IterationCompleted(2, 85.0, 0.5, 5, 60),
      ConvergenceDetected(3, "cost_delta_below_threshold"),
      WarningEvent(1, "Test warning", "low")
    )

    val summary = ClusteringSummary(events, 0, 1000)
    val report  = summary.report

    assert(report.contains("Clustering Summary"))
    assert(report.contains("Iterations: 3"))
    assert(report.contains("Final Cost"))
    assert(report.contains("Cost Improvement"))
    assert(report.contains("Converged: Yes"))
    assert(report.contains("Warnings"))
    assert(report.contains("Per-Iteration Metrics"))
  }

  test("ClusteringSummary should handle empty events") {
    val summary = ClusteringSummary(Seq.empty, 0, 1000)

    assert(summary.numIterations == 0)
    assert(summary.finalCost.isEmpty)
    assert(summary.costImprovement.isEmpty)
    assert(summary.avgIterationDuration == 0.0)
    assert(summary.avgCenterMovement == 0.0)
    assert(!summary.converged)
  }

  test("ClusteringSummary should compute total duration") {
    val summary = ClusteringSummary(Seq.empty, 1000, 5000)

    assert(summary.totalDuration == 4000)
  }

  test("IterationTracker should record start and complete") {
    val sink    = SummarySink()
    val tracker = IterationTracker(sink, 0)

    // Start event should be recorded
    assert(sink.size == 1)
    assert(sink.getEvents.head.isInstanceOf[IterationStarted])

    // Complete the iteration
    tracker.complete(cost = 100.0, centerMovement = 1.5, assignmentChanges = 50)

    // Complete event should be recorded
    assert(sink.size == 2)
    val events = sink.getEvents
    assert(events(1).isInstanceOf[IterationCompleted])

    val completed = events(1).asInstanceOf[IterationCompleted]
    assert(completed.cost == 100.0)
    assert(completed.centerMovement == 1.5)
    assert(completed.assignmentChanges == 50)
    assert(completed.duration >= 0)
  }

  test("IterationTracker should measure duration") {
    val sink    = SummarySink()
    val tracker = IterationTracker(sink, 0)

    Thread.sleep(50) // Simulate work

    tracker.complete(100.0, 1.5, 50)

    val completed = sink.getEvents(1).asInstanceOf[IterationCompleted]
    assert(completed.duration >= 50) // Should be at least 50ms
  }

  test("ClusteringEvent should have correct event types") {
    assert(IterationStarted(0).eventType == "iteration_started")
    assert(IterationCompleted(0, 100.0, 1.5, 50, 100).eventType == "iteration_completed")
    assert(ConvergenceDetected(0, "test").eventType == "convergence_detected")
    assert(EmptyClustersDetected(0, Set(1), "test").eventType == "empty_clusters_detected")
    assert(WarningEvent(0, "test").eventType == "warning")
    assert(InitializationCompleted("test", 100).eventType == "initialization_completed")
    assert(TrainingCompleted(0, 100.0, 1000).eventType == "training_completed")
  }

  test("SummarySink should be thread-safe") {
    val sink = SummarySink()

    // Simulate concurrent access
    val threads = (0 until 10).map { i =>
      new Thread(() => {
        for (j <- 0 until 10) {
          sink.record(IterationCompleted(i * 10 + j, 100.0, 1.0, 10, 50))
        }
      })
    }

    threads.foreach(_.start())
    threads.foreach(_.join())

    // Should have recorded all 100 events
    assert(sink.size == 100)
  }

  test("SummarySink should be serializable") {
    val sink = SummarySink()
    sink.record(IterationStarted(0))

    val stream = new java.io.ByteArrayOutputStream()
    val oos    = new java.io.ObjectOutputStream(stream)
    oos.writeObject(sink)
    oos.close()

    val bytes = stream.toByteArray
    assert(bytes.nonEmpty)
  }

  test("ClusteringSummary should be serializable") {
    val summary = ClusteringSummary(
      Seq(IterationCompleted(0, 100.0, 1.0, 10, 100)),
      0,
      1000
    )

    val stream = new java.io.ByteArrayOutputStream()
    val oos    = new java.io.ObjectOutputStream(stream)
    oos.writeObject(summary)
    oos.close()

    val bytes = stream.toByteArray
    assert(bytes.nonEmpty)
  }
}
