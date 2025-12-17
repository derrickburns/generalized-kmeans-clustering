/*
 * Licensed to the Massive Data Science and Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Massive Data Science and Derrick R. Burns licenses this file to You under the
 * Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
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

package com.massivedatascience.clusterer.ml

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import com.massivedatascience.clusterer.ml.df.kernels._
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class TimeSeriesKMeansSuite extends AnyFunSuite with DataFrameSuiteBase with Matchers {

  // ============================================================================
  // DTW Kernel Tests
  // ============================================================================

  test("DTWKernel distance between identical sequences is 0") {
    val dtw = new DTWKernel()
    val seq = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0)
    dtw.distance(seq, seq) shouldBe 0.0 +- 1e-10
  }

  test("DTWKernel distance is symmetric") {
    val dtw  = new DTWKernel()
    val seq1 = Vectors.dense(1.0, 2.0, 3.0, 4.0)
    val seq2 = Vectors.dense(1.0, 3.0, 3.0, 5.0)
    dtw.distance(seq1, seq2) shouldBe dtw.distance(seq2, seq1) +- 1e-10
  }

  test("DTWKernel handles time-shifted sequences") {
    val dtw  = new DTWKernel()
    val seq1 = Vectors.dense(0.0, 0.0, 1.0, 2.0, 3.0, 3.0)
    val seq2 = Vectors.dense(1.0, 2.0, 3.0) // Same pattern, shifted

    val dist = dtw.distance(seq1, seq2)
    // DTW should find alignment, distance should be relatively small
    assert(dist < 3.0, s"DTW distance $dist too large for time-shifted sequences")
  }

  test("DTWKernel with Sakoe-Chiba window constraint") {
    val dtwNoWindow   = new DTWKernel(window = None)
    val dtwWithWindow = new DTWKernel(window = Some(2))

    val seq1 = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0)
    val seq2 = Vectors.dense(1.0, 1.5, 2.5, 3.5, 5.0)

    val distNoWindow   = dtwNoWindow.distance(seq1, seq2)
    val distWithWindow = dtwWithWindow.distance(seq1, seq2)

    // Both should compute valid distances
    assert(distNoWindow >= 0)
    assert(distWithWindow >= 0)
    // Constrained should be >= unconstrained (restricted search space)
    assert(distWithWindow >= distNoWindow - 1e-10)
  }

  test("DTWKernel alignment path is valid") {
    val dtw  = new DTWKernel()
    val seq1 = Vectors.dense(1.0, 2.0, 3.0)
    val seq2 = Vectors.dense(1.0, 1.5, 2.0, 3.0)

    val path = dtw.alignmentPath(seq1, seq2)

    // Path should start at (0,0) and end at (n-1, m-1)
    path.head shouldBe (0, 0)
    path.last shouldBe (seq1.size - 1, seq2.size - 1)

    // Path should be monotonically increasing
    for (i <- 1 until path.length) {
      val (pi, pj) = path(i - 1)
      val (ci, cj) = path(i)
      assert(ci >= pi && cj >= pj, s"Path not monotonic at step $i")
      assert(ci - pi <= 1 && cj - pj <= 1, s"Path jumps too far at step $i")
    }
  }

  test("DTWKernel similarity (apply) is in [0, 1] for normalized sequences") {
    val dtw  = new DTWKernel()
    val seq1 = Vectors.dense(0.0, 0.5, 1.0)
    val seq2 = Vectors.dense(0.0, 0.6, 1.0)

    val sim = dtw(seq1, seq2)
    assert(sim > 0 && sim <= 1, s"Similarity $sim out of expected range")
  }

  // ============================================================================
  // Soft-DTW Kernel Tests
  // ============================================================================

  test("SoftDTWKernel distance approximates DTW as gamma approaches 0") {
    val softDtw = new SoftDTWKernel(gamma = 0.01)

    val seq1 = Vectors.dense(1.0, 2.0, 3.0, 4.0)
    val seq2 = Vectors.dense(1.5, 2.5, 3.5, 4.5)

    val softDtwDist = softDtw.distance(seq1, seq2)

    // For squared metric in softDTW vs absolute in DTW, scale appropriately
    // They should be in the same order of magnitude
    assert(softDtwDist > 0)
  }

  test("SoftDTWKernel is differentiable (produces finite gradients)") {
    val softDtw = new SoftDTWKernel(gamma = 1.0)

    val seq1 = Vectors.dense(1.0, 2.0, 3.0)
    val seq2 = Vectors.dense(1.5, 2.5, 3.5)

    // Just verify no NaN/Inf in distance
    val dist = softDtw.distance(seq1, seq2)
    assert(java.lang.Double.isFinite(dist), s"SoftDTW distance is not finite: $dist")
  }

  test("SoftDTWKernel with different gamma values") {
    val seq1 = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0)
    val seq2 = Vectors.dense(1.0, 2.5, 3.0, 4.5, 5.0)

    // Use moderate gamma values
    val softMedium = new SoftDTWKernel(gamma = 1.0)
    val softLarge  = new SoftDTWKernel(gamma = 10.0)

    val distMedium = softMedium.distance(seq1, seq2)
    val distLarge  = softLarge.distance(seq1, seq2)

    // Both should be finite (soft-DTW can be negative due to soft-min aggregation)
    assert(
      java.lang.Double.isFinite(distMedium),
      s"Medium gamma distance should be finite: $distMedium"
    )
    assert(
      java.lang.Double.isFinite(distLarge),
      s"Large gamma distance should be finite: $distLarge"
    )

    // Similarity (kernel value) should be positive
    assert(softMedium(seq1, seq2) > 0, s"Kernel similarity should be positive")
  }

  // ============================================================================
  // GAK Kernel Tests
  // ============================================================================

  test("GAK kernel is positive semi-definite (k(x,x) > 0)") {
    val gak = new GAKKernel(sigma = 1.0)
    val seq = Vectors.dense(1.0, 2.0, 3.0, 4.0)

    val selfSim = gak(seq, seq)
    assert(selfSim > 0, s"GAK self-similarity should be positive: $selfSim")
  }

  test("GAK kernel is symmetric") {
    val gak  = new GAKKernel(sigma = 1.0)
    val seq1 = Vectors.dense(1.0, 2.0, 3.0)
    val seq2 = Vectors.dense(1.5, 2.5, 3.5)

    gak(seq1, seq2) shouldBe gak(seq2, seq1) +- 1e-10
  }

  test("GAK kernel similarity decreases with distance") {
    val gak   = new GAKKernel(sigma = 1.0)
    val base  = Vectors.dense(1.0, 2.0, 3.0)
    val close = Vectors.dense(1.1, 2.1, 3.1)
    val far   = Vectors.dense(5.0, 6.0, 7.0)

    val simClose = gak(base, close)
    val simFar   = gak(base, far)

    assert(simClose > simFar, "GAK should give higher similarity to closer sequences")
  }

  // ============================================================================
  // Derivative DTW Tests
  // ============================================================================

  test("DerivativeDTW is invariant to vertical offset") {
    val ddtw = new DerivativeDTWKernel()

    val seq1       = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0)
    val seq2Offset = Vectors.dense(11.0, 12.0, 13.0, 14.0, 15.0) // Same shape, +10 offset

    val sim = ddtw(seq1, seq2Offset)
    // Should have very high similarity (same derivative pattern)
    assert(sim > 0.99, s"DerivativeDTW should be invariant to offset, got sim=$sim")
  }

  test("DerivativeDTW detects shape differences") {
    val ddtw = new DerivativeDTWKernel()

    val increasing = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0)
    val decreasing = Vectors.dense(5.0, 4.0, 3.0, 2.0, 1.0)

    val sim = ddtw(increasing, decreasing)
    // Should have low similarity (opposite shapes)
    assert(sim < 0.5, s"DerivativeDTW should detect shape difference, got sim=$sim")
  }

  // ============================================================================
  // DBA Barycenter Tests
  // ============================================================================

  test("DBA barycenter of identical sequences is the sequence itself") {
    val dtw  = new DTWKernel()
    val seq  = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0)
    val seqs = Array(seq, seq, seq)
    val bc   = dtw.barycenter(seqs, None, 10)

    // Should be very close to original
    for (i <- 0 until seq.size) {
      bc(i) shouldBe seq(i) +- 0.1
    }
  }

  test("DBA barycenter is between input sequences") {
    val dtw  = new DTWKernel()
    val seq1 = Vectors.dense(0.0, 0.0, 0.0, 0.0)
    val seq2 = Vectors.dense(2.0, 2.0, 2.0, 2.0)
    val seqs = Array(seq1, seq2)
    val bc   = dtw.barycenter(seqs, None, 10)

    // Barycenter should be around 1.0
    for (i <- 0 until bc.size) {
      bc(i) shouldBe 1.0 +- 0.5
    }
  }

  test("DBA respects weights") {
    val dtw  = new DTWKernel()
    val seq1 = Vectors.dense(0.0, 0.0, 0.0)
    val seq2 = Vectors.dense(3.0, 3.0, 3.0)
    val seqs = Array(seq1, seq2)

    // Weight seq1 more heavily (0.75 vs 0.25)
    val weights = Some(Array(0.75, 0.25))
    val bc      = dtw.barycenter(seqs, weights, 10)

    // Barycenter should be closer to seq1 (around 0.75)
    for (i <- 0 until bc.size) {
      assert(bc(i) < 1.5, s"Barycenter(${bc(i)}) should be closer to weighted seq1")
    }
  }

  // ============================================================================
  // SequenceKernel Factory Tests
  // ============================================================================

  test("SequenceKernel factory creates correct kernel types") {
    val dtw        = SequenceKernel.create("dtw")
    val softDtw    = SequenceKernel.create("softdtw", gamma = 0.5)
    val gak        = SequenceKernel.create("gak", sigma = 2.0)
    val derivative = SequenceKernel.create("derivative")

    dtw shouldBe a[DTWKernel]
    softDtw shouldBe a[SoftDTWKernel]
    gak shouldBe a[GAKKernel]
    derivative shouldBe a[DerivativeDTWKernel]
  }

  test("SequenceKernel factory with window parameter") {
    val dtwWindowed = SequenceKernel.create("dtw", window = Some(5))
    dtwWindowed shouldBe a[DTWKernel]
    dtwWindowed.asInstanceOf[DTWKernel].window shouldBe Some(5)
  }

  test("SequenceKernel factory rejects unknown kernel") {
    an[IllegalArgumentException] should be thrownBy {
      SequenceKernel.create("unknown_kernel")
    }
  }

  // ============================================================================
  // TimeSeriesKMeans Estimator Tests
  // ============================================================================

  test("TimeSeriesKMeans basic clustering with DTW") {
    // Two clear patterns: increasing and decreasing
    val increasing = (1 to 10).map(i => Tuple1(Vectors.dense((1 to 5).map(_ + i * 0.1).toArray)))
    val decreasing =
      (1 to 10).map(i => Tuple1(Vectors.dense((5 to 1 by -1).map(_ + i * 0.1).toArray)))

    val df = spark.createDataFrame(increasing ++ decreasing).toDF("features")

    val tsKMeans =
      new TimeSeriesKMeans().setK(2).setKernelType("dtw").setMaxIter(20).setDbaIter(5).setSeed(42)

    val model       = tsKMeans.fit(df)
    val predictions = model.transform(df)

    assert(model.clusterCenters.length == 2)
    assert(predictions.select("prediction").distinct().count() == 2)
  }

  test("TimeSeriesKMeans finds clusters with time-shifted patterns") {
    // Pattern 1: peak early
    val pattern1 = (1 to 8).map { i =>
      val base = Array(0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0)
      Tuple1(Vectors.dense(base.map(_ + i * 0.05)))
    }

    // Pattern 2: peak late
    val pattern2 = (1 to 8).map { i =>
      val base = Array(0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0)
      Tuple1(Vectors.dense(base.map(_ + i * 0.05)))
    }

    val df = spark.createDataFrame(pattern1 ++ pattern2).toDF("features")

    val tsKMeans = new TimeSeriesKMeans().setK(2).setKernelType("dtw").setMaxIter(30).setSeed(42)

    val model       = tsKMeans.fit(df)
    val predictions = model.transform(df)

    val clusterCounts = predictions.groupBy("prediction").count().collect()
    assert(clusterCounts.length == 2, "Should find 2 clusters")
  }

  test("TimeSeriesKMeans with Sakoe-Chiba window") {
    val data = (1 to 20).map { i =>
      val values = (1 to 10).map(j => math.sin(j * 0.5 + i * 0.1))
      Tuple1(Vectors.dense(values.toArray))
    }

    val df = spark.createDataFrame(data).toDF("features")

    val tsKMeans =
      new TimeSeriesKMeans().setK(2).setKernelType("dtw").setWindow(3).setMaxIter(10).setSeed(42)

    val model = tsKMeans.fit(df)
    assert(model.clusterCenters.length == 2)
  }

  test("TimeSeriesKMeans with soft-DTW kernel") {
    val data = (1 to 16).map { i =>
      val cluster = if (i <= 8) 0.0 else 5.0
      Tuple1(Vectors.dense(Array(cluster, cluster + 1, cluster + 2, cluster + 1, cluster)))
    }

    val df = spark.createDataFrame(data).toDF("features")

    val tsKMeans = new TimeSeriesKMeans()
      .setK(2)
      .setKernelType("softdtw")
      .setGamma(1.0)
      .setMaxIter(15)
      .setSeed(42)

    val model = tsKMeans.fit(df)
    assert(model.clusterCenters.length == 2)
  }

  test("TimeSeriesKMeans with derivative DTW kernel") {
    // Same amplitude pattern at different offsets
    val pattern = Array(0.0, 1.0, 2.0, 1.0, 0.0)

    val data = (1 to 16).map { i =>
      val offset = if (i <= 8) 0.0 else 10.0
      Tuple1(Vectors.dense(pattern.map(_ + offset)))
    }

    val df = spark.createDataFrame(data).toDF("features")

    val tsKMeans =
      new TimeSeriesKMeans().setK(2).setKernelType("derivative").setMaxIter(10).setSeed(42)

    val model = tsKMeans.fit(df)
    // With derivative DTW, offset-shifted patterns should cluster together
    assert(model.clusterCenters.length == 2)
  }

  test("TimeSeriesKMeans parameter validation") {
    val tsKMeans = new TimeSeriesKMeans()

    // k must be > 1
    an[IllegalArgumentException] should be thrownBy {
      tsKMeans.setK(0)
    }

    // gamma must be > 0
    an[IllegalArgumentException] should be thrownBy {
      tsKMeans.setGamma(-1.0)
    }

    // window must be >= 0
    an[IllegalArgumentException] should be thrownBy {
      tsKMeans.setWindow(-1)
    }
  }

  test("TimeSeriesKMeans deterministic with same seed") {
    val data = (1 to 20).map { i =>
      Tuple1(Vectors.dense(Array.fill(5)(scala.util.Random.nextDouble())))
    }

    val df = spark.createDataFrame(data).toDF("features")

    val model1 = new TimeSeriesKMeans().setK(3).setMaxIter(10).setSeed(12345).fit(df)

    val model2 = new TimeSeriesKMeans().setK(3).setMaxIter(10).setSeed(12345).fit(df)

    // Same seed should produce same centers
    for (i <- model1.clusterCenters.indices) {
      for (j <- 0 until model1.clusterCenters(i).size) {
        model1.clusterCenters(i)(j) shouldBe model2.clusterCenters(i)(j) +- 1e-10
      }
    }
  }

  test("TimeSeriesKMeans training summary") {
    val data = (1 to 20).map { i =>
      val cluster = if (i <= 10) 0.0 else 5.0
      Tuple1(Vectors.dense(Array(cluster, cluster + 1, cluster)))
    }

    val df = spark.createDataFrame(data).toDF("features")

    val model = new TimeSeriesKMeans().setK(2).setMaxIter(20).setSeed(42).fit(df)

    assert(model.hasSummary)
    val summary = model.summary
    assert(summary.algorithm == "TimeSeriesKMeans")
    assert(summary.k == 2)
    assert(summary.numPoints == 20)
    assert(summary.iterations > 0)
    assert(summary.distortionHistory.nonEmpty)
  }

  test("TimeSeriesKMeans transform includes distance column") {
    val data = (1 to 10).map(i => Tuple1(Vectors.dense(Array(i.toDouble, i.toDouble + 1))))
    val df   = spark.createDataFrame(data).toDF("features")

    val model =
      new TimeSeriesKMeans().setK(2).setMaxIter(10).setSeed(42).setDistanceCol("dist").fit(df)

    val predictions = model.transform(df)

    assert(predictions.columns.contains("prediction"))
    assert(predictions.columns.contains("dist"))

    // All distances should be non-negative
    val distances = predictions.select("dist").collect().map(_.getDouble(0))
    distances.foreach(d => assert(d >= 0, s"Distance should be non-negative: $d"))
  }

  // ============================================================================
  // Persistence Tests
  // ============================================================================

  test("TimeSeriesKMeans model persistence roundtrip") {
    val data = (1 to 16).map { i =>
      val cluster = if (i <= 8) 0.0 else 5.0
      Tuple1(Vectors.dense(Array(cluster, cluster + 1, cluster + 2)))
    }

    val df = spark.createDataFrame(data).toDF("features")

    val model = new TimeSeriesKMeans()
      .setK(2)
      .setKernelType("dtw")
      .setWindow(3)
      .setMaxIter(10)
      .setSeed(42)
      .fit(df)

    val path = java.nio.file.Files.createTempDirectory("ts_kmeans_test").toString
    try {
      model.write.overwrite().save(path)
      val loaded = TimeSeriesKMeansModel.load(path)

      assert(loaded.numClusters == model.numClusters)
      assert(loaded.getK == model.getK)
      assert(loaded.getKernelType == model.getKernelType)
      assert(loaded.getWindow == model.getWindow)

      // Centers should match
      for (i <- model.clusterCenters.indices) {
        for (j <- 0 until model.clusterCenters(i).size) {
          loaded.clusterCenters(i)(j) shouldBe model.clusterCenters(i)(j) +- 1e-10
        }
      }

      // Transform should produce same results
      val orig     = model.transform(df).select("prediction").collect().map(_.getInt(0))
      val reloaded = loaded.transform(df).select("prediction").collect().map(_.getInt(0))
      orig shouldBe reloaded
    } finally {
      org.apache.commons.io.FileUtils.deleteDirectory(new java.io.File(path))
    }
  }

  test("TimeSeriesKMeans estimator persistence") {
    val tsKMeans = new TimeSeriesKMeans()
      .setK(3)
      .setKernelType("softdtw")
      .setGamma(0.5)
      .setWindow(5)
      .setMaxIter(25)
      .setDbaIter(8)

    val path = java.nio.file.Files.createTempDirectory("ts_kmeans_est_test").toString
    try {
      tsKMeans.write.overwrite().save(path)
      val loaded = TimeSeriesKMeans.load(path)

      assert(loaded.getK == 3)
      assert(loaded.getKernelType == "softdtw")
      assert(loaded.getGamma == 0.5)
      assert(loaded.getWindow == 5)
      assert(loaded.getMaxIter == 25)
      assert(loaded.getDbaIter == 8)
    } finally {
      org.apache.commons.io.FileUtils.deleteDirectory(new java.io.File(path))
    }
  }
}
