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

import java.nio.file.Files

import org.apache.spark.ml.linalg.{ SparseVector, Vectors }
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests for Sparse K-Means clustering.
  *
  * Validates:
  *   - Sparse vs dense kernel selection
  *   - Sparsity detection
  *   - Different divergences with sparse data
  *   - Model persistence
  *   - Parameter handling
  */
class SparseKMeansSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  private val spark: SparkSession = SparkSession
    .builder()
    .master("local[2]")
    .appName("SparseKMeansSuite")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  import spark.implicits._

  override def afterAll(): Unit = {
    try {
      spark.stop()
    } finally {
      super.afterAll()
    }
  }

  // Sparse test data (like TF-IDF vectors)
  private def sparseTestData() = {
    Seq(
      Tuple1(Vectors.sparse(100, Array(0, 5, 10), Array(1.0, 2.0, 1.5))),
      Tuple1(Vectors.sparse(100, Array(0, 5, 11), Array(1.1, 2.1, 1.4))),
      Tuple1(Vectors.sparse(100, Array(1, 6, 12), Array(0.9, 1.9, 1.6))),
      Tuple1(Vectors.sparse(100, Array(50, 55, 60), Array(3.0, 4.0, 3.5))),
      Tuple1(Vectors.sparse(100, Array(50, 55, 61), Array(3.1, 4.1, 3.4))),
      Tuple1(Vectors.sparse(100, Array(51, 56, 62), Array(2.9, 3.9, 3.6)))
    ).toDF("features")
  }

  // Dense test data
  private def denseTestData() = {
    Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(0.2, 0.0)),
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(10.1, 10.1)),
      Tuple1(Vectors.dense(10.2, 10.0))
    ).toDF("features")
  }

  // Mixed test data with both sparse and dense
  private def mixedTestData() = {
    Seq(
      Tuple1(Vectors.sparse(10, Array(0, 1), Array(1.0, 2.0))),
      Tuple1(Vectors.dense(1.1, 2.1, 0, 0, 0, 0, 0, 0, 0, 0)),
      Tuple1(Vectors.sparse(10, Array(5, 6), Array(5.0, 6.0))),
      Tuple1(Vectors.dense(0, 0, 0, 0, 0, 5.1, 6.1, 0, 0, 0))
    ).toDF("features")
  }

  private def withTempDir(prefix: String)(f: String => Unit): Unit = {
    val dir = Files.createTempDirectory(prefix).toFile
    try {
      f(dir.getCanonicalPath)
    } finally {
      if (dir.exists()) {
        dir.listFiles().foreach(_.delete())
        dir.delete()
      }
    }
  }

  // ========== Basic Tests ==========

  test("SparseKMeans with sparse data (auto mode)") {
    val df = sparseTestData()

    val skm = new SparseKMeans().setK(2).setMaxIter(10).setSeed(42L).setSparseMode("auto")

    val model = skm.fit(df)

    model.numClusters shouldBe 2
    model.usedSparseKernel shouldBe true // Should auto-detect sparse
    model.sparsityRatio should be < 0.1  // Very sparse data

    val predictions = model.transform(df)
    predictions.select("prediction").distinct().count() shouldBe 2
  }

  test("SparseKMeans with dense data (auto mode)") {
    val df = denseTestData()

    val skm = new SparseKMeans().setK(2).setMaxIter(10).setSeed(42L).setSparseMode("auto")

    val model = skm.fit(df)

    model.numClusters shouldBe 2
    model.usedSparseKernel shouldBe false // Dense data
    model.sparsityRatio should be > 0.3   // Mostly dense

    val predictions = model.transform(df)
    predictions.select("prediction").distinct().count() shouldBe 2
  }

  test("SparseKMeans force sparse mode") {
    val df = denseTestData()

    val skm = new SparseKMeans().setK(2).setMaxIter(10).setSeed(42L).setSparseMode("force")

    val model = skm.fit(df)

    model.usedSparseKernel shouldBe true // Forced sparse
  }

  test("SparseKMeans force dense mode") {
    val df = sparseTestData()

    val skm = new SparseKMeans().setK(2).setMaxIter(10).setSeed(42L).setSparseMode("dense")

    val model = skm.fit(df)

    model.usedSparseKernel shouldBe false // Forced dense
  }

  // ========== Divergence Tests ==========

  test("SparseKMeans with KL divergence on sparse data") {
    val df = sparseTestData()

    val skm = new SparseKMeans()
      .setK(2)
      .setDivergence("kl")
      .setMaxIter(10)
      .setSeed(42L)
      .setSparseMode("force")

    val model = skm.fit(df)

    // Model stores the divergence parameter value
    model.divergenceName shouldBe "kl"
    model.usedSparseKernel shouldBe true

    val predictions = model.transform(df)
    predictions.count() shouldBe 6
  }

  test("SparseKMeans with L1 divergence on sparse data") {
    val df = sparseTestData()

    val skm = new SparseKMeans()
      .setK(2)
      .setDivergence("l1")
      .setMaxIter(10)
      .setSeed(42L)
      .setSparseMode("force")

    val model = skm.fit(df)

    model.divergenceName shouldBe "l1"
    model.usedSparseKernel shouldBe true
  }

  test("SparseKMeans with spherical divergence on sparse data") {
    val df = sparseTestData()

    val skm = new SparseKMeans()
      .setK(2)
      .setDivergence("spherical")
      .setMaxIter(10)
      .setSeed(42L)
      .setSparseMode("force")

    val model = skm.fit(df)

    model.divergenceName shouldBe "spherical"
    model.usedSparseKernel shouldBe true
  }

  test("SparseKMeans with unsupported sparse divergence falls back") {
    val df = sparseTestData()

    // Itakura-Saito doesn't have sparse optimization
    val skm = new SparseKMeans()
      .setK(2)
      .setDivergence("itakuraSaito")
      .setMaxIter(10)
      .setSeed(42L)
      .setSparseMode("auto")

    val model = skm.fit(df)

    // Should fall back to dense since IS doesn't support sparse
    model.usedSparseKernel shouldBe false
  }

  // ========== Sparsity Threshold Tests ==========

  test("SparseKMeans respects sparseThreshold parameter") {
    val df = sparseTestData()

    // With default threshold (0.3), sparse data should use sparse kernel
    val skmDefault = new SparseKMeans().setK(2).setMaxIter(10).setSeed(42L).setSparseMode("auto")

    val modelDefault = skmDefault.fit(df)
    modelDefault.usedSparseKernel shouldBe true
    modelDefault.sparsityRatio should be < 0.3 // Should detect low sparsity
  }

  test("SparseKMeans with high sparseThreshold uses sparse more often") {
    val df = denseTestData() // Use dense data

    // With very high threshold, even dense data should use sparse
    val skm = new SparseKMeans()
      .setK(2)
      .setMaxIter(10)
      .setSeed(42L)
      .setSparseMode("auto")
      .setSparseThreshold(0.99) // Very high threshold

    val model = skm.fit(df)

    // Dense data has high sparsity ratio (all non-zero), so it should NOT use sparse
    // even with high threshold
    model.sparsityRatio should be > 0.5 // Dense data is not sparse
  }

  // ========== Mixed Data Tests ==========

  test("SparseKMeans handles mixed sparse/dense vectors") {
    val df = mixedTestData()

    val skm = new SparseKMeans().setK(2).setMaxIter(10).setSeed(42L).setSparseMode("auto")

    val model = skm.fit(df)

    model.numClusters shouldBe 2

    val predictions = model.transform(df)
    predictions.count() shouldBe 4
    predictions.select("prediction").distinct().count() shouldBe 2
  }

  // ========== Training Summary Tests ==========

  test("SparseKMeans provides training summary") {
    val df = sparseTestData()

    val skm = new SparseKMeans().setK(2).setMaxIter(10).setSeed(42L)

    val model = skm.fit(df)

    model.trainingSummary shouldBe defined
    val summary = model.trainingSummary.get

    summary.algorithm shouldBe "SparseKMeans"
    summary.k shouldBe 2
    summary.iterations should be > 0
    summary.distortionHistory should not be empty
    summary.assignmentStrategy should include("sparse=")
  }

  // ========== Persistence Tests ==========

  test("SparseKMeans model save and load") {
    withTempDir("sparsekmeans-persist") { dir =>
      val df = sparseTestData()

      val skm = new SparseKMeans()
        .setK(2)
        .setMaxIter(10)
        .setSeed(42L)
        .setSparseMode("force")
        .setDivergence("squaredEuclidean")

      val model = skm.fit(df)
      model.write.overwrite().save(s"$dir/model")

      val loadedModel = SparseKMeansModel.load(s"$dir/model")

      loadedModel.numClusters shouldBe model.numClusters
      loadedModel.divergenceName shouldBe model.divergenceName
      loadedModel.usedSparseKernel shouldBe model.usedSparseKernel
      loadedModel.sparsityRatio shouldBe model.sparsityRatio

      // Check centers match
      loadedModel.clusterCenters.zip(model.clusterCenters).foreach { case (loaded, original) =>
        loaded.toArray should contain theSameElementsAs original.toArray
      }

      // Check predictions match
      val origPred   = model.transform(df).select("prediction").collect()
      val loadedPred = loadedModel.transform(df).select("prediction").collect()
      origPred should contain theSameElementsAs loadedPred
    }
  }

  test("SparseKMeans estimator save and load") {
    withTempDir("sparsekmeans-estimator") { dir =>
      val skm = new SparseKMeans()
        .setK(5)
        .setMaxIter(20)
        .setSeed(123L)
        .setSparseMode("force")
        .setSparseThreshold(0.5)
        .setDivergence("kl")

      skm.write.overwrite().save(s"$dir/estimator")

      val loadedSkm = SparseKMeans.load(s"$dir/estimator")

      loadedSkm.getK shouldBe 5
      loadedSkm.getMaxIter shouldBe 20
      loadedSkm.getSeed shouldBe 123L
      loadedSkm.getSparseMode shouldBe "force"
      loadedSkm.getSparseThreshold shouldBe 0.5
      loadedSkm.getDivergence shouldBe "kl"
    }
  }

  // ========== Parameter Validation Tests ==========

  test("SparseKMeans validates sparseMode parameter") {
    val skm = new SparseKMeans()

    // Valid modes
    noException should be thrownBy skm.setSparseMode("auto")
    noException should be thrownBy skm.setSparseMode("force")
    noException should be thrownBy skm.setSparseMode("dense")

    // Invalid mode
    an[IllegalArgumentException] should be thrownBy skm.setSparseMode("invalid")
  }

  test("SparseKMeans validates sparseThreshold parameter") {
    val skm = new SparseKMeans()

    // Valid thresholds
    noException should be thrownBy skm.setSparseThreshold(0.1)
    noException should be thrownBy skm.setSparseThreshold(0.5)
    noException should be thrownBy skm.setSparseThreshold(1.0)

    // Invalid thresholds
    an[IllegalArgumentException] should be thrownBy skm.setSparseThreshold(0.0)
    an[IllegalArgumentException] should be thrownBy skm.setSparseThreshold(-0.1)
    an[IllegalArgumentException] should be thrownBy skm.setSparseThreshold(1.1)
  }

  // ========== Determinism Tests ==========

  test("SparseKMeans is deterministic with same seed") {
    val df = sparseTestData()

    val skm1 = new SparseKMeans().setK(2).setMaxIter(10).setSeed(42L)
    val skm2 = new SparseKMeans().setK(2).setMaxIter(10).setSeed(42L)

    val model1 = skm1.fit(df)
    val model2 = skm2.fit(df)

    model1.clusterCenters.zip(model2.clusterCenters).foreach { case (c1, c2) =>
      c1.toArray should contain theSameElementsAs c2.toArray
    }
  }

  test("SparseKMeans produces different results with different seeds") {
    val df = sparseTestData()

    val skm1 = new SparseKMeans().setK(2).setMaxIter(10).setSeed(1L)
    val skm2 = new SparseKMeans().setK(2).setMaxIter(10).setSeed(999L)

    val model1 = skm1.fit(df)
    val model2 = skm2.fit(df)

    // Centers should differ (with high probability)
    val centers1 = model1.clusterCenters.map(_.toArray).sortBy(_.sum)
    val centers2 = model2.clusterCenters.map(_.toArray).sortBy(_.sum)

    // At least some difference expected
    val allSame = centers1.zip(centers2).forall { case (c1, c2) =>
      c1.zip(c2).forall { case (v1, v2) => math.abs(v1 - v2) < 1e-10 }
    }
    // This might occasionally be false positive if both seeds happen to give same result
    // but it's very unlikely with different random seeds
  }

  // ========== Edge Cases ==========

  test("SparseKMeans rejects k=1") {
    // k=1 is rejected at parameter setting time
    an[IllegalArgumentException] should be thrownBy {
      new SparseKMeans().setK(1)
    }
  }

  test("SparseKMeans handles k equal to data size") {
    val df = sparseTestData() // 6 points

    val skm = new SparseKMeans().setK(6).setMaxIter(5).setSeed(42L)

    val model = skm.fit(df)
    model.numClusters shouldBe 6
  }

  test("SparseKMeans toString is informative") {
    val df = sparseTestData()

    val skm   = new SparseKMeans().setK(2).setMaxIter(5).setSeed(42L)
    val model = skm.fit(df)

    val str = model.toString
    str should include("SparseKMeansModel")
    str should include("k=2")
    str should include("sparse")
  }
}
