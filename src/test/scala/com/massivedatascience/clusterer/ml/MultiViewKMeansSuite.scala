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

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests for Multi-View K-Means clustering.
  *
  * Validates:
  *   - Multi-view clustering with different divergences
  *   - View weight handling
  *   - Combine strategies (weighted, max, min)
  *   - Model persistence
  *   - Parameter handling
  */
class MultiViewKMeansSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  private val spark: SparkSession = SparkSession
    .builder()
    .master("local[2]")
    .appName("MultiViewKMeansSuite")
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

  // Test data with two views
  private def twoViewTestData() = {
    Seq(
      // Cluster 0: similar in both views
      (Vectors.dense(0.0, 0.0), Vectors.dense(1.0, 1.0)),
      (Vectors.dense(0.1, 0.1), Vectors.dense(1.1, 0.9)),
      (Vectors.dense(0.2, 0.0), Vectors.dense(0.9, 1.1)),
      // Cluster 1: similar in both views
      (Vectors.dense(10.0, 10.0), Vectors.dense(5.0, 5.0)),
      (Vectors.dense(10.1, 10.1), Vectors.dense(5.1, 4.9)),
      (Vectors.dense(10.2, 10.0), Vectors.dense(4.9, 5.1))
    ).toDF("view1", "view2")
  }

  // Test data with three views
  private def threeViewTestData() = {
    Seq(
      // Cluster 0
      (Vectors.dense(0.0, 0.0), Vectors.dense(1.0), Vectors.dense(2.0, 2.0, 2.0)),
      (Vectors.dense(0.1, 0.1), Vectors.dense(1.1), Vectors.dense(2.1, 2.1, 2.1)),
      // Cluster 1
      (Vectors.dense(10.0, 10.0), Vectors.dense(5.0), Vectors.dense(8.0, 8.0, 8.0)),
      (Vectors.dense(10.1, 10.1), Vectors.dense(5.1), Vectors.dense(8.1, 8.1, 8.1))
    ).toDF("view1", "view2", "view3")
  }

  private def withTempDir(prefix: String)(f: String => Unit): Unit = {
    val dir = Files.createTempDirectory(prefix).toFile
    try {
      f(dir.getCanonicalPath)
    } finally {
      deleteRecursively(dir)
    }
  }

  private def deleteRecursively(file: java.io.File): Unit = {
    if (file.isDirectory) {
      file.listFiles().foreach(deleteRecursively)
    }
    val _ = file.delete()
  }

  // ========== Basic Tests ==========

  test("MultiViewKMeans with two views") {
    val df = twoViewTestData()

    val views = Seq(
      ViewSpec("view1", weight = 1.0, divergence = "squaredEuclidean"),
      ViewSpec("view2", weight = 1.0, divergence = "squaredEuclidean")
    )

    val mvkm = new MultiViewKMeans().setK(2).setViews(views).setMaxIter(10).setSeed(42L)

    val model = mvkm.fit(df)

    model.numClusters shouldBe 2
    model.numViews shouldBe 2

    val predictions = model.transform(df)
    predictions.select("prediction").distinct().count() shouldBe 2
  }

  test("MultiViewKMeans with three views") {
    val df = threeViewTestData()

    val views = Seq(
      ViewSpec("view1"),
      ViewSpec("view2"),
      ViewSpec("view3")
    )

    val mvkm = new MultiViewKMeans().setK(2).setViews(views).setMaxIter(10).setSeed(42L)

    val model = mvkm.fit(df)

    model.numClusters shouldBe 2
    model.numViews shouldBe 3

    // Check that we can get centers for each view
    model.getCentersForView(0).length shouldBe 2
    model.getCentersForView(1).length shouldBe 2
    model.getCentersForView(2).length shouldBe 2

    // Check getting centers by column name
    model.getCentersForView("view1").length shouldBe 2
  }

  test("MultiViewKMeans with different divergences per view") {
    val df = twoViewTestData()

    val views = Seq(
      ViewSpec("view1", weight = 1.0, divergence = "squaredEuclidean"),
      ViewSpec("view2", weight = 1.0, divergence = "spherical")
    )

    val mvkm = new MultiViewKMeans().setK(2).setViews(views).setMaxIter(10).setSeed(42L)

    val model = mvkm.fit(df)

    model.numClusters shouldBe 2

    val predictions = model.transform(df)
    predictions.count() shouldBe 6
  }

  // ========== Weight Tests ==========

  test("MultiViewKMeans with different view weights") {
    val df = twoViewTestData()

    val views = Seq(
      ViewSpec("view1", weight = 2.0), // Dominant view
      ViewSpec("view2", weight = 0.5)
    )

    val mvkm = new MultiViewKMeans().setK(2).setViews(views).setMaxIter(10).setSeed(42L)

    val model = mvkm.fit(df)
    model.numClusters shouldBe 2

    // Normalized weights should sum to 1
    model.weights.sum should be(1.0 +- 0.001)
    model.weights(0) should be > model.weights(1)
  }

  test("MultiViewKMeans normalizeWeights=false preserves original weights") {
    val df = twoViewTestData()

    val views = Seq(
      ViewSpec("view1", weight = 2.0),
      ViewSpec("view2", weight = 3.0)
    )

    val mvkm = new MultiViewKMeans()
      .setK(2)
      .setViews(views)
      .setMaxIter(10)
      .setSeed(42L)
      .setNormalizeWeights(false)

    val model = mvkm.fit(df)

    model.weights(0) shouldBe 2.0
    model.weights(1) shouldBe 3.0
  }

  // ========== Combine Strategy Tests ==========

  test("MultiViewKMeans with weighted combine strategy") {
    val df = twoViewTestData()

    val views = Seq(
      ViewSpec("view1"),
      ViewSpec("view2")
    )

    val mvkm = new MultiViewKMeans()
      .setK(2)
      .setViews(views)
      .setCombineStrategy("weighted")
      .setMaxIter(10)
      .setSeed(42L)

    val model = mvkm.fit(df)
    model.numClusters shouldBe 2
  }

  test("MultiViewKMeans with max combine strategy") {
    val df = twoViewTestData()

    val views = Seq(
      ViewSpec("view1"),
      ViewSpec("view2")
    )

    val mvkm = new MultiViewKMeans()
      .setK(2)
      .setViews(views)
      .setCombineStrategy("max")
      .setMaxIter(10)
      .setSeed(42L)

    val model = mvkm.fit(df)
    model.numClusters shouldBe 2

    val predictions = model.transform(df)
    predictions.select("prediction").distinct().count() shouldBe 2
  }

  test("MultiViewKMeans with min combine strategy") {
    val df = twoViewTestData()

    val views = Seq(
      ViewSpec("view1"),
      ViewSpec("view2")
    )

    val mvkm = new MultiViewKMeans()
      .setK(2)
      .setViews(views)
      .setCombineStrategy("min")
      .setMaxIter(10)
      .setSeed(42L)

    val model = mvkm.fit(df)
    model.numClusters shouldBe 2
  }

  // ========== Training Summary Tests ==========

  test("MultiViewKMeans provides training summary") {
    val df = twoViewTestData()

    val views = Seq(
      ViewSpec("view1"),
      ViewSpec("view2")
    )

    val mvkm = new MultiViewKMeans().setK(2).setViews(views).setMaxIter(10).setSeed(42L)

    val model = mvkm.fit(df)

    model.trainingSummary shouldBe defined
    val summary = model.trainingSummary.get

    summary.algorithm shouldBe "MultiViewKMeans"
    summary.k shouldBe 2
    summary.iterations should be > 0
    summary.distortionHistory should not be empty
    summary.assignmentStrategy should include("multiview")
    summary.divergence should include("squaredEuclidean")
  }

  // ========== Persistence Tests ==========

  test("MultiViewKMeans model save and load") {
    withTempDir("mvkmeans-persist") { dir =>
      val df = twoViewTestData()

      val views = Seq(
        ViewSpec("view1", weight = 2.0, divergence = "squaredEuclidean"),
        ViewSpec("view2", weight = 1.0, divergence = "spherical")
      )

      val mvkm = new MultiViewKMeans()
        .setK(2)
        .setViews(views)
        .setMaxIter(10)
        .setSeed(42L)
        .setCombineStrategy("weighted")

      val model = mvkm.fit(df)
      model.write.overwrite().save(s"$dir/model")

      val loadedModel = MultiViewKMeansModel.load(s"$dir/model")

      loadedModel.numClusters shouldBe model.numClusters
      loadedModel.numViews shouldBe model.numViews
      loadedModel.viewSpecs.length shouldBe model.viewSpecs.length

      // Check view specs match
      loadedModel.viewSpecs.zip(model.viewSpecs).foreach { case (loaded, original) =>
        loaded.featuresCol shouldBe original.featuresCol
        loaded.divergence shouldBe original.divergence
      }

      // Check predictions match
      val origPred   = model.transform(df).select("prediction").collect()
      val loadedPred = loadedModel.transform(df).select("prediction").collect()
      origPred should contain theSameElementsAs loadedPred
    }
  }

  test("MultiViewKMeans estimator save and load") {
    withTempDir("mvkmeans-estimator") { dir =>
      val views = Seq(
        ViewSpec("view1", weight = 2.0, divergence = "kl"),
        ViewSpec("view2", weight = 1.0, divergence = "l1")
      )

      val mvkm = new MultiViewKMeans()
        .setK(5)
        .setViews(views)
        .setMaxIter(20)
        .setSeed(123L)
        .setCombineStrategy("max")
        .setNormalizeWeights(false)

      mvkm.write.overwrite().save(s"$dir/estimator")

      val loadedMvkm = MultiViewKMeans.load(s"$dir/estimator")

      loadedMvkm.getK shouldBe 5
      loadedMvkm.getMaxIter shouldBe 20
      loadedMvkm.getSeed shouldBe 123L
      loadedMvkm.getCombineStrategy shouldBe "max"
      loadedMvkm.getNormalizeWeights shouldBe false
      loadedMvkm.getViews.length shouldBe 2
      loadedMvkm.getViews(0).divergence shouldBe "kl"
      loadedMvkm.getViews(1).divergence shouldBe "l1"
    }
  }

  // ========== Parameter Validation Tests ==========

  test("MultiViewKMeans validates combineStrategy parameter") {
    val mvkm = new MultiViewKMeans()

    noException should be thrownBy mvkm.setCombineStrategy("weighted")
    noException should be thrownBy mvkm.setCombineStrategy("max")
    noException should be thrownBy mvkm.setCombineStrategy("min")

    an[IllegalArgumentException] should be thrownBy mvkm.setCombineStrategy("invalid")
  }

  test("MultiViewKMeans requires views to be set") {
    val df   = twoViewTestData()
    val mvkm = new MultiViewKMeans().setK(2)

    an[IllegalArgumentException] should be thrownBy mvkm.fit(df)
  }

  test("MultiViewKMeans validates view columns exist") {
    val df = twoViewTestData()

    val views = Seq(
      ViewSpec("view1"),
      ViewSpec("nonexistent_view")
    )

    val mvkm = new MultiViewKMeans().setK(2).setViews(views)

    an[IllegalArgumentException] should be thrownBy mvkm.fit(df)
  }

  test("ViewSpec validates weight is positive") {
    an[IllegalArgumentException] should be thrownBy ViewSpec("features", weight = 0.0)
    an[IllegalArgumentException] should be thrownBy ViewSpec("features", weight = -1.0)
  }

  test("ViewSpec validates featuresCol is not empty") {
    an[IllegalArgumentException] should be thrownBy ViewSpec("")
  }

  // ========== Determinism Tests ==========

  test("MultiViewKMeans is deterministic with same seed") {
    val df = twoViewTestData()

    val views = Seq(
      ViewSpec("view1"),
      ViewSpec("view2")
    )

    val mvkm1 = new MultiViewKMeans().setK(2).setViews(views).setMaxIter(10).setSeed(42L)
    val mvkm2 = new MultiViewKMeans().setK(2).setViews(views).setMaxIter(10).setSeed(42L)

    val model1 = mvkm1.fit(df)
    val model2 = mvkm2.fit(df)

    // Centers should match for each view
    model1.clusterCenters.zip(model2.clusterCenters).foreach { case (view1Centers, view2Centers) =>
      view1Centers.zip(view2Centers).foreach { case (c1, c2) =>
        c1.toArray should contain theSameElementsAs c2.toArray
      }
    }
  }

  // ========== Edge Cases ==========

  test("MultiViewKMeans rejects k=1") {
    an[IllegalArgumentException] should be thrownBy {
      new MultiViewKMeans().setK(1)
    }
  }

  test("MultiViewKMeans toString is informative") {
    val df = twoViewTestData()

    val views = Seq(
      ViewSpec("view1", divergence = "kl"),
      ViewSpec("view2", divergence = "spherical")
    )

    val mvkm  = new MultiViewKMeans().setK(2).setViews(views).setMaxIter(5).setSeed(42L)
    val model = mvkm.fit(df)

    val str = model.toString
    str should include("MultiViewKMeansModel")
    str should include("k=2")
    str should include("views=2")
  }

  test("MultiViewKMeans with single view behaves like GeneralizedKMeans") {
    val df = twoViewTestData()

    val views = Seq(ViewSpec("view1"))

    val mvkm = new MultiViewKMeans().setK(2).setViews(views).setMaxIter(10).setSeed(42L)

    val model = mvkm.fit(df)

    model.numClusters shouldBe 2
    model.numViews shouldBe 1

    val predictions = model.transform(df)
    predictions.select("prediction").distinct().count() shouldBe 2
  }

  test("MultiViewKMeans getCentersForView throws for invalid view index") {
    val df = twoViewTestData()

    val views = Seq(ViewSpec("view1"), ViewSpec("view2"))

    val mvkm  = new MultiViewKMeans().setK(2).setViews(views).setMaxIter(5).setSeed(42L)
    val model = mvkm.fit(df)

    an[IllegalArgumentException] should be thrownBy model.getCentersForView(5)
    an[IllegalArgumentException] should be thrownBy model.getCentersForView(-1)
    an[IllegalArgumentException] should be thrownBy model.getCentersForView("nonexistent")
  }
}
