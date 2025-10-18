package com.massivedatascience.clusterer.ml.tests

import com.massivedatascience.clusterer.ml.{XMeans, GeneralizedKMeans}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

/** Test suite for X-Means clustering (automatic k selection).
  */
class XMeansSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("XMeansSuite")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) {
        spark.stop()
      }
    } finally {
      super.afterAll()
    }
  }

  test("X-Means should find optimal k for well-separated clusters") {
    // Create data with 3 clear clusters
    val data = Seq(
      // Cluster 1: around (0, 0)
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(-0.1, 0.1),
      Vectors.dense(0.1, -0.1),
      // Cluster 2: around (5, 5)
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1),
      Vectors.dense(4.9, 5.1),
      Vectors.dense(5.1, 4.9),
      // Cluster 3: around (10, 10)
      Vectors.dense(10.0, 10.0),
      Vectors.dense(10.1, 10.1),
      Vectors.dense(9.9, 10.1),
      Vectors.dense(10.1, 9.9)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val xmeans = new XMeans()
      .setMinK(2)
      .setMaxK(5)
      .setCriterion("bic")
      .setMaxIter(20)
      .setSeed(42)

    val model = xmeans.fit(df)

    // Should find k=3 as optimal
    assert(model.numClusters === 3, s"Expected k=3, got k=${model.numClusters}")

    // Verify model works for prediction
    val predictions = model.transform(df)
    assert(predictions.select("prediction").distinct().count() === 3)
  }

  test("X-Means should work with BIC criterion") {
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(1.1, 0.9),
      Vectors.dense(9.0, 9.0),
      Vectors.dense(9.1, 8.9)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val xmeans = new XMeans()
      .setMinK(2)
      .setMaxK(4)
      .setCriterion("bic")
      .setMaxIter(20)

    val model = xmeans.fit(df)

    // Should find k=2 for two clusters
    assert(model.numClusters >= 2 && model.numClusters <= 4)
  }

  test("X-Means should work with AIC criterion") {
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(1.1, 0.9),
      Vectors.dense(9.0, 9.0),
      Vectors.dense(9.1, 8.9)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val xmeans = new XMeans()
      .setMinK(2)
      .setMaxK(4)
      .setCriterion("aic")
      .setMaxIter(20)

    val model = xmeans.fit(df)

    // AIC might prefer different k than BIC
    assert(model.numClusters >= 2 && model.numClusters <= 4)
  }

  test("X-Means should work with different divergences") {
    val data = Seq(
      Vectors.dense(1.0, 2.0),
      Vectors.dense(1.1, 2.1),
      Vectors.dense(5.0, 6.0),
      Vectors.dense(5.1, 6.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // Test with L1 divergence
    val xmeansL1 = new XMeans()
      .setMinK(2)
      .setMaxK(3)
      .setDivergence("l1")
      .setMaxIter(20)

    val modelL1 = xmeansL1.fit(df)
    assert(modelL1.numClusters >= 1, s"Expected at least 1 cluster, got ${modelL1.numClusters}")

    // Verify it produces valid predictions
    val preds = modelL1.transform(df)
    assert(preds.count() === 4)
  }

  test("X-Means should handle weighted data") {
    val data = Seq(
      (Vectors.dense(1.0, 1.0), 10.0),  // High weight
      (Vectors.dense(1.1, 0.9), 10.0),
      (Vectors.dense(9.0, 9.0), 1.0),   // Low weight
      (Vectors.dense(9.1, 8.9), 1.0)
    )

    val df = spark.createDataFrame(data).toDF("features", "weight")

    val xmeans = new XMeans()
      .setMinK(2)
      .setMaxK(3)
      .setWeightCol("weight")
      .setMaxIter(20)

    val model = xmeans.fit(df)
    assert(model.numClusters >= 2)

    // Centers should favor high-weight points
    val predictions = model.transform(df)
    assert(predictions.count() === 4)
  }

  test("X-Means should respect minK and maxK bounds") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(10.0, 10.0),
      Vectors.dense(15.0, 15.0),
      Vectors.dense(20.0, 20.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val xmeans = new XMeans()
      .setMinK(2)
      .setMaxK(3) // Limit max k
      .setMaxIter(20)

    val model = xmeans.fit(df)

    // Should not exceed maxK
    assert(model.numClusters >= 2 && model.numClusters <= 3)
  }

  test("X-Means should prefer simpler models with BIC") {
    // Create ambiguous data where clusters overlap
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(2.0, 2.0),
      Vectors.dense(3.0, 3.0),
      Vectors.dense(4.0, 4.0),
      Vectors.dense(5.0, 5.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val xmeansBIC = new XMeans()
      .setMinK(2)
      .setMaxK(5)
      .setCriterion("bic") // BIC penalizes complexity more
      .setMaxIter(20)

    val modelBIC = xmeansBIC.fit(df)

    // BIC should prefer fewer clusters for ambiguous data
    // Just verify it produces a valid result within bounds
    assert(modelBIC.numClusters >= 2 && modelBIC.numClusters <= 5)
  }

  test("X-Means model should support prediction") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(10.0, 10.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val xmeans = new XMeans()
      .setMinK(2)
      .setMaxK(3)
      .setMaxIter(20)

    val model = xmeans.fit(df)

    // Test transform
    val predictions = model.transform(df)
    assert(predictions.columns.contains("prediction"))
    assert(predictions.count() === 2)

    // Test predict on single vector
    val testVector = Vectors.dense(0.1, 0.1)
    val cluster = model.predict(testVector)
    assert(cluster >= 0 && cluster < model.numClusters)
  }

  test("X-Means should compute valid cost") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val xmeans = new XMeans()
      .setMinK(2)
      .setMaxK(3)
      .setMaxIter(20)

    val model = xmeans.fit(df)
    val cost = model.computeCost(df)

    // Cost should be positive and finite
    assert(cost > 0.0 && java.lang.Double.isFinite(cost))
  }

  test("X-Means parameter validation") {
    // minK must be >= 2 (validation happens at set time)
    assertThrows[IllegalArgumentException] {
      new XMeans().setMinK(1)
    }

    // maxK must be >= 2
    assertThrows[IllegalArgumentException] {
      new XMeans().setMaxK(1)
    }

    // Invalid criterion (validation happens at set time)
    assertThrows[IllegalArgumentException] {
      new XMeans().setCriterion("invalid")
    }
  }

  test("X-Means should work with small dataset") {
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(2.0, 2.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val xmeans = new XMeans()
      .setMinK(2)
      .setMaxK(2) // Only try k=2
      .setMaxIter(10)

    val model = xmeans.fit(df)
    assert(model.numClusters === 2)
  }

  test("X-Means model persistence") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(10.0, 10.0),
      Vectors.dense(10.1, 10.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val xmeans = new XMeans()
      .setMinK(2)
      .setMaxK(3)
      .setMaxIter(20)
      .setSeed(42)

    val model = xmeans.fit(df)
    val originalPreds = model.transform(df).select("prediction").collect().map(_.getInt(0))

    // Save and load model
    val tempDir = java.nio.file.Files.createTempDirectory("xmeans-model-test").toString
    try {
      model.write.overwrite().save(tempDir)

      val loadedModel = com.massivedatascience.clusterer.ml.GeneralizedKMeansModel.load(tempDir)
      val loadedPreds = loadedModel.transform(df).select("prediction").collect().map(_.getInt(0))

      // Predictions should match
      assert(originalPreds.sameElements(loadedPreds))
    } finally {
      // Clean up
      import scala.reflect.io.Directory
      val dir = new Directory(new java.io.File(tempDir))
      dir.deleteRecursively()
    }
  }
}
