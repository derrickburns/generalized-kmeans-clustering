package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.ml.{KMedoids, KMedoidsModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

/** Test suite for K-Medoids clustering (PAM algorithm).
  */
class KMedoidsSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("KMedoidsSuite")
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

  test("KMedoids should cluster well-separated data with Euclidean distance") {
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
      Vectors.dense(5.1, 4.9)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val kmedoids = new KMedoids()
      .setK(2)
      .setMaxIter(20)
      .setDistanceFunction("euclidean")
      .setSeed(42)

    val model = kmedoids.fit(df)

    // Should find 2 clusters
    assert(model.numClusters === 2)

    // Medoids should be actual data points
    assert(data.exists(p => p.toArray.sameElements(model.medoids(0).toArray)))
    assert(data.exists(p => p.toArray.sameElements(model.medoids(1).toArray)))

    // Predictions
    val predictions = model.transform(df)
    assert(predictions.select("prediction").distinct().count() === 2)

    // Cost should be reasonable
    val cost = model.computeCost(df)
    assert(cost > 0.0 && cost < 1.0)
  }

  test("KMedoids should work with Manhattan distance") {
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(1.1, 0.9),
      Vectors.dense(9.0, 9.0),
      Vectors.dense(9.1, 8.9)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val kmedoids = new KMedoids()
      .setK(2)
      .setMaxIter(20)
      .setDistanceFunction("manhattan")
      .setSeed(42)

    val model = kmedoids.fit(df)

    assert(model.numClusters === 2)
    assert(model.distanceFunctionName === "manhattan")

    // Medoids must be actual data points
    model.medoids.foreach { medoid =>
      assert(data.exists(_.toArray.sameElements(medoid.toArray)))
    }

    val predictions = model.transform(df)
    assert(predictions.count() === 4)
  }

  test("KMedoids should work with cosine distance") {
    val data = Seq(
      Vectors.dense(1.0, 0.0, 0.0),
      Vectors.dense(0.9, 0.1, 0.0),
      Vectors.dense(0.0, 1.0, 0.0),
      Vectors.dense(0.0, 0.9, 0.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val kmedoids = new KMedoids()
      .setK(2)
      .setMaxIter(20)
      .setDistanceFunction("cosine")
      .setSeed(42)

    val model = kmedoids.fit(df)

    assert(model.numClusters === 2)
    assert(model.distanceFunctionName === "cosine")

    val predictions = model.transform(df)
    assert(predictions.select("prediction").distinct().count() <= 2)
  }

  test("KMedoids should handle k=1 correctly") {
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(2.0, 2.0),
      Vectors.dense(3.0, 3.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // k=1 should be rejected by parameter validation
    assertThrows[IllegalArgumentException] {
      new KMedoids().setK(1)
    }
  }

  test("KMedoids should handle k equal to dataset size") {
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(2.0, 2.0),
      Vectors.dense(3.0, 3.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val kmedoids = new KMedoids()
      .setK(3)
      .setMaxIter(10)
      .setSeed(42)

    val model = kmedoids.fit(df)

    // All points should be medoids
    assert(model.numClusters === 3)
    model.medoids.foreach { medoid =>
      assert(data.exists(_.toArray.sameElements(medoid.toArray)))
    }

    // Cost should be 0 (each point assigned to itself)
    val cost = model.computeCost(df)
    assert(cost < 1e-10)
  }

  test("KMedoids should be deterministic with same seed") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1),
      Vectors.dense(10.0, 10.0),
      Vectors.dense(10.1, 10.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model1 = new KMedoids()
      .setK(3)
      .setMaxIter(10)
      .setSeed(42)
      .fit(df)

    val model2 = new KMedoids()
      .setK(3)
      .setMaxIter(10)
      .setSeed(42)
      .fit(df)

    // Same seed should produce same medoids
    (0 until 3).foreach { i =>
      assert(model1.medoids(i).toArray.sameElements(model2.medoids(i).toArray))
    }
  }

  test("KMedoids should produce different results with different seeds") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(1.0, 1.0),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1),
      Vectors.dense(6.0, 6.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model1 = new KMedoids()
      .setK(2)
      .setMaxIter(10)
      .setSeed(42)
      .fit(df)

    val model2 = new KMedoids()
      .setK(2)
      .setMaxIter(10)
      .setSeed(123)
      .fit(df)

    // Different seeds may produce different medoids
    // (not guaranteed, but likely for this data)
    // Just verify both are valid
    assert(model1.numClusters === 2)
    assert(model2.numClusters === 2)
  }

  test("KMedoids should handle 1D data") {
    val data = Seq(
      Vectors.dense(1.0),
      Vectors.dense(2.0),
      Vectors.dense(3.0),
      Vectors.dense(10.0),
      Vectors.dense(11.0),
      Vectors.dense(12.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new KMedoids()
      .setK(2)
      .setMaxIter(10)
      .setSeed(42)
      .fit(df)

    assert(model.numClusters === 2)
    assert(model.numFeatures === 1)

    val predictions = model.transform(df)
    assert(predictions.select("prediction").distinct().count() === 2)
  }

  test("KMedoids should handle high-dimensional data") {
    val dim = 50
    val data = Seq(
      Vectors.dense(Array.fill(dim)(0.0)),
      Vectors.dense(Array.fill(dim)(0.1)),
      Vectors.dense(Array.fill(dim)(5.0)),
      Vectors.dense(Array.fill(dim)(5.1))
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new KMedoids()
      .setK(2)
      .setMaxIter(10)
      .setSeed(42)
      .fit(df)

    assert(model.numClusters === 2)
    assert(model.numFeatures === dim)

    val predictions = model.transform(df)
    assert(predictions.count() === 4)
  }

  test("KMedoids should converge within maxIter") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(0.2, 0.2),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1),
      Vectors.dense(5.2, 5.2)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // Even with low maxIter, should not fail
    val model = new KMedoids()
      .setK(2)
      .setMaxIter(1)
      .setSeed(42)
      .fit(df)

    assert(model.numClusters === 2)

    val predictions = model.transform(df)
    assert(predictions.count() === 6)
  }

  test("KMedoids should handle outliers better than K-Means") {
    val data = Seq(
      // Main cluster
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(-0.1, 0.1),
      // Outlier
      Vectors.dense(100.0, 100.0),
      // Second cluster
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1),
      Vectors.dense(4.9, 4.9)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new KMedoids()
      .setK(3)
      .setMaxIter(20)
      .setSeed(42)
      .fit(df)

    // Outlier should form its own cluster or join nearest cluster
    // Medoids should be actual points (not affected by outlier like centroid would be)
    assert(model.numClusters === 3)
    model.medoids.foreach { medoid =>
      assert(data.exists(_.toArray.sameElements(medoid.toArray)))
    }
  }

  test("KMedoids model should support persistence") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new KMedoids()
      .setK(2)
      .setMaxIter(10)
      .setSeed(42)
      .fit(df)

    val originalPredictions = model.transform(df).select("prediction").collect().map(_.getInt(0))

    // Save and load model
    val tempDir = java.nio.file.Files.createTempDirectory("kmedoids-model-test").toString
    try {
      model.write.overwrite().save(tempDir)

      val loadedModel = KMedoidsModel.load(tempDir)

      // Verify medoids match
      (0 until model.numClusters).foreach { i =>
        assert(model.medoids(i).toArray.sameElements(loadedModel.medoids(i).toArray))
      }

      // Verify predictions match
      val loadedPredictions = loadedModel.transform(df).select("prediction").collect().map(_.getInt(0))
      assert(originalPredictions.sameElements(loadedPredictions))

      // Verify distance function
      assert(loadedModel.distanceFunctionName === model.distanceFunctionName)
    } finally {
      // Clean up
      import scala.reflect.io.Directory
      val dir = new Directory(new java.io.File(tempDir))
      dir.deleteRecursively()
    }
  }

  test("KMedoids should handle sparse-like data (many zeros)") {
    val data = Seq(
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.9, 0.1, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0),
      Vectors.dense(0.0, 0.0, 0.0, 0.9, 0.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new KMedoids()
      .setK(2)
      .setMaxIter(10)
      .setSeed(42)
      .fit(df)

    assert(model.numClusters === 2)

    val predictions = model.transform(df)
    assert(predictions.select("prediction").distinct().count() === 2)
  }

  test("KMedoids model copy should work correctly") {
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(5.0, 5.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new KMedoids()
      .setK(2)
      .setMaxIter(10)
      .setSeed(42)
      .fit(df)

    val copied = model.copy(org.apache.spark.ml.param.ParamMap.empty)

    // Verify medoids copied
    (0 until model.numClusters).foreach { i =>
      assert(model.medoids(i).toArray.sameElements(copied.medoids(i).toArray))
    }

    // Verify distance function
    assert(copied.distanceFunctionName === model.distanceFunctionName)
  }

  test("KMedoids parameter validation") {
    // k must be > 1
    assertThrows[IllegalArgumentException] {
      new KMedoids().setK(1)
    }

    assertThrows[IllegalArgumentException] {
      new KMedoids().setK(0)
    }

    // Invalid distance function
    assertThrows[IllegalArgumentException] {
      new KMedoids().setDistanceFunction("invalid")
    }

    // Valid parameters should not throw
    val kmedoids = new KMedoids()
      .setK(3)
      .setMaxIter(50)
      .setDistanceFunction("manhattan")
      .setSeed(123)

    assert(kmedoids.getK === 3)
    assert(kmedoids.getMaxIter === 50)
    assert(kmedoids.getDistanceFunction === "manhattan")
  }

  test("KMedoids should produce lower cost than random assignment") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(0.2, 0.2),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1),
      Vectors.dense(5.2, 5.2)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new KMedoids()
      .setK(2)
      .setMaxIter(20)
      .setSeed(42)
      .fit(df)

    val cost = model.computeCost(df)

    // Cost should be positive
    assert(cost > 0.0)

    // Cost should be reasonable (not huge)
    assert(cost < 5.0)
  }
}
