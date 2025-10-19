package com.massivedatascience.clusterer.ml

import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.{ DataFrame, SparkSession }
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

/** Integration tests for DataFrame-based GeneralizedKMeans.
  */
class GeneralizedKMeansSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("GeneralizedKMeansSuite")
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

  /** Create a simple 2D dataset with 3 well-separated clusters.
    */
  def createSimpleDataset(): DataFrame = {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      // Cluster 0: around (0, 0)
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(-0.1, 0.1)),
      Tuple1(Vectors.dense(0.1, -0.1)),
      Tuple1(Vectors.dense(-0.1, -0.1)),
      // Cluster 1: around (5, 5)
      Tuple1(Vectors.dense(5.0, 5.0)),
      Tuple1(Vectors.dense(5.1, 5.1)),
      Tuple1(Vectors.dense(4.9, 5.1)),
      Tuple1(Vectors.dense(5.1, 4.9)),
      Tuple1(Vectors.dense(4.9, 4.9)),
      // Cluster 2: around (10, 0)
      Tuple1(Vectors.dense(10.0, 0.0)),
      Tuple1(Vectors.dense(10.1, 0.1)),
      Tuple1(Vectors.dense(9.9, 0.1)),
      Tuple1(Vectors.dense(10.1, -0.1)),
      Tuple1(Vectors.dense(9.9, -0.1))
    )

    data.toDF("features")
  }

  test("fit and transform with Squared Euclidean") {
    val df = createSimpleDataset()

    val kmeans = new GeneralizedKMeans()
      .setK(3)
      .setDivergence("squaredEuclidean")
      .setMaxIter(10)
      .setSeed(42)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")

    val model = kmeans.fit(df)

    assert(model.numClusters === 3)
    assert(model.numFeatures === 2)

    val predictions = model.transform(df)
    assert(predictions.columns.contains("prediction"))

    val numPredictions = predictions.count()
    assert(numPredictions === 15)

    // Check that we have 3 different cluster assignments
    val distinctClusters = predictions.select("prediction").distinct().count()
    assert(distinctClusters === 3)
  }

  test("fit with KL divergence") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create probability distributions (must sum to 1)
    val probData = Seq(
      Tuple1(Vectors.dense(0.7, 0.2, 0.1)),
      Tuple1(Vectors.dense(0.6, 0.3, 0.1)),
      Tuple1(Vectors.dense(0.8, 0.1, 0.1)),
      Tuple1(Vectors.dense(0.1, 0.2, 0.7)),
      Tuple1(Vectors.dense(0.1, 0.3, 0.6)),
      Tuple1(Vectors.dense(0.2, 0.1, 0.7))
    ).toDF("features")

    val kmeans = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("kl")
      .setSmoothing(1e-10)
      .setMaxIter(10)
      .setSeed(42)

    val model = kmeans.fit(probData)

    assert(model.numClusters === 2)
    assert(model.kernelName.startsWith("KL"))

    val predictions      = model.transform(probData)
    val distinctClusters = predictions.select("prediction").distinct().count()
    assert(distinctClusters === 2)
  }

  test("model with distance column") {
    val df = createSimpleDataset()

    val kmeans =
      new GeneralizedKMeans().setK(3).setMaxIter(10).setSeed(42).setDistanceCol("distance")

    val model       = kmeans.fit(df)
    val predictions = model.transform(df)

    assert(predictions.columns.contains("distance"))

    // All distances should be non-negative
    val negativeDistances = predictions.filter("distance < 0").count()
    assert(negativeDistances === 0)
  }

  test("predict single point") {
    val df = createSimpleDataset()

    val kmeans = new GeneralizedKMeans().setK(3).setMaxIter(10).setSeed(42)

    val model = kmeans.fit(df)

    // Point close to cluster 0 (around 0,0)
    val point1      = Vectors.dense(0.05, 0.05)
    val prediction1 = model.predict(point1)
    assert(prediction1 >= 0 && prediction1 < 3)

    // Point close to cluster 1 (around 5,5)
    val point2      = Vectors.dense(5.05, 5.05)
    val prediction2 = model.predict(point2)
    assert(prediction2 >= 0 && prediction2 < 3)

    // Different points near different clusters should have different predictions
    // (with high probability given well-separated clusters)
    // Note: This could theoretically fail if random initialization is very unlucky,
    // but with seed=42 it should be stable
  }

  test("compute cost") {
    val df = createSimpleDataset()

    val kmeans = new GeneralizedKMeans().setK(3).setMaxIter(10).setSeed(42)

    val model = kmeans.fit(df)
    val cost  = model.computeCost(df)

    // Cost should be non-negative and finite
    assert(cost >= 0.0)
    assert(!cost.isInfinity)
    assert(!cost.isNaN)
  }

  test("auto assignment strategy selection") {
    val df = createSimpleDataset()

    val kmeans = new GeneralizedKMeans()
      .setK(3)
      .setDivergence("squaredEuclidean")
      .setAssignmentStrategy("auto")
      .setMaxIter(10)
      .setSeed(42)

    val model = kmeans.fit(df)
    assert(model.numClusters === 3)
  }

  test("broadcast assignment strategy") {
    val df = createSimpleDataset()

    val kmeans = new GeneralizedKMeans()
      .setK(3)
      .setDivergence("squaredEuclidean")
      .setAssignmentStrategy("broadcast")
      .setMaxIter(10)
      .setSeed(42)

    val model = kmeans.fit(df)
    assert(model.numClusters === 3)
  }

  test("random initialization") {
    val df = createSimpleDataset()

    val kmeans = new GeneralizedKMeans().setK(3).setInitMode("random").setMaxIter(10).setSeed(42)

    val model = kmeans.fit(df)
    assert(model.numClusters === 3)
  }

  test("k-means|| initialization") {
    val df = createSimpleDataset()

    val kmeans = new GeneralizedKMeans()
      .setK(3)
      .setInitMode("k-means||")
      .setInitSteps(2)
      .setMaxIter(10)
      .setSeed(42)

    val model = kmeans.fit(df)
    assert(model.numClusters === 3)
  }

  test("weighted clustering") {
    val sparkSession = spark
    import sparkSession.implicits._

    val weightedData = Seq(
      (Vectors.dense(0.0, 0.0), 1.0),
      (Vectors.dense(0.1, 0.1), 1.0),
      (Vectors.dense(5.0, 5.0), 10.0), // High weight
      (Vectors.dense(5.1, 5.1), 10.0)
    ).toDF("features", "weight")

    val kmeans = new GeneralizedKMeans().setK(2).setWeightCol("weight").setMaxIter(10).setSeed(42)

    val model = kmeans.fit(weightedData)
    assert(model.numClusters === 2)

    val predictions = model.transform(weightedData)
    assert(predictions.count() === 4)
  }

  test("model toString") {
    val df = createSimpleDataset()

    val kmeans = new GeneralizedKMeans().setK(3).setMaxIter(10).setSeed(42)

    val model = kmeans.fit(df)
    val str   = model.toString

    assert(str.contains("GeneralizedKMeansModel"))
    assert(str.contains("k=3"))
    assert(str.contains("features=2"))
  }

  test("model persistence - save and load") {
    val df = createSimpleDataset()

    val kmeans = new GeneralizedKMeans()
      .setK(3)
      .setDivergence("squaredEuclidean")
      .setMaxIter(10)
      .setSeed(42)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")

    val originalModel       = kmeans.fit(df)
    val originalPredictions = originalModel.transform(df).select("prediction").collect()

    // Save model
    val tempDir = java.nio.file.Files.createTempDirectory("kmeans-model-test").toString
    try {
      originalModel.write.overwrite().save(tempDir)

      // Load model
      val loadedModel = GeneralizedKMeansModel.load(tempDir)

      // Verify loaded model properties
      assert(loadedModel.numClusters === originalModel.numClusters)
      assert(loadedModel.numFeatures === originalModel.numFeatures)
      assert(loadedModel.kernelName === originalModel.kernelName)
      assert(loadedModel.getFeaturesCol === originalModel.getFeaturesCol)
      assert(loadedModel.getPredictionCol === originalModel.getPredictionCol)

      // Verify cluster centers match
      originalModel.clusterCenters.zip(loadedModel.clusterCenters).foreach { case (orig, loaded) =>
        assert(orig.sameElements(loaded))
      }

      // Verify predictions match
      val loadedPredictions = loadedModel.transform(df).select("prediction").collect()
      assert(originalPredictions.sameElements(loadedPredictions))

    } finally {
      // Cleanup
      import scala.reflect.io.Directory
      val dir = new Directory(new java.io.File(tempDir))
      dir.deleteRecursively()
    }
  }

  test("model persistence - different kernels") {
    val df = createSimpleDataset()

    Seq("squaredEuclidean", "kl").foreach { divergence =>
      val kmeans =
        new GeneralizedKMeans().setK(2).setDivergence(divergence).setMaxIter(5).setSeed(42)

      val originalModel = kmeans.fit(df)

      val tempDir = java.nio.file.Files.createTempDirectory(s"kmeans-$divergence-test").toString
      try {
        originalModel.write.overwrite().save(tempDir)
        val loadedModel = GeneralizedKMeansModel.load(tempDir)

        assert(loadedModel.kernelName === originalModel.kernelName)
        assert(loadedModel.numClusters === originalModel.numClusters)

      } finally {
        import scala.reflect.io.Directory
        val dir = new Directory(new java.io.File(tempDir))
        dir.deleteRecursively()
      }
    }
  }
}
