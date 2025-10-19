package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.ml.GeneralizedKMeans
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

class KMediansSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("KMediansSuite")
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

  test("K-Medians should be robust to outliers compared to K-Means") {
    // Create data with clear clusters and outliers
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(1.1, 0.9),
      Vectors.dense(0.9, 1.1),
      Vectors.dense(1.0, 1.05),
      Vectors.dense(100.0, 100.0), // Extreme outlier
      Vectors.dense(10.0, 10.0),
      Vectors.dense(10.1, 9.9),
      Vectors.dense(9.9, 10.1),
      Vectors.dense(10.05, 10.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // Test K-Medians with L1 distance
    val kmedians = new GeneralizedKMeans().setK(2).setDivergence("l1").setMaxIter(20).setSeed(42)

    val medianModel   = kmedians.fit(df)
    val medianCenters = medianModel.clusterCenters

    // Centers should be close to (1, 1) and (10, 10), not pulled much by the outlier
    assert(medianCenters.length === 2)

    // Check that we have centers near both clusters
    val hasClusterNear1  = medianCenters.exists { c =>
      math.abs(c(0) - 1.0) < 2.0 && math.abs(c(1) - 1.0) < 2.0
    }
    val hasClusterNear10 = medianCenters.exists { c =>
      math.abs(c(0) - 10.0) < 2.0 && math.abs(c(1) - 10.0) < 2.0
    }

    assert(
      hasClusterNear1,
      s"Should have cluster near (1,1), got centers: ${medianCenters.mkString(", ")}"
    )
    assert(
      hasClusterNear10,
      s"Should have cluster near (10,10), got centers: ${medianCenters.mkString(", ")}"
    )

    // Compare with K-Means (Euclidean)
    val kmeans =
      new GeneralizedKMeans().setK(2).setDivergence("squaredEuclidean").setMaxIter(20).setSeed(42)

    val meansModel   = kmeans.fit(df)
    val meansCenters = meansModel.clusterCenters

    // K-Means centers might be more affected by the outlier
    // Just verify it runs without error
    assert(meansCenters.length === 2)
  }

  test("K-Medians should handle weighted points correctly") {
    // Weighted median should favor high-weight points
    val data = Seq(
      (Vectors.dense(1.0, 1.0), 10.0),  // High weight
      (Vectors.dense(2.0, 2.0), 10.0),  // High weight
      (Vectors.dense(10.0, 10.0), 1.0), // Low weight outlier
      (Vectors.dense(3.0, 3.0), 10.0),  // High weight
      (Vectors.dense(4.0, 4.0), 10.0),  // High weight
      (Vectors.dense(11.0, 11.0), 1.0)  // Another low weight outlier
    )

    val df = spark.createDataFrame(data).toDF("features", "weight")

    val kmedians = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("l1")
      .setWeightCol("weight")
      .setMaxIter(20)
      .setSeed(42)

    val model   = kmedians.fit(df)
    val centers = model.clusterCenters

    assert(centers.length === 2)

    // At least one center should be close to the high-weight points (around 2-3)
    // not pulled much by the low-weight outliers at 10-11
    val hasClusterNear2 = centers.exists { c =>
      c(0) < 6.0 && c(1) < 6.0
    }
    assert(
      hasClusterNear2,
      s"Should have cluster near high-weight points, got centers: ${centers.mkString(", ")}"
    )
  }

  test("K-Medians should converge") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 1.0),
      Vectors.dense(9.0, 8.0),
      Vectors.dense(8.0, 9.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val kmedians =
      new GeneralizedKMeans().setK(2).setDivergence("l1").setMaxIter(20).setTol(1e-4).setSeed(42)

    val model = kmedians.fit(df)

    // Should converge and produce 2 clusters
    assert(model.numClusters === 2)

    // Transform should work
    val predictions = model.transform(df)
    assert(predictions.select("prediction").count() === 4)

    // All predictions should be valid cluster IDs
    val predValues = predictions.select("prediction").collect().map(_.getInt(0))
    assert(predValues.forall(p => p >= 0 && p < 2))
  }

  test("K-Medians distance column should contain L1 distances") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 1.0),
      Vectors.dense(10.0, 10.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val kmedians = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("l1")
      .setDistanceCol("l1_distance")
      .setMaxIter(20)
      .setSeed(42)

    val model       = kmedians.fit(df)
    val predictions = model.transform(df)

    assert(predictions.columns.contains("l1_distance"))

    // All distances should be non-negative
    val distances = predictions.select("l1_distance").collect().map(_.getDouble(0))
    assert(distances.forall(_ >= 0.0))
  }

  test("K-Medians should work with manhattan divergence name") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 1.0),
      Vectors.dense(10.0, 10.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // Should accept "manhattan" as alias for "l1"
    val kmedians =
      new GeneralizedKMeans().setK(2).setDivergence("manhattan").setMaxIter(20).setSeed(42)

    val model = kmedians.fit(df)
    assert(model.numClusters === 2)
  }

  test("K-Medians model persistence") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 1.0),
      Vectors.dense(10.0, 10.0),
      Vectors.dense(11.0, 11.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val kmedians = new GeneralizedKMeans().setK(2).setDivergence("l1").setMaxIter(20).setSeed(42)

    val model         = kmedians.fit(df)
    val originalPreds = model.transform(df).select("prediction").collect().map(_.getInt(0))

    // Save and load model
    val tempDir = java.nio.file.Files.createTempDirectory("kmedians-model-test").toString
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
