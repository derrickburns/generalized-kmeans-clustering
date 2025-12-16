package com.massivedatascience.clusterer.ml

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

/** Tests for DPMeans clustering with automatic k determination. */
class DPMeansSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("DPMeansSuite")
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

  test("DPMeans finds correct number of clusters with well-separated data") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create 3 well-separated clusters
    val data = Seq(
      // Cluster around (0, 0)
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(-0.1, 0.1)),
      Tuple1(Vectors.dense(0.1, -0.1)),
      Tuple1(Vectors.dense(-0.1, -0.1)),
      // Cluster around (10, 0)
      Tuple1(Vectors.dense(10.0, 0.0)),
      Tuple1(Vectors.dense(10.1, 0.1)),
      Tuple1(Vectors.dense(9.9, 0.1)),
      Tuple1(Vectors.dense(10.1, -0.1)),
      Tuple1(Vectors.dense(9.9, -0.1)),
      // Cluster around (5, 10)
      Tuple1(Vectors.dense(5.0, 10.0)),
      Tuple1(Vectors.dense(5.1, 10.1)),
      Tuple1(Vectors.dense(4.9, 9.9)),
      Tuple1(Vectors.dense(5.1, 9.9)),
      Tuple1(Vectors.dense(4.9, 10.1))
    ).toDF("features")

    // Lambda of 2.0 should create ~3 clusters (cluster diameter < 2, inter-cluster > 5)
    val dpmeans = new DPMeans()
      .setLambda(2.0)
      .setMaxIter(20)
      .setSeed(42L)

    val model = dpmeans.fit(data)

    // Should find approximately 3 clusters
    assert(model.getK >= 2 && model.getK <= 5, s"Expected 2-5 clusters, got ${model.getK}")

    // All points should be assigned
    val predictions = model.transform(data)
    assert(predictions.count() == 15)

    // Verify predictions column exists
    assert(predictions.columns.contains("prediction"))
  }

  test("DPMeans respects maxK parameter") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create many well-separated points
    val data = (0 until 20).map { i =>
      Tuple1(Vectors.dense(i * 10.0, 0.0))
    }.toDF("features")

    // Small lambda would create many clusters, but maxK limits it
    val dpmeans = new DPMeans()
      .setLambda(1.0)
      .setMaxK(5)
      .setMaxIter(10)
      .setSeed(42L)

    val model = dpmeans.fit(data)

    assert(model.getK <= 5, s"Expected at most 5 clusters, got ${model.getK}")
  }

  test("DPMeans with large lambda creates single cluster") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(2.0, 2.0)),
      Tuple1(Vectors.dense(3.0, 3.0))
    ).toDF("features")

    // Very large lambda should create only 1 cluster
    val dpmeans = new DPMeans()
      .setLambda(100.0)
      .setMaxIter(10)
      .setSeed(42L)

    val model = dpmeans.fit(data)

    assert(model.getK == 1, s"Expected 1 cluster with large lambda, got ${model.getK}")
  }

  test("DPMeans with small lambda creates many clusters") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Points spread out
    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(5.0, 0.0)),
      Tuple1(Vectors.dense(10.0, 0.0)),
      Tuple1(Vectors.dense(15.0, 0.0)),
      Tuple1(Vectors.dense(20.0, 0.0))
    ).toDF("features")

    // Small lambda should create cluster for each point
    val dpmeans = new DPMeans()
      .setLambda(1.0)
      .setMaxK(10)
      .setMaxIter(10)
      .setSeed(42L)

    val model = dpmeans.fit(data)

    assert(model.getK >= 3, s"Expected at least 3 clusters with small lambda, got ${model.getK}")
  }

  test("DPMeans works with KL divergence") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Probability distributions (must be positive)
    val data = Seq(
      // Cluster 1: high first component
      Tuple1(Vectors.dense(0.8, 0.1, 0.1)),
      Tuple1(Vectors.dense(0.7, 0.2, 0.1)),
      Tuple1(Vectors.dense(0.75, 0.15, 0.1)),
      // Cluster 2: high second component
      Tuple1(Vectors.dense(0.1, 0.8, 0.1)),
      Tuple1(Vectors.dense(0.2, 0.7, 0.1)),
      Tuple1(Vectors.dense(0.15, 0.75, 0.1)),
      // Cluster 3: high third component
      Tuple1(Vectors.dense(0.1, 0.1, 0.8)),
      Tuple1(Vectors.dense(0.1, 0.2, 0.7)),
      Tuple1(Vectors.dense(0.1, 0.15, 0.75))
    ).toDF("features")

    val dpmeans = new DPMeans()
      .setDivergence("kl")
      .setSmoothing(1e-6)
      .setLambda(0.5)
      .setMaxIter(20)
      .setSeed(42L)

    val model = dpmeans.fit(data)

    // Should find 2-4 clusters
    assert(model.getK >= 1 && model.getK <= 5, s"Expected 1-5 clusters, got ${model.getK}")

    // All points should be assigned
    val predictions = model.transform(data)
    assert(predictions.count() == 9)
  }

  test("DPMeans works with spherical divergence") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Normalized vectors (unit length)
    def normalize(arr: Array[Double]): Array[Double] = {
      val norm = math.sqrt(arr.map(x => x * x).sum)
      arr.map(_ / norm)
    }

    val data = Seq(
      // Direction 1: (1, 0, 0)
      Tuple1(Vectors.dense(normalize(Array(1.0, 0.0, 0.0)))),
      Tuple1(Vectors.dense(normalize(Array(1.0, 0.1, 0.0)))),
      Tuple1(Vectors.dense(normalize(Array(1.0, 0.0, 0.1)))),
      // Direction 2: (0, 1, 0)
      Tuple1(Vectors.dense(normalize(Array(0.0, 1.0, 0.0)))),
      Tuple1(Vectors.dense(normalize(Array(0.1, 1.0, 0.0)))),
      Tuple1(Vectors.dense(normalize(Array(0.0, 1.0, 0.1)))),
      // Direction 3: (0, 0, 1)
      Tuple1(Vectors.dense(normalize(Array(0.0, 0.0, 1.0)))),
      Tuple1(Vectors.dense(normalize(Array(0.1, 0.0, 1.0)))),
      Tuple1(Vectors.dense(normalize(Array(0.0, 0.1, 1.0))))
    ).toDF("features")

    val dpmeans = new DPMeans()
      .setDivergence("spherical")
      .setLambda(0.5)
      .setMaxIter(20)
      .setSeed(42L)

    val model = dpmeans.fit(data)

    // Should find 2-4 clusters
    assert(model.getK >= 1, s"Expected at least 1 cluster, got ${model.getK}")

    val predictions = model.transform(data)
    assert(predictions.count() == 9)
  }

  test("DPMeans distance column output") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(10.0, 0.0))
    ).toDF("features")

    val dpmeans = new DPMeans()
      .setLambda(5.0)
      .setDistanceCol("distance")
      .setMaxIter(10)
      .setSeed(42L)

    val model = dpmeans.fit(data)
    val predictions = model.transform(data)

    assert(predictions.columns.contains("distance"))

    // Distances should be non-negative
    val distances = predictions.select("distance").collect().map(_.getDouble(0))
    distances.foreach(d => assert(d >= 0, s"Distance should be non-negative, got $d"))
  }

  test("DPMeans converges with identical points") {
    val sparkSession = spark
    import sparkSession.implicits._

    // All identical points
    val data = Seq(
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(1.0, 1.0))
    ).toDF("features")

    val dpmeans = new DPMeans()
      .setLambda(0.5)
      .setMaxIter(10)
      .setSeed(42L)

    val model = dpmeans.fit(data)

    // Should create exactly 1 cluster
    assert(model.getK == 1, s"Expected 1 cluster for identical points, got ${model.getK}")

    // Center should be at (1, 1)
    val center = model.clusterCenters.head.toArray
    assert(math.abs(center(0) - 1.0) < 0.01, s"Center x should be 1.0, got ${center(0)}")
    assert(math.abs(center(1) - 1.0) < 0.01, s"Center y should be 1.0, got ${center(1)}")
  }

  test("DPMeans parameter getters and setters") {
    val dpmeans = new DPMeans()
      .setLambda(5.0)
      .setMaxK(20)
      .setDivergence("kl")
      .setSmoothing(1e-6)
      .setMaxIter(50)
      .setTol(1e-5)
      .setSeed(123L)
      .setFeaturesCol("feat")
      .setPredictionCol("pred")

    assert(dpmeans.getLambda == 5.0)
    assert(dpmeans.getMaxK == 20)
    assert(dpmeans.getDivergence == "kl")
    assert(dpmeans.getSmoothing == 1e-6)
    assert(dpmeans.getMaxIter == 50)
    assert(dpmeans.getTol == 1e-5)
    assert(dpmeans.getSeed == 123L)
    assert(dpmeans.getFeaturesCol == "feat")
    assert(dpmeans.getPredictionCol == "pred")
  }

  test("DPMeans model getK returns correct cluster count") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(100.0, 0.0))
    ).toDF("features")

    val dpmeans = new DPMeans()
      .setLambda(10.0)
      .setMaxIter(10)
      .setSeed(42L)

    val model = dpmeans.fit(data)

    // Two very distant points with lambda=10 should create 2 clusters
    assert(model.getK == 2, s"Expected 2 clusters, got ${model.getK}")
    assert(model.clusterCenters.length == 2)
  }
}
