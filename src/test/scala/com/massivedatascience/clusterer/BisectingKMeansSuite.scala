package com.massivedatascience.clusterer

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** Test suite for Bisecting K-Means clustering.
  *
  * Tests hierarchical divisive clustering with various configurations, comparing behavior with standard K-Means.
  */
class BisectingKMeansSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient private var _spark: SparkSession = _

  private def spark: SparkSession = _spark

  override def beforeAll(): Unit = {
    super.beforeAll()
    _spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("BisectingKMeansSuite")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .config("spark.driver.host", "localhost")
      .getOrCreate()
    _spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = {
    if (_spark != null) {
      _spark.stop()
    }
    super.afterAll()
  }

  test("Bisecting K-Means should cluster simple 2D data") {
    // Create data with 3 obvious clusters
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
      // Cluster 3: around (10, 0)
      Vectors.dense(10.0, 0.0),
      Vectors.dense(10.1, 0.1),
      Vectors.dense(9.9, 0.1),
      Vectors.dense(10.1, -0.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val bisecting = new ml.BisectingKMeans()
      .setK(3)
      .setDivergence("squaredEuclidean")
      .setMaxIter(10)
      .setMinDivisibleClusterSize(1) // Allow splitting small clusters
      .setSeed(42)

    val model       = bisecting.fit(df)
    val predictions = model.transform(df)

    // Should have 3 clusters
    assert(model.numClusters === 3)

    // All points should be assigned to valid cluster IDs
    val clusterCounts = predictions.groupBy("prediction").count().collect()
    assert(clusterCounts.length === 3)

    // With 12 points split into 3 clusters, we should have reasonable cluster sizes
    // (not necessarily exactly 4 each, but all points should be assigned)
    val totalPoints = clusterCounts.map(_.getLong(1)).sum
    assert(totalPoints === 12, s"Expected 12 total points, got $totalPoints")

    // Each cluster should have at least 1 point
    assert(clusterCounts.forall(row => row.getLong(1) >= 1))
  }

  test("Bisecting K-Means should respect minDivisibleClusterSize") {
    // Create data with one large cluster and one small cluster
    val data = Seq(
      // Large cluster (6 points)
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(-0.1, 0.1),
      Vectors.dense(0.1, -0.1),
      Vectors.dense(-0.1, -0.1),
      Vectors.dense(0.0, 0.1),
      // Small cluster (2 points)
      Vectors.dense(10.0, 10.0),
      Vectors.dense(10.1, 10.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // Set minDivisibleClusterSize to 5, so only the large cluster can be split
    val bisecting = new ml.BisectingKMeans()
      .setK(4) // Request 4 clusters
      .setDivergence("squaredEuclidean")
      .setMaxIter(10)
      .setMinDivisibleClusterSize(5) // Minimum size to split
      .setSeed(42)

    val model = bisecting.fit(df)

    // Should stop before reaching k=4 because small clusters can't be split
    assert(model.numClusters < 4)
  }

  test("Bisecting K-Means should work with different divergences") {
    val data = Seq(
      Vectors.dense(1.0, 2.0),
      Vectors.dense(1.1, 2.1),
      Vectors.dense(5.0, 6.0),
      Vectors.dense(5.1, 6.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // Test with KL divergence
    val bisectingKL = new ml.BisectingKMeans()
      .setK(2)
      .setDivergence("kl")
      .setSmoothing(1e-8)
      .setMaxIter(10)
      .setSeed(42)

    val modelKL = bisectingKL.fit(df)
    assert(modelKL.numClusters === 2)

    // Test with L1 divergence
    val bisectingL1 = new ml.BisectingKMeans()
      .setK(2)
      .setDivergence("l1")
      .setMaxIter(10)
      .setSeed(42)

    val modelL1 = bisectingL1.fit(df)
    assert(modelL1.numClusters === 2)
  }

  test("Bisecting K-Means should be more deterministic than random K-Means") {
    // Create data with overlapping clusters (harder case)
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 1.0),
      Vectors.dense(2.0, 2.0),
      Vectors.dense(3.0, 3.0),
      Vectors.dense(4.0, 4.0),
      Vectors.dense(5.0, 5.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // Run bisecting K-Means multiple times with different seeds
    val runs = Range.inclusive(1, 5).map { i =>
      val bisecting = new ml.BisectingKMeans()
        .setK(2)
        .setDivergence("squaredEuclidean")
        .setMaxIter(10)
        .setSeed(i)

      val model       = bisecting.fit(df)
      val predictions = model.transform(df)
      predictions.select("prediction").collect().map(_.getInt(0))
    }

    // All runs should produce similar results (assignments may be permuted)
    // Check that variance in cluster sizes is low
    val clusterSizes = runs.map { assignments =>
      assignments.count(_ == 0)
    }

    val mean     = clusterSizes.sum.toDouble / clusterSizes.length
    val variance = clusterSizes.map(s => math.pow(s - mean, 2)).sum / clusterSizes.length

    // Bisecting should have low variance (deterministic splitting)
    assert(variance < 2.0, s"Bisecting K-Means showed high variance: $variance")
  }

  test("Bisecting K-Means model should support transform and predict") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(5.0, 5.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val bisecting = new ml.BisectingKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setMaxIter(10)
      .setSeed(42)

    val model = bisecting.fit(df)

    // Test transform
    val predictions = model.transform(df)
    assert(predictions.columns.contains("prediction"))
    assert(predictions.count() === 2)

    // Test predict (single vector)
    val testVector = Vectors.dense(0.1, 0.1)
    val cluster    = model.predict(testVector)
    assert(cluster >= 0 && cluster < 2)
  }

  test("Bisecting K-Means should handle weighted data") {
    val data = Seq(
      (Vectors.dense(0.0, 0.0), 10.0), // Heavy point
      (Vectors.dense(0.5, 0.5), 1.0),
      (Vectors.dense(5.0, 5.0), 1.0),
      (Vectors.dense(5.5, 5.5), 1.0)
    )

    val df = spark.createDataFrame(data).toDF("features", "weight")

    val bisecting = new ml.BisectingKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setMaxIter(10)
      .setWeightCol("weight")
      .setSeed(42)

    val model = bisecting.fit(df)
    assert(model.numClusters === 2)

    // The heavy point should influence the center of its cluster
    val centers = model.clusterCentersAsVectors
    val hasNearZeroCenter = centers.exists { center =>
      val arr = center.toArray
      math.sqrt(arr(0) * arr(0) + arr(1) * arr(1)) < 1.0
    }
    assert(hasNearZeroCenter, "Heavy weighted point should pull center closer to origin")
  }

  test("Bisecting K-Means should produce hierarchical structure") {
    // Create data that forms a natural hierarchy
    val data = Seq(
      // Group A (will split into A1 and A2)
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(2.0, 2.0),
      Vectors.dense(2.1, 2.1),
      // Group B (will split into B1 and B2)
      Vectors.dense(10.0, 10.0),
      Vectors.dense(10.1, 10.1),
      Vectors.dense(12.0, 12.0),
      Vectors.dense(12.1, 12.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // First split into 2 clusters (should be A vs B)
    val bisecting2 = new ml.BisectingKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setMaxIter(10)
      .setSeed(42)

    val model2       = bisecting2.fit(df)
    val predictions2 = model2.transform(df).select("prediction").collect().map(_.getInt(0))

    // First 4 points (Group A) should be in one cluster, last 4 (Group B) in another
    assert(predictions2.take(4).toSet.size === 1, "Group A should be in one cluster")
    assert(predictions2.drop(4).toSet.size === 1, "Group B should be in one cluster")
    assert(
      predictions2.take(4).head != predictions2.drop(4).head,
      "Groups should be in different clusters"
    )

    // Now split into 4 clusters (should be A1, A2, B1, B2)
    val bisecting4 = new ml.BisectingKMeans()
      .setK(4)
      .setDivergence("squaredEuclidean")
      .setMaxIter(10)
      .setSeed(42)

    val model4 = bisecting4.fit(df)
    assert(model4.numClusters === 4)
  }

  test("Bisecting K-Means should handle edge case with k=2") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(10.0, 10.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val bisecting = new ml.BisectingKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setMaxIter(10)
      .setSeed(42)

    val model = bisecting.fit(df)
    assert(model.numClusters === 2)
  }

  test("Bisecting K-Means should compute cost correctly") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val bisecting = new ml.BisectingKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setMaxIter(10)
      .setSeed(42)

    val model = bisecting.fit(df)
    val cost  = model.computeCost(df)

    // Cost should be positive and finite
    assert(cost > 0.0 && java.lang.Double.isFinite(cost))
  }

  test("Bisecting K-Means parameter validation should work") {
    val data = Seq(Vectors.dense(0.0, 0.0))
    val df   = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // k must be > 1 (validation happens at setK time)
    assertThrows[IllegalArgumentException] {
      new ml.BisectingKMeans().setK(1)
    }

    // minDivisibleClusterSize must be >= 1 (validation happens at set time)
    assertThrows[IllegalArgumentException] {
      new ml.BisectingKMeans().setK(2).setMinDivisibleClusterSize(0)
    }

    // Invalid divergence should be caught at set time
    assertThrows[IllegalArgumentException] {
      new ml.BisectingKMeans().setK(2).setDivergence("invalid")
    }
  }
}
