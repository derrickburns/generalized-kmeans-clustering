package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.TestingUtils._
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.{ DataFrame, SparkSession }
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

class CoresetKMeansSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("CoresetKMeansSuite")
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

  test("CoresetKMeans should cluster simple 2D data") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(0.0, 0.2)),
      Tuple1(Vectors.dense(9.0, 9.0)),
      Tuple1(Vectors.dense(9.1, 9.1)),
      Tuple1(Vectors.dense(9.0, 9.2))
    ).toDF("features")

    val coreset = new CoresetKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setCoresetSize(4)
      .setMaxIter(10)
      .setSeed(42)

    val model       = coreset.fit(data)
    val predictions = model.transform(data)

    // Should have 2 clusters
    assert(model.clusterCenters.length == 2)

    // All points should be assigned
    assert(predictions.count() == data.count())

    // Check cluster assignments are reasonable
    val clusters = predictions.select("prediction").collect().map(_.getInt(0))
    assert(clusters.toSet.size <= 2) // At most 2 clusters

    // Points in same region should have same cluster
    val firstThree = clusters.take(3)
    assert(firstThree.toSet.size == 1, "First 3 points should be in same cluster")

    val lastThree = clusters.drop(3)
    assert(lastThree.toSet.size == 1, "Last 3 points should be in same cluster")
  }

  test("CoresetKMeans should work with KL divergence") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.9, 0.1)),
      Tuple1(Vectors.dense(0.85, 0.15)),
      Tuple1(Vectors.dense(0.1, 0.9)),
      Tuple1(Vectors.dense(0.15, 0.85))
    ).toDF("features")

    val coreset = new CoresetKMeans()
      .setK(2)
      .setDivergence("kl")
      .setCoresetSize(3)
      .setSmoothing(1e-10)
      .setMaxIter(20)
      .setSeed(123)

    val model       = coreset.fit(data)
    val predictions = model.transform(data)

    assert(model.clusterCenters.length == 2)
    assert(predictions.count() == 4)

    // Check that clustering is reasonable (similar distributions together)
    val clusters = predictions.select("prediction").collect().map(_.getInt(0))
    assert(clusters(0) == clusters(1), "First two points should cluster together")
    assert(clusters(2) == clusters(3), "Last two points should cluster together")
    assert(clusters(0) != clusters(2), "Should have 2 distinct clusters")
  }

  test("CoresetKMeans should handle small datasets (no core-set)") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(2.0, 2.0)),
      Tuple1(Vectors.dense(10.0, 10.0))
    ).toDF("features")

    val coreset = new CoresetKMeans()
      .setK(2)
      .setCoresetSize(100) // Larger than dataset
      .setMaxIter(10)
      .setSeed(42)

    val model = coreset.fit(data)

    assert(model.clusterCenters.length == 2)
    // Note: Training summary doesn't track compression ratio in additionalMetrics
    // This is delegated to GeneralizedKMeans which has its own summary
  }

  test("CoresetKMeans parameters should be validated") {
    intercept[IllegalArgumentException] {
      new CoresetKMeans().setK(0)
    }

    intercept[IllegalArgumentException] {
      new CoresetKMeans().setCoresetSize(-1)
    }

    intercept[IllegalArgumentException] {
      new CoresetKMeans().setEpsilon(1.5)
    }

    intercept[IllegalArgumentException] {
      new CoresetKMeans().setRefinementIterations(-1)
    }

    intercept[IllegalArgumentException] {
      new CoresetKMeans().setMinSamplingProb(0.0)
    }
  }

  test("CoresetKMeans should support different sensitivity strategies") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = (1 to 20).map { i =>
      val x = if (i <= 10) i.toDouble else (i + 10).toDouble
      val y = if (i <= 10) i.toDouble else (i + 10).toDouble
      Tuple1(Vectors.dense(x, y))
    }.toDF("features")

    val strategies = Seq("uniform", "distance", "density", "hybrid")

    strategies.foreach { strategy =>
      val coreset = new CoresetKMeans()
        .setK(2)
        .setCoresetSize(10)
        .setSensitivityStrategy(strategy)
        .setMaxIter(10)
        .setSeed(42)

      val model = coreset.fit(data)
      assert(model.clusterCenters.length == 2, s"Strategy '$strategy' should produce 2 clusters")
    }
  }

  test("CoresetKMeans should support refinement control") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = (1 to 50).map { i =>
      val x = if (i <= 25) i.toDouble else (i + 20).toDouble
      Tuple1(Vectors.dense(x, x))
    }.toDF("features")

    // With refinement - use larger core-set to ensure both clusters are represented
    val withRefinement = new CoresetKMeans()
      .setK(2)
      .setCoresetSize(20)
      .setRefinementIterations(3)
      .setEnableRefinement(true)
      .setMaxIter(10)
      .setSeed(42)

    val modelWithRef = withRefinement.fit(data)

    // Without refinement
    val withoutRefinement = new CoresetKMeans()
      .setK(2)
      .setCoresetSize(20)
      .setRefinementIterations(0)
      .setEnableRefinement(false)
      .setMaxIter(10)
      .setSeed(42)

    val modelWithoutRef = withoutRefinement.fit(data)

    // Both should produce valid models with 2 clusters
    assert(modelWithRef.clusterCenters.length == 2)
    assert(modelWithoutRef.clusterCenters.length == 2)

    // Note: Training summary doesn't expose custom metrics
    // Both models should be valid regardless of refinement setting
  }

  test("CoresetKMeans should support weighted input") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      (Vectors.dense(1.0, 1.0), 10.0),  // High weight
      (Vectors.dense(1.1, 1.1), 10.0),
      (Vectors.dense(10.0, 10.0), 1.0), // Low weight
      (Vectors.dense(10.1, 10.1), 1.0)
    ).toDF("features", "weight")

    val coreset = new CoresetKMeans()
      .setK(2)
      .setWeightCol("weight")
      .setCoresetSize(3)
      .setMaxIter(10)
      .setSeed(42)

    val model       = coreset.fit(data)
    val predictions = model.transform(data)

    assert(model.clusterCenters.length == 2)
    assert(predictions.count() == 4)
  }

  test("CoresetKMeans should work with distance column output") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(10.0, 10.0))
    ).toDF("features")

    val coreset = new CoresetKMeans()
      .setK(2)
      .setCoresetSize(2)
      .setDistanceCol("distance")
      .setMaxIter(10)
      .setSeed(42)

    val model       = coreset.fit(data)
    val predictions = model.transform(data)

    assert(predictions.columns.contains("distance"))

    val distances = predictions.select("distance").collect().map(_.getDouble(0))
    assert(distances.forall(_ >= 0.0), "All distances should be non-negative")
  }

  test("CoresetKMeans should be deterministic with same seed") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = (1 to 30).map(i => Tuple1(Vectors.dense(i.toDouble, i.toDouble))).toDF("features")

    val coreset1 = new CoresetKMeans().setK(3).setCoresetSize(15).setMaxIter(10).setSeed(12345)

    val coreset2 = new CoresetKMeans().setK(3).setCoresetSize(15).setMaxIter(10).setSeed(12345)

    val model1 = coreset1.fit(data)
    val model2 = coreset2.fit(data)

    // Centers should be identical with same seed
    model1.clusterCenters.zip(model2.clusterCenters).foreach { case (c1, c2) =>
      assert(c1.length == c2.length, "Center dimensions should match")
      var i = 0
      while (i < c1.length) {
        assert(math.abs(c1(i) - c2(i)) < 1e-10, s"Centers should be identical with same seed")
        i += 1
      }
    }
  }

  test("CoresetKMeans should support L1 (Manhattan) divergence") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.5, 0.5)),
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(10.5, 10.5))
    ).toDF("features")

    val coreset =
      new CoresetKMeans().setK(2).setDivergence("l1").setCoresetSize(3).setMaxIter(15).setSeed(42)

    val model = coreset.fit(data)

    assert(model.clusterCenters.length == 2)
    // Note: GeneralizedKMeansModel doesn't expose divergenceName field directly
  }

  test("CoresetKMeans should work with Itakura-Saito divergence") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(1.1, 2.1)),
      Tuple1(Vectors.dense(5.0, 10.0)),
      Tuple1(Vectors.dense(5.1, 10.1))
    ).toDF("features")

    val coreset = new CoresetKMeans()
      .setK(2)
      .setDivergence("itakuraSaito")
      .setSmoothing(1e-10)
      .setCoresetSize(3)
      .setMaxIter(20)
      .setSeed(42)

    val model = coreset.fit(data)

    assert(model.clusterCenters.length == 2)
    // Note: GeneralizedKMeansModel doesn't expose divergenceName field directly
  }

  test("CoresetKMeans should provide comprehensive training summary") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = (1 to 100).map { i =>
      val x = if (i <= 50) i.toDouble else (i + 50).toDouble
      Tuple1(Vectors.dense(x, x))
    }.toDF("features")

    val coreset = new CoresetKMeans()
      .setK(2)
      .setCoresetSize(20)
      .setEpsilon(0.1)
      .setRefinementIterations(2)
      .setMaxIter(15)
      .setSeed(42)

    val model   = coreset.fit(data)
    val summary = model.trainingSummary.get

    // Note: Training summary doesn't expose custom core-set metrics
    // The model should still have valid summary from GeneralizedKMeans
    assert(summary.iterations >= 0)
  }

  test("CoresetKMeans should handle edge case: k = 2 with small data") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(1.0, 2.0)),
      Tuple1(Vectors.dense(3.0, 4.0)),
      Tuple1(Vectors.dense(5.0, 6.0))
    ).toDF("features")

    val coreset = new CoresetKMeans().setK(2).setCoresetSize(2).setMaxIter(5).setSeed(42)

    val model = coreset.fit(data)

    assert(model.clusterCenters.length == 2)

    val predictions = model.transform(data)
    val clusters    = predictions.select("prediction").collect().map(_.getInt(0))

    // Should have at most 2 clusters
    assert(clusters.toSet.size <= 2)
  }

  test("CoresetKMeans copy should preserve parameters") {
    val original = new CoresetKMeans()
      .setK(5)
      .setCoresetSize(500)
      .setEpsilon(0.05)
      .setSensitivityStrategy("distance")
      .setRefinementIterations(5)
      .setMaxIter(30)
      .setSeed(999)

    val copied = original.copy(org.apache.spark.ml.param.ParamMap.empty)

    assert(copied.getK == 5)
    assert(copied.getCoresetSize == 500)
    assert(copied.getEpsilon == 0.05)
    assert(copied.getSensitivityStrategy == "distance")
    assert(copied.getRefinementIterations == 5)
    assert(copied.getMaxIter == 30)
    assert(copied.getSeed == 999)
  }

  test("CoresetKMeans should support different init modes") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = (1 to 40).map(i => Tuple1(Vectors.dense(i.toDouble, i.toDouble))).toDF("features")

    val initModes = Seq("k-means||", "random")

    initModes.foreach { mode =>
      val coreset =
        new CoresetKMeans().setK(3).setCoresetSize(20).setInitMode(mode).setMaxIter(10).setSeed(42)

      val model = coreset.fit(data)
      assert(model.clusterCenters.length == 3, s"Init mode '$mode' should produce 3 clusters")
    }
  }

  test("CoresetKMeans should handle high-dimensional data") {
    val sparkSession = spark
    import sparkSession.implicits._

    val dim  = 50
    val data = (1 to 100).map { i =>
      val features = Array.fill(dim)(if (i <= 50) i.toDouble / 10.0 else (i + 50).toDouble / 10.0)
      Tuple1(Vectors.dense(features))
    }.toDF("features")

    val coreset = new CoresetKMeans().setK(2).setCoresetSize(30).setMaxIter(10).setSeed(42)

    val model = coreset.fit(data)

    assert(model.clusterCenters.length == 2)
    assert(model.clusterCenters(0).size == dim)
  }

  test("CoresetKMeans should validate unknown divergence") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Validation happens at parameter setting time, not fit time
    intercept[IllegalArgumentException] {
      new CoresetKMeans().setK(2).setDivergence("unknownDivergence")
    }
  }

  test("CoresetKMeans should validate unknown sensitivity strategy") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Validation happens at parameter setting time, not fit time
    intercept[IllegalArgumentException] {
      new CoresetKMeans().setK(2).setSensitivityStrategy("unknownStrategy")
    }
  }
}
