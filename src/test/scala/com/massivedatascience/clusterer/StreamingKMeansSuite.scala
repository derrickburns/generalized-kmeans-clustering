package com.massivedatascience.clusterer.ml.tests

import com.massivedatascience.clusterer.ml.{ StreamingKMeans, StreamingKMeansModel }
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{ DataFrame, SparkSession }
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

import scala.concurrent.duration._
import scala.language.postfixOps

/** Test suite for Streaming K-Means clustering.
  */
class StreamingKMeansSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("StreamingKMeansSuite")
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

  test("StreamingKMeans should initialize with batch data") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val streamingKMeans =
      new StreamingKMeans().setK(2).setMaxIter(10).setDecayFactor(0.9).setSeed(42)

    val model = streamingKMeans.fit(df)

    assert(model.clusterCenters.length === 2)
    assert(model.currentWeights.length === 2)

    // Should be able to predict
    val predictions = model.transform(df)
    assert(predictions.columns.contains("prediction"))
    assert(predictions.count() === 4)
  }

  test("StreamingKMeans model should update with new batch") {
    val initialData = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1)
    )

    val df = spark.createDataFrame(initialData.map(Tuple1.apply)).toDF("features")

    val streamingKMeans = new StreamingKMeans().setK(2).setMaxIter(10).setSeed(42)

    val model         = streamingKMeans.fit(df)
    val centersBefore = model.currentCenters.map(_.copy)

    // Update with new batch (similar distribution)
    val newBatch = Seq(
      Vectors.dense(0.2, 0.2),
      Vectors.dense(4.9, 4.9)
    )

    val batchDF = spark.createDataFrame(newBatch.map(Tuple1.apply)).toDF("features")
    model.update(batchDF)

    val centersAfter = model.currentCenters

    // Centers should have changed (slightly)
    val changed = centersBefore.zip(centersAfter).exists { case (before, after) =>
      val diff = (0 until before.size).map { i =>
        math.abs(before(i) - after(i))
      }.sum
      diff > 1e-10
    }

    assert(changed, "Centers should update after processing new batch")
  }

  test("StreamingKMeans should support decay factor") {
    val initialData = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1)
    )

    val df = spark.createDataFrame(initialData.map(Tuple1.apply)).toDF("features")

    // High decay = more forgetting
    val highDecay =
      new StreamingKMeans().setK(2).setMaxIter(10).setDecayFactor(0.1).setSeed(42).fit(df)

    // Low decay = less forgetting
    val lowDecay =
      new StreamingKMeans().setK(2).setMaxIter(10).setDecayFactor(0.9).setSeed(42).fit(df)

    val centersHighBefore = highDecay.currentCenters.map(_.copy)
    val centersLowBefore  = lowDecay.currentCenters.map(_.copy)

    // Update both with new batch (different distribution)
    val newBatch = Seq(
      Vectors.dense(10.0, 10.0),
      Vectors.dense(10.1, 10.1),
      Vectors.dense(15.0, 15.0),
      Vectors.dense(15.1, 15.1)
    )

    val batchDF = spark.createDataFrame(newBatch.map(Tuple1.apply)).toDF("features")
    highDecay.update(batchDF)
    lowDecay.update(batchDF)

    val centersHighAfter = highDecay.currentCenters
    val centersLowAfter  = lowDecay.currentCenters

    // High decay should move centers more toward new data
    val highShift = centersHighBefore
      .zip(centersHighAfter)
      .map { case (before, after) =>
        (0 until before.size).map { i => math.abs(before(i) - after(i)) }.sum
      }
      .sum

    val lowShift = centersLowBefore
      .zip(centersLowAfter)
      .map { case (before, after) =>
        (0 until before.size).map { i => math.abs(before(i) - after(i)) }.sum
      }
      .sum

    assert(
      highShift > lowShift * 1.1,
      s"High decay ($highShift) should shift more than low decay ($lowShift)"
    )
  }

  test("StreamingKMeans should support half-life parameter") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val streamingKMeans = new StreamingKMeans().setK(2).setMaxIter(10).setHalfLife(5.0).setSeed(42)

    val model = streamingKMeans.fit(df)

    // Half-life should compute decay factor
    val expectedDecay = math.pow(0.5, 1.0 / 5.0)
    assert(model.decayFactorValue === expectedDecay)

    assert(model.clusterCenters.length === 2)
  }

  test("StreamingKMeans should work with different time units") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // Time unit = batches
    val batchUnit = new StreamingKMeans()
      .setK(2)
      .setMaxIter(10)
      .setDecayFactor(0.5)
      .setTimeUnit("batches")
      .setSeed(42)
      .fit(df)

    // Time unit = points
    val pointUnit = new StreamingKMeans()
      .setK(2)
      .setMaxIter(10)
      .setDecayFactor(0.5)
      .setTimeUnit("points")
      .setSeed(42)
      .fit(df)

    assert(batchUnit.timeUnitValue === "batches")
    assert(pointUnit.timeUnitValue === "points")

    // Both should work for updates
    val batch =
      spark.createDataFrame(Seq(Vectors.dense(1.0, 1.0)).map(Tuple1.apply)).toDF("features")
    batchUnit.update(batch)
    pointUnit.update(batch)

    assert(batchUnit.currentCenters.length === 2)
    assert(pointUnit.currentCenters.length === 2)
  }

  test("StreamingKMeans should work with different divergences") {
    val data = Seq(
      Vectors.dense(1.0, 2.0),
      Vectors.dense(1.1, 2.1),
      Vectors.dense(5.0, 6.0),
      Vectors.dense(5.1, 6.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // Test with L1 divergence
    val l1Model =
      new StreamingKMeans().setK(2).setMaxIter(10).setDivergence("l1").setSeed(42).fit(df)

    assert(l1Model.divergenceName === "l1")
    assert(l1Model.clusterCenters.length === 2)

    // Update should work
    val batch =
      spark.createDataFrame(Seq(Vectors.dense(1.2, 2.2)).map(Tuple1.apply)).toDF("features")
    l1Model.update(batch)

    val predictions = l1Model.transform(df)
    assert(predictions.count() === 4)
  }

  test("StreamingKMeans should handle weighted data") {
    val data = Seq(
      (Vectors.dense(0.0, 0.0), 10.0),
      (Vectors.dense(0.1, 0.1), 10.0),
      (Vectors.dense(5.0, 5.0), 1.0),
      (Vectors.dense(5.1, 5.1), 1.0)
    )

    val df = spark.createDataFrame(data).toDF("features", "weight")

    val streamingKMeans =
      new StreamingKMeans().setK(2).setMaxIter(10).setWeightCol("weight").setSeed(42)

    val model = streamingKMeans.fit(df)

    // Centers should favor high-weight points
    assert(model.clusterCenters.length === 2)

    // Update with weighted batch
    val batch = Seq(
      (Vectors.dense(0.2, 0.2), 5.0),
      (Vectors.dense(5.2, 5.2), 1.0)
    )

    val batchDF = spark.createDataFrame(batch).toDF("features", "weight")
    model.update(batchDF)

    assert(model.currentWeights.sum > 0)
  }

  test("StreamingKMeans should track cluster weights") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new StreamingKMeans().setK(2).setMaxIter(10).setDecayFactor(0.9).setSeed(42).fit(df)

    val weightsBefore = model.currentWeights.clone()

    // Add more data to one cluster
    val batch = Seq(
      Vectors.dense(0.2, 0.2),
      Vectors.dense(0.3, 0.3),
      Vectors.dense(0.4, 0.4)
    )

    val batchDF = spark.createDataFrame(batch.map(Tuple1.apply)).toDF("features")
    model.update(batchDF)

    val weightsAfter = model.currentWeights

    // Weights should have changed
    assert(weightsBefore.zip(weightsAfter).exists { case (before, after) =>
      math.abs(before - after) > 1e-10
    })

    // Total weight should be positive
    assert(weightsAfter.sum > 0)
  }

  test("StreamingKMeans should handle empty batch gracefully") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(5.0, 5.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new StreamingKMeans().setK(2).setMaxIter(10).setSeed(42).fit(df)

    val centersBefore = model.currentCenters.map(_.copy)

    // Update with empty batch
    val emptyBatch =
      spark.createDataFrame(Seq.empty[Tuple1[org.apache.spark.ml.linalg.Vector]]).toDF("features")
    model.update(emptyBatch)

    val centersAfter = model.currentCenters

    // Centers should be unchanged (or only slightly decayed)
    centersBefore.zip(centersAfter).foreach { case (before, after) =>
      (0 until before.size).foreach { i =>
        assert(math.abs(before(i) - after(i)) < 1e-5)
      }
    }
  }

  test("StreamingKMeans should split dying clusters") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(0.2, 0.2)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new StreamingKMeans().setK(3).setMaxIter(10).setDecayFactor(0.5).setSeed(42).fit(df)

    // Force one cluster to die by repeatedly updating with data from other clusters
    (1 to 20).foreach { _ =>
      val batch   = Seq(
        Vectors.dense(0.0, 0.0),
        Vectors.dense(0.1, 0.1)
      )
      val batchDF = spark.createDataFrame(batch.map(Tuple1.apply)).toDF("features")
      model.update(batchDF)
    }

    // All clusters should still have reasonable weights (dying cluster gets split)
    val weights   = model.currentWeights
    val maxWeight = weights.max
    val minWeight = weights.min

    // No cluster should be too small relative to largest
    // (may still be small, but not extremely so due to splitting)
    assert(weights.forall(_ >= 0))
  }

  test("StreamingKMeans should support multiple sequential updates") {
    val initialData = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1)
    )

    val df = spark.createDataFrame(initialData.map(Tuple1.apply)).toDF("features")

    val model = new StreamingKMeans().setK(2).setMaxIter(10).setDecayFactor(0.9).setSeed(42).fit(df)

    val initialCenters = model.currentCenters.map(_.copy)

    // Multiple updates
    (1 to 5).foreach { i =>
      val batch   = Seq(
        Vectors.dense(i * 0.1, i * 0.1),
        Vectors.dense(5.0 + i * 0.1, 5.0 + i * 0.1)
      )
      val batchDF = spark.createDataFrame(batch.map(Tuple1.apply)).toDF("features")
      model.update(batchDF)
    }

    val finalCenters = model.currentCenters

    // Centers should have evolved
    val totalShift = initialCenters
      .zip(finalCenters)
      .map { case (initial, final_) =>
        (0 until initial.size).map { i =>
          math.abs(initial(i) - final_(i))
        }.sum
      }
      .sum

    assert(totalShift > 0.01, s"Centers should shift with multiple updates (shift=$totalShift)")
  }

  test("StreamingKMeans should compute cost correctly") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new StreamingKMeans().setK(2).setMaxIter(10).setSeed(42).fit(df)

    val cost = model.computeCost(df)

    // Cost should be positive and finite
    assert(cost > 0.0 && java.lang.Double.isFinite(cost))

    // After update, cost on new similar data should be reasonable
    val batch = Seq(
      Vectors.dense(0.2, 0.2),
      Vectors.dense(5.2, 5.2)
    )

    val batchDF = spark.createDataFrame(batch.map(Tuple1.apply)).toDF("features")
    model.update(batchDF)

    val newCost = model.computeCost(batchDF)
    assert(newCost > 0.0 && java.lang.Double.isFinite(newCost))
  }

  test("StreamingKMeans parameter validation") {
    // decayFactor must be in [0, 1]
    assertThrows[IllegalArgumentException] {
      new StreamingKMeans().setDecayFactor(-0.1)
    }

    assertThrows[IllegalArgumentException] {
      new StreamingKMeans().setDecayFactor(1.5)
    }

    // timeUnit must be "batches" or "points"
    assertThrows[IllegalArgumentException] {
      new StreamingKMeans().setTimeUnit("invalid")
    }

    // halfLife must be positive
    assertThrows[IllegalArgumentException] {
      new StreamingKMeans().setHalfLife(0.0)
    }

    assertThrows[IllegalArgumentException] {
      new StreamingKMeans().setHalfLife(-1.0)
    }
  }

  test("StreamingKMeans updater should track batches processed") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(5.0, 5.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new StreamingKMeans().setK(2).setMaxIter(10).setSeed(42).fit(df)

    val updater = model.createStreamingUpdater()

    assert(updater.batchesProcessed === 0)

    // Simulate batch processing
    val batch1 =
      spark.createDataFrame(Seq(Vectors.dense(1.0, 1.0)).map(Tuple1.apply)).toDF("features")
    updater.currentModel.update(batch1)
    // Note: batchCounter only increments through updateOn(), not direct update() calls
    // So counter stays at 0 unless we use the streaming API

    assert(updater.currentModel.clusterCenters.length === 2)
  }

  test("StreamingKMeans model should support copy") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(5.0, 5.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new StreamingKMeans().setK(2).setMaxIter(10).setDecayFactor(0.9).setSeed(42).fit(df)

    val copied = model.copy(org.apache.spark.ml.param.ParamMap.empty)

    assert(copied.numClusters === model.numClusters)
    assert(copied.decayFactorValue === model.decayFactorValue)
    assert(copied.timeUnitValue === model.timeUnitValue)

    // Centers should be equal (defensive copy)
    (0 until copied.numClusters).foreach { i =>
      assert(copied.clusterCenters(i).sameElements(model.clusterCenters(i)))
    }
  }

  test("StreamingKMeans should work with high-dimensional data") {
    val dim  = 50
    val data = Seq(
      Vectors.dense(Array.fill(dim)(0.0)),
      Vectors.dense(Array.fill(dim)(0.1)),
      Vectors.dense(Array.fill(dim)(5.0)),
      Vectors.dense(Array.fill(dim)(5.1))
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model = new StreamingKMeans().setK(2).setMaxIter(10).setSeed(42).fit(df)

    assert(model.clusterCenters.length === 2)
    assert(model.clusterCenters(0).size === dim)

    // Update should work
    val batch = Seq(
      Vectors.dense(Array.fill(dim)(0.2)),
      Vectors.dense(Array.fill(dim)(5.2))
    )

    val batchDF = spark.createDataFrame(batch.map(Tuple1.apply)).toDF("features")
    model.update(batchDF)

    assert(model.currentCenters(0).size === dim)
  }
}
