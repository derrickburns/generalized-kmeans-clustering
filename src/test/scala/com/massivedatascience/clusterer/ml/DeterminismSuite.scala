package com.massivedatascience.clusterer.ml

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.scalatest.matchers.should.Matchers

/** Tests for determinism - same seed produces identical results.
  *
  * These tests verify that all clustering estimators are fully deterministic when given the same
  * seed, which is critical for reproducibility in production environments.
  *
  * Tests cover all 5 main estimators:
  *   - GeneralizedKMeans
  *   - BisectingGeneralizedKMeans
  *   - XMeans
  *   - SoftKMeans
  *   - StreamingKMeans
  */
class DeterminismSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  private val spark: SparkSession = SparkSession
    .builder()
    .master("local[2]")
    .appName("DeterminismSuite")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  import spark.implicits._

  override def beforeAll(): Unit = {
    super.beforeAll()
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

  /** Create test DataFrame with well-separated clusters */
  private def testDF() = {
    Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.5, 0.5)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(9.0, 9.0)),
      Tuple1(Vectors.dense(9.5, 9.5)),
      Tuple1(Vectors.dense(10.0, 10.0))
    ).toDF("features")
  }

  test("GeneralizedKMeans: same seed produces identical centers") {
    val df = testDF()

    val model1 = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setSeed(1234)
      .setMaxIter(10)
      .fit(df)

    val model2 = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setSeed(1234)
      .setMaxIter(10)
      .fit(df)

    // Centers should be identical within epsilon
    model1.clusterCenters.length shouldBe model2.clusterCenters.length
    model1.clusterCenters.zip(model2.clusterCenters).foreach { case (c1, c2) =>
      c1.zip(c2).foreach { case (x1, x2) =>
        math.abs(x1 - x2) should be < 1e-10
      }
    }

    // Predictions should be identical
    val pred1 = model1.transform(df).select("prediction").collect().map(_.getInt(0))
    val pred2 = model2.transform(df).select("prediction").collect().map(_.getInt(0))
    pred1 should contain theSameElementsInOrderAs pred2
  }

  test("GeneralizedKMeans: same seed with KL divergence produces identical centers") {
    val df = Seq(
      Tuple1(Vectors.dense(0.1, 0.2)),
      Tuple1(Vectors.dense(0.15, 0.25)),
      Tuple1(Vectors.dense(0.9, 0.8)),
      Tuple1(Vectors.dense(0.95, 0.85))
    ).toDF("features")

    val model1 = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("kl")
      .setSmoothing(1e-6)
      .setSeed(5678)
      .setMaxIter(10)
      .fit(df)

    val model2 = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("kl")
      .setSmoothing(1e-6)
      .setSeed(5678)
      .setMaxIter(10)
      .fit(df)

    // Centers should be identical
    model1.clusterCenters.length shouldBe model2.clusterCenters.length
    model1.clusterCenters.zip(model2.clusterCenters).foreach { case (c1, c2) =>
      c1.zip(c2).foreach { case (x1, x2) =>
        math.abs(x1 - x2) should be < 1e-8
      }
    }
  }

  test("BisectingKMeans: same seed produces identical centers") {
    val df = testDF()

    val model1 = new BisectingKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setSeed(2345)
      .setMaxIter(10)
      .fit(df)

    val model2 = new BisectingKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setSeed(2345)
      .setMaxIter(10)
      .fit(df)

    // Centers should be identical
    model1.clusterCenters.length shouldBe model2.clusterCenters.length
    model1.clusterCenters.zip(model2.clusterCenters).foreach { case (c1, c2) =>
      c1.zip(c2).foreach { case (x1, x2) =>
        math.abs(x1 - x2) should be < 1e-10
      }
    }

    // Predictions should be identical
    val pred1 = model1.transform(df).select("prediction").collect().map(_.getInt(0))
    val pred2 = model2.transform(df).select("prediction").collect().map(_.getInt(0))
    pred1 should contain theSameElementsInOrderAs pred2
  }

  test("XMeans: same seed produces identical k and centers") {
    val df = testDF()

    val model1 = new XMeans()
      .setMinK(2)
      .setMaxK(4)
      .setDivergence("squaredEuclidean")
      .setSeed(3456)
      .setMaxIter(10)
      .fit(df)

    val model2 = new XMeans()
      .setMinK(2)
      .setMaxK(4)
      .setDivergence("squaredEuclidean")
      .setSeed(3456)
      .setMaxIter(10)
      .fit(df)

    // Should find same number of clusters
    model1.numClusters shouldBe model2.numClusters

    // Centers should be identical
    model1.clusterCenters.zip(model2.clusterCenters).foreach { case (c1, c2) =>
      c1.zip(c2).foreach { case (x1, x2) =>
        math.abs(x1 - x2) should be < 1e-10
      }
    }

    // Predictions should be identical
    val pred1 = model1.transform(df).select("prediction").collect().map(_.getInt(0))
    val pred2 = model2.transform(df).select("prediction").collect().map(_.getInt(0))
    pred1 should contain theSameElementsInOrderAs pred2
  }

  test("SoftKMeans: same seed produces identical centers and probabilities") {
    val df = testDF()

    val model1 = new SoftKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setBeta(1.5)
      .setSeed(4567)
      .setMaxIter(10)
      .fit(df)

    val model2 = new SoftKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setBeta(1.5)
      .setSeed(4567)
      .setMaxIter(10)
      .fit(df)

    // Centers should be identical (SoftKMeans uses Array[Vector])
    model1.clusterCenters.length shouldBe model2.clusterCenters.length
    model1.clusterCenters.zip(model2.clusterCenters).foreach { case (c1, c2) =>
      c1.toArray.zip(c2.toArray).foreach { case (x1, x2) =>
        math.abs(x1 - x2) should be < 1e-10
      }
    }

    // Predictions should be identical
    val pred1 = model1.transform(df).select("prediction").collect().map(_.getInt(0))
    val pred2 = model2.transform(df).select("prediction").collect().map(_.getInt(0))
    pred1 should contain theSameElementsInOrderAs pred2

    // Probabilities should be identical
    val probs1 = model1.transform(df).select("probabilities").collect()
    val probs2 = model2.transform(df).select("probabilities").collect()
    probs1.zip(probs2).foreach { case (p1, p2) =>
      val arr1 = p1.getAs[org.apache.spark.ml.linalg.Vector](0).toArray
      val arr2 = p2.getAs[org.apache.spark.ml.linalg.Vector](0).toArray
      arr1.zip(arr2).foreach { case (a1, a2) =>
        math.abs(a1 - a2) should be < 1e-10
      }
    }
  }

  test("StreamingKMeans: same seed produces identical centers") {
    val df = testDF()

    val model1 = new StreamingKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setSeed(5678)
      .setDecayFactor(0.9)
      .setMaxIter(10)
      .fit(df)

    val model2 = new StreamingKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setSeed(5678)
      .setDecayFactor(0.9)
      .setMaxIter(10)
      .fit(df)

    // Centers should be identical
    model1.clusterCenters.length shouldBe model2.clusterCenters.length
    model1.clusterCenters.zip(model2.clusterCenters).foreach { case (c1, c2) =>
      c1.zip(c2).foreach { case (x1, x2) =>
        math.abs(x1 - x2) should be < 1e-10
      }
    }

    // Predictions should be identical
    val pred1 = model1.transform(df).select("prediction").collect().map(_.getInt(0))
    val pred2 = model2.transform(df).select("prediction").collect().map(_.getInt(0))
    pred1 should contain theSameElementsInOrderAs pred2
  }

  test("KMedoids: same seed produces identical medoids") {
    val df = testDF()

    val model1 =
      new KMedoids().setK(2).setDistanceFunction("euclidean").setSeed(6789).setMaxIter(10).fit(df)

    val model2 =
      new KMedoids().setK(2).setDistanceFunction("euclidean").setSeed(6789).setMaxIter(10).fit(df)

    // Medoid indices should be identical
    model1.medoidIndices.length shouldBe model2.medoidIndices.length
    model1.medoidIndices.zip(model2.medoidIndices).foreach { case (i1, i2) =>
      i1 shouldBe i2
    }

    // Medoid vectors should be identical
    model1.medoids.zip(model2.medoids).foreach { case (m1, m2) =>
      m1.toArray.zip(m2.toArray).foreach { case (x1, x2) =>
        math.abs(x1 - x2) should be < 1e-10
      }
    }

    // Predictions should be identical
    val pred1 = model1.transform(df).select("prediction").collect().map(_.getInt(0))
    val pred2 = model2.transform(df).select("prediction").collect().map(_.getInt(0))
    pred1 should contain theSameElementsInOrderAs pred2
  }

  test("GeneralizedKMeans: different seeds produce different results") {
    val df = testDF()

    val model1 = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setSeed(1111)
      .setMaxIter(10)
      .fit(df)

    val model2 = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setSeed(2222)
      .setMaxIter(10)
      .fit(df)

    // Centers should be different (at least one coordinate should differ)
    val allIdentical = model1.clusterCenters.zip(model2.clusterCenters).forall { case (c1, c2) =>
      c1.zip(c2).forall { case (x1, x2) => math.abs(x1 - x2) < 1e-10 }
    }
    allIdentical shouldBe false
  }
}
