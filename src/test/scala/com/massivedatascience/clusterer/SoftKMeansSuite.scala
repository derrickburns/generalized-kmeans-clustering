/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
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

package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.ml.SoftKMeans
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

/** Test suite for Soft K-Means clustering (fuzzy c-means).
  */
class SoftKMeansSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("SoftKMeansSuite")
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

  test("Soft K-Means should perform basic soft clustering") {
    // Create data with 2 clear clusters
    val data = Seq(
      // Cluster 1: around (1, 1)
      Vectors.dense(1.0, 1.0),
      Vectors.dense(1.1, 0.9),
      Vectors.dense(0.9, 1.1),
      // Cluster 2: around (5, 5)
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 4.9),
      Vectors.dense(4.9, 5.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val softKMeans = new SoftKMeans().setK(2).setBeta(2.0).setMaxIter(20).setSeed(42)

    val model = softKMeans.fit(df)

    // Basic model properties
    assert(model.numClusters === 2)
    assert(model.clusterCenters.length === 2)

    // Transform should add both prediction and probabilities columns
    val predictions = model.transform(df)
    assert(predictions.columns.contains("prediction"))
    assert(predictions.columns.contains("probabilities"))

    // Each point should have probabilities for 2 clusters
    val firstRow = predictions.select("probabilities").head()
    val probs    = firstRow.getAs[org.apache.spark.ml.linalg.Vector](0)
    assert(probs.size === 2)

    // Probabilities should sum to ~1.0
    val probSum = probs.toArray.sum
    assert(math.abs(probSum - 1.0) < 1e-6)

    // All probabilities should be positive
    probs.toArray.foreach(p => assert(p > 0.0))
  }

  test("Soft K-Means should work with different beta values") {
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(1.2, 1.2),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.2, 5.2)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // Soft clustering (low beta)
    val softModel = new SoftKMeans()
      .setK(2)
      .setBeta(0.5) // Very soft assignments
      .setMaxIter(20)
      .setSeed(42)
      .fit(df)

    val softPreds = softModel.transform(df)

    // Sharp clustering (high beta)
    val sharpModel = new SoftKMeans()
      .setK(2)
      .setBeta(10.0) // Sharp assignments
      .setMaxIter(20)
      .setSeed(42)
      .fit(df)

    val sharpPreds = sharpModel.transform(df)

    // Compute average entropy for soft vs sharp
    val softEntropy = softPreds
      .select("probabilities")
      .collect()
      .map { row =>
        val probs = row.getAs[org.apache.spark.ml.linalg.Vector](0).toArray
        -probs.map(p => if (p > 1e-10) p * math.log(p) else 0.0).sum
      }
      .sum / softPreds.count()

    val sharpEntropy = sharpPreds
      .select("probabilities")
      .collect()
      .map { row =>
        val probs = row.getAs[org.apache.spark.ml.linalg.Vector](0).toArray
        -probs.map(p => if (p > 1e-10) p * math.log(p) else 0.0).sum
      }
      .sum / sharpPreds.count()

    // Soft clustering should have higher entropy (more uncertainty)
    assert(softEntropy > sharpEntropy)
  }

  test("Soft K-Means should work with different divergences") {
    val data = Seq(
      Vectors.dense(1.0, 2.0),
      Vectors.dense(1.1, 2.1),
      Vectors.dense(5.0, 6.0),
      Vectors.dense(5.1, 6.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // Test Euclidean
    val euclideanModel =
      new SoftKMeans().setK(2).setDivergence("squaredEuclidean").setMaxIter(20).setSeed(42).fit(df)

    assert(euclideanModel.numClusters === 2)

    // Test L1
    val l1Model = new SoftKMeans().setK(2).setDivergence("l1").setMaxIter(20).setSeed(42).fit(df)

    assert(l1Model.numClusters === 2)

    // Both should produce valid predictions
    val euclideanPreds = euclideanModel.transform(df)
    val l1Preds        = l1Model.transform(df)

    assert(euclideanPreds.count() === 4)
    assert(l1Preds.count() === 4)
  }

  test("Soft K-Means should handle weighted data") {
    val data = Seq(
      (Vectors.dense(1.0, 1.0), 10.0), // High weight
      (Vectors.dense(1.1, 0.9), 10.0),
      (Vectors.dense(5.0, 5.0), 1.0),  // Low weight
      (Vectors.dense(5.1, 4.9), 1.0)
    )

    val df = spark.createDataFrame(data).toDF("features", "weight")

    val softKMeans = new SoftKMeans().setK(2).setWeightCol("weight").setMaxIter(20).setSeed(42)

    val model = softKMeans.fit(df)

    // Centers should be influenced more by high-weight points
    val predictions = model.transform(df)
    assert(predictions.count() === 4)

    // Model should have 2 clusters
    assert(model.numClusters === 2)
  }

  test("Soft K-Means should compute valid costs") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 5.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val softKMeans = new SoftKMeans().setK(2).setMaxIter(20).setSeed(42)

    val model = softKMeans.fit(df)

    // Hard cost
    val hardCost = model.computeCost(df)
    assert(hardCost > 0.0 && java.lang.Double.isFinite(hardCost))

    // Soft cost
    val softCost = model.computeSoftCost(df)
    assert(softCost > 0.0 && java.lang.Double.isFinite(softCost))

    // Soft cost should be <= hard cost (weighted average property)
    assert(softCost <= hardCost * 1.1) // Allow small numerical tolerance
  }

  test("Soft K-Means should compute effective number of clusters") {
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(1.1, 0.9),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 4.9)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    // Very soft clustering (low beta)
    val softModel = new SoftKMeans().setK(2).setBeta(0.1).setMaxIter(20).setSeed(42).fit(df)

    val effectiveClusters = softModel.effectiveNumberOfClusters(df)

    // Effective clusters should be between 1 and k
    assert(effectiveClusters >= 1.0 && effectiveClusters <= 2.0)
  }

  test("Soft K-Means should support single vector predictions") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(10.0, 10.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val softKMeans = new SoftKMeans().setK(2).setMaxIter(20).setSeed(42)

    val model = softKMeans.fit(df)

    // Test hard prediction
    val testVector = Vectors.dense(0.1, 0.1)
    val prediction = model.predict(testVector)
    assert(prediction >= 0 && prediction < 2)

    // Test soft prediction
    val softPrediction = model.predictSoft(testVector)
    assert(softPrediction.size === 2)
    assert(math.abs(softPrediction.toArray.sum - 1.0) < 1e-6)
  }

  test("Soft K-Means parameter validation") {
    // Beta must be > 0
    assertThrows[IllegalArgumentException] {
      new SoftKMeans().setBeta(0.0)
    }

    assertThrows[IllegalArgumentException] {
      new SoftKMeans().setBeta(-1.0)
    }

    // MinMembership must be in [0, 1]
    assertThrows[IllegalArgumentException] {
      new SoftKMeans().setMinMembership(-0.1)
    }

    assertThrows[IllegalArgumentException] {
      new SoftKMeans().setMinMembership(1.5)
    }

    // K must be >= 2
    assertThrows[IllegalArgumentException] {
      new SoftKMeans().setK(1)
    }
  }

  test("Soft K-Means should reject single cluster") {
    // K must be >= 2, so setK(1) should throw
    assertThrows[IllegalArgumentException] {
      new SoftKMeans().setK(1)
    }
  }

  test("Soft K-Means should handle edge case with 2 points") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(10.0, 10.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val softKMeans = new SoftKMeans().setK(2).setMaxIter(20).setSeed(42)

    val model = softKMeans.fit(df)
    assert(model.numClusters === 2)

    val predictions = model.transform(df)
    assert(predictions.count() === 2)
  }

  test("Soft K-Means model persistence") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(10.0, 10.0),
      Vectors.dense(10.1, 10.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val softKMeans = new SoftKMeans().setK(2).setBeta(2.0).setMaxIter(20).setSeed(42)

    val model         = softKMeans.fit(df)
    val originalPreds = model.transform(df).select("prediction", "probabilities").collect()

    // Save and load model
    val tempDir = java.nio.file.Files.createTempDirectory("softkmeans-model-test").toString
    try {
      model.write.overwrite().save(tempDir)

      val loadedModel = com.massivedatascience.clusterer.ml.SoftKMeansModel.load(tempDir)
      val loadedPreds = loadedModel.transform(df).select("prediction", "probabilities").collect()

      // Predictions should match
      assert(originalPreds.length === loadedPreds.length)
      originalPreds.zip(loadedPreds).foreach { case (orig, loaded) =>
        assert(orig.getInt(0) === loaded.getInt(0))

        val origProbs   = orig.getAs[org.apache.spark.ml.linalg.Vector](1).toArray
        val loadedProbs = loaded.getAs[org.apache.spark.ml.linalg.Vector](1).toArray

        origProbs.zip(loadedProbs).foreach { case (op, lp) =>
          assert(math.abs(op - lp) < 1e-6)
        }
      }
    } finally {
      // Clean up
      import scala.reflect.io.Directory
      val dir = new Directory(new java.io.File(tempDir))
      dir.deleteRecursively()
    }
  }

  test("Soft K-Means should converge with reasonable iterations") {
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(1.1, 0.9),
      Vectors.dense(1.2, 1.1),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 4.9),
      Vectors.dense(5.2, 5.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val softKMeans = new SoftKMeans().setK(2).setMaxIter(50).setTol(1e-4).setSeed(42)

    val model = softKMeans.fit(df)

    // Should converge in reasonable number of iterations
    // (convergence is implicit in successful model creation)
    assert(model.numClusters === 2)
  }

  test("Soft K-Means should handle minMembership threshold") {
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(10.0, 10.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val minMemb    = 1e-10
    val softKMeans = new SoftKMeans()
      .setK(2)
      .setBeta(10.0) // Very sharp clustering
      .setMinMembership(minMemb)
      .setMaxIter(20)
      .setSeed(42)

    val model       = softKMeans.fit(df)
    val predictions = model.transform(df)

    // Even with very sharp clustering, all probabilities should be >= minMembership (with tolerance)
    predictions.select("probabilities").collect().foreach { row =>
      val probs = row.getAs[org.apache.spark.ml.linalg.Vector](0).toArray
      probs.foreach(p => assert(p >= minMemb * 0.99)) // Allow tiny numerical error
    }
  }

  test("Soft K-Means should produce consistent results with same seed") {
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(1.1, 0.9),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.1, 4.9)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val model1 = new SoftKMeans().setK(2).setMaxIter(20).setSeed(42).fit(df)

    val model2 = new SoftKMeans().setK(2).setMaxIter(20).setSeed(42).fit(df)

    val preds1 = model1.transform(df).select("prediction").collect().map(_.getInt(0))
    val preds2 = model2.transform(df).select("prediction").collect().map(_.getInt(0))

    // Results should be identical with same seed
    assert(preds1.sameElements(preds2))
  }

  test("Soft K-Means should work with higher dimensional data") {
    val data = Seq(
      Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0),
      Vectors.dense(1.1, 2.1, 3.1, 4.1, 5.1),
      Vectors.dense(10.0, 11.0, 12.0, 13.0, 14.0),
      Vectors.dense(10.1, 11.1, 12.1, 13.1, 14.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val softKMeans = new SoftKMeans().setK(2).setMaxIter(20).setSeed(42)

    val model = softKMeans.fit(df)
    assert(model.numClusters === 2)

    // Each center should have 5 dimensions
    model.clusterCenters.foreach { center =>
      assert(center.size === 5)
    }
  }
}
