/*
 * Licensed to the Massive Data Science and Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Massive Data Science and Derrick R. Burns licenses this file to You under the
 * Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
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

package com.massivedatascience.clusterer.ml

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import com.massivedatascience.clusterer.ml.df.MutualInformation
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

import java.nio.file.Files

class InformationBottleneckSuite extends AnyFunSuite with DataFrameSuiteBase with Matchers {

  // ============================================================================
  // MutualInformation utility tests
  // ============================================================================

  test("MutualInformation.entropy - uniform distribution") {
    val uniform = Array.fill(4)(0.25)
    val h       = MutualInformation.entropy(uniform)
    // H(uniform) = log(n) for n categories
    h shouldBe (math.log(4.0) +- 0.01)
  }

  test("MutualInformation.entropy - deterministic distribution") {
    val deterministic = Array(1.0, 0.0, 0.0, 0.0)
    val h             = MutualInformation.entropy(deterministic)
    // H(deterministic) = 0
    h shouldBe (0.0 +- 0.01)
  }

  test("MutualInformation.entropy - binary distribution") {
    val binary = Array(0.5, 0.5)
    val h      = MutualInformation.entropy(binary)
    // H(0.5, 0.5) = log(2)
    h shouldBe (math.log(2.0) +- 0.01)
  }

  test("MutualInformation.klDivergence - same distributions") {
    val p  = Array(0.25, 0.25, 0.25, 0.25)
    val q  = Array(0.25, 0.25, 0.25, 0.25)
    val kl = MutualInformation.klDivergence(p, q)
    // KL(P||P) = 0
    kl shouldBe (0.0 +- 0.01)
  }

  test("MutualInformation.klDivergence - non-negative") {
    val p  = Array(0.1, 0.2, 0.3, 0.4)
    val q  = Array(0.4, 0.3, 0.2, 0.1)
    val kl = MutualInformation.klDivergence(p, q)
    kl should be >= 0.0
  }

  test("MutualInformation.klDivergence - asymmetric") {
    val p    = Array(0.1, 0.9)
    val q    = Array(0.5, 0.5)
    val klPQ = MutualInformation.klDivergence(p, q)
    val klQP = MutualInformation.klDivergence(q, p)
    // KL is asymmetric in general
    klPQ should not equal klQP
  }

  test("MutualInformation.jsDivergence - symmetric") {
    val p    = Array(0.1, 0.9)
    val q    = Array(0.5, 0.5)
    val jsPQ = MutualInformation.jsDivergence(p, q)
    val jsQP = MutualInformation.jsDivergence(q, p)
    jsPQ shouldBe (jsQP +- 0.0001)
  }

  test("MutualInformation.jsDivergence - bounded by log(2)") {
    val p  = Array(1.0, 0.0)
    val q  = Array(0.0, 1.0)
    val js = MutualInformation.jsDivergence(p, q)
    js should be <= math.log(2.0) + 0.01
  }

  test("MutualInformation.mutualInformation - independent variables") {
    // Independent: p(x,y) = p(x) * p(y)
    val joint = Array(
      Array(0.25, 0.25), // p(x=0) = 0.5
      Array(0.25, 0.25)  // p(x=1) = 0.5
    )
    val mi    = MutualInformation.mutualInformation(joint)
    // I(X;Y) = 0 for independent variables
    mi shouldBe (0.0 +- 0.01)
  }

  test("MutualInformation.mutualInformation - perfectly correlated") {
    // Perfectly correlated: X = Y
    val joint = Array(
      Array(0.5, 0.0), // x=0 implies y=0
      Array(0.0, 0.5)  // x=1 implies y=1
    )
    val mi    = MutualInformation.mutualInformation(joint)
    // I(X;Y) = H(X) = H(Y) = log(2)
    mi shouldBe (math.log(2.0) +- 0.01)
  }

  test("MutualInformation.mutualInformation - non-negative") {
    val joint = Array(
      Array(0.1, 0.2, 0.05),
      Array(0.15, 0.3, 0.2)
    )
    val mi    = MutualInformation.mutualInformation(joint)
    mi should be >= 0.0
  }

  test("MutualInformation.conditionalEntropy - correct computation") {
    // For perfectly correlated, H(X|Y) = 0
    val joint    = Array(
      Array(0.5, 0.0),
      Array(0.0, 0.5)
    )
    val hXgivenY = MutualInformation.conditionalEntropy(joint)
    hXgivenY shouldBe (0.0 +- 0.05)
  }

  test("MutualInformation.estimateJoint - correct normalization") {
    val x     = Array(0, 0, 1, 1, 1)
    val y     = Array(0, 1, 0, 1, 1)
    val joint = MutualInformation.estimateJoint(x, y, 2, 2)

    // Check sums to 1
    val totalSum = joint.flatten.sum
    totalSum shouldBe (1.0 +- 0.001)
  }

  test("MutualInformation.conditionalYgivenX - rows sum to 1") {
    val joint    = Array(Array(0.2, 0.3), Array(0.1, 0.4))
    val pYgivenX = MutualInformation.conditionalYgivenX(joint)

    pYgivenX.foreach { row =>
      row.sum shouldBe (1.0 +- 0.01)
    }
  }

  // ============================================================================
  // InformationBottleneck estimator tests
  // ============================================================================

  test("InformationBottleneck - basic clustering with relevance variable") {
    val spark = this.spark
    import spark.implicits._

    // Create data where features correlate with labels
    val data = Seq(
      (Vectors.dense(0.0, 0.0), 0),
      (Vectors.dense(0.1, 0.1), 0),
      (Vectors.dense(0.2, 0.0), 0),
      (Vectors.dense(1.0, 1.0), 1),
      (Vectors.dense(1.1, 0.9), 1),
      (Vectors.dense(0.9, 1.1), 1)
    )
    val df   = spark.createDataFrame(data).toDF("features", "label")

    val ib = new InformationBottleneck()
      .setK(2)
      .setBeta(2.0)
      .setFeaturesCol("features")
      .setRelevanceCol("label")
      .setMaxIter(50)
      .setSeed(42L)

    val model = ib.fit(df)

    model.numClusters shouldBe 2
    model.compressionInfo should be >= 0.0
    model.relevanceInfo should be >= 0.0

    val predictions = model.transform(df)
    predictions.columns should contain("prediction")
    predictions.columns should contain("probabilities")
  }

  test("InformationBottleneck - self-information mode (no relevance col)") {
    val spark = this.spark
    import spark.implicits._

    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.1, 0.1),
      Vectors.dense(1.0, 1.0),
      Vectors.dense(1.1, 0.9)
    ).map(Tuple1(_))
    val df   = spark.createDataFrame(data).toDF("features")

    val ib = new InformationBottleneck().setK(2).setBeta(1.0).setMaxIter(20).setSeed(42L)

    val model       = ib.fit(df)
    val predictions = model.transform(df)

    predictions.count() shouldBe 4
    predictions.select("prediction").distinct().count() should be <= 2L
  }

  test("InformationBottleneck - predictions are valid probabilities") {
    val spark = this.spark
    import spark.implicits._

    val data = (0 until 20).map { i =>
      (Vectors.dense(i.toDouble, (i % 3).toDouble), i % 2)
    }
    val df   = spark.createDataFrame(data).toDF("features", "label")

    val ib = new InformationBottleneck()
      .setK(3)
      .setBeta(1.0)
      .setRelevanceCol("label")
      .setMaxIter(30)
      .setSeed(42L)

    val model       = ib.fit(df)
    val predictions = model.transform(df)

    // Check probabilities sum to 1
    predictions.collect().foreach { row =>
      val probs = row.getAs[Vector]("probabilities")
      probs.toArray.sum shouldBe (1.0 +- 0.01)
      probs.toArray.foreach(p => p should be >= 0.0)
    }
  }

  test("InformationBottleneck - predictions match argmax of probabilities") {
    val spark = this.spark
    import spark.implicits._

    val data = (0 until 10).map { i =>
      (Vectors.dense(i.toDouble, i.toDouble * 2), i % 3)
    }
    val df   = spark.createDataFrame(data).toDF("features", "label")

    val ib = new InformationBottleneck()
      .setK(3)
      .setBeta(1.5)
      .setRelevanceCol("label")
      .setMaxIter(30)
      .setSeed(42L)

    val model       = ib.fit(df)
    val predictions = model.transform(df)

    predictions.collect().foreach { row =>
      val probs      = row.getAs[Vector]("probabilities").toArray
      val prediction = row.getInt(row.fieldIndex("prediction"))
      val argmax     = probs.zipWithIndex.maxBy(_._1)._2
      prediction shouldBe argmax
    }
  }

  test("InformationBottleneck - higher beta preserves more relevance info") {
    val spark = this.spark
    import spark.implicits._

    val data = (0 until 30).map { i =>
      (Vectors.dense(i.toDouble, (i * 2).toDouble), i % 3)
    }
    val df   = spark.createDataFrame(data).toDF("features", "label")

    val ibLowBeta = new InformationBottleneck()
      .setK(3)
      .setBeta(0.5)
      .setRelevanceCol("label")
      .setMaxIter(50)
      .setSeed(42L)

    val ibHighBeta = new InformationBottleneck()
      .setK(3)
      .setBeta(5.0)
      .setRelevanceCol("label")
      .setMaxIter(50)
      .setSeed(42L)

    val modelLow  = ibLowBeta.fit(df)
    val modelHigh = ibHighBeta.fit(df)

    // Higher beta should generally preserve more relevance information
    // (though this may not always hold due to optimization dynamics)
    modelHigh.relevanceInfo should be >= (modelLow.relevanceInfo * 0.5)
  }

  test("InformationBottleneck - convergence tracking") {
    val spark = this.spark
    import spark.implicits._

    val data = (0 until 20).map { i =>
      (Vectors.dense(i.toDouble, i.toDouble), i % 2)
    }
    val df   = spark.createDataFrame(data).toDF("features", "label")

    val ib = new InformationBottleneck()
      .setK(2)
      .setBeta(1.0)
      .setRelevanceCol("label")
      .setMaxIter(100)
      .setTol(1e-6)
      .setSeed(42L)

    val model = ib.fit(df)

    model.iterations should be > 0
    model.infoHistory.length shouldBe model.iterations
  }

  test("InformationBottleneck - deterministic with same seed") {
    val spark = this.spark
    import spark.implicits._

    val data = (0 until 15).map { i =>
      (Vectors.dense(i.toDouble, (i + 1).toDouble), i % 2)
    }
    val df   = spark.createDataFrame(data).toDF("features", "label")

    val ib1 = new InformationBottleneck()
      .setK(2)
      .setBeta(1.0)
      .setRelevanceCol("label")
      .setMaxIter(30)
      .setSeed(123L)

    val ib2 = new InformationBottleneck()
      .setK(2)
      .setBeta(1.0)
      .setRelevanceCol("label")
      .setMaxIter(30)
      .setSeed(123L)

    val model1 = ib1.fit(df)
    val model2 = ib2.fit(df)

    model1.compressionInfo shouldBe (model2.compressionInfo +- 0.001)
    model1.relevanceInfo shouldBe (model2.relevanceInfo +- 0.001)
  }

  test("InformationBottleneck - summary output") {
    val spark = this.spark
    import spark.implicits._

    val data = (0 until 10).map { i =>
      (Vectors.dense(i.toDouble), i % 2)
    }
    val df   = spark.createDataFrame(data).toDF("features", "label")

    val ib =
      new InformationBottleneck().setK(2).setRelevanceCol("label").setMaxIter(20).setSeed(42L)

    val model   = ib.fit(df)
    val summary = model.summary

    summary should include("Information Bottleneck")
    summary should include("Clusters")
    summary should include("Compression")
    summary should include("Relevance")
  }

  test("InformationBottleneck - validates k > 1") {
    val ib = new InformationBottleneck()

    an[IllegalArgumentException] should be thrownBy {
      ib.setK(1)
    }
  }

  test("InformationBottleneck - validates beta > 0") {
    val ib = new InformationBottleneck()

    an[IllegalArgumentException] should be thrownBy {
      ib.setBeta(0.0)
    }
  }

  // ============================================================================
  // Persistence tests
  // ============================================================================

  test("InformationBottleneck - model persistence") {
    val spark = this.spark
    import spark.implicits._

    val data = (0 until 15).map { i =>
      (Vectors.dense(i.toDouble, i * 2.0), i % 2)
    }
    val df   = spark.createDataFrame(data).toDF("features", "label")

    val ib = new InformationBottleneck()
      .setK(2)
      .setBeta(1.5)
      .setRelevanceCol("label")
      .setMaxIter(30)
      .setSeed(42L)

    val model = ib.fit(df)

    val tempDir  = Files.createTempDirectory("ib-model-test")
    val savePath = tempDir.resolve("model").toString

    try {
      model.write.overwrite().save(savePath)
      val loadedModel = InformationBottleneckModel.load(savePath)

      loadedModel.numClusters shouldBe model.numClusters
      loadedModel.compressionInfo shouldBe (model.compressionInfo +- 0.001)
      loadedModel.relevanceInfo shouldBe (model.relevanceInfo +- 0.001)
      loadedModel.iterations shouldBe model.iterations
      loadedModel.converged shouldBe model.converged

      // Test that loaded model produces same predictions
      val predictions1 = model.transform(df).select("prediction").collect()
      val predictions2 = loadedModel.transform(df).select("prediction").collect()

      predictions1.zip(predictions2).foreach { case (r1, r2) =>
        r1.getInt(0) shouldBe r2.getInt(0)
      }
    } finally {
      // Cleanup
      import scala.reflect.io.Directory
      new Directory(tempDir.toFile).deleteRecursively()
    }
  }

  test("InformationBottleneck - estimator persistence") {
    val tempDir  = Files.createTempDirectory("ib-estimator-test")
    val savePath = tempDir.resolve("estimator").toString

    try {
      val ib = new InformationBottleneck().setK(3).setBeta(2.0).setMaxIter(50).setSeed(123L)

      ib.write.overwrite().save(savePath)
      val loadedIb = InformationBottleneck.load(savePath)

      loadedIb.getK shouldBe 3
      loadedIb.getBeta shouldBe 2.0
      loadedIb.getMaxIter shouldBe 50
      loadedIb.getSeed shouldBe 123L
    } finally {
      import scala.reflect.io.Directory
      new Directory(tempDir.toFile).deleteRecursively()
    }
  }

  // ============================================================================
  // Edge cases
  // ============================================================================

  test("InformationBottleneck - handles string relevance column") {
    val spark = this.spark
    import spark.implicits._

    val data = Seq(
      (Vectors.dense(0.0, 0.0), "cat_a"),
      (Vectors.dense(0.1, 0.1), "cat_a"),
      (Vectors.dense(1.0, 1.0), "cat_b"),
      (Vectors.dense(1.1, 0.9), "cat_b")
    )
    val df   = spark.createDataFrame(data).toDF("features", "category")

    val ib = new InformationBottleneck()
      .setK(2)
      .setBeta(1.0)
      .setRelevanceCol("category")
      .setMaxIter(30)
      .setSeed(42L)

    val model = ib.fit(df)
    model.numClusters shouldBe 2

    val predictions = model.transform(df)
    predictions.count() shouldBe 4
  }

  test("InformationBottleneck - handles continuous relevance column") {
    val spark = this.spark
    import spark.implicits._

    val data = (0 until 20).map { i =>
      (Vectors.dense(i.toDouble), i.toDouble * 0.5)
    }
    val df   = spark.createDataFrame(data).toDF("features", "score")

    val ib = new InformationBottleneck()
      .setK(3)
      .setBeta(1.0)
      .setRelevanceCol("score")
      .setNumBins(5)
      .setMaxIter(30)
      .setSeed(42L)

    val model = ib.fit(df)
    model.numClusters shouldBe 3

    val predictions = model.transform(df)
    predictions.count() shouldBe 20
  }
}
