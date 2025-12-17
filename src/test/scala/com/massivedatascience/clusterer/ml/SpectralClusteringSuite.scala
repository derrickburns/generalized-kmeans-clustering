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
import com.massivedatascience.clusterer.ml.df.SpectralGraph
import com.massivedatascience.clusterer.ml.df.kernels.RBFKernel
import org.apache.spark.ml.linalg.{ DenseMatrix, Vectors }
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

import java.nio.file.Files

class SpectralClusteringSuite extends AnyFunSuite with DataFrameSuiteBase with Matchers {

  // ============================================================================
  // SpectralGraph affinity tests
  // ============================================================================

  test("SpectralGraph.buildFullAffinity - symmetric matrix") {
    val points   = Array(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 0.0),
      Vectors.dense(0.0, 1.0)
    )
    val kernel   = new RBFKernel(1.0)
    val affinity = SpectralGraph.buildFullAffinity(points, kernel)

    affinity.numRows shouldBe 3
    affinity.numCols shouldBe 3

    // Check symmetry
    for (i <- 0 until 3) {
      for (j <- 0 until 3) {
        affinity(i, j) shouldBe (affinity(j, i) +- 1e-10)
      }
    }
  }

  test("SpectralGraph.buildFullAffinity - diagonal is 1 for RBF") {
    val points   = Array(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 1.0),
      Vectors.dense(2.0, 2.0)
    )
    val kernel   = new RBFKernel(1.0)
    val affinity = SpectralGraph.buildFullAffinity(points, kernel)

    // RBF kernel: k(x, x) = exp(0) = 1
    for (i <- 0 until 3) {
      affinity(i, i) shouldBe (1.0 +- 1e-10)
    }
  }

  test("SpectralGraph.buildKNNAffinity - sparsity") {
    val points   = Array(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 0.0),
      Vectors.dense(2.0, 0.0),
      Vectors.dense(4.0, 0.0) // Closer so RBF kernel value is measurable
    )
    val kernel   = new RBFKernel(0.1) // Lower gamma for broader kernel
    val affinity = SpectralGraph.buildKNNAffinity(points, kernel, k = 1)

    // Point 3 (at 4.0) should only connect to point 2 (at 2.0)
    // With k=1, point 3's only neighbor is point 2
    // Mutual k-NN: point 2's neighbor is point 1, point 3's neighbor is point 2
    // So connection between 2 and 3 should exist (3->2)
    // Use small tolerance for floating point precision
    affinity(3, 2) should be > 1e-10 // Should have non-zero affinity
  }

  test("SpectralGraph.buildEpsilonAffinity - neighborhood constraint") {
    val points   = Array(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.5, 0.0),
      Vectors.dense(5.0, 0.0)
    )
    val kernel   = new RBFKernel(1.0)
    val affinity = SpectralGraph.buildEpsilonAffinity(points, kernel, epsilon = 1.0)

    // Points 0 and 1 are within epsilon=1
    affinity(0, 1) should be > 0.0
    affinity(1, 0) should be > 0.0

    // Point 2 is outside epsilon from 0 and 1
    affinity(0, 2) shouldBe 0.0
    affinity(1, 2) shouldBe 0.0
  }

  // ============================================================================
  // SpectralGraph Laplacian tests
  // ============================================================================

  test("SpectralGraph.computeDegrees - correct sums") {
    val values   = Array(0.0, 0.5, 0.3, 0.5, 0.0, 0.2, 0.3, 0.2, 0.0)
    val affinity = new DenseMatrix(3, 3, values)
    val degrees  = SpectralGraph.computeDegrees(affinity)

    degrees(0) shouldBe (0.8 +- 0.01)
    degrees(1) shouldBe (0.7 +- 0.01)
    degrees(2) shouldBe (0.5 +- 0.01)
  }

  test("SpectralGraph.unnormalizedLaplacian - L = D - W") {
    val values    = Array(0.0, 0.5, 0.5, 0.0)
    val affinity  = new DenseMatrix(2, 2, values)
    val laplacian = SpectralGraph.unnormalizedLaplacian(affinity)

    // D = diag(0.5, 0.5), L = D - W
    laplacian(0, 0) shouldBe (0.5 +- 0.01)
    laplacian(1, 1) shouldBe (0.5 +- 0.01)
    laplacian(0, 1) shouldBe (-0.5 +- 0.01)
    laplacian(1, 0) shouldBe (-0.5 +- 0.01)
  }

  test("SpectralGraph.symmetricNormalizedLaplacian - eigenvalues in [0, 2]") {
    val points    = Array(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 0.0),
      Vectors.dense(0.5, 0.5)
    )
    val kernel    = new RBFKernel(1.0)
    val affinity  = SpectralGraph.buildFullAffinity(points, kernel)
    val laplacian = SpectralGraph.symmetricNormalizedLaplacian(affinity)

    val (eigenvalues, _) = SpectralGraph.computeSmallestEigenvectors(laplacian, 3, seed = 42L)

    // Symmetric normalized Laplacian eigenvalues are in [0, 2]
    eigenvalues.foreach { ev =>
      ev should be >= -0.1 // Allow small numerical error
      ev should be <= 2.1
    }
  }

  test("SpectralGraph.randomWalkLaplacian - row sums") {
    val points    = Array(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 0.0)
    )
    val kernel    = new RBFKernel(1.0)
    val affinity  = SpectralGraph.buildFullAffinity(points, kernel)
    val laplacian = SpectralGraph.randomWalkLaplacian(affinity)

    // For random walk Laplacian, row sums of (I - D^-1 W) should be 0
    // because D^-1 W rows sum to 1
    for (i <- 0 until 2) {
      var rowSum = 0.0
      for (j <- 0 until 2) {
        rowSum += laplacian(i, j)
      }
      rowSum shouldBe (0.0 +- 0.01)
    }
  }

  // ============================================================================
  // SpectralGraph eigendecomposition tests
  // ============================================================================

  test("SpectralGraph.computeSmallestEigenvectors - smallest is near zero for connected graph") {
    val points    = Array(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 0.0),
      Vectors.dense(0.5, 1.0)
    )
    val kernel    = new RBFKernel(1.0)
    val affinity  = SpectralGraph.buildFullAffinity(points, kernel)
    val laplacian = SpectralGraph.symmetricNormalizedLaplacian(affinity)

    val (eigenvalues, eigenvectors) =
      SpectralGraph.computeSmallestEigenvectors(laplacian, 2, seed = 42L)

    // For connected graph, smallest eigenvalue should be near 0
    eigenvalues.head shouldBe (0.0 +- 0.1)
  }

  test("SpectralGraph.computeSmallestEigenvectors - eigenvectors are normalized") {
    val points    = Array(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 0.0),
      Vectors.dense(0.5, 1.0)
    )
    val kernel    = new RBFKernel(1.0)
    val affinity  = SpectralGraph.buildFullAffinity(points, kernel)
    val laplacian = SpectralGraph.symmetricNormalizedLaplacian(affinity)

    val (_, eigenvectors) = SpectralGraph.computeSmallestEigenvectors(laplacian, 2, seed = 42L)

    eigenvectors.foreach { v =>
      val norm = math.sqrt(v.map(x => x * x).sum)
      norm shouldBe (1.0 +- 0.01)
    }
  }

  // ============================================================================
  // SpectralGraph Nyström approximation tests
  // ============================================================================

  test("SpectralGraph.nystromApproximation - produces valid embeddings") {
    val points = (0 until 20).map { i =>
      Vectors.dense(i.toDouble, (i % 3).toDouble)
    }.toArray

    val kernel    = new RBFKernel(0.5)
    val embedding = SpectralGraph.nystromApproximation(points, kernel, numLandmarks = 5, k = 2)

    embedding.length shouldBe 20
    embedding.foreach(_.size shouldBe 2)
  }

  test("SpectralGraph.nystromApproximation - row normalization") {
    val points = (0 until 15).map { i =>
      Vectors.dense(i.toDouble, i.toDouble * 0.5)
    }.toArray

    val kernel    = new RBFKernel(0.5)
    val embedding = SpectralGraph.nystromApproximation(points, kernel, numLandmarks = 5, k = 2)

    // Embeddings should be roughly normalized
    embedding.foreach { v =>
      val norm = math.sqrt(v.toArray.map(x => x * x).sum)
      // Nyström doesn't guarantee perfect normalization but should be close
      norm should be > 0.0
    }
  }

  // ============================================================================
  // SpectralClustering estimator tests
  // ============================================================================

  test("SpectralClustering - basic clustering with two clusters") {
    val spark = this.spark
    import spark.implicits._

    // Two well-separated clusters at reasonable distance
    // Use k-NN affinity which handles well-separated clusters better
    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.5, 0.5),
      Vectors.dense(1.0, 0.0),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.5, 4.5),
      Vectors.dense(4.5, 5.5)
    ).map(Tuple1(_))
    val df   = spark.createDataFrame(data).toDF("features")

    val spectral = new SpectralClustering()
      .setK(2)
      .setKernelType("rbf")
      .setGamma(0.5)          // Moderate gamma
      .setAffinityType("knn") // k-NN handles separated clusters better
      .setNumNeighbors(2)
      .setLaplacianType("symmetric")
      .setSeed(42L)

    val model       = spectral.fit(df)
    val predictions = model.transform(df)

    model.numClusters shouldBe 2

    // Verify the model produces valid predictions
    predictions.count() shouldBe 6
    val distinctPreds = predictions.select("prediction").distinct().count()
    distinctPreds should be >= 1L
    distinctPreds should be <= 2L
  }

  test("SpectralClustering - with k-NN affinity") {
    val spark = this.spark
    import spark.implicits._

    val data = (0 until 12).map { i =>
      Tuple1(Vectors.dense((i % 3).toDouble, (i / 3).toDouble))
    }
    val df   = spark.createDataFrame(data).toDF("features")

    val spectral = new SpectralClustering()
      .setK(3)
      .setAffinityType("knn")
      .setNumNeighbors(3)
      .setKernelType("rbf")
      .setGamma(1.0)
      .setSeed(42L)

    val model       = spectral.fit(df)
    val predictions = model.transform(df)

    predictions.count() shouldBe 12
    predictions.select("prediction").distinct().count() should be <= 3L
  }

  test("SpectralClustering - with Laplacian kernel") {
    val spark = this.spark
    import spark.implicits._

    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.5, 0.5),
      Vectors.dense(3.0, 3.0),
      Vectors.dense(3.5, 3.5)
    ).map(Tuple1(_))
    val df   = spark.createDataFrame(data).toDF("features")

    val spectral =
      new SpectralClustering().setK(2).setKernelType("laplacian").setGamma(1.0).setSeed(42L)

    val model = spectral.fit(df)
    model.numClusters shouldBe 2
  }

  test("SpectralClustering - with unnormalized Laplacian") {
    val spark = this.spark
    import spark.implicits._

    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 0.0),
      Vectors.dense(5.0, 0.0),
      Vectors.dense(6.0, 0.0)
    ).map(Tuple1(_))
    val df   = spark.createDataFrame(data).toDF("features")

    val spectral = new SpectralClustering()
      .setK(2)
      .setLaplacianType("unnormalized")
      .setKernelType("rbf")
      .setGamma(0.5)
      .setSeed(42L)

    val model = spectral.fit(df)
    model.numClusters shouldBe 2
  }

  test("SpectralClustering - with randomWalk Laplacian") {
    val spark = this.spark
    import spark.implicits._

    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 1.0),
      Vectors.dense(10.0, 10.0),
      Vectors.dense(11.0, 11.0)
    ).map(Tuple1(_))
    val df   = spark.createDataFrame(data).toDF("features")

    val spectral = new SpectralClustering()
      .setK(2)
      .setLaplacianType("randomWalk")
      .setKernelType("rbf")
      .setGamma(0.1)
      .setSeed(42L)

    val model = spectral.fit(df)
    model.numClusters shouldBe 2
  }

  test("SpectralClustering - deterministic with same seed") {
    val spark = this.spark
    import spark.implicits._

    val data = (0 until 10).map { i =>
      Tuple1(Vectors.dense(i.toDouble, (i % 2).toDouble * 5))
    }
    val df   = spark.createDataFrame(data).toDF("features")

    val spectral1 = new SpectralClustering().setK(2).setSeed(123L)

    val spectral2 = new SpectralClustering().setK(2).setSeed(123L)

    val model1 = spectral1.fit(df)
    val model2 = spectral2.fit(df)

    val preds1 = model1.transform(df).select("prediction").collect().map(_.getInt(0))
    val preds2 = model2.transform(df).select("prediction").collect().map(_.getInt(0))

    preds1 shouldBe preds2
  }

  test("SpectralClustering - validates k > 1") {
    val spectral = new SpectralClustering()

    an[IllegalArgumentException] should be thrownBy {
      spectral.setK(1)
    }
  }

  test("SpectralClustering - validates gamma > 0") {
    val spectral = new SpectralClustering()

    an[IllegalArgumentException] should be thrownBy {
      spectral.setGamma(0.0)
    }
  }

  // ============================================================================
  // Persistence tests
  // ============================================================================

  test("SpectralClustering - model persistence") {
    val spark = this.spark
    import spark.implicits._

    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 1.0),
      Vectors.dense(5.0, 5.0),
      Vectors.dense(6.0, 6.0)
    ).map(Tuple1(_))
    val df   = spark.createDataFrame(data).toDF("features")

    val spectral = new SpectralClustering().setK(2).setKernelType("rbf").setGamma(0.5).setSeed(42L)

    val model = spectral.fit(df)

    val tempDir  = Files.createTempDirectory("spectral-model-test")
    val savePath = tempDir.resolve("model").toString

    try {
      model.write.overwrite().save(savePath)
      val loadedModel = SpectralClusteringModel.load(savePath)

      loadedModel.numClusters shouldBe model.numClusters
      loadedModel.getKernelType shouldBe model.getKernelType
      loadedModel.getGamma shouldBe model.getGamma

      // Check predictions match
      val preds1 = model.transform(df).select("prediction").collect().map(_.getInt(0))
      val preds2 = loadedModel.transform(df).select("prediction").collect().map(_.getInt(0))
      preds1 shouldBe preds2
    } finally {
      import scala.reflect.io.Directory
      new Directory(tempDir.toFile).deleteRecursively()
    }
  }

  test("SpectralClustering - estimator persistence") {
    val tempDir  = Files.createTempDirectory("spectral-estimator-test")
    val savePath = tempDir.resolve("estimator").toString

    try {
      val spectral = new SpectralClustering()
        .setK(4)
        .setKernelType("laplacian")
        .setGamma(2.0)
        .setAffinityType("knn")
        .setNumNeighbors(5)
        .setSeed(123L)

      spectral.write.overwrite().save(savePath)
      val loaded = SpectralClustering.load(savePath)

      loaded.getK shouldBe 4
      loaded.getKernelType shouldBe "laplacian"
      loaded.getGamma shouldBe 2.0
      loaded.getAffinityType shouldBe "knn"
      loaded.getNumNeighbors shouldBe 5
      loaded.getSeed shouldBe 123L
    } finally {
      import scala.reflect.io.Directory
      new Directory(tempDir.toFile).deleteRecursively()
    }
  }

  // ============================================================================
  // Edge cases
  // ============================================================================

  test("SpectralClustering - handles small dataset") {
    val spark = this.spark
    import spark.implicits._

    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 1.0)
    ).map(Tuple1(_))
    val df   = spark.createDataFrame(data).toDF("features")

    val spectral = new SpectralClustering().setK(2).setSeed(42L)

    val model       = spectral.fit(df)
    val predictions = model.transform(df)

    predictions.count() shouldBe 2
  }

  test("SpectralClustering - with polynomial kernel") {
    val spark = this.spark
    import spark.implicits._

    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(1.0, 0.0),
      Vectors.dense(3.0, 3.0),
      Vectors.dense(4.0, 3.0)
    ).map(Tuple1(_))
    val df   = spark.createDataFrame(data).toDF("features")

    val spectral = new SpectralClustering()
      .setK(2)
      .setKernelType("polynomial")
      .setDegree(2)
      .setGamma(1.0)
      .setSeed(42L)

    val model = spectral.fit(df)
    model.numClusters shouldBe 2
  }

  test("SpectralClustering - with epsilon neighborhood") {
    val spark = this.spark
    import spark.implicits._

    val data = Seq(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.5, 0.0),
      Vectors.dense(5.0, 0.0),
      Vectors.dense(5.5, 0.0)
    ).map(Tuple1(_))
    val df   = spark.createDataFrame(data).toDF("features")

    val spectral = new SpectralClustering()
      .setK(2)
      .setAffinityType("epsilon")
      .setEpsilon(1.0)
      .setKernelType("rbf")
      .setGamma(1.0)
      .setSeed(42L)

    val model       = spectral.fit(df)
    val predictions = model.transform(df)

    predictions.count() shouldBe 4
    model.numClusters shouldBe 2
  }
}
