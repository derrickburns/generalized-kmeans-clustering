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

package com.massivedatascience.clusterer.ml

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.scalatest.matchers.should.Matchers
import java.nio.file.Files

/** Tests for model persistence (save/load) across Spark and Scala versions.
  *
  * These tests verify that models can be saved and loaded correctly, maintaining all parameters and
  * cluster centers.
  */
class PersistenceSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  private val spark: SparkSession = SparkSession
    .builder()
    .master("local[2]")
    .appName("PersistenceSuite")
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

  /** Create a small test DataFrame */
  private def tinyDF() = {
    Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(9.0, 9.0)),
      Tuple1(Vectors.dense(10.0, 10.0))
    ).toDF("features")
  }

  test("GeneralizedKMeans save-load roundtrip with Squared Euclidean") {
    val df = tinyDF()

    val gkm =
      new GeneralizedKMeans().setK(2).setDivergence("squaredEuclidean").setSeed(42).setMaxIter(10)

    val model = gkm.fit(df)
    val tmp   = Files.createTempDirectory("gkm-model-se").toFile.getCanonicalPath

    // Save model
    model.write.overwrite().save(tmp)

    // Load model
    val loaded = GeneralizedKMeansModel.load(tmp)

    // Verify parameters
    loaded.getDivergence shouldBe "squaredEuclidean"
    loaded.numClusters shouldBe 2
    loaded.numFeatures shouldBe 2
    loaded.getMaxIter shouldBe 10
    loaded.getSeed shouldBe 42

    // Verify centers match
    loaded.clusterCenters.length shouldBe model.clusterCenters.length
    loaded.clusterCenters.zip(model.clusterCenters).foreach { case (loadedCenter, origCenter) =>
      loadedCenter should contain theSameElementsInOrderAs origCenter
    }

    // Verify transform works
    val pred = loaded.transform(df)
    pred.count() shouldBe 4

    // Clean up
    new java.io.File(tmp).listFiles().foreach(_.delete())
    new java.io.File(tmp).delete()
  }

  test("GeneralizedKMeans save-load roundtrip with KL divergence") {
    val df = tinyDF()

    val gkm = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("kl")
      .setSmoothing(1e-6)
      .setSeed(7)
      .setMaxIter(5)
      .setDistanceCol("distance")

    val model = gkm.fit(df)
    val tmp   = Files.createTempDirectory("gkm-model-kl").toFile.getCanonicalPath

    // Save model
    model.write.overwrite().save(tmp)

    // Load model
    val loaded = GeneralizedKMeansModel.load(tmp)

    // Verify parameters
    loaded.getDivergence shouldBe "kl"
    loaded.getSmoothing shouldBe 1e-6
    loaded.numClusters shouldBe 2
    loaded.getDistanceCol shouldBe "distance"
    loaded.hasDistanceCol shouldBe true

    // Verify transform works and includes distance column
    val pred = loaded.transform(df)
    pred.count() shouldBe 4
    pred.columns should contain("distance")

    // Clean up
    new java.io.File(tmp).listFiles().foreach(_.delete())
    new java.io.File(tmp).delete()
  }

  test("GeneralizedKMeans save-load roundtrip with all parameters") {
    val df = tinyDF()

    val gkm = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setMaxIter(15)
      .setTol(1e-5)
      .setSeed(123)
      .setAssignmentStrategy("auto")
      .setEmptyClusterStrategy("reseedRandom")
      .setCheckpointInterval(5)
      .setInitMode("k-means||")
      .setInitSteps(3)
      .setFeaturesCol("features")
      .setPredictionCol("cluster")
      .setDistanceCol("dist")

    val model = gkm.fit(df)
    val tmp   = Files.createTempDirectory("gkm-model-full").toFile.getCanonicalPath

    // Save model
    model.write.overwrite().save(tmp)

    // Load model
    val loaded = GeneralizedKMeansModel.load(tmp)

    // Verify all parameters
    loaded.getK shouldBe 2
    loaded.getDivergence shouldBe "squaredEuclidean"
    loaded.getMaxIter shouldBe 15
    loaded.getTol shouldBe 1e-5
    loaded.getSeed shouldBe 123
    loaded.getAssignmentStrategy shouldBe "auto"
    loaded.getEmptyClusterStrategy shouldBe "reseedRandom"
    loaded.getCheckpointInterval shouldBe 5
    loaded.getInitMode shouldBe "k-means||"
    loaded.getInitSteps shouldBe 3
    loaded.getFeaturesCol shouldBe "features"
    loaded.getPredictionCol shouldBe "cluster"
    loaded.getDistanceCol shouldBe "dist"

    // Verify transform works with custom column names
    val pred = loaded.transform(df)
    pred.count() shouldBe 4
    pred.columns should contain("cluster")
    pred.columns should contain("dist")

    // Clean up
    new java.io.File(tmp).listFiles().foreach(_.delete())
    new java.io.File(tmp).delete()
  }

  test("GeneralizedKMeans metadata JSON structure") {
    val df = tinyDF()

    val gkm = new GeneralizedKMeans().setK(2).setDivergence("squaredEuclidean").setSeed(42)

    val model = gkm.fit(df)
    val tmp   = Files.createTempDirectory("gkm-metadata-test").toFile.getCanonicalPath

    // Save model
    model.write.overwrite().save(tmp)

    // Read metadata.json
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    val metadataJson = readMetadata(tmp)

    // Verify it's valid JSON and contains expected fields
    metadataJson should include("layoutVersion")
    metadataJson should include("GeneralizedKMeansModel")
    metadataJson should include("sparkMLVersion")
    metadataJson should include("scalaBinaryVersion")
    metadataJson should include("squaredEuclidean")
    metadataJson should include("centers")
    metadataJson should include("checksums")

    // Clean up
    new java.io.File(tmp).listFiles().foreach(_.delete())
    new java.io.File(tmp).delete()
  }

  test("GeneralizedKMeans centers.parquet structure") {
    val df = tinyDF()

    val gkm = new GeneralizedKMeans().setK(2).setDivergence("squaredEuclidean").setSeed(42)

    val model = gkm.fit(df)
    val tmp   = Files.createTempDirectory("gkm-centers-test").toFile.getCanonicalPath

    // Save model
    model.write.overwrite().save(tmp)

    // Read centers.parquet
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    val centersDF = readCenters(spark, tmp)

    // Verify structure
    centersDF.columns should contain("center_id")
    centersDF.columns should contain("weight")
    centersDF.columns should contain("vector")
    centersDF.count() shouldBe 2

    // Verify ordering (center_id should be 0, 1)
    val centerIds = centersDF.select("center_id").collect().map(_.getInt(0))
    centerIds should contain theSameElementsInOrderAs Seq(0, 1)

    // Clean up
    new java.io.File(tmp).listFiles().foreach(_.delete())
    new java.io.File(tmp).delete()
  }
}
