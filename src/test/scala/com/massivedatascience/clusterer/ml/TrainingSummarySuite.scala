package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df.LloydResult
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class TrainingSummarySuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  private val spark: SparkSession = SparkSession
    .builder()
    .master("local[2]")
    .appName("TrainingSummarySuite")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")
  import spark.implicits._

  override def afterAll(): Unit = {
    spark.stop()
  }

  test("TrainingSummary creation from LloydResult") {
    val result = LloydResult(
      centers = Array(Array(0.0, 0.0), Array(10.0, 10.0)),
      iterations = 5,
      distortionHistory = Array(100.0, 50.0, 25.0, 12.5, 10.0),
      movementHistory = Array(5.0, 2.5, 1.0, 0.5, 0.1),
      converged = true,
      emptyClusterEvents = 0
    )

    val summary = TrainingSummary.fromLloydResult(
      algorithm = "GeneralizedKMeans",
      result = result,
      k = 2,
      dim = 2,
      numPoints = 100,
      assignmentStrategy = "AutoAssignment",
      divergence = "squaredEuclidean",
      elapsedMillis = 1000
    )

    summary.algorithm shouldBe "GeneralizedKMeans"
    summary.k shouldBe 2
    summary.effectiveK shouldBe 2
    summary.dim shouldBe 2
    summary.numPoints shouldBe 100
    summary.iterations shouldBe 5
    summary.converged shouldBe true
    summary.distortionHistory.length shouldBe 5
    summary.movementHistory.length shouldBe 5
    summary.finalDistortion shouldBe 10.0
    summary.avgIterationMillis shouldBe 200.0
    summary.hasEmptyClusters shouldBe false
    summary.assignmentStrategy shouldBe "AutoAssignment"
    summary.divergence shouldBe "squaredEuclidean"
  }

  test("TrainingSummary with empty clusters") {
    val result = LloydResult(
      centers = Array(Array(0.0, 0.0)), // Only 1 center instead of requested 2
      iterations = 10,
      distortionHistory = Array(50.0, 40.0),
      movementHistory = Array(1.0, 0.5),
      converged = false,
      emptyClusterEvents = 1
    )

    val summary = TrainingSummary.fromLloydResult(
      algorithm = "GeneralizedKMeans",
      result = result,
      k = 2, // Requested 2
      dim = 2,
      numPoints = 10,
      assignmentStrategy = "BroadcastUDF",
      divergence = "kl",
      elapsedMillis = 500
    )

    summary.k shouldBe 2
    summary.effectiveK shouldBe 1 // Only got 1
    summary.hasEmptyClusters shouldBe true
    summary.converged shouldBe false
  }

  test("TrainingSummary toString produces readable output") {
    val result = LloydResult(
      centers = Array(Array(0.0, 0.0), Array(10.0, 10.0)),
      iterations = 5,
      distortionHistory = Array(100.0, 50.0),
      movementHistory = Array(5.0, 0.1),
      converged = true,
      emptyClusterEvents = 0
    )

    val summary = TrainingSummary.fromLloydResult(
      algorithm = "GeneralizedKMeans",
      result = result,
      k = 2,
      dim = 2,
      numPoints = 100,
      assignmentStrategy = "SECrossJoin",
      divergence = "squaredEuclidean",
      elapsedMillis = 1234
    )

    val str = summary.toString
    str should include("GeneralizedKMeans")
    str should include("Clusters: 2/2")
    str should include("converged")
    str should include("squaredEuclidean")
    str should include("SECrossJoin")
  }

  test("TrainingSummary convergenceReport") {
    val result = LloydResult(
      centers = Array(Array(0.0, 0.0)),
      iterations = 3,
      distortionHistory = Array(100.0, 50.0, 25.0),
      movementHistory = Array(10.0, 5.0, 0.1),
      converged = true,
      emptyClusterEvents = 0
    )

    val summary = TrainingSummary.fromLloydResult(
      algorithm = "Test",
      result = result,
      k = 1,
      dim = 2,
      numPoints = 10,
      assignmentStrategy = "Test",
      divergence = "test",
      elapsedMillis = 100
    )

    val report = summary.convergenceReport
    report should include("100.000 → 25.000")
    report should include("75.0% reduction")
    report should include("Max center movement:")
    report should include("Final center movement:")
  }

  test("TrainingSummary toDF creates valid DataFrame") {
    val result = LloydResult(
      centers = Array(Array(0.0, 0.0)),
      iterations = 5,
      distortionHistory = Array(100.0, 50.0),
      movementHistory = Array(5.0, 0.1),
      converged = true,
      emptyClusterEvents = 0
    )

    val summary = TrainingSummary.fromLloydResult(
      algorithm = "GeneralizedKMeans",
      result = result,
      k = 1,
      dim = 2,
      numPoints = 100,
      assignmentStrategy = "AutoAssignment",
      divergence = "squaredEuclidean",
      elapsedMillis = 1000
    )

    val df = summary.toDF(spark)

    df.count() shouldBe 1
    df.columns should contain allOf (
      "algorithm",
      "k",
      "effectiveK",
      "dim",
      "numPoints",
      "iterations",
      "converged",
      "finalDistortion",
      "assignmentStrategy",
      "divergence",
      "elapsedMillis",
      "avgIterationMillis",
      "hasEmptyClusters",
      "trainedAt",
      "distortionHistory",
      "movementHistory"
    )

    val row = df.first()
    row.getAs[String]("algorithm") shouldBe "GeneralizedKMeans"
    row.getAs[Int]("k") shouldBe 1
    row.getAs[Int]("effectiveK") shouldBe 1
    row.getAs[Int]("iterations") shouldBe 5
    row.getAs[Boolean]("converged") shouldBe true
  }

  test("GeneralizedKMeansModel has summary after training") {
    val df = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(9.0, 9.0)),
      Tuple1(Vectors.dense(10.0, 10.0))
    ).toDF("features")

    val gkm   = new GeneralizedKMeans().setK(2).setDivergence("squaredEuclidean").setSeed(42)
    val model = gkm.fit(df)

    // Model should have summary after training
    model.hasSummary shouldBe true
    val summary = model.summary

    // Verify summary contents
    summary.algorithm shouldBe "GeneralizedKMeans"
    summary.k shouldBe 2
    summary.effectiveK should be <= 2
    summary.effectiveK should be >= 1
    summary.dim shouldBe 2
    summary.numPoints shouldBe 4
    summary.iterations should be >= 1
    summary.divergence shouldBe "squaredEuclidean"
    summary.distortionHistory.length should be >= 1
    summary.movementHistory.length should be >= 1
    summary.elapsedMillis should be >= 0L
  }

  test("GeneralizedKMeansModel loaded from disk has no summary") {
    val df = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(9.0, 9.0)),
      Tuple1(Vectors.dense(10.0, 10.0))
    ).toDF("features")

    val gkm    = new GeneralizedKMeans().setK(2).setDivergence("squaredEuclidean").setSeed(42)
    val model  = gkm.fit(df)
    val tmpDir = java.nio.file.Files.createTempDirectory("gkm-summary-test")

    try {
      val savePath = tmpDir.resolve("model").toString
      model.write.overwrite().save(savePath)

      val loaded = GeneralizedKMeansModel.load(savePath)

      // Loaded model should NOT have summary
      loaded.hasSummary shouldBe false

      // Accessing summary should throw exception
      val exception = intercept[NoSuchElementException] {
        loaded.summary
      }
      exception.getMessage should include("not available")
      exception.getMessage should include("loaded from disk")

    } finally {
      // Cleanup
      import scala.util.Try
      Try {
        def deleteRecursively(path: java.nio.file.Path): Unit = {
          if (java.nio.file.Files.isDirectory(path)) {
            java.nio.file.Files.list(path).forEach(deleteRecursively)
          }
          java.nio.file.Files.deleteIfExists(path)
        }
        deleteRecursively(tmpDir)
      }
    }
  }
}
