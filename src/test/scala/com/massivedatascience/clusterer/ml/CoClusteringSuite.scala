package com.massivedatascience.clusterer.ml

import org.apache.spark.sql.{ DataFrame, SparkSession }
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import java.nio.file.{ Files, Paths }
import scala.util.Try

/** Integration tests for DataFrame-based CoClustering.
  *
  * Tests co-clustering functionality following the Spark ML pattern.
  */
class CoClusteringSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("CoClusteringSuite")
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

  /** Create a simple matrix dataset with 2x2 block structure.
    *
    * Matrix structure:
    * {{{
    *        col0  col1 | col2  col3
    * row0   1.0   1.1  | 5.0   5.1
    * row1   1.2   0.9  | 4.9   5.2
    * -----------------------
    * row2   4.8   5.0  | 1.0   1.1
    * row3   5.1   4.9  | 0.9   1.2
    * }}}
    *
    * Should result in:
    *   - Row clusters: {row0, row1} -> 0, {row2, row3} -> 1
    *   - Col clusters: {col0, col1} -> 0, {col2, col3} -> 1
    *   - Block centers: [[1.05, 5.05], [4.95, 1.05]]
    */
  def createBlockStructuredMatrix(): DataFrame = {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      // Upper left block: ~1.0
      (0L, 0L, 1.0),
      (0L, 1L, 1.1),
      (1L, 0L, 1.2),
      (1L, 1L, 0.9),
      // Upper right block: ~5.0
      (0L, 2L, 5.0),
      (0L, 3L, 5.1),
      (1L, 2L, 4.9),
      (1L, 3L, 5.2),
      // Lower left block: ~5.0
      (2L, 0L, 4.8),
      (2L, 1L, 5.0),
      (3L, 0L, 5.1),
      (3L, 1L, 4.9),
      // Lower right block: ~1.0
      (2L, 2L, 1.0),
      (2L, 3L, 1.1),
      (3L, 2L, 0.9),
      (3L, 3L, 1.2)
    )

    data.toDF("row", "col", "value")
  }

  test("fit and transform basic co-clustering") {
    val df = createBlockStructuredMatrix()

    val coClustering = new CoClustering()
      .setNumRowClusters(2)
      .setNumColClusters(2)
      .setRowIndexCol("row")
      .setColIndexCol("col")
      .setValueCol("value")
      .setMaxIter(50)
      .setSeed(42)

    val model = coClustering.fit(df)

    assert(model.numRowClustersActual <= 2, "Should have at most 2 row clusters")
    assert(model.numColClustersActual <= 2, "Should have at most 2 column clusters")
    assert(model.rowClusters.size === 4, "Should have 4 rows")
    assert(model.colClusters.size === 4, "Should have 4 columns")

    val predictions = model.transform(df)
    assert(predictions.columns.contains("rowPrediction"))
    assert(predictions.columns.contains("colPrediction"))
    assert(predictions.count() === 16)
  }

  test("training summary available after fit") {
    val df = createBlockStructuredMatrix()

    val coClustering = new CoClustering()
      .setNumRowClusters(2)
      .setNumColClusters(2)
      .setRowIndexCol("row")
      .setColIndexCol("col")
      .setValueCol("value")
      .setMaxIter(20)
      .setSeed(42)

    val model = coClustering.fit(df)

    assert(model.hasSummary)
    val summary = model.summary
    assert(summary.iterations > 0)
    assert(summary.finalObjective >= 0.0)
  }

  test("block value prediction") {
    val df = createBlockStructuredMatrix()

    val coClustering = new CoClustering()
      .setNumRowClusters(2)
      .setNumColClusters(2)
      .setRowIndexCol("row")
      .setColIndexCol("col")
      .setValueCol("value")
      .setMaxIter(30)
      .setSeed(42)

    val model = coClustering.fit(df)

    // Predict block value for a known cell
    val predicted = model.predictBlockValue(0L, 0L)
    assert(predicted > 0.0, "Predicted value should be positive")
  }

  test("model persistence round-trip") {
    val df = createBlockStructuredMatrix()

    val coClustering = new CoClustering()
      .setNumRowClusters(2)
      .setNumColClusters(2)
      .setRowIndexCol("row")
      .setColIndexCol("col")
      .setValueCol("value")
      .setMaxIter(20)
      .setSeed(42)

    val model = coClustering.fit(df)

    // Save to temp directory
    val tmpDir    = Files.createTempDirectory("coclustering-test")
    val modelPath = tmpDir.resolve("model").toString

    try {
      model.write.overwrite().save(modelPath)

      // Load and verify
      val loadedModel = CoClusteringModel.load(modelPath)

      assert(loadedModel.rowClusters === model.rowClusters)
      assert(loadedModel.colClusters === model.colClusters)
      assert(loadedModel.blockCenters.length === model.blockCenters.length)

      for (r <- model.blockCenters.indices; c <- model.blockCenters(r).indices) {
        assert(
          math.abs(loadedModel.blockCenters(r)(c) - model.blockCenters(r)(c)) < 1e-9,
          s"Block center ($r, $c) mismatch"
        )
      }

      // Verify loaded model can transform
      val predictions = loadedModel.transform(df)
      assert(predictions.count() === 16)
    } finally {
      // Cleanup
      deleteRecursively(tmpDir.toFile)
    }
  }

  test("custom column names") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      (0L, 0L, 1.0),
      (0L, 1L, 2.0),
      (1L, 0L, 2.0),
      (1L, 1L, 1.0)
    )

    val df = data.toDF("myRow", "myCol", "myVal")

    val coClustering = new CoClustering()
      .setNumRowClusters(2)
      .setNumColClusters(2)
      .setRowIndexCol("myRow")
      .setColIndexCol("myCol")
      .setValueCol("myVal")
      .setRowPredictionCol("rCluster")
      .setColPredictionCol("cCluster")
      .setMaxIter(10)
      .setSeed(42)

    val model       = coClustering.fit(df)
    val predictions = model.transform(df)

    assert(predictions.columns.contains("rCluster"))
    assert(predictions.columns.contains("cCluster"))
    assert(!predictions.columns.contains("rowPrediction"))
  }

  test("estimator persistence") {
    val tmpDir        = Files.createTempDirectory("coclustering-estimator-test")
    val estimatorPath = tmpDir.resolve("estimator").toString

    try {
      val coClustering = new CoClustering().setNumRowClusters(3).setNumColClusters(4).setMaxIter(50)

      coClustering.write.overwrite().save(estimatorPath)

      val loaded = CoClustering.load(estimatorPath)
      assert(loaded.getNumRowClusters === 3)
      assert(loaded.getNumColClusters === 4)
      assert(loaded.getMaxIter === 50)
    } finally {
      deleteRecursively(tmpDir.toFile)
    }
  }

  test("convergence before max iterations") {
    val df = createBlockStructuredMatrix()

    val coClustering = new CoClustering()
      .setNumRowClusters(2)
      .setNumColClusters(2)
      .setRowIndexCol("row")
      .setColIndexCol("col")
      .setValueCol("value")
      .setMaxIter(1000)   // High max iter
      .setTolerance(1e-3) // Reasonable tolerance
      .setSeed(42)

    val model = coClustering.fit(df)

    // With good block structure, should converge well before 1000 iterations
    assert(model.hasSummary)
    assert(
      model.summary.iterations < 100,
      s"Should converge quickly, but took ${model.summary.iterations} iterations"
    )
  }

  private def deleteRecursively(file: java.io.File): Unit = {
    if (file.isDirectory) {
      file.listFiles().foreach(deleteRecursively)
    }
    file.delete()
  }
}
