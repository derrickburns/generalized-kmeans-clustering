package com.massivedatascience.clusterer.ml.df

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

/** Tests for AdaptiveBroadcastAssignment with memory-aware chunk sizing. */
class AdaptiveBroadcastAssignmentSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("AdaptiveBroadcastAssignmentSuite")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .config("spark.executor.memory", "512m") // Set explicit memory for testing
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

  test("AdaptiveBroadcastAssignment produces correct results") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(11.0, 11.0))
    ).toDF("features")

    val centers = Array(
      Array(0.5, 0.5),
      Array(10.5, 10.5)
    )

    val kernel   = new SquaredEuclideanKernel()
    val assigner = new AdaptiveBroadcastAssignment()

    val result = assigner.assign(data, "features", None, centers, kernel)

    // Verify assignments
    val assignments = result.select("cluster").collect().map(_.getInt(0))
    assert(assignments(0) == 0) // (0,0) -> cluster 0
    assert(assignments(1) == 0) // (1,1) -> cluster 0
    assert(assignments(2) == 1) // (10,10) -> cluster 1
    assert(assignments(3) == 1) // (11,11) -> cluster 1
  }

  test("AdaptiveBroadcastAssignment works with KL divergence") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.9, 0.1)),
      Tuple1(Vectors.dense(0.8, 0.2)),
      Tuple1(Vectors.dense(0.1, 0.9)),
      Tuple1(Vectors.dense(0.2, 0.8))
    ).toDF("features")

    val centers = Array(
      Array(0.85, 0.15),
      Array(0.15, 0.85)
    )

    val kernel   = new KLDivergenceKernel(1e-10)
    val assigner = new AdaptiveBroadcastAssignment()

    val result = assigner.assign(data, "features", None, centers, kernel)

    val assignments = result.select("cluster").collect().map(_.getInt(0))
    // First two points should go to cluster 0, last two to cluster 1
    assert(assignments(0) == 0)
    assert(assignments(1) == 0)
    assert(assignments(2) == 1)
    assert(assignments(3) == 1)
  }

  test("AdaptiveBroadcastAssignment uses SE fast path for squared Euclidean") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(10.0, 10.0))
    ).toDF("features")

    val centers = Array(
      Array(0.0, 0.0),
      Array(10.0, 10.0)
    )

    val kernel   = new SquaredEuclideanKernel()
    val assigner = new AdaptiveBroadcastAssignment()

    // Should use SE fast path (kernel.supportsExpressionOptimization = true)
    val result = assigner.assign(data, "features", None, centers, kernel)
    assert(result.count() == 2)
    assert(result.columns.contains("cluster"))
  }

  test("AdaptiveBroadcastAssignment handles large k gracefully") {
    val sparkSession = spark
    import sparkSession.implicits._

    val numPoints = 100
    val data      = (0 until numPoints).map { i =>
      Tuple1(Vectors.dense(i.toDouble, i.toDouble))
    }.toDF("features")

    // Create many centers (more than typical chunk size)
    val k       = 200
    val centers = (0 until k).map { i =>
      Array(i.toDouble, i.toDouble)
    }.toArray

    val kernel = new KLDivergenceKernel(1e-10) // Non-SE to test chunking path

    // Use small maxChunkSize to force chunking
    val assigner = new AdaptiveBroadcastAssignment(
      broadcastFraction = 0.1,
      safetyFactor = 2.0,
      minChunkSize = 10,
      maxChunkSize = 50 // Force chunking with 50 centers per chunk
    )

    val result = assigner.assign(data, "features", None, centers, kernel)

    // Should complete without error
    assert(result.count() == numPoints)
    assert(result.columns.contains("cluster"))

    // All assignments should be valid cluster IDs
    val assignments = result.select("cluster").collect().map(_.getInt(0))
    assignments.foreach { cluster =>
      assert(cluster >= 0 && cluster < k, s"Invalid cluster ID: $cluster")
    }
  }

  test("AdaptiveBroadcastAssignment respects minChunkSize and maxChunkSize") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0))
    ).toDF("features")

    val centers = Array(
      Array(0.0, 0.0),
      Array(1.0, 1.0)
    )

    val kernel = new KLDivergenceKernel(1e-10)

    // Test with constrained chunk size
    val assigner = new AdaptiveBroadcastAssignment(
      broadcastFraction = 0.1,
      safetyFactor = 2.0,
      minChunkSize = 5,
      maxChunkSize = 20
    )

    val result = assigner.assign(data, "features", None, centers, kernel)
    assert(result.count() == 1)
  }

  test("AdaptiveBroadcastAssignment validates parameters") {
    // broadcastFraction must be in (0, 1]
    assertThrows[IllegalArgumentException] {
      new AdaptiveBroadcastAssignment(broadcastFraction = 0.0)
    }
    assertThrows[IllegalArgumentException] {
      new AdaptiveBroadcastAssignment(broadcastFraction = 1.5)
    }

    // safetyFactor must be >= 1.0
    assertThrows[IllegalArgumentException] {
      new AdaptiveBroadcastAssignment(safetyFactor = 0.5)
    }

    // minChunkSize must be > 0
    assertThrows[IllegalArgumentException] {
      new AdaptiveBroadcastAssignment(minChunkSize = 0)
    }

    // maxChunkSize must be >= minChunkSize
    assertThrows[IllegalArgumentException] {
      new AdaptiveBroadcastAssignment(minChunkSize = 100, maxChunkSize = 50)
    }
  }

  test("AdaptiveBroadcastAssignment handles empty centers") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0))
    ).toDF("features")

    val centers  = Array.empty[Array[Double]]
    val kernel   = new SquaredEuclideanKernel()
    val assigner = new AdaptiveBroadcastAssignment()

    val result = assigner.assign(data, "features", None, centers, kernel)

    // Should return with default cluster 0
    assert(result.count() == 1)
    assert(result.select("cluster").head().getInt(0) == 0)
  }

  test("AdaptiveBroadcastAssignment produces same results as BroadcastUDF") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create test data
    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.5, 0.5)),
      Tuple1(Vectors.dense(1.0, 0.0)),
      Tuple1(Vectors.dense(5.0, 5.0)),
      Tuple1(Vectors.dense(5.5, 5.5)),
      Tuple1(Vectors.dense(6.0, 5.0))
    ).toDF("features")

    val centers = Array(
      Array(0.5, 0.25),
      Array(5.5, 5.25)
    )

    val kernel = new KLDivergenceKernel(1e-6)

    // Run both assignment strategies
    val adaptiveAssigner  = new AdaptiveBroadcastAssignment()
    val broadcastAssigner = new BroadcastUDFAssignment()

    val adaptiveResult = adaptiveAssigner
      .assign(data, "features", None, centers, kernel)
      .select("cluster")
      .collect()
      .map(_.getInt(0))

    val broadcastResult = broadcastAssigner
      .assign(data, "features", None, centers, kernel)
      .select("cluster")
      .collect()
      .map(_.getInt(0))

    // Results should be identical
    assert(
      adaptiveResult.toSeq == broadcastResult.toSeq,
      s"Adaptive: ${adaptiveResult.toSeq}, Broadcast: ${broadcastResult.toSeq}"
    )
  }
}
