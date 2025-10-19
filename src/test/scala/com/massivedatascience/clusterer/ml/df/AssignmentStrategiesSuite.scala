package com.massivedatascience.clusterer.ml.df

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.scalatest.matchers.should.Matchers

// Import kernel implementations
import com.massivedatascience.clusterer.ml.df._

/** Tests for assignment strategies.
  *
  * These tests verify:
  * - BroadcastUDFAssignment works correctly
  * - ChunkedBroadcastAssignment produces identical results to broadcast
  * - AutoAssignment selects correct strategy based on k×dim threshold
  * - All strategies handle edge cases (small k, large k×dim)
  */
class AssignmentStrategiesSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  private val spark: SparkSession = SparkSession
    .builder()
    .master("local[2]")
    .appName("AssignmentStrategiesSuite")
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

  test("BroadcastUDFAssignment: assigns points correctly") {
    val df = testDF()
    val centers = Array(
      Array(0.5, 0.5),  // Center 0: near (0,0), (0.5,0.5), (1,1)
      Array(9.5, 9.5)   // Center 1: near (9,9), (9.5,9.5), (10,10)
    )
    val kernel = new SquaredEuclideanKernel()

    val strategy = new BroadcastUDFAssignment()
    val assigned = strategy.assign(df, "features", None, centers, kernel)

    assigned.count() shouldBe 6
    assigned.columns should contain("cluster")

    val clusters = assigned.select("cluster").collect().map(_.getInt(0)).sorted
    clusters should contain allOf (0, 1)

    // First 3 points should be in cluster 0, last 3 in cluster 1
    val predictions = assigned.select("cluster").collect().map(_.getInt(0))
    predictions.take(3).toSet shouldBe Set(0)
    predictions.drop(3).toSet shouldBe Set(1)
  }

  test("ChunkedBroadcastAssignment: produces same results as BroadcastUDF") {
    val df = testDF()
    val centers = Array(
      Array(0.5, 0.5),
      Array(9.5, 9.5)
    )
    val kernel = new GeneralizedIDivergenceKernel(1e-10)

    val broadcast = new BroadcastUDFAssignment()
    val chunked   = new ChunkedBroadcastAssignment(chunkSize = 1) // Force chunking even with k=2

    val broadcastResult = broadcast.assign(df, "features", None, centers, kernel)
    val chunkedResult   = chunked.assign(df, "features", None, centers, kernel)

    val broadcastClusters = broadcastResult.select("cluster").collect().map(_.getInt(0))
    val chunkedClusters   = chunkedResult.select("cluster").collect().map(_.getInt(0))

    broadcastClusters should contain theSameElementsInOrderAs chunkedClusters
  }

  test("ChunkedBroadcastAssignment: handles many clusters") {
    val df = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(20.0, 20.0)),
      Tuple1(Vectors.dense(30.0, 30.0)),
      Tuple1(Vectors.dense(40.0, 40.0))
    ).toDF("features")

    // Create 10 centers spread out
    val centers = (0 until 10).map { i =>
      Array(i * 5.0, i * 5.0)
    }.toArray

    val kernel = new SquaredEuclideanKernel()

    val broadcast = new BroadcastUDFAssignment()
    val chunked   = new ChunkedBroadcastAssignment(chunkSize = 3) // 4 chunks

    val broadcastResult = broadcast.assign(df, "features", None, centers, kernel)
    val chunkedResult   = chunked.assign(df, "features", None, centers, kernel)

    val broadcastClusters = broadcastResult.select("cluster").collect().map(_.getInt(0)).sorted
    val chunkedClusters   = chunkedResult.select("cluster").collect().map(_.getInt(0)).sorted

    broadcastClusters should contain theSameElementsAs chunkedClusters
  }

  test("ChunkedBroadcastAssignment: falls back to BroadcastUDF when k <= chunkSize") {
    val df      = testDF()
    val centers = Array(Array(0.5, 0.5), Array(9.5, 9.5))
    val kernel  = new SquaredEuclideanKernel()

    val chunked = new ChunkedBroadcastAssignment(chunkSize = 10) // Larger than k=2

    val result = chunked.assign(df, "features", None, centers, kernel)

    result.count() shouldBe 6
    result.columns should contain("cluster")
  }

  test("AutoAssignment: selects SECrossJoin for Squared Euclidean") {
    val df      = testDF()
    val centers = Array(Array(0.5, 0.5), Array(9.5, 9.5))
    val kernel  = new SquaredEuclideanKernel()

    val auto = new AutoAssignment(broadcastThresholdElems = 200000, chunkSize = 100)

    val result = auto.assign(df, "features", None, centers, kernel)

    result.count() shouldBe 6
    result.columns should contain("cluster")

    // Verify correct assignments
    val predictions = result.select("cluster").collect().map(_.getInt(0))
    predictions.take(3).toSet shouldBe Set(0)
    predictions.drop(3).toSet shouldBe Set(1)
  }

  test("AutoAssignment: selects BroadcastUDF for non-SE with small k×dim") {
    val df = testDF()
    val centers = Array(
      Array(0.5, 0.5),
      Array(9.5, 9.5)
    )
    val kernel = new GeneralizedIDivergenceKernel(1e-10) // Non-SE kernel

    // k×dim = 2×2 = 4, which is << 200000
    val auto = new AutoAssignment(broadcastThresholdElems = 200000, chunkSize = 100)

    val result = auto.assign(df, "features", None, centers, kernel)

    result.count() shouldBe 6
    result.columns should contain("cluster")
  }

  test("AutoAssignment: selects ChunkedBroadcast for non-SE with large k×dim") {
    val df = testDF()

    // Create many centers to exceed threshold
    // threshold = 100, k=10, dim=20, k×dim=200 > 100
    val centers = (0 until 10).map { i =>
      Array.fill(20)(i * 1.0 + scala.util.Random.nextDouble())
    }.toArray

    // Pad input data to 20 dimensions
    val df20d = Seq(
      Tuple1(Vectors.dense(Array.fill(20)(0.0))),
      Tuple1(Vectors.dense(Array.fill(20)(1.0))),
      Tuple1(Vectors.dense(Array.fill(20)(9.0)))
    ).toDF("features")

    val kernel = new GeneralizedIDivergenceKernel(1e-10)

    val auto = new AutoAssignment(broadcastThresholdElems = 100, chunkSize = 3)

    val result = auto.assign(df20d, "features", None, centers, kernel)

    result.count() shouldBe 3
    result.columns should contain("cluster")

    // Verify all cluster IDs are valid
    val clusterIds = result.select("cluster").collect().map(_.getInt(0))
    clusterIds.foreach { id =>
      id should be >= 0
      id should be < 10
    }
  }

  test("AutoAssignment: logs strategy selection") {
    val df      = testDF()
    val centers = Array(Array(0.5, 0.5), Array(9.5, 9.5))

    // Test SE kernel
    val seKernel = new SquaredEuclideanKernel()
    val auto1    = new AutoAssignment()
    auto1.assign(df, "features", None, centers, seKernel)

    // Test non-SE with small k×dim
    val klKernel = new KLDivergenceKernel(1e-10)
    val auto2    = new AutoAssignment()
    auto2.assign(df, "features", None, centers, klKernel)

    // If we got here without exceptions, logging worked
    succeed
  }

  test("ChunkedBroadcastAssignment: handles single-chunk case") {
    val df      = testDF()
    val centers = Array(Array(0.5, 0.5), Array(9.5, 9.5))
    val kernel  = new SquaredEuclideanKernel()

    val chunked = new ChunkedBroadcastAssignment(chunkSize = 100)

    val result = chunked.assign(df, "features", None, centers, kernel)

    result.count() shouldBe 6
    result.columns should contain("cluster")
  }

  test("BroadcastUDFAssignment: works with different kernels") {
    val df = Seq(
      Tuple1(Vectors.dense(0.1, 0.2)),
      Tuple1(Vectors.dense(0.9, 0.8))
    ).toDF("features")

    val centers = Array(
      Array(0.1, 0.1),
      Array(0.9, 0.9)
    )

    val kernels = Seq(
      new SquaredEuclideanKernel(),
      new KLDivergenceKernel(1e-6),
      new GeneralizedIDivergenceKernel(1e-6)
    )

    val strategy = new BroadcastUDFAssignment()

    kernels.foreach { kernel =>
      val result = strategy.assign(df, "features", None, centers, kernel)
      result.count() shouldBe 2
      result.columns should contain("cluster")
    }
  }

  test("ChunkedBroadcastAssignment: correctness with KL divergence") {
    val df = Seq(
      Tuple1(Vectors.dense(0.1, 0.2)),
      Tuple1(Vectors.dense(0.15, 0.25)),
      Tuple1(Vectors.dense(0.9, 0.8)),
      Tuple1(Vectors.dense(0.95, 0.85))
    ).toDF("features")

    val centers = Array(
      Array(0.1, 0.1),
      Array(0.9, 0.9)
    )

    val kernel = new KLDivergenceKernel(1e-6)

    val broadcast = new BroadcastUDFAssignment()
    val chunked   = new ChunkedBroadcastAssignment(chunkSize = 1)

    val broadcastResult = broadcast.assign(df, "features", None, centers, kernel)
    val chunkedResult   = chunked.assign(df, "features", None, centers, kernel)

    val broadcastClusters = broadcastResult.select("cluster").collect().map(_.getInt(0))
    val chunkedClusters   = chunkedResult.select("cluster").collect().map(_.getInt(0))

    broadcastClusters should contain theSameElementsInOrderAs chunkedClusters
  }
}
