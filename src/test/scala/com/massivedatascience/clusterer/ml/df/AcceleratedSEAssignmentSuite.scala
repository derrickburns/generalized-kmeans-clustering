package com.massivedatascience.clusterer.ml.df

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

/** Tests for AcceleratedSEAssignment with triangle inequality pruning. */
class AcceleratedSEAssignmentSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("AcceleratedSEAssignmentSuite")
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

  test("accelerated assignment produces same results as standard assignment") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create well-separated clusters for good pruning
    val data = Seq(
      // Cluster 0: around (0, 0)
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(-0.1, 0.1)),
      // Cluster 1: around (10, 0)
      Tuple1(Vectors.dense(10.0, 0.0)),
      Tuple1(Vectors.dense(10.1, 0.1)),
      Tuple1(Vectors.dense(9.9, 0.1)),
      // Cluster 2: around (5, 10)
      Tuple1(Vectors.dense(5.0, 10.0)),
      Tuple1(Vectors.dense(5.1, 10.1)),
      Tuple1(Vectors.dense(4.9, 9.9)),
      // Cluster 3: around (0, 10)
      Tuple1(Vectors.dense(0.0, 10.0)),
      Tuple1(Vectors.dense(0.1, 10.1)),
      Tuple1(Vectors.dense(-0.1, 9.9)),
      // Cluster 4: around (10, 10)
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(10.1, 10.1)),
      Tuple1(Vectors.dense(9.9, 9.9))
    ).toDF("features")

    val centers = Array(
      Array(0.0, 0.0),   // Cluster 0
      Array(10.0, 0.0),  // Cluster 1
      Array(5.0, 10.0),  // Cluster 2
      Array(0.0, 10.0),  // Cluster 3
      Array(10.0, 10.0)  // Cluster 4
    )

    val kernel = new SquaredEuclideanKernel()

    // Run standard assignment
    val standardAssigner = new BroadcastUDFAssignment()
    val standardResult = standardAssigner.assign(data, "features", None, centers, kernel)
    val standardAssignments = standardResult.select("cluster").collect().map(_.getInt(0))

    // Run accelerated assignment
    val acceleratedAssigner = new AcceleratedSEAssignment()
    val acceleratedResult = acceleratedAssigner.assign(data, "features", None, centers, kernel)
    val acceleratedAssignments = acceleratedResult.select("cluster").collect().map(_.getInt(0))

    // Verify same results
    assert(standardAssignments.toSeq === acceleratedAssignments.toSeq,
      "Accelerated assignment should produce identical results to standard assignment")
  }

  test("accelerated assignment falls back to standard for small k") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(5.0, 5.0))
    ).toDF("features")

    val centers = Array(
      Array(0.0, 0.0),
      Array(5.0, 5.0)
    )

    val kernel = new SquaredEuclideanKernel()
    val assigner = new AcceleratedSEAssignment()

    // Should complete without error (falls back internally)
    val result = assigner.assign(data, "features", None, centers, kernel)
    assert(result.count() === 3)
  }

  test("accelerated assignment rejects non-SE kernels") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.5, 0.5))
    ).toDF("features")

    val centers = Array(
      Array(0.3, 0.7),
      Array(0.6, 0.4),
      Array(0.2, 0.8),
      Array(0.8, 0.2),
      Array(0.5, 0.5)
    )

    val klKernel = new KLDivergenceKernel()
    val assigner = new AcceleratedSEAssignment()

    val thrown = intercept[IllegalArgumentException] {
      assigner.assign(data, "features", None, centers, klKernel)
    }

    assert(thrown.getMessage.contains("Squared Euclidean"))
  }

  test("AcceleratedAssignment factory returns correct strategy") {
    val seKernel = new SquaredEuclideanKernel()
    val klKernel = new KLDivergenceKernel()

    // SE kernel with large k should use accelerated
    val seStrategy = AcceleratedAssignment.forKernel(seKernel, 10)
    assert(seStrategy.isInstanceOf[AcceleratedSEAssignment])

    // SE kernel with small k should use standard
    val seSmallK = AcceleratedAssignment.forKernel(seKernel, 3)
    assert(seSmallK.isInstanceOf[BroadcastUDFAssignment])

    // Non-SE kernel should use standard
    val klStrategy = AcceleratedAssignment.forKernel(klKernel, 10)
    assert(klStrategy.isInstanceOf[BroadcastUDFAssignment])
  }

  test("accelerated assignment handles single center") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(2.0, 2.0))
    ).toDF("features")

    val kernel = new SquaredEuclideanKernel()
    val assigner = new AcceleratedSEAssignment()

    // Single center - falls back to standard
    val single = Array(Array(1.5, 1.5))
    val singleResult = assigner.assign(data, "features", None, single, kernel)
    assert(singleResult.select("cluster").distinct().count() === 1)
  }

  test("pruning provides speedup for well-separated clusters") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create many well-separated clusters
    val numClusters = 20
    val pointsPerCluster = 50
    val separation = 100.0  // Large separation for high pruning rate

    val points = (0 until numClusters).flatMap { c =>
      val cx = (c % 5) * separation
      val cy = (c / 5) * separation
      (0 until pointsPerCluster).map { _ =>
        Tuple1(Vectors.dense(cx + math.random() * 2 - 1, cy + math.random() * 2 - 1))
      }
    }

    val data = points.toDF("features")

    val centers = (0 until numClusters).map { c =>
      val cx = (c % 5) * separation
      val cy = (c / 5) * separation
      Array(cx, cy)
    }.toArray

    val kernel = new SquaredEuclideanKernel()
    val assigner = new AcceleratedSEAssignment()

    // Run and verify correctness
    val result = assigner.assign(data, "features", None, centers, kernel)
    val count = result.count()

    assert(count === numClusters * pointsPerCluster)

    // Verify assignments are reasonable (each cluster should have roughly pointsPerCluster)
    val clusterCounts = result.groupBy("cluster").count().collect()
    assert(clusterCounts.length === numClusters)
  }

  test("assignment correctness for overlapping clusters") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create overlapping clusters where pruning is less effective
    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 0.0)),
      Tuple1(Vectors.dense(2.0, 0.0)),
      Tuple1(Vectors.dense(3.0, 0.0)),
      Tuple1(Vectors.dense(4.0, 0.0)),
      Tuple1(Vectors.dense(0.5, 0.0)),
      Tuple1(Vectors.dense(1.5, 0.0)),
      Tuple1(Vectors.dense(2.5, 0.0)),
      Tuple1(Vectors.dense(3.5, 0.0))
    ).toDF("features")

    // Centers close together
    val centers = Array(
      Array(0.0, 0.0),
      Array(1.0, 0.0),
      Array(2.0, 0.0),
      Array(3.0, 0.0),
      Array(4.0, 0.0)
    )

    val kernel = new SquaredEuclideanKernel()

    val standard = new BroadcastUDFAssignment()
    val accelerated = new AcceleratedSEAssignment()

    val standardResult = standard.assign(data, "features", None, centers, kernel)
      .select("cluster").collect().map(_.getInt(0))

    val acceleratedResult = accelerated.assign(data, "features", None, centers, kernel)
      .select("cluster").collect().map(_.getInt(0))

    // Must produce identical results even with overlapping clusters
    assert(standardResult.toSeq === acceleratedResult.toSeq)
  }
}
