package com.massivedatascience.clusterer.ml.df

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

/** Tests for ElkanLloydsIterator with cross-iteration bounds tracking. */
class ElkanLloydsIteratorSuite extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("ElkanLloydsIteratorSuite")
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

  /** Create test config with required strategies. */
  def createConfig(k: Int, maxIter: Int = 100, tol: Double = 1e-4): LloydsConfig = {
    LloydsConfig(
      k = k,
      maxIter = maxIter,
      tol = tol,
      kernel = new SquaredEuclideanKernel(),
      assigner = new BroadcastUDFAssignment(), // Not used by Elkan iterator
      updater = new GradMeanUDAFUpdate(),
      emptyHandler = new ReseedRandomHandler(),
      convergence = new MovementConvergence(),
      validator = new StandardInputValidator(),
      checkpointInterval = 0,
      checkpointDir = None
    )
  }

  test("Elkan iterator produces same results as default iterator") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create well-separated clusters
    val data = Seq(
      // Cluster 0: around (0, 0)
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(-0.1, 0.1)),
      Tuple1(Vectors.dense(0.1, -0.1)),
      Tuple1(Vectors.dense(-0.1, -0.1)),
      // Cluster 1: around (10, 0)
      Tuple1(Vectors.dense(10.0, 0.0)),
      Tuple1(Vectors.dense(10.1, 0.1)),
      Tuple1(Vectors.dense(9.9, 0.1)),
      Tuple1(Vectors.dense(10.1, -0.1)),
      Tuple1(Vectors.dense(9.9, -0.1)),
      // Cluster 2: around (5, 10)
      Tuple1(Vectors.dense(5.0, 10.0)),
      Tuple1(Vectors.dense(5.1, 10.1)),
      Tuple1(Vectors.dense(4.9, 9.9)),
      Tuple1(Vectors.dense(5.1, 9.9)),
      Tuple1(Vectors.dense(4.9, 10.1)),
      // Cluster 3: around (0, 10)
      Tuple1(Vectors.dense(0.0, 10.0)),
      Tuple1(Vectors.dense(0.1, 10.1)),
      Tuple1(Vectors.dense(-0.1, 9.9)),
      Tuple1(Vectors.dense(0.1, 9.9)),
      Tuple1(Vectors.dense(-0.1, 10.1)),
      // Cluster 4: around (10, 10)
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(10.1, 10.1)),
      Tuple1(Vectors.dense(9.9, 9.9)),
      Tuple1(Vectors.dense(10.1, 9.9)),
      Tuple1(Vectors.dense(9.9, 10.1))
    ).toDF("features")

    val initialCenters = Array(
      Array(0.5, 0.5),    // Near cluster 0
      Array(9.5, 0.5),    // Near cluster 1
      Array(5.5, 9.5),    // Near cluster 2
      Array(0.5, 9.5),    // Near cluster 3
      Array(9.5, 9.5)     // Near cluster 4
    )

    val config = createConfig(k = 5)

    // Run default iterator
    val defaultIterator = new DefaultLloydsIterator()
    val defaultResult = defaultIterator.run(data, "features", None, initialCenters.map(_.clone()), config)

    // Run Elkan iterator
    val elkanIterator = new ElkanLloydsIterator()
    val elkanResult = elkanIterator.run(data, "features", None, initialCenters.map(_.clone()), config)

    // Both should converge
    assert(defaultResult.converged, "Default iterator should converge")
    assert(elkanResult.converged, "Elkan iterator should converge")

    // Centers should be approximately equal
    for (i <- defaultResult.centers.indices) {
      for (j <- defaultResult.centers(i).indices) {
        assert(
          math.abs(defaultResult.centers(i)(j) - elkanResult.centers(i)(j)) < 0.1,
          s"Center $i dimension $j differs: default=${defaultResult.centers(i)(j)}, elkan=${elkanResult.centers(i)(j)}"
        )
      }
    }

    // Final distortion should be similar
    val defaultDistortion = defaultResult.distortionHistory.last
    val elkanDistortion = elkanResult.distortionHistory.last
    assert(
      math.abs(defaultDistortion - elkanDistortion) / defaultDistortion < 0.01,
      s"Distortion differs: default=$defaultDistortion, elkan=$elkanDistortion"
    )
  }

  test("Elkan iterator falls back for small k") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(11.0, 11.0))
    ).toDF("features")

    val initialCenters = Array(
      Array(0.5, 0.5),
      Array(10.5, 10.5)
    )

    val config = createConfig(k = 2)

    val elkanIterator = new ElkanLloydsIterator()
    val result = elkanIterator.run(data, "features", None, initialCenters, config)

    // Should complete successfully (falls back to default)
    assert(result.converged || result.iterations == config.maxIter)
    assert(result.centers.length == 2)
  }

  test("Elkan iterator rejects non-SE kernels") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.5, 0.5))
    ).toDF("features")

    val initialCenters = Array(
      Array(0.3, 0.7),
      Array(0.6, 0.4),
      Array(0.2, 0.8),
      Array(0.8, 0.2),
      Array(0.5, 0.5)
    )

    val config = LloydsConfig(
      k = 5,
      maxIter = 10,
      tol = 1e-4,
      kernel = new KLDivergenceKernel(),  // Non-SE kernel
      assigner = new BroadcastUDFAssignment(),
      updater = new GradMeanUDAFUpdate(),
      emptyHandler = new ReseedRandomHandler(),
      convergence = new MovementConvergence(),
      validator = new StandardInputValidator(),
      checkpointInterval = 0,
      checkpointDir = None
    )

    val elkanIterator = new ElkanLloydsIterator()

    val thrown = intercept[IllegalArgumentException] {
      elkanIterator.run(data, "features", None, initialCenters, config)
    }

    assert(thrown.getMessage.contains("Squared Euclidean"))
  }

  test("LloydsIteratorFactory creates appropriate iterator") {
    val seKernel = new SquaredEuclideanKernel()
    val klKernel = new KLDivergenceKernel()

    // SE kernel with large k should use Elkan
    val elkanIterator = LloydsIteratorFactory.create(seKernel, 10, useAcceleration = true)
    assert(elkanIterator.isInstanceOf[ElkanLloydsIterator])

    // SE kernel with small k should use default
    val defaultSmallK = LloydsIteratorFactory.create(seKernel, 3, useAcceleration = true)
    assert(defaultSmallK.isInstanceOf[DefaultLloydsIterator])

    // Non-SE kernel should use default
    val defaultKL = LloydsIteratorFactory.create(klKernel, 10, useAcceleration = true)
    assert(defaultKL.isInstanceOf[DefaultLloydsIterator])

    // Acceleration disabled should use default
    val defaultDisabled = LloydsIteratorFactory.create(seKernel, 10, useAcceleration = false)
    assert(defaultDisabled.isInstanceOf[DefaultLloydsIterator])
  }

  test("Elkan iterator converges on synthetic data") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create synthetic data with clear cluster structure
    val numClusters = 10
    val pointsPerCluster = 20
    val separation = 50.0

    val random = new scala.util.Random(42)
    val points = (0 until numClusters).flatMap { c =>
      val cx = (c % 5) * separation
      val cy = (c / 5) * separation
      (0 until pointsPerCluster).map { _ =>
        Tuple1(Vectors.dense(cx + random.nextGaussian() * 2, cy + random.nextGaussian() * 2))
      }
    }

    val data = points.toDF("features")

    // Initialize centers near true centers
    val initialCenters = (0 until numClusters).map { c =>
      val cx = (c % 5) * separation
      val cy = (c / 5) * separation
      Array(cx + random.nextGaussian() * 5, cy + random.nextGaussian() * 5)
    }.toArray

    val config = createConfig(k = numClusters, maxIter = 50)

    val elkanIterator = new ElkanLloydsIterator()
    val result = elkanIterator.run(data, "features", None, initialCenters, config)

    // Should converge
    assert(result.converged, s"Should converge but took ${result.iterations} iterations")

    // Distortion should decrease monotonically (or nearly so)
    for (i <- 1 until result.distortionHistory.length) {
      val prev = result.distortionHistory(i - 1)
      val curr = result.distortionHistory(i)
      // Allow small increases due to empty cluster reseeding
      assert(curr <= prev * 1.01, s"Distortion increased significantly at iteration $i: $prev -> $curr")
    }
  }

  test("Elkan iterator handles edge case: all points in one cluster") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(-0.1, 0.1)),
      Tuple1(Vectors.dense(0.1, -0.1)),
      Tuple1(Vectors.dense(-0.1, -0.1))
    ).toDF("features")

    // Centers spread out but data is concentrated
    val initialCenters = Array(
      Array(0.0, 0.0),
      Array(10.0, 0.0),
      Array(0.0, 10.0),
      Array(10.0, 10.0),
      Array(5.0, 5.0)
    )

    val config = createConfig(k = 5, maxIter = 20)

    val elkanIterator = new ElkanLloydsIterator()
    val result = elkanIterator.run(data, "features", None, initialCenters, config)

    // Should complete without error - centers may collapse when all points
    // are assigned to one cluster (existing limitation in EmptyClusterHandler
    // which doesn't know the original k)
    assert(result.iterations > 0)
    assert(result.centers.nonEmpty)
    // The final center should be close to the data centroid (0, 0)
    val finalCenter = result.centers.head
    assert(math.abs(finalCenter(0)) < 0.2, s"Center x=${finalCenter(0)} should be near 0")
    assert(math.abs(finalCenter(1)) < 0.2, s"Center y=${finalCenter(1)} should be near 0")
  }
}
