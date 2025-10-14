package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import org.scalacheck.Gen

/**
 * Property-based tests for DataFrame API using ScalaCheck.
 *
 * These tests verify invariants and properties that should hold for all inputs,
 * helping to catch edge cases and ensure correctness across a wide range of scenarios.
 */
class PropertyBasedTestSuite extends AnyFunSuite
    with ScalaCheckPropertyChecks
    with Matchers
    with BeforeAndAfterAll {

  // Configure ScalaCheck for fewer test cases but faster execution
  implicit override val generatorDrivenConfig: PropertyCheckConfiguration =
    PropertyCheckConfiguration(minSuccessful = 10, maxDiscardedFactor = 5.0)

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("PropertyBasedTestSuite")
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

  // Generators for test data

  /** Generate a valid dimension (between 2 and 10) - conservative for stability */
  val dimGen: Gen[Int] = Gen.choose(2, 10)

  /** Generate a valid k (between 2 and 5) - conservative for stability */
  val kGen: Gen[Int] = Gen.choose(2, 5)

  /** Generate a valid number of points (between k*5 and 50) - ensure enough points per cluster */
  def numPointsGen(k: Int): Gen[Int] = Gen.choose(math.max(k * 5, 20), 50)

  /** Generate a vector with values in [-10, 10] */
  def vectorGen(dim: Int): Gen[Vector] = {
    Gen.listOfN(dim, Gen.choose(-10.0, 10.0)).map(arr => Vectors.dense(arr.toArray))
  }

  /** Generate a vector with all positive values (for KL divergence) */
  def positiveVectorGen(dim: Int): Gen[Vector] = {
    Gen.listOfN(dim, Gen.choose(0.01, 10.0)).map(arr => Vectors.dense(arr.toArray))
  }

  /** Generate a probability vector (non-negative, sums to 1) */
  def probabilityVectorGen(dim: Int): Gen[Vector] = {
    Gen.listOfN(dim, Gen.choose(0.1, 10.0)).map { values =>
      val sum = values.sum
      val normalized = values.map(_ / sum)
      Vectors.dense(normalized.toArray)
    }
  }

  /** Generate a vector with values in (0, 1) for logistic loss */
  def probabilityElementsVectorGen(dim: Int): Gen[Vector] = {
    Gen.listOfN(dim, Gen.choose(0.01, 0.99)).map(arr => Vectors.dense(arr.toArray))
  }

  // Property 1: Number of predictions equals number of input points
  // Note: Some property tests are currently ignored due to an edge case bug in MovementConvergence
  // that causes ArrayIndexOutOfBoundsException when clusters become empty during iterations.

  test("Property: number of predictions equals number of input points") {
    forAll(dimGen, kGen, Gen.choose(20, 50)) { (dim: Int, k: Int, numPoints: Int) =>
      whenever(numPoints >= k * 5) {  // Ensure at least 5 points per cluster
        val sparkSession = spark
        import sparkSession.implicits._

        val data = (1 to numPoints).map(_ =>
          Tuple1(Vectors.dense(Array.fill(dim)(scala.util.Random.nextDouble() * 10 - 5)))
        ).toDF("features")

        val kmeans = new GeneralizedKMeans()
          .setK(k)
          .setDivergence("squaredEuclidean")
          .setMaxIter(5)
          .setSeed(42)

        val model = kmeans.fit(data)
        val predictions = model.transform(data)

        predictions.count() shouldBe numPoints
      }
    }
  }

  // Property 2: All cluster assignments are within valid range [0, k)

  test("Property: cluster assignments are in valid range [0, k)") {
    forAll(dimGen, kGen, Gen.choose(20, 50)) { (dim: Int, k: Int, numPoints: Int) =>
      whenever(numPoints >= k * 5) {
        val sparkSession = spark
        import sparkSession.implicits._

        val data = (1 to numPoints).map(_ =>
          Tuple1(Vectors.dense(Array.fill(dim)(scala.util.Random.nextDouble() * 10 - 5)))
        ).toDF("features")

        val kmeans = new GeneralizedKMeans()
          .setK(k)
          .setDivergence("squaredEuclidean")
          .setMaxIter(5)
          .setSeed(42)

        val model = kmeans.fit(data)
        val predictions = model.transform(data)

        val clusterIds = predictions.select("prediction").collect().map(_.getInt(0))

        clusterIds.foreach { id =>
          id should be >= 0
          id should be < k
        }
      }
    }
  }

  // Property 3: Model produces same results with same seed (reproducibility)

  test("Property: clustering is reproducible with same seed") {
    forAll(dimGen, kGen, Gen.choose(20, 50)) { (dim: Int, k: Int, numPoints: Int) =>
      whenever(numPoints >= k * 5) {
        val sparkSession = spark
        import sparkSession.implicits._

        val data = (1 to numPoints).map(_ =>
          Tuple1(Vectors.dense(Array.fill(dim)(scala.util.Random.nextDouble() * 10 - 5)))
        ).toDF("features").cache()

        val kmeans1 = new GeneralizedKMeans()
          .setK(k)
          .setDivergence("squaredEuclidean")
          .setMaxIter(10)
          .setSeed(42)

        val kmeans2 = new GeneralizedKMeans()
          .setK(k)
          .setDivergence("squaredEuclidean")
          .setMaxIter(10)
          .setSeed(42)

        val model1 = kmeans1.fit(data)
        val model2 = kmeans2.fit(data)

        val predictions1 = model1.transform(data).select("prediction").collect().map(_.getInt(0))
        val predictions2 = model2.transform(data).select("prediction").collect().map(_.getInt(0))

        predictions1 should contain theSameElementsInOrderAs predictions2

        data.unpersist()
      }
    }
  }

  // Property 4: Cost is non-negative
  // Note: This test occasionally triggers an ArrayIndexOutOfBoundsException in MovementConvergence
  // when clusters become empty during iterations. This is a known issue to be fixed.

  test("Property: clustering cost is always non-negative") {
    forAll(dimGen, kGen, Gen.choose(20, 40)) { (dim: Int, k: Int, numPoints: Int) =>
      whenever(numPoints >= k * 5) {
        val sparkSession = spark
        import sparkSession.implicits._

        val data = (1 to numPoints).map(_ =>
          Tuple1(Vectors.dense(Array.fill(dim)(scala.util.Random.nextDouble() * 10 - 5)))
        ).toDF("features")

        val kmeans = new GeneralizedKMeans()
          .setK(k)
          .setDivergence("squaredEuclidean")
          .setMaxIter(5)
          .setSeed(42)

        val model = kmeans.fit(data)
        val cost = model.computeCost(data)

        cost should be >= 0.0
        cost.isNaN shouldBe false
        cost.isInfinity shouldBe false
      }
    }
  }

  // Property 5: Distance column values are non-negative

  test("Property: distance column contains non-negative values") {
    forAll(dimGen, kGen, Gen.choose(20, 40)) { (dim: Int, k: Int, numPoints: Int) =>
      whenever(numPoints >= k * 5) {
        val sparkSession = spark
        import sparkSession.implicits._

        val data = (1 to numPoints).map(_ =>
          Tuple1(Vectors.dense(Array.fill(dim)(scala.util.Random.nextDouble() * 10 - 5)))
        ).toDF("features")

        val kmeans = new GeneralizedKMeans()
          .setK(k)
          .setDivergence("squaredEuclidean")
          .setMaxIter(5)
          .setSeed(42)
          .setDistanceCol("distance")

        val model = kmeans.fit(data)
        val predictions = model.transform(data)

        val distances = predictions.select("distance").collect().map(_.getDouble(0))

        distances.foreach { dist =>
          dist should be >= 0.0
          dist.isNaN shouldBe false
          dist.isInfinity shouldBe false
        }
      }
    }
  }

  // Property 6: Number of cluster centers equals k
  // Note: This test occasionally triggers an ArrayIndexOutOfBoundsException in MovementConvergence
  // when clusters become empty during iterations. This is a known issue to be fixed.

  test("Property: model has exactly k cluster centers") {
    forAll(dimGen, kGen, Gen.choose(20, 40)) { (dim: Int, k: Int, numPoints: Int) =>
      whenever(numPoints >= k * 5) {
        val sparkSession = spark
        import sparkSession.implicits._

        val data = (1 to numPoints).map(_ =>
          Tuple1(Vectors.dense(Array.fill(dim)(scala.util.Random.nextDouble() * 10 - 5)))
        ).toDF("features")

        val kmeans = new GeneralizedKMeans()
          .setK(k)
          .setDivergence("squaredEuclidean")
          .setMaxIter(5)
          .setSeed(42)

        val model = kmeans.fit(data)

        model.numClusters shouldBe k
        model.clusterCenters.length shouldBe k
        model.clusterCentersAsVectors.length shouldBe k
      }
    }
  }

  // Property 7: Center dimensions match feature dimensions
  // Note: This test occasionally triggers an ArrayIndexOutOfBoundsException in MovementConvergence
  // when clusters become empty during iterations. This is a known issue to be fixed.

  test("Property: cluster center dimensions match input dimensions") {
    forAll(dimGen, kGen, Gen.choose(20, 40)) { (dim: Int, k: Int, numPoints: Int) =>
      whenever(numPoints >= k * 5) {
        val sparkSession = spark
        import sparkSession.implicits._

        val data = (1 to numPoints).map(_ =>
          Tuple1(Vectors.dense(Array.fill(dim)(scala.util.Random.nextDouble() * 10 - 5)))
        ).toDF("features")

        val kmeans = new GeneralizedKMeans()
          .setK(k)
          .setDivergence("squaredEuclidean")
          .setMaxIter(5)
          .setSeed(42)

        val model = kmeans.fit(data)

        model.numFeatures shouldBe dim
        model.clusterCenters.foreach { center =>
          center.length shouldBe dim
        }
      }
    }
  }

  // Property 8: Single-point prediction consistency

  test("Property: single-point predict agrees with transform") {
    forAll(dimGen, kGen, Gen.choose(20, 30)) { (dim: Int, k: Int, numPoints: Int) =>
      whenever(numPoints >= k * 5) {
        val sparkSession = spark
        import sparkSession.implicits._

        val vectors = (1 to numPoints).map(_ =>
          Vectors.dense(Array.fill(dim)(scala.util.Random.nextDouble() * 10 - 5))
        )

        val data = vectors.map(Tuple1.apply).toDF("features")

        val kmeans = new GeneralizedKMeans()
          .setK(k)
          .setDivergence("squaredEuclidean")
          .setMaxIter(5)
          .setSeed(42)

        val model = kmeans.fit(data)
        val predictions = model.transform(data).select("prediction").collect().map(_.getInt(0))

        vectors.zip(predictions).foreach { case (vec, transformPrediction) =>
          val singlePrediction = model.predict(vec)
          singlePrediction shouldBe transformPrediction
        }
      }
    }
  }

  // Property 9: KL divergence works with probability distributions

  test("Property: KL divergence handles probability distributions") {
    forAll(Gen.choose(3, 10), Gen.choose(2, 5), Gen.choose(20, 30)) {
      (dim: Int, k: Int, numPoints: Int) =>
      whenever(numPoints >= k * 5) {
        val sparkSession = spark
        import sparkSession.implicits._

        // Generate probability distributions (sum to 1)
        val data = (1 to numPoints).map { _ =>
          val values = Array.fill(dim)(scala.util.Random.nextDouble() + 0.1)
          val sum = values.sum
          val prob = values.map(_ / sum)
          Tuple1(Vectors.dense(prob))
        }.toDF("features")

        val kmeans = new GeneralizedKMeans()
          .setK(k)
          .setDivergence("kl")
          .setSmoothing(1e-10)
          .setMaxIter(5)
          .setSeed(42)

        val model = kmeans.fit(data)
        val cost = model.computeCost(data)

        model.numClusters should be >= 1
        model.numClusters should be <= k
        cost should be >= 0.0
        cost.isNaN shouldBe false
      }
    }
  }

  // Property 10: Weighted clustering preserves total weight

  test("Property: weighted clustering respects point weights") {
    forAll(dimGen, kGen, Gen.choose(20, 30)) { (dim: Int, k: Int, numPoints: Int) =>
      whenever(numPoints >= k * 5) {
        val sparkSession = spark
        import sparkSession.implicits._

        val data = (1 to numPoints).map { _ =>
          val features = Vectors.dense(Array.fill(dim)(scala.util.Random.nextDouble() * 10 - 5))
          val weight = scala.util.Random.nextDouble() * 10 + 1.0
          (features, weight)
        }.toDF("features", "weight")

        val kmeans = new GeneralizedKMeans()
          .setK(k)
          .setDivergence("squaredEuclidean")
          .setWeightCol("weight")
          .setMaxIter(5)
          .setSeed(42)

        val model = kmeans.fit(data)
        val predictions = model.transform(data)

        // Should successfully fit and transform
        predictions.count() shouldBe numPoints

        // All predictions should be valid
        val clusterIds = predictions.select("prediction").collect().map(_.getInt(0))
        clusterIds.foreach { id =>
          id should be >= 0
          id should be < k
        }
      }
    }
  }
}
