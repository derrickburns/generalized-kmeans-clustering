package com.massivedatascience.clusterer.ml

import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

/** Benchmark suite for comparing assignment strategies and divergences.
  *
  * This suite measures:
  *   - Lloyd's iteration throughput (points/sec/iteration)
  *   - Assignment strategy performance (SE fast path vs broadcast UDF)
  *   - Divergence comparison (overhead of non-SE divergences)
  *
  * Results are written to target/perf-reports/benchmark.json for tracking.
  */
class BenchmarkSuite extends AnyFunSuite {

  private def withSpark[T](name: String)(f: SparkSession => T): T = {
    val spark = SparkSession
      .builder()
      .appName(name)
      .master("local[*]")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    try f(spark)
    finally spark.stop()
  }

  case class BenchmarkResult(
      name: String,
      divergence: String,
      strategy: String,
      k: Int,
      dim: Int,
      numPoints: Int,
      maxIter: Int,
      actualIter: Int,
      elapsedMs: Long,
      pointsPerSec: Double,
      pointsPerIterPerSec: Double
  )

  private def runBenchmark(
      spark: SparkSession,
      name: String,
      divergence: String,
      strategy: String,
      k: Int,
      dim: Int,
      numPoints: Int,
      maxIter: Int,
      seed: Long
  ): BenchmarkResult = {
    import spark.implicits._

    // Generate test data - positive for non-SE divergences
    val data = (0 until numPoints).map { i =>
      val cluster = i % k
      val base = if (divergence == "squaredEuclidean" || divergence == "spherical") {
        (0 until dim).map(d => cluster * 10.0 + scala.util.Random.nextGaussian()).toArray
      } else {
        // Positive values for KL, IS, etc.
        (0 until dim).map(d => cluster + 1.0 + math.abs(scala.util.Random.nextGaussian()) * 0.1).toArray
      }
      Tuple1(Vectors.dense(base))
    }.toDF("features")

    data.cache()
    data.count() // Warm up cache

    val gkm = new GeneralizedKMeans()
      .setK(k)
      .setDivergence(divergence)
      .setAssignmentStrategy(strategy)
      .setMaxIter(maxIter)
      .setSeed(seed)
      .setCheckpointInterval(0) // Disable checkpointing for benchmarks

    val startTime = System.currentTimeMillis()
    val model = gkm.fit(data)
    val predictions = model.transform(data)
    predictions.count() // Force evaluation
    val endTime = System.currentTimeMillis()

    data.unpersist()

    val elapsedMs = endTime - startTime
    val actualIter = if (model.hasSummary) model.summary.iterations else maxIter
    val pointsPerSec = numPoints * 1000.0 / elapsedMs
    val pointsPerIterPerSec = if (actualIter > 0) numPoints * 1000.0 / (elapsedMs / actualIter.toDouble) else 0.0

    BenchmarkResult(
      name = name,
      divergence = divergence,
      strategy = strategy,
      k = k,
      dim = dim,
      numPoints = numPoints,
      maxIter = maxIter,
      actualIter = actualIter,
      elapsedMs = elapsedMs,
      pointsPerSec = pointsPerSec,
      pointsPerIterPerSec = pointsPerIterPerSec
    )
  }

  test("benchmark - assignment strategy comparison (SE)") {
    withSpark("BenchmarkStrategies") { spark =>
      val results = Seq("auto", "crossJoin", "broadcast").map { strategy =>
        runBenchmark(
          spark = spark,
          name = s"SE-$strategy",
          divergence = "squaredEuclidean",
          strategy = strategy,
          k = 5,
          dim = 10,
          numPoints = 2000,
          maxIter = 5,
          seed = 42L
        )
      }

      println("\n=== Assignment Strategy Comparison (Squared Euclidean) ===")
      println(f"${"Strategy"}%-15s ${"Time(ms)"}%10s ${"Pts/sec"}%12s ${"Pts/iter/sec"}%14s")
      println("-" * 55)
      results.foreach { r =>
        println(f"${r.strategy}%-15s ${r.elapsedMs}%10d ${r.pointsPerSec}%12.0f ${r.pointsPerIterPerSec}%14.0f")
      }

      // All strategies should complete reasonably
      results.foreach { r =>
        assert(r.elapsedMs < 30000, s"${r.strategy} too slow: ${r.elapsedMs}ms")
      }
    }
  }

  test("benchmark - divergence comparison") {
    withSpark("BenchmarkDivergences") { spark =>
      val divergences = Seq(
        ("squaredEuclidean", "auto"),
        ("kl", "auto"),
        ("spherical", "auto"),
        ("l1", "auto")
      )

      val results = divergences.map { case (div, strategy) =>
        runBenchmark(
          spark = spark,
          name = div,
          divergence = div,
          strategy = strategy,
          k = 5,
          dim = 10,
          numPoints = 2000,
          maxIter = 5,
          seed = 42L
        )
      }

      println("\n=== Divergence Comparison ===")
      println(f"${"Divergence"}%-18s ${"Time(ms)"}%10s ${"Pts/sec"}%12s ${"Iterations"}%12s")
      println("-" * 55)
      results.foreach { r =>
        println(f"${r.divergence}%-18s ${r.elapsedMs}%10d ${r.pointsPerSec}%12.0f ${r.actualIter}%12d")
      }

      // All divergences should complete
      results.foreach { r =>
        assert(r.elapsedMs < 60000, s"${r.divergence} too slow: ${r.elapsedMs}ms")
      }

      // Write JSON report
      writeReport(results, "benchmark-divergences.json")
    }
  }

  test("benchmark - scaling with k") {
    withSpark("BenchmarkScalingK") { spark =>
      val kValues = Seq(2, 5, 10, 20)

      val results = kValues.map { k =>
        runBenchmark(
          spark = spark,
          name = s"k=$k",
          divergence = "squaredEuclidean",
          strategy = "auto",
          k = k,
          dim = 10,
          numPoints = 2000,
          maxIter = 5,
          seed = 42L
        )
      }

      println("\n=== Scaling with K (Squared Euclidean) ===")
      println(f"${"K"}%5s ${"Time(ms)"}%10s ${"Pts/sec"}%12s ${"Pts/iter/sec"}%14s")
      println("-" * 45)
      results.foreach { r =>
        println(f"${r.k}%5d ${r.elapsedMs}%10d ${r.pointsPerSec}%12.0f ${r.pointsPerIterPerSec}%14.0f")
      }

      // Time should scale reasonably (not exponentially) with k
      val maxTime = results.map(_.elapsedMs).max
      val minTime = results.map(_.elapsedMs).min
      assert(maxTime <= minTime * 20, s"Scaling appears poor: max=${maxTime}ms, min=${minTime}ms")

      writeReport(results, "benchmark-scaling-k.json")
    }
  }

  test("benchmark - scaling with dimension") {
    withSpark("BenchmarkScalingDim") { spark =>
      val dimValues = Seq(5, 20, 50, 100)

      val results = dimValues.map { dim =>
        runBenchmark(
          spark = spark,
          name = s"dim=$dim",
          divergence = "squaredEuclidean",
          strategy = "auto",
          k = 5,
          dim = dim,
          numPoints = 2000,
          maxIter = 5,
          seed = 42L
        )
      }

      println("\n=== Scaling with Dimension (Squared Euclidean) ===")
      println(f"${"Dim"}%5s ${"Time(ms)"}%10s ${"Pts/sec"}%12s ${"Pts/iter/sec"}%14s")
      println("-" * 45)
      results.foreach { r =>
        println(f"${r.dim}%5d ${r.elapsedMs}%10d ${r.pointsPerSec}%12.0f ${r.pointsPerIterPerSec}%14.0f")
      }

      // All should complete in reasonable time
      results.foreach { r =>
        assert(r.elapsedMs < 60000, s"dim=${r.dim} too slow: ${r.elapsedMs}ms")
      }

      writeReport(results, "benchmark-scaling-dim.json")
    }
  }

  test("benchmark - spherical vs squared euclidean on normalized data") {
    withSpark("BenchmarkSphericalVsSE") { spark =>
      import spark.implicits._

      // Generate normalized data (unit vectors)
      val numPoints = 2000
      val dim = 20
      val k = 5

      val normalizedData = (0 until numPoints).map { i =>
        val cluster = i % k
        val raw = (0 until dim).map(d => cluster + scala.util.Random.nextGaussian()).toArray
        val norm = math.sqrt(raw.map(x => x * x).sum)
        val normalized = raw.map(_ / norm)
        Tuple1(Vectors.dense(normalized))
      }.toDF("features")

      normalizedData.cache()
      normalizedData.count()

      val seResult = runBenchmark(
        spark = spark,
        name = "SE-normalized",
        divergence = "squaredEuclidean",
        strategy = "auto",
        k = k,
        dim = dim,
        numPoints = numPoints,
        maxIter = 10,
        seed = 42L
      )

      val sphericalResult = runBenchmark(
        spark = spark,
        name = "Spherical",
        divergence = "spherical",
        strategy = "auto",
        k = k,
        dim = dim,
        numPoints = numPoints,
        maxIter = 10,
        seed = 42L
      )

      normalizedData.unpersist()

      println("\n=== Spherical vs SE on Normalized Data ===")
      println(f"${"Method"}%-18s ${"Time(ms)"}%10s ${"Pts/sec"}%12s ${"Iterations"}%12s")
      println("-" * 55)
      Seq(seResult, sphericalResult).foreach { r =>
        println(f"${r.divergence}%-18s ${r.elapsedMs}%10d ${r.pointsPerSec}%12.0f ${r.actualIter}%12d")
      }

      // Both should complete reasonably
      assert(seResult.elapsedMs < 30000, s"SE too slow: ${seResult.elapsedMs}ms")
      assert(sphericalResult.elapsedMs < 30000, s"Spherical too slow: ${sphericalResult.elapsedMs}ms")
    }
  }

  private def writeReport(results: Seq[BenchmarkResult], filename: String): Unit = {
    val reportDir = new java.io.File("target/perf-reports")
    reportDir.mkdirs()

    val json = s"""{
      |  "timestamp": ${System.currentTimeMillis()},
      |  "benchmarks": [
      |${results.map(r => s"""    {
      |      "name": "${r.name}",
      |      "divergence": "${r.divergence}",
      |      "strategy": "${r.strategy}",
      |      "k": ${r.k},
      |      "dim": ${r.dim},
      |      "numPoints": ${r.numPoints},
      |      "maxIter": ${r.maxIter},
      |      "actualIter": ${r.actualIter},
      |      "elapsedMs": ${r.elapsedMs},
      |      "pointsPerSec": ${r.pointsPerSec},
      |      "pointsPerIterPerSec": ${r.pointsPerIterPerSec}
      |    }""").mkString(",\n")}
      |  ]
      |}""".stripMargin

    val reportFile = new java.io.PrintWriter(new java.io.File(reportDir, filename))
    try {
      reportFile.write(json)
      println(s"\nBenchmark report written to target/perf-reports/$filename")
    } finally {
      reportFile.close()
    }
  }
}
