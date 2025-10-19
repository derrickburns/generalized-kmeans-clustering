package com.massivedatascience.clusterer

import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.sql.SparkSession
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
import org.apache.spark.ml.linalg.Vectors

class PerfSanitySuite extends AnyFunSuite {

  private def withSpark[T](name: String)(f: SparkSession => T): T = {
    val spark = SparkSession
      .builder()
      .appName(name)
      .master("local[*]")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .getOrCreate()
    try f(spark)
    finally spark.stop()
  }

  test("perf sanity - SE and KL paths") {
    withSpark("PerfSanity") { spark =>
      import spark.implicits._
      val numPoints = 2000
      val data      = (0 until numPoints).map { i =>
        val base = if (i % 2 == 0) 0.0 else 10.0
        Tuple1(Vectors.dense(base + (i % 5) * 0.1, base + (i % 7) * 0.1))
      }.toDF("features")

      // Squared Euclidean benchmark
      val t0     = System.nanoTime()
      val se     =
        new GeneralizedKMeans().setK(2).setDivergence("squaredEuclidean").setMaxIter(5).setSeed(1)
      val mSe    = se.fit(data)
      val predSe = mSe.transform(data)
      val _      = predSe.count()
      val t1     = System.nanoTime()

      val seSec        = (t1 - t0) / 1e9
      val seThroughput = numPoints / seSec

      // KL divergence benchmark
      val t2     = System.nanoTime()
      val kl     = new GeneralizedKMeans().setK(2).setDivergence("kl").setMaxIter(3).setSeed(2)
      val mKl    = kl.fit(data)
      val predKl = mKl.transform(data)
      val _2     = predKl.count()
      val t3     = System.nanoTime()

      val klSec        = (t3 - t2) / 1e9
      val klThroughput = numPoints / klSec

      // Structured output for CI (grep-able):
      println(f"perf_sanity_seconds=SE:${seSec}%.3f")
      println(f"perf_sanity_seconds=KL:${klSec}%.3f")
      println(f"perf_sanity_throughput=SE:${seThroughput}%.0f")
      println(f"perf_sanity_throughput=KL:${klThroughput}%.0f")

      // JSON output for machine parsing:
      val jsonReport = s"""{
        |  "timestamp": ${System.currentTimeMillis()},
        |  "numPoints": $numPoints,
        |  "benchmarks": [
        |    {
        |      "name": "SquaredEuclidean",
        |      "divergence": "squaredEuclidean",
        |      "k": 2,
        |      "maxIter": 5,
        |      "elapsedSeconds": ${seSec},
        |      "throughputPointsPerSec": ${seThroughput}
        |    },
        |    {
        |      "name": "KL",
        |      "divergence": "kl",
        |      "k": 2,
        |      "maxIter": 3,
        |      "elapsedSeconds": ${klSec},
        |      "throughputPointsPerSec": ${klThroughput}
        |    }
        |  ]
        |}""".stripMargin

      // Write JSON report for artifact upload
      val reportDir  = new java.io.File("target/perf-reports")
      reportDir.mkdirs()
      val reportFile = new java.io.PrintWriter(new java.io.File(reportDir, "perf-sanity.json"))
      try {
        reportFile.write(jsonReport)
      } finally {
        reportFile.close()
      }

      println(s"\nPerformance report written to target/perf-reports/perf-sanity.json")

      // Regression thresholds (fail if significantly slower than expected)
      val seThreshold = 10.0 // SE should complete in < 10 seconds
      val klThreshold = 15.0 // KL should complete in < 15 seconds

      assert(
        seSec < seThreshold,
        f"SE performance regression: ${seSec}%.3fs > ${seThreshold}s threshold"
      )
      assert(
        klSec < klThreshold,
        f"KL performance regression: ${klSec}%.3fs > ${klThreshold}s threshold"
      )
    }
  }
}
