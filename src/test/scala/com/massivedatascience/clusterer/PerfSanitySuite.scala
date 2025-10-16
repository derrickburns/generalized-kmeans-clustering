package com.massivedatascience.clusterer

import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.sql.SparkSession
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
import org.apache.spark.ml.linalg.Vectors

class PerfSanitySuite extends AnyFunSuite {

  private def withSpark[T](name: String)(f: SparkSession => T): T = {
    val spark = SparkSession.builder().appName(name).master("local[*]")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .getOrCreate()
    try f(spark) finally spark.stop()
  }

  test("perf sanity - SE and KL paths") {
    withSpark("PerfSanity") { spark =>
      import spark.implicits._
      val data = (0 until 2000).map { i =>
        val base = if (i % 2 == 0) 0.0 else 10.0
        Tuple1(Vectors.dense(base + (i % 5) * 0.1, base + (i % 7) * 0.1))
      }.toDF("features")

      val t0 = System.nanoTime()
      val se = new GeneralizedKMeans().setK(2).setDivergence("squaredEuclidean").setMaxIter(5).setSeed(1)
      val mSe = se.fit(data)
      val _ = mSe.transform(data).count()
      val t1 = System.nanoTime()

      val kl = new GeneralizedKMeans().setK(2).setDivergence("kl").setInputTransform("epsilonShift").setShiftValue(1e-6).setMaxIter(3).setSeed(2)
      val mKl = kl.fit(data)
      val _2 = mKl.transform(data).count()
      val t2 = System.nanoTime()

      val seSec = (t1 - t0) / 1e9
      val klSec = (t2 - t1) / 1e9

      // CI will grep these lines:
      println(f"perf_sanity_seconds=SE:${seSec}%.3f")
      println(f"perf_sanity_seconds=KL:${klSec}%.3f")
    }
  }
}
