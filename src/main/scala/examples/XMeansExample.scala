package examples

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.XMeans

object XMeansExample extends App {
  val spark = SparkSession.builder().appName("XMeansExample").master("local[*]").getOrCreate()
  import spark.implicits._

  val df = Seq(
    Tuple1(Vectors.dense(0.0, 0.0)),
    Tuple1(Vectors.dense(1.0, 1.0)),
    Tuple1(Vectors.dense(9.0, 9.0)),
    Tuple1(Vectors.dense(10.0, 10.0))
  ).toDF("features")

  val xm     = new XMeans().setMinK(2).setMaxK(3).setDivergence("squaredEuclidean").setSeed(7)
  val model  = xm.fit(df)
  val kFound = model.getK
  assert(kFound >= 2 && kFound <= 3, s"XMeans returned invalid k=$kFound")

  // Demonstrate training summary usage (summary is from final k value)
  if (model.hasSummary) {
    val summary = model.summary
    println(s"\nTraining Summary (for optimal k=$kFound):")
    println(s"  Algorithm: ${summary.algorithm}")
    println(s"  Clusters: ${summary.effectiveK}/${summary.k}")
    println(s"  Iterations: ${summary.iterations} (converged=${summary.converged})")
    println(s"  Final distortion: ${summary.finalDistortion}")
    println(s"  Training time: ${summary.elapsedMillis}ms")

    assert(summary.k == kFound, s"summary k should match found k")
    assert(summary.iterations >= 1, "should have at least 1 iteration")
  }

  println(s"\nexamples.XMeansExample OK (k=$kFound)")
  spark.stop()
}
