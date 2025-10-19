package examples

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.SoftKMeans

object SoftKMeansExample extends App {
  val spark = SparkSession.builder().appName("SoftKMeansExample").master("local[*]").getOrCreate()
  import spark.implicits._

  val df = Seq(
    Tuple1(Vectors.dense(0.0, 0.0)),
    Tuple1(Vectors.dense(1.0, 1.0)),
    Tuple1(Vectors.dense(9.0, 9.0)),
    Tuple1(Vectors.dense(10.0, 10.0))
  ).toDF("features")

  val soft  = new SoftKMeans().setK(2).setBeta(1.5).setDivergence("squaredEuclidean").setSeed(11)
  val model = soft.fit(df)
  val pred  = model.transform(df)
  assert(pred.columns.contains("probabilities"), "probabilities column missing")
  assert(pred.columns.contains("prediction"), "prediction column missing")

  // Demonstrate training summary usage
  if (model.hasSummary) {
    val summary = model.summary
    println(s"\nTraining Summary:")
    println(s"  Algorithm: ${summary.algorithm}")
    println(s"  Clusters: ${summary.effectiveK}/${summary.k}")
    println(s"  Iterations: ${summary.iterations} (converged=${summary.converged})")
    println(s"  Final distortion: ${summary.finalDistortion}")
    println(s"  Training time: ${summary.elapsedMillis}ms")
    println(s"  Assignment strategy: ${summary.assignmentStrategy}")

    assert(summary.iterations >= 1, "should have at least 1 iteration")
    assert(summary.effectiveK <= summary.k, "effective k should be <= requested k")
    assert(summary.assignmentStrategy == "SoftEM", "should use SoftEM strategy")
  }

  println("\nexamples.SoftKMeansExample OK")
  spark.stop()
}
