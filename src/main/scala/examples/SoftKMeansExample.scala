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

  println("examples.SoftKMeansExample OK")
  spark.stop()
}
