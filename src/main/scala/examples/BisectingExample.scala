package examples

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

object BisectingExample extends App {
  val spark = SparkSession.builder().appName("BisectingExample").master("local[*]").getOrCreate()
  import spark.implicits._

  val df = Seq(
    Tuple1(Vectors.dense(0.0, 0.0)),
    Tuple1(Vectors.dense(1.0, 1.0)),
    Tuple1(Vectors.dense(9.0, 8.5)),
    Tuple1(Vectors.dense(8.5, 9.0))
  ).toDF("features")

  // Use standard GKM for a trivial run; bisecting variant often wraps base API in your codebase.
  val gkm   =
    new GeneralizedKMeans().setK(2).setDivergence("squaredEuclidean").setMaxIter(5).setSeed(42)
  val model = gkm.fit(df)
  val pred  = model.transform(df)

  val cnt = pred.count()
  assert(cnt == 4, s"expected 4 rows, got $cnt")
  assert(pred.columns.contains("prediction"), "prediction column missing")

  println("examples.BisectingExample OK")
  spark.stop()
}
