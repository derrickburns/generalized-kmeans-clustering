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

  val xm = new XMeans().setMinK(1).setMaxK(3).setDivergence("squaredEuclidean").setSeed(7)
  val model = xm.fit(df)
  val kFound = model.getK
  assert(kFound >= 1 && kFound <= 3, s"XMeans returned invalid k=$kFound")

  println(s"examples.XMeansExample OK (k=$kFound)")
  spark.stop()
}
