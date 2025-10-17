package examples

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

/**
 * Usage:
 *   sbt -Dspark.version=3.4.3 "runMain examples.PersistenceRoundTrip save ./tmp_model_34"
 *   sbt -Dspark.version=3.5.1 "runMain examples.PersistenceRoundTrip load ./tmp_model_34"
 */
object PersistenceRoundTrip {
  def main(args: Array[String]): Unit = {
    require(args.length == 2, "args: save|load <path>")
    val mode = args(0)
    val path = args(1)

    val spark = SparkSession.builder().appName("PersistenceRoundTrip").master("local[*]").getOrCreate()
    import spark.implicits._

    val df = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(9.0, 9.0)),
      Tuple1(Vectors.dense(10.0, 10.0))
    ).toDF("features")

    mode match {
      case "save" =>
        val gkm = new GeneralizedKMeans()
          .setK(2)
          .setDivergence("squaredEuclidean")
          .setSeed(123)
        val model = gkm.fit(df)
        model.write.overwrite().save(path)
        println(s"Saved model to $path")

      case "load" =>
        val loaded = com.massivedatascience.clusterer.ml.GeneralizedKMeansModel.load(path)
        val preds = loaded.transform(df)
        val n = preds.count()
        assert(n == 4, s"expected 4 rows after load, got $n")
        println(s"Loaded model from $path; predictions=$n")
      case other =>
        sys.error(s"Unknown mode: $other")
    }

    spark.stop()
  }
}
