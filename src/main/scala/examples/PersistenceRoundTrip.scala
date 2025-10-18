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

        // Assertions to verify roundtrip correctness
        assert(loaded.numClusters == 2, s"Expected k=2, got ${loaded.numClusters}")
        assert(loaded.clusterCenters.length == 2, s"Expected 2 centers, got ${loaded.clusterCenters.length}")
        assert(loaded.numFeatures == 2, s"Expected dim=2, got ${loaded.numFeatures}")

        // Verify predictions work
        val preds = loaded.transform(df)
        val n = preds.count()
        assert(n == 4, s"expected 4 rows after load, got $n")

        // Verify center values are reasonable (should be near (0.5, 0.5) and (9.5, 9.5))
        val centers = loaded.clusterCenters.sortBy(_.apply(0))
        assert(math.abs(centers(0)(0) - 0.5) < 1.0, s"Center 0 x-coord should be near 0.5, got ${centers(0)(0)}")
        assert(math.abs(centers(1)(0) - 9.5) < 1.0, s"Center 1 x-coord should be near 9.5, got ${centers(1)(0)}")

        println(s"✅ Loaded model from $path; predictions=$n; all assertions passed")
      case other =>
        sys.error(s"Unknown mode: $other")
    }

    spark.stop()
  }
}
