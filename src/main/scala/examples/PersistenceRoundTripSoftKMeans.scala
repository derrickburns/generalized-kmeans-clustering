package examples

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.SoftKMeans

/** Usage: sbt -Dspark.version=3.4.3 "runMain examples.PersistenceRoundTripSoftKMeans save
  * ./tmp_soft_34" sbt -Dspark.version=3.5.1 "runMain examples.PersistenceRoundTripSoftKMeans load
  * ./tmp_soft_34"
  */
object PersistenceRoundTripSoftKMeans {
  def main(args: Array[String]): Unit = {
    require(args.length == 2, "args: save|load <path>")
    val mode = args(0)
    val path = args(1)

    val spark = SparkSession
      .builder()
      .appName("PersistenceRoundTripSoftKMeans")
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._

    val df = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(9.0, 9.0)),
      Tuple1(Vectors.dense(9.1, 9.1)),
      Tuple1(Vectors.dense(10.0, 10.0))
    ).toDF("features")

    mode match {
      case "save" =>
        val softKMeans = new SoftKMeans()
          .setK(2)
          .setDivergence("squaredEuclidean")
          .setBeta(2.0)
          .setMinMembership(0.01)
          .setSeed(789)
        val model      = softKMeans.fit(df)
        model.write.overwrite().save(path)
        println(s"Saved SoftKMeans model to $path")
        println(s"  Centers: ${model.clusterCenters.mkString(", ")}")
        println(s"  Beta: ${model.betaValue}")

      case "load" =>
        val loaded = com.massivedatascience.clusterer.ml.SoftKMeansModel.load(path)

        // Assertions to verify roundtrip correctness
        assert(loaded.numClusters == 2, s"Expected k=2, got ${loaded.numClusters}")
        assert(
          loaded.clusterCenters.length == 2,
          s"Expected 2 centers, got ${loaded.clusterCenters.length}"
        )
        assert(
          loaded.clusterCenters(0).size == 2,
          s"Expected dim=2, got ${loaded.clusterCenters(0).size}"
        )

        // Verify soft clustering parameters
        assert(
          math.abs(loaded.betaValue - 2.0) < 0.001,
          s"Expected beta=2.0, got ${loaded.betaValue}"
        )
        assert(
          math.abs(loaded.minMembershipValue - 0.01) < 0.001,
          s"Expected minMembership=0.01, got ${loaded.minMembershipValue}"
        )

        // Verify predictions work and include probability column
        val preds = loaded.transform(df)
        val n     = preds.count()
        assert(n == 6, s"expected 6 rows after load, got $n")

        // Verify probabilities column exists
        assert(
          preds.columns.contains("probabilities"),
          "Expected 'probabilities' column in predictions"
        )

        // Verify centers are reasonable (one near 0, one near 9-10)
        val centers = loaded.clusterCenters.sortBy(_.apply(0))
        assert(centers(0)(0) < 2.0, s"Center 0 should be near cluster at (0,0), got ${centers(0)}")
        assert(
          centers(1)(0) > 8.0,
          s"Center 1 should be near cluster at (9-10,9-10), got ${centers(1)}"
        )

        println(s"✅ Loaded SoftKMeans model from $path; predictions=$n; all assertions passed")
        println(s"  Centers: ${loaded.clusterCenters.mkString(", ")}")
        println(s"  Beta: ${loaded.betaValue}, MinMembership: ${loaded.minMembershipValue}")

      case other =>
        sys.error(s"Unknown mode: $other")
    }

    spark.stop()
  }
}
