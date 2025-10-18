package examples

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.KMedoids

/** Usage: sbt -Dspark.version=3.4.3 "runMain examples.PersistenceRoundTripKMedoids save
  * ./tmp_kmedoids_34" sbt -Dspark.version=3.5.1 "runMain examples.PersistenceRoundTripKMedoids load
  * ./tmp_kmedoids_34"
  */
object PersistenceRoundTripKMedoids {
  def main(args: Array[String]): Unit = {
    require(args.length == 2, "args: save|load <path>")
    val mode = args(0)
    val path = args(1)

    val spark = SparkSession
      .builder()
      .appName("PersistenceRoundTripKMedoids")
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
        val kmedoids = new KMedoids().setK(2).setMaxIter(10).setSeed(456)
        val model    = kmedoids.fit(df)
        model.write.overwrite().save(path)
        println(s"Saved KMedoids model to $path")
        println(s"  Medoids: ${model.medoids.mkString(", ")}")
        println(s"  Medoid indices: ${model.medoidIndices.mkString(", ")}")

      case "load" =>
        val loaded = com.massivedatascience.clusterer.ml.KMedoidsModel.load(path)

        // Assertions to verify roundtrip correctness
        assert(loaded.numClusters == 2, s"Expected k=2, got ${loaded.numClusters}")
        assert(loaded.medoids.length == 2, s"Expected 2 medoids, got ${loaded.medoids.length}")
        assert(
          loaded.medoidIndices.length == 2,
          s"Expected 2 medoid indices, got ${loaded.medoidIndices.length}"
        )
        assert(loaded.numFeatures == 2, s"Expected dim=2, got ${loaded.numFeatures}")

        // Verify predictions work
        val preds = loaded.transform(df)
        val n     = preds.count()
        assert(n == 6, s"expected 6 rows after load, got $n")

        // Verify medoids are actual data points (one near 0, one near 9-10)
        val medoids = loaded.medoids.sortBy(_.apply(0))
        assert(medoids(0)(0) < 2.0, s"Medoid 0 should be near cluster at (0,0), got ${medoids(0)}")
        assert(
          medoids(1)(0) > 8.0,
          s"Medoid 1 should be near cluster at (9-10,9-10), got ${medoids(1)}"
        )

        // Verify medoid indices are valid
        assert(
          loaded.medoidIndices.forall(i => i >= 0 && i < 6),
          s"Medoid indices should be in [0,5], got ${loaded.medoidIndices.mkString(", ")}"
        )

        println(s"✅ Loaded KMedoids model from $path; predictions=$n; all assertions passed")
        println(s"  Medoids: ${loaded.medoids.mkString(", ")}")
        println(s"  Medoid indices: ${loaded.medoidIndices.mkString(", ")}")

      case other =>
        sys.error(s"Unknown mode: $other")
    }

    spark.stop()
  }
}
