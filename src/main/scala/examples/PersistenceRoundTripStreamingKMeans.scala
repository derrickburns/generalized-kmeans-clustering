package examples

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.StreamingKMeans

/** Usage: sbt -Dspark.version=3.4.3 "runMain examples.PersistenceRoundTripStreamingKMeans save
  * ./tmp_streaming_34" sbt -Dspark.version=3.5.1 "runMain
  * examples.PersistenceRoundTripStreamingKMeans load ./tmp_streaming_34"
  */
object PersistenceRoundTripStreamingKMeans {
  def main(args: Array[String]): Unit = {
    require(args.length == 2, "args: save|load <path>")
    val mode = args(0)
    val path = args(1)

    val spark = SparkSession
      .builder()
      .appName("PersistenceRoundTripStreamingKMeans")
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._

    val df1 = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(1.0, 1.0))
    ).toDF("features")

    val df2 = Seq(
      Tuple1(Vectors.dense(9.0, 9.0)),
      Tuple1(Vectors.dense(9.1, 9.1)),
      Tuple1(Vectors.dense(10.0, 10.0))
    ).toDF("features")

    mode match {
      case "save" =>
        val streamingKMeans = new StreamingKMeans()
          .setK(2)
          .setDivergence("squaredEuclidean")
          .setDecayFactor(0.9)
          .setSmoothing(1e-9)
          .setSeed(42)

        // Initialize model with first batch
        val model1 = streamingKMeans.fit(df1)
        println(s"After batch 1 - Centers: ${model1.clusterCenters.mkString(", ")}")
        println(s"After batch 1 - Weights: ${model1.currentWeights.mkString(", ")}")

        // Simulate streaming update with second batch
        val model2 = model1.update(df2)
        println(s"After batch 2 - Centers: ${model2.clusterCenters.mkString(", ")}")
        println(s"After batch 2 - Weights: ${model2.currentWeights.mkString(", ")}")

        model2.write.overwrite().save(path)
        println(s"Saved StreamingKMeans model to $path")

      case "load" =>
        val loaded = com.massivedatascience.clusterer.ml.StreamingKMeansModel.load(path)

        // Assertions to verify roundtrip correctness
        assert(loaded.numClusters == 2, s"Expected k=2, got ${loaded.numClusters}")
        assert(
          loaded.clusterCenters.length == 2,
          s"Expected 2 centers, got ${loaded.clusterCenters.length}"
        )
        assert(loaded.numFeatures == 2, s"Expected dim=2, got ${loaded.numFeatures}")

        // Verify streaming-specific parameters
        assert(
          loaded.divergenceName == "squaredEuclidean",
          s"Expected squaredEuclidean divergence, got ${loaded.divergenceName}"
        )
        assert(
          math.abs(loaded.decayFactorValue - 0.9) < 0.001,
          s"Expected decayFactor=0.9, got ${loaded.decayFactorValue}"
        )
        assert(
          math.abs(loaded.smoothingValue - 1e-9) < 1e-10,
          s"Expected smoothing=1e-9, got ${loaded.smoothingValue}"
        )

        // CRITICAL: Verify cluster weights were restored (essential for streaming!)
        val currentWeights = loaded.currentWeights
        assert(
          currentWeights.length == 2,
          s"Expected 2 cluster weights, got ${currentWeights.length}"
        )
        assert(
          currentWeights.forall(_ > 0),
          s"Cluster weights should be positive, got ${currentWeights.mkString(", ")}"
        )
        println(s"Cluster weights restored: ${currentWeights.mkString(", ")}")

        // Verify predictions work
        val preds = loaded.transform(df1)
        val n     = preds.count()
        assert(n == 3, s"expected 3 rows after load, got $n")

        // Verify we can continue streaming after load
        val continued = loaded.update(df2)
        assert(
          continued.clusterCenters.length == 2,
          "Should be able to continue streaming after load"
        )
        println(s"After continued update - Centers: ${continued.clusterCenters.mkString(", ")}")
        println(s"After continued update - Weights: ${continued.currentWeights.mkString(", ")}")

        println(s"✅ Loaded StreamingKMeans model from $path; predictions=$n; all assertions passed")
        println(s"  Centers: ${loaded.clusterCenters.mkString(", ")}")
        println(s"  Weights: ${loaded.currentWeights.mkString(", ")}")

      case other =>
        sys.error(s"Unknown mode: $other")
    }

    spark.stop()
  }
}
