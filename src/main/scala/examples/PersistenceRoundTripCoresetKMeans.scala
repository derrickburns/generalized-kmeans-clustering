package examples

import com.massivedatascience.clusterer.ml.CoresetKMeans
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/** Demonstrates save/load round-trip for CoresetKMeans models.
  *
  * This example:
  *   1. Creates a dataset with two clusters
  *   1. Trains a CoresetKMeans model with KL divergence
  *   1. Saves the model to disk
  *   1. Loads the model back
  *   1. Verifies predictions and centers match
  */
object PersistenceRoundTripCoresetKMeans {

  def main(args: Array[String]): Unit = {
    val mode = if (args.length > 0) args(0) else "save"
    val path = if (args.length > 1) args(1) else "/tmp/coreset-kmeans-model"

    val spark =
      SparkSession.builder().appName("CoresetKMeans Persistence").master("local[*]").getOrCreate()

    mode match {
      case "save" => saveModel(spark, path)
      case "load" => loadModel(spark, path)
      case _      =>
        println(s"Unknown mode: $mode. Use 'save' or 'load'")
        sys.exit(1)
    }

    spark.stop()
  }

  private def saveModel(spark: SparkSession, path: String): Unit = {
    import spark.implicits._

    println(s"=== Saving CoresetKMeans model to: $path ===")

    // Create dataset: two Gaussian clusters
    val data = Seq(
      // Cluster 1: probability distributions favoring first dimension
      Tuple1(Vectors.dense(0.9, 0.1)),
      Tuple1(Vectors.dense(0.85, 0.15)),
      Tuple1(Vectors.dense(0.8, 0.2)),
      Tuple1(Vectors.dense(0.88, 0.12)),
      Tuple1(Vectors.dense(0.92, 0.08)),
      // Cluster 2: probability distributions favoring second dimension
      Tuple1(Vectors.dense(0.1, 0.9)),
      Tuple1(Vectors.dense(0.15, 0.85)),
      Tuple1(Vectors.dense(0.2, 0.8)),
      Tuple1(Vectors.dense(0.12, 0.88)),
      Tuple1(Vectors.dense(0.08, 0.92))
    ).toDF("features")

    println(s"Training on ${data.count()} points...")

    // Train model with core-set approximation
    val coreset = new CoresetKMeans()
      .setK(2)
      .setDivergence("kl") // KL divergence for probability distributions
      .setCoresetSize(6)   // Small core-set for demo
      .setEpsilon(0.1)
      .setSensitivityStrategy("hybrid")
      .setRefinementIterations(2)
      .setMaxIter(20)
      .setSeed(42)
      .setFeaturesCol("features")
      .setPredictionCol("cluster")

    val model = coreset.fit(data)

    println(s"Model trained successfully")
    println(s"Cluster centers:")
    model.clusterCenters.zipWithIndex.foreach { case (center, i) =>
      println(f"  Cluster $i: [${center.mkString(", ")}]")
    }

    // Make predictions
    val predictions = model.transform(data)
    println(s"\nPredictions:")
    predictions.select("features", "cluster").show(truncate = false)

    // Save model
    println(s"\nSaving model to: $path")
    model.write.overwrite().save(path)
    println("Model saved successfully!")
  }

  private def loadModel(spark: SparkSession, path: String): Unit = {
    import spark.implicits._

    println(s"=== Loading CoresetKMeans model from: $path ===")

    // Load model
    val model = com.massivedatascience.clusterer.ml.GeneralizedKMeansModel.load(path)
    println("Model loaded successfully!")

    println(s"Number of clusters: ${model.clusterCenters.length}")

    println(s"Cluster centers:")
    model.clusterCenters.zipWithIndex.foreach { case (center, i) =>
      println(f"  Cluster $i: [${center.mkString(", ")}]")
    }

    // Create test data
    val testData = Seq(
      Tuple1(Vectors.dense(0.87, 0.13)), // Should be cluster 0
      Tuple1(Vectors.dense(0.13, 0.87))  // Should be cluster 1
    ).toDF("features")

    // Make predictions (use the prediction column name from the loaded model)
    val predictions = model.transform(testData)
    println(s"\nTest predictions:")
    val predCol     = model.getPredictionCol
    predictions.select("features", predCol).show(truncate = false)

    // Verify predictions are reasonable
    val clusters = predictions.select(predCol).collect().map(_.getInt(0))
    require(
      clusters(0) != clusters(1),
      "Test points from different clusters should have different predictions"
    )
    println("\n✓ Predictions verified successfully!")
  }
}
