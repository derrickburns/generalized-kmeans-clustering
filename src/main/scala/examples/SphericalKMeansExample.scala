package examples

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

/** Demonstrates Spherical K-Means clustering using cosine similarity.
  *
  * Spherical K-Means is ideal for:
  *   - Text/document clustering (TF-IDF, word embeddings)
  *   - Image feature clustering (CNN embeddings)
  *   - Recommendation systems (user/item vectors)
  *   - Any data where direction matters more than magnitude
  *
  * This example simulates clustering document embeddings from two topics.
  */
object SphericalKMeansExample extends App {
  val spark = SparkSession.builder().appName("SphericalKMeansExample").master("local[*]").getOrCreate()
  import spark.implicits._

  // Simulate document embeddings from 2 topics
  // Topic A: vectors pointing roughly in direction (1, 1, 0)
  // Topic B: vectors pointing roughly in direction (0, 0, 1)
  val embeddings = Seq(
    // Topic A documents (similar directions)
    Tuple1(Vectors.dense(0.8, 0.6, 0.0)),
    Tuple1(Vectors.dense(0.9, 0.5, 0.1)),
    Tuple1(Vectors.dense(0.7, 0.7, 0.1)),
    Tuple1(Vectors.dense(0.85, 0.55, 0.05)),
    // Topic B documents (different direction)
    Tuple1(Vectors.dense(0.1, 0.2, 0.95)),
    Tuple1(Vectors.dense(0.0, 0.3, 0.9)),
    Tuple1(Vectors.dense(0.15, 0.1, 0.98)),
    Tuple1(Vectors.dense(0.05, 0.25, 0.92))
  ).toDF("features")

  println("Spherical K-Means Example: Clustering document embeddings")
  println("=" * 60)

  // Use spherical (cosine) divergence
  val sphericalKMeans = new GeneralizedKMeans()
    .setK(2)
    .setDivergence("spherical")  // or "cosine" - both work
    .setMaxIter(20)
    .setSeed(42)

  val model = sphericalKMeans.fit(embeddings)
  val predictions = model.transform(embeddings)

  // Verify basic functionality
  val cnt = predictions.count()
  assert(cnt == 8, s"expected 8 rows, got $cnt")
  assert(predictions.columns.contains("prediction"), "prediction column missing")

  // Show predictions
  println("\nPredictions:")
  predictions.show(truncate = false)

  // Verify clustering quality: documents should be grouped by topic
  val predictions0 = predictions.filter($"prediction" === 0).count()
  val predictions1 = predictions.filter($"prediction" === 1).count()
  println(s"\nCluster distribution: cluster0=$predictions0, cluster1=$predictions1")

  // With well-separated directions, each cluster should have roughly 4 documents
  assert(predictions0 >= 3 && predictions0 <= 5, s"cluster 0 should have ~4 docs, got $predictions0")
  assert(predictions1 >= 3 && predictions1 <= 5, s"cluster 1 should have ~4 docs, got $predictions1")

  // Show cluster centers (normalized)
  println("\nCluster centers (normalized vectors):")
  model.clusterCenters.zipWithIndex.foreach { case (center, i) =>
    val norm = math.sqrt(center.map(x => x * x).sum)
    println(s"  Cluster $i: [${center.map(x => f"$x%.4f").mkString(", ")}] (norm=${"%.4f".format(norm)})")
  }

  // Demonstrate training summary
  if (model.hasSummary) {
    val summary = model.summary
    println(s"\nTraining Summary:")
    println(s"  Algorithm: ${summary.algorithm}")
    println(s"  Clusters: ${summary.effectiveK}")
    println(s"  Iterations: ${summary.iterations} (converged=${summary.converged})")
    println(s"  Final distortion: ${"%.6f".format(summary.finalDistortion)}")
    println(s"  Training time: ${summary.elapsedMillis}ms")
    println(s"  Divergence: ${summary.divergence}")

    // Spherical distance should be small (good cosine similarity within clusters)
    assert(summary.finalDistortion < 1.0, s"distortion should be low for well-separated data")
    assert(summary.divergence == "spherical", s"divergence should be 'spherical'")
  }

  // Demonstrate using "cosine" alias (same behavior)
  val cosineKMeans = new GeneralizedKMeans()
    .setK(2)
    .setDivergence("cosine")
    .setMaxIter(10)
    .setSeed(42)

  val cosineModel = cosineKMeans.fit(embeddings)
  assert(cosineModel.clusterCenters.length == 2, "cosine alias should work")
  println("\n'cosine' divergence alias also works correctly")

  println("\nexamples.SphericalKMeansExample OK")
  spark.stop()
}
