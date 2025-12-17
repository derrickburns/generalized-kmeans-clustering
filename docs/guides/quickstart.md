# Quick Start Guide

Get up and running with Generalized K-Means in under 5 minutes.

## Installation

Add to your `build.sbt`:

```scala
libraryDependencies += "com.massivedatascience" %% "massivedatascience-clusterer" % "0.7.0"
```

**Compatibility:**
- Spark 4.0.x (Scala 2.13 only)
- Spark 3.5.x / 3.4.x (Scala 2.13 or 2.12)
- Java 17

## Your First Clustering

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

// Create Spark session
val spark = SparkSession.builder()
  .appName("QuickStart")
  .master("local[*]")
  .getOrCreate()

import spark.implicits._

// Create sample data
val df = Seq(
  Tuple1(Vectors.dense(0.0, 0.0)),
  Tuple1(Vectors.dense(0.1, 0.1)),
  Tuple1(Vectors.dense(0.2, 0.0)),
  Tuple1(Vectors.dense(10.0, 10.0)),
  Tuple1(Vectors.dense(10.1, 10.1)),
  Tuple1(Vectors.dense(10.2, 10.0))
).toDF("features")

// Train model
val gkm = new GeneralizedKMeans()
  .setK(2)
  .setMaxIter(20)
  .setSeed(42)

val model = gkm.fit(df)

// Make predictions
val predictions = model.transform(df)
predictions.show()
```

Output:
```
+----------+----------+
|  features|prediction|
+----------+----------+
| [0.0,0.0]|         0|
| [0.1,0.1]|         0|
| [0.2,0.0]|         0|
|[10.0,10.0]|        1|
|[10.1,10.1]|        1|
|[10.2,10.0]|        1|
+----------+----------+
```

## Choosing a Divergence

Different data types benefit from different distance measures:

```scala
// For general numeric data (default)
new GeneralizedKMeans().setDivergence("squaredEuclidean")

// For probability distributions, topic models
new GeneralizedKMeans().setDivergence("kl").setSmoothing(1e-6)

// For text embeddings, document vectors
new GeneralizedKMeans().setDivergence("spherical")

// For outlier-robust clustering
new GeneralizedKMeans().setDivergence("l1")

// For spectral/audio data
new GeneralizedKMeans().setDivergence("itakuraSaito").setSmoothing(1e-6)
```

See the [Divergence Selection Guide](divergence-selection.md) for detailed recommendations.

## Common Variants

### Automatic K Selection (X-Means)

Don't know how many clusters? Let X-Means find the optimal k:

```scala
import com.massivedatascience.clusterer.ml.XMeans

val xmeans = new XMeans()
  .setMinK(2)
  .setMaxK(10)
  .setScoringMethod("bic")  // or "aic"

val model = xmeans.fit(df)
println(s"Optimal k: ${model.getK}")
```

### Soft/Fuzzy Clustering

Get probability distributions over clusters:

```scala
import com.massivedatascience.clusterer.ml.SoftKMeans

val soft = new SoftKMeans()
  .setK(3)
  .setBeta(2.0)  // Fuzziness parameter

val model = soft.fit(df)
val predictions = model.transform(df)
// Contains: prediction, probability, probabilities columns
```

### High-Dimensional Sparse Data

For TF-IDF vectors, embeddings, or other sparse data:

```scala
import com.massivedatascience.clusterer.ml.SparseKMeans

val sparse = new SparseKMeans()
  .setK(100)
  .setDivergence("kl")
  .setSparseMode("auto")  // Auto-detect sparsity

val model = sparse.fit(tfidfVectors)
```

### Multi-View Clustering

Cluster data with multiple feature representations:

```scala
import com.massivedatascience.clusterer.ml.{MultiViewKMeans, ViewSpec}

val views = Seq(
  ViewSpec("content_features", weight = 2.0, divergence = "kl"),
  ViewSpec("metadata_features", weight = 1.0, divergence = "squaredEuclidean")
)

val mvkm = new MultiViewKMeans()
  .setK(10)
  .setViews(views)

val model = mvkm.fit(multiViewData)
```

## Saving and Loading Models

```scala
// Save
model.write.overwrite().save("/path/to/model")

// Load
import com.massivedatascience.clusterer.ml.GeneralizedKMeansModel
val loaded = GeneralizedKMeansModel.load("/path/to/model")
```

## Training Summary

Access training metrics after fitting:

```scala
val model = gkm.fit(df)

if (model.hasSummary) {
  val summary = model.summary
  println(s"Iterations: ${summary.iterations}")
  println(s"Converged: ${summary.converged}")
  println(s"Final distortion: ${summary.finalDistortion}")
  println(s"Training time: ${summary.elapsedMillis}ms")
}
```

## Next Steps

- [Divergence Selection Guide](divergence-selection.md) - Choose the right distance measure
- [X-Means Auto-K Demo](xmeans-auto-k.md) - Automatic cluster count selection
- [Soft Clustering Guide](soft-clustering.md) - Interpret probabilistic memberships
- [API Reference](../api/) - Full API documentation
