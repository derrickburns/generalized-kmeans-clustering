# X-Means: Automatic Cluster Count Selection

X-Means automatically determines the optimal number of clusters using information-theoretic criteria (BIC or AIC), eliminating the need to manually specify k.

## The K Selection Problem

Traditional K-Means requires you to specify the number of clusters beforehand. Choosing k is often:
- **Difficult**: No ground truth for unlabeled data
- **Critical**: Wrong k leads to poor clustering quality
- **Time-consuming**: Manual trial-and-error with elbow plots

X-Means solves this by automatically searching for the optimal k within a specified range.

## How X-Means Works

1. **Start** with minimum k clusters
2. **Fit** standard K-Means
3. **For each cluster**, try splitting it into two
4. **Accept split** if it improves the scoring criterion (BIC/AIC)
5. **Repeat** until maximum k is reached or no splits improve the score
6. **Return** the model with the best score

## Basic Usage

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.XMeans

val spark = SparkSession.builder()
  .appName("XMeansDemo")
  .master("local[*]")
  .getOrCreate()

import spark.implicits._

// Create data with 3 natural clusters
val df = Seq(
  // Cluster 1
  Tuple1(Vectors.dense(0.0, 0.0)),
  Tuple1(Vectors.dense(0.5, 0.5)),
  Tuple1(Vectors.dense(1.0, 0.0)),
  // Cluster 2
  Tuple1(Vectors.dense(10.0, 10.0)),
  Tuple1(Vectors.dense(10.5, 10.5)),
  Tuple1(Vectors.dense(11.0, 10.0)),
  // Cluster 3
  Tuple1(Vectors.dense(0.0, 20.0)),
  Tuple1(Vectors.dense(0.5, 20.5)),
  Tuple1(Vectors.dense(1.0, 20.0))
).toDF("features")

// X-Means will find k=3
val xmeans = new XMeans()
  .setMinK(2)
  .setMaxK(10)
  .setScoringMethod("bic")
  .setSeed(42)

val model = xmeans.fit(df)

println(s"Optimal k found: ${model.getK}")
// Output: Optimal k found: 3
```

## Scoring Methods

### BIC (Bayesian Information Criterion)

```scala
new XMeans().setScoringMethod("bic")
```

**BIC = -2 * log(L) + k * log(n)**

- Penalizes model complexity more strongly
- Tends to select **fewer clusters**
- Recommended for: larger datasets, when parsimony is important

### AIC (Akaike Information Criterion)

```scala
new XMeans().setScoringMethod("aic")
```

**AIC = -2 * log(L) + 2k**

- Lighter penalty on complexity
- May select **more clusters**
- Recommended for: smaller datasets, when capturing fine structure matters

### Comparison

| Criterion | Penalty | Tends to select | Best for |
|-----------|---------|-----------------|----------|
| BIC | log(n) | Fewer clusters | Large datasets, parsimony |
| AIC | 2 | More clusters | Small datasets, fine detail |

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `minK` | Minimum number of clusters to consider | 2 |
| `maxK` | Maximum number of clusters to consider | 10 |
| `scoringMethod` | "bic" or "aic" | "bic" |
| `maxIter` | Max iterations per K-Means fit | 20 |
| `tol` | Convergence tolerance | 1e-4 |
| `seed` | Random seed for reproducibility | - |
| `divergence` | Bregman divergence to use | "squaredEuclidean" |

## Complete Example with Analysis

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.XMeans

val spark = SparkSession.builder()
  .appName("XMeansComplete")
  .master("local[*]")
  .getOrCreate()

import spark.implicits._

// Generate synthetic data with 4 clusters
val random = new scala.util.Random(42)
def cluster(cx: Double, cy: Double, n: Int) =
  (1 to n).map(_ => Vectors.dense(cx + random.nextGaussian() * 0.5,
                                   cy + random.nextGaussian() * 0.5))

val data = (
  cluster(0, 0, 50) ++      // Cluster 1
  cluster(5, 0, 50) ++      // Cluster 2
  cluster(0, 5, 50) ++      // Cluster 3
  cluster(5, 5, 50)         // Cluster 4
).map(v => Tuple1(v)).toDF("features")

// Run X-Means with BIC
val xmeansBIC = new XMeans()
  .setMinK(2)
  .setMaxK(8)
  .setScoringMethod("bic")
  .setSeed(42)

val modelBIC = xmeansBIC.fit(data)
println(s"BIC selected k = ${modelBIC.getK}")

// Run X-Means with AIC
val xmeansAIC = new XMeans()
  .setMinK(2)
  .setMaxK(8)
  .setScoringMethod("aic")
  .setSeed(42)

val modelAIC = xmeansAIC.fit(data)
println(s"AIC selected k = ${modelAIC.getK}")

// Analyze the BIC model
if (modelBIC.hasSummary) {
  val summary = modelBIC.summary
  println(s"\n=== Training Summary (k=${modelBIC.getK}) ===")
  println(s"Iterations: ${summary.iterations}")
  println(s"Converged: ${summary.converged}")
  println(s"Final distortion: ${summary.finalDistortion}")
  println(s"Training time: ${summary.elapsedMillis}ms")
}

// Show cluster sizes
val predictions = modelBIC.transform(data)
println("\n=== Cluster Sizes ===")
predictions.groupBy("prediction").count().orderBy("prediction").show()

// Show cluster centers
println("\n=== Cluster Centers ===")
modelBIC.clusterCenters.zipWithIndex.foreach { case (center, i) =>
  println(s"Cluster $i: (${center(0).formatted("%.2f")}, ${center(1).formatted("%.2f")})")
}
```

## Use Cases

### 1. Customer Segmentation

When you don't know how many customer segments exist:

```scala
val xmeans = new XMeans()
  .setMinK(3)
  .setMaxK(15)
  .setScoringMethod("bic")

val model = xmeans.fit(customerFeatures)
println(s"Found ${model.getK} customer segments")
```

### 2. Document Topic Discovery

Find the natural number of topics in a document corpus:

```scala
val xmeans = new XMeans()
  .setMinK(5)
  .setMaxK(50)
  .setDivergence("kl")
  .setSmoothing(1e-6)
  .setScoringMethod("aic")  // AIC for finer topics

val model = xmeans.fit(tfidfVectors)
println(s"Found ${model.getK} topics")
```

### 3. Image Color Quantization

Automatically determine optimal palette size:

```scala
val xmeans = new XMeans()
  .setMinK(8)
  .setMaxK(256)
  .setScoringMethod("bic")

val model = xmeans.fit(pixelColors)
println(s"Optimal palette: ${model.getK} colors")
```

## Comparison with Manual K Selection

### Traditional Elbow Method (Manual)

```scala
// Manual approach - run multiple times
val results = (2 to 10).map { k =>
  val kmeans = new GeneralizedKMeans().setK(k)
  val model = kmeans.fit(data)
  (k, model.computeCost(data))
}
// Then manually inspect elbow plot...
```

### X-Means (Automatic)

```scala
// Automatic - single call
val model = new XMeans()
  .setMinK(2)
  .setMaxK(10)
  .fit(data)

println(s"Optimal k: ${model.getK}")  // Done!
```

## Tips and Best Practices

### 1. Set Reasonable Bounds

```scala
// Don't set maxK too high - increases computation
new XMeans()
  .setMinK(2)      // At least 2 clusters
  .setMaxK(20)     // Reasonable upper bound
```

### 2. Use BIC for Large Datasets

```scala
// BIC's stronger penalty prevents over-clustering
new XMeans()
  .setScoringMethod("bic")  // Better for n > 1000
```

### 3. Use AIC for Small Datasets or Fine Structure

```scala
// AIC may find more nuanced structure
new XMeans()
  .setScoringMethod("aic")  // Better for n < 500
```

### 4. Combine with Domain Knowledge

```scala
// If you know there are at least 5 categories
new XMeans()
  .setMinK(5)
  .setMaxK(15)
```

### 5. Set Seed for Reproducibility

```scala
// Always set seed for consistent results
new XMeans()
  .setSeed(42)
```

## Limitations

1. **Computational cost**: Tests multiple k values, slower than single K-Means
2. **Greedy splits**: May not find global optimum
3. **Gaussian assumption**: BIC/AIC assume Gaussian clusters
4. **No guarantee**: The "optimal" k is model-based, not ground truth

## Model Persistence

```scala
// Save
model.write.overwrite().save("/path/to/xmeans-model")

// Load
import com.massivedatascience.clusterer.ml.XMeansModel
val loaded = XMeansModel.load("/path/to/xmeans-model")
println(s"Loaded model with k=${loaded.getK}")
```

## Summary

- X-Means automatically selects k using BIC or AIC
- Use **BIC** for parsimony (fewer, larger clusters)
- Use **AIC** for detail (more, smaller clusters)
- Set reasonable `minK` and `maxK` bounds
- Always set `seed` for reproducibility
