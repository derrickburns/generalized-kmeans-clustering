---
title: "Comparison with Spark MLlib"
---

# Comparison with Spark MLlib

When should you use this library vs. built-in Spark MLlib KMeans?

---

## Feature Comparison

| Feature | Spark MLlib KMeans | This Library |
|---------|-------------------|--------------|
| **Basic K-Means** | ✓ | ✓ |
| **Divergences** | Squared Euclidean only | 8 divergences |
| **KL Divergence** | — | ✓ |
| **Cosine Distance** | — | ✓ |
| **Itakura-Saito** | — | ✓ |
| **Automatic K (X-Means)** | — | ✓ |
| **Soft/Fuzzy Clustering** | — | ✓ |
| **Streaming Updates** | ✓ (deprecated) | ✓ |
| **Bisecting K-Means** | ✓ | ✓ |
| **K-Medoids** | — | ✓ |
| **Balanced Clusters** | — | ✓ |
| **Constrained Clustering** | — | ✓ |
| **Outlier Detection** | — | ✓ |
| **Mini-Batch** | — | ✓ |
| **Time Series (DTW)** | — | ✓ |
| **Spectral Clustering** | — | ✓ |

---

## When to Use Spark MLlib

Use built-in `org.apache.spark.ml.clustering.KMeans` when:

1. **Simple Euclidean clustering** — Standard k-means on numeric data
2. **Minimal dependencies** — You want zero external JARs
3. **Proven stability** — You need the most battle-tested option
4. **Basic use case** — You know k, have clean data, just need clusters

```scala
// Spark MLlib - simple and built-in
import org.apache.spark.ml.clustering.KMeans

val kmeans = new KMeans()
  .setK(5)
  .setMaxIter(20)
  .setSeed(42)

val model = kmeans.fit(data)
```

---

## When to Use This Library

Use `GeneralizedKMeans` when:

### 1. You Need a Different Distance Measure

```scala
// Clustering probability distributions
val kmeans = new GeneralizedKMeans()
  .setK(10)
  .setDivergence("kl")  // Not possible in MLlib
```

### 2. You Don't Know How Many Clusters

```scala
// Automatic k selection
val xmeans = new XMeans()
  .setMinK(2)
  .setMaxK(20)
  .setCriterion("bic")

val model = xmeans.fit(data)
println(s"Optimal k: ${model.k}")  // Discovered automatically
```

### 3. Points Can Belong to Multiple Clusters

```scala
// Soft/fuzzy memberships
val soft = new SoftKMeans()
  .setK(5)
  .setBeta(2.0)

val model = soft.fit(data)
// Output includes probability of belonging to each cluster
```

### 4. You Have Outliers

```scala
// Robust clustering with outlier detection
val robust = new RobustKMeans()
  .setK(5)
  .setRobustMode("noise_cluster")
  .setTrimFraction(0.05)

val model = robust.fit(noisyData)
// Outliers assigned to cluster -1
```

### 5. You Need Equal-Sized Clusters

```scala
// Balanced cluster sizes
val balanced = new BalancedKMeans()
  .setK(5)
  .setBalanceMode("hard")

val model = balanced.fit(data)
// All clusters have approximately equal size
```

### 6. You're Clustering Text Documents

```scala
// Cosine similarity for TF-IDF vectors
val kmeans = new GeneralizedKMeans()
  .setK(20)
  .setDivergence("cosine")

val model = kmeans.fit(tfidfVectors)
```

---

## Performance Comparison

Both libraries have similar performance for basic squared Euclidean k-means:

| Dataset Size | MLlib | This Library | Notes |
|-------------|-------|--------------|-------|
| 100K × 100 | ~30s | ~30s | Equivalent |
| 1M × 100 | ~2min | ~2min | Equivalent |
| 10M × 100 | ~15min | ~15min | Equivalent |

**Performance is equivalent** because both use the same underlying Spark DataFrame operations.

This library adds:
- **Elkan acceleration** for squared Euclidean (can be 2-5x faster)
- **CrossJoin strategy** for large k (faster than broadcast for k > 1000)

---

## API Compatibility

This library follows the same Estimator/Model pattern as MLlib:

```scala
// MLlib pattern
val mllibModel = new org.apache.spark.ml.clustering.KMeans()
  .setK(5)
  .fit(data)
val mllibPredictions = mllibModel.transform(data)

// This library - identical pattern
val gkmModel = new GeneralizedKMeans()
  .setK(5)
  .fit(data)
val gkmPredictions = gkmModel.transform(data)
```

Both produce the same output schema: `prediction` column with cluster IDs.

---

## Migration from MLlib

Switching from MLlib is straightforward:

```scala
// Before (MLlib)
import org.apache.spark.ml.clustering.KMeans
val model = new KMeans().setK(5).fit(data)

// After (this library) - just change import
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
val model = new GeneralizedKMeans().setK(5).fit(data)
```

The default divergence is `squaredEuclidean`, so behavior is identical.

---

## Summary

| Use Case | Recommendation |
|----------|---------------|
| Basic clustering, no dependencies | Spark MLlib |
| Need KL/cosine/other divergence | This library |
| Don't know optimal k | This library (X-Means) |
| Soft cluster memberships | This library (SoftKMeans) |
| Outlier handling | This library (RobustKMeans) |
| Equal cluster sizes | This library (BalancedKMeans) |
| Text/document clustering | This library (cosine) |
| Probability distributions | This library (kl) |

---

[Back to Explanation](index.html) | [Home](../)
