---
title: "Cluster Validity"
---

# Cluster Validity

How to evaluate clustering quality.

---

## The Challenge

Unlike supervised learning, there's no "ground truth" to compare against. We need intrinsic measures of cluster quality.

---

## Key Metrics

### Within-Cluster Sum of Squares (WCSS)

**What it measures:** Compactness — how tight are the clusters?

```
WCSS = Σ_j Σ_{x ∈ cluster j} ||x - c_j||²
```

**Interpretation:**
- Lower is better
- Always decreases as k increases
- Used in elbow method

```scala
val model = kmeans.fit(data)
println(s"WCSS: ${model.summary.trainingCost}")
```

---

### Silhouette Score

**What it measures:** Separation vs cohesion — are clusters well-separated?

For each point:
```
a(x) = average distance to points in same cluster
b(x) = average distance to points in nearest other cluster
s(x) = (b(x) - a(x)) / max(a(x), b(x))
```

**Interpretation:**
- Range: [-1, 1]
- +1 = perfect (dense, well-separated)
- 0 = overlapping clusters
- -1 = misassigned points

```scala
val silhouette = model.summary.silhouette
println(s"Silhouette: $silhouette")
```

---

### Calinski-Harabasz Index

**What it measures:** Ratio of between-cluster to within-cluster variance.

```
CH = [BCSS / (k-1)] / [WCSS / (n-k)]
```

Where BCSS = between-cluster sum of squares.

**Interpretation:**
- Higher is better
- Favors convex, dense clusters
- Can compare different k values

```scala
val ch = model.summary.calinskiHarabasz
println(s"Calinski-Harabasz: $ch")
```

---

### Davies-Bouldin Index

**What it measures:** Average similarity between clusters.

```
DB = (1/k) Σ_i max_{j≠i} (σ_i + σ_j) / d(c_i, c_j)
```

Where σ_i = average distance of points in cluster i to center.

**Interpretation:**
- Lower is better
- 0 = perfect separation
- Penalizes clusters that are close together

```scala
val db = model.summary.daviesBouldin
println(s"Davies-Bouldin: $db")
```

---

## Choosing k: Elbow Method

Plot WCSS vs k, look for "elbow":

```scala
val metrics = (2 to 15).map { k =>
  val model = new GeneralizedKMeans().setK(k).fit(data)
  (k, model.summary.trainingCost)
}

// Plot and find elbow
metrics.foreach { case (k, wcss) =>
  println(f"k=$k%2d  WCSS=$wcss%.2f  ${"█" * (wcss/1000).toInt}")
}
```

Output:
```
k= 2  WCSS=15234.00  ███████████████
k= 3  WCSS=8456.00   ████████
k= 4  WCSS=5123.00   █████        ← Elbow here
k= 5  WCSS=4567.00   ████
k= 6  WCSS=4234.00   ████
```

---

## Choosing k: Silhouette Method

Pick k with highest silhouette:

```scala
val metrics = (2 to 10).map { k =>
  val model = new GeneralizedKMeans().setK(k).fit(data)
  (k, model.summary.silhouette)
}

val bestK = metrics.maxBy(_._2)._1
println(s"Best k by silhouette: $bestK")
```

---

## Choosing k: X-Means (Automatic)

Let the algorithm decide using BIC/AIC:

```scala
val xmeans = new XMeans()
  .setMinK(2)
  .setMaxK(20)
  .setCriterion("bic")

val model = xmeans.fit(data)
println(s"Optimal k: ${model.k}")
```

---

## Metric Comparison

| Metric | Range | Best Value | Pros | Cons |
|--------|-------|------------|------|------|
| WCSS | [0, ∞) | Low | Simple, fast | Always improves with k |
| Silhouette | [-1, 1] | High (+1) | Interpretable | O(n²) naive |
| Calinski-Harabasz | [0, ∞) | High | Fast | Favors convex clusters |
| Davies-Bouldin | [0, ∞) | Low (0) | Intuitive | Sensitive to outliers |

---

## Complete Evaluation Example

```scala
import com.massivedatascience.clusterer.ml.{GeneralizedKMeans, ClusteringMetrics}

// Train model
val model = new GeneralizedKMeans()
  .setK(5)
  .fit(data)

// Get all metrics
val summary = model.summary
println(s"""
  |Clustering Evaluation:
  |  WCSS:              ${summary.trainingCost}
  |  Silhouette:        ${summary.silhouette}
  |  Calinski-Harabasz: ${summary.calinskiHarabasz}
  |  Davies-Bouldin:    ${summary.daviesBouldin}
  |  Cluster sizes:     ${summary.clusterSizes.mkString(", ")}
""".stripMargin)
```

---

## External Validation

If you have ground truth labels:

```scala
// Adjusted Rand Index
val ari = ClusteringMetrics.adjustedRandIndex(predictions, labels)

// Normalized Mutual Information
val nmi = ClusteringMetrics.normalizedMutualInfo(predictions, labels)
```

| Metric | Range | Perfect |
|--------|-------|---------|
| Adjusted Rand Index | [-1, 1] | 1 |
| Normalized Mutual Information | [0, 1] | 1 |

---

[Back to Explanation](index.html) | [Home](../)
