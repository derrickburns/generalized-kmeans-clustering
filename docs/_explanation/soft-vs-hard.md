---
title: "Soft vs Hard Clustering"
---

# Soft vs Hard Clustering

When points can belong to multiple clusters.

---

## Hard Clustering (Standard)

Each point belongs to exactly one cluster:

```
Point → Cluster 2
```

Output: Single integer prediction per point.

```scala
val model = new GeneralizedKMeans().setK(3).fit(data)
model.transform(data).select("prediction").show()
// +----------+
// |prediction|
// +----------+
// |         0|
// |         2|
// |         1|
// +----------+
```

---

## Soft Clustering (Fuzzy)

Each point has a probability of belonging to each cluster:

```
Point → [0.7, 0.2, 0.1]  (70% cluster 0, 20% cluster 1, 10% cluster 2)
```

Output: Probability vector per point.

```scala
val model = new SoftKMeans().setK(3).setBeta(2.0).fit(data)
model.transform(data).select("prediction", "probabilities").show()
// +----------+--------------------+
// |prediction|       probabilities|
// +----------+--------------------+
// |         0|[0.85, 0.10, 0.05]|
// |         2|[0.05, 0.15, 0.80]|
// |         1|[0.10, 0.75, 0.15]|
// +----------+--------------------+
```

---

## When to Use Soft Clustering

### 1. Overlapping Groups

Customer segments that aren't mutually exclusive:
- "Budget-conscious" AND "Tech enthusiast"

### 2. Uncertainty Quantification

Know confidence in cluster assignment:
- "90% likely cluster A, 10% likely cluster B"

### 3. Downstream Probabilistic Models

Feed membership probabilities into other models:
```scala
// Use soft memberships as features
val withProbs = softModel.transform(data)
val downstream = logisticRegression.fit(withProbs)
```

### 4. Smooth Boundaries

When cluster boundaries are fuzzy, not sharp.

---

## The Beta Parameter

Controls "softness" of assignments:

| Beta | Behavior |
|------|----------|
| 1.0 | Very soft (nearly uniform) |
| 2.0 | Default (balanced) |
| 5.0 | Harder (more peaked) |
| ∞ | Hard clustering (one-hot) |

```scala
// Softer assignments
new SoftKMeans().setBeta(1.5)

// Harder assignments
new SoftKMeans().setBeta(5.0)
```

### Visual Example

Point at distance 1 from center A, distance 2 from center B:

| Beta | P(A) | P(B) |
|------|------|------|
| 1.5 | 0.59 | 0.41 |
| 2.0 | 0.67 | 0.33 |
| 3.0 | 0.80 | 0.20 |
| 5.0 | 0.94 | 0.06 |

---

## Mathematical Formulation

### Hard Assignment

```
assign(x) = argmin_j d(x, c_j)
```

### Soft Assignment

```
P(cluster j | x) = exp(-d(x, c_j) / β) / Σ_k exp(-d(x, c_k) / β)
```

This is a softmax over negative distances, with temperature β.

---

## Center Updates

### Hard K-Means

```
c_j = (1/n_j) Σ_{x ∈ cluster j} x
```

### Soft K-Means

```
c_j = Σ_i P(j|x_i)^m × x_i / Σ_i P(j|x_i)^m
```

Where m is the fuzziness exponent (typically 2).

---

## Code Example

```scala
import com.massivedatascience.clusterer.ml.SoftKMeans

// Train soft clustering model
val soft = new SoftKMeans()
  .setK(5)
  .setBeta(2.0)
  .setMaxIter(50)

val model = soft.fit(data)

// Get predictions with probabilities
val results = model.transform(data)

// Find points with high uncertainty (no dominant cluster)
val uncertain = results.filter(
  array_max(col("probabilities")) < 0.5
)

// Find points strongly in one cluster
val confident = results.filter(
  array_max(col("probabilities")) > 0.9
)
```

---

## Use Cases

| Scenario | Hard or Soft? |
|----------|---------------|
| Customer segmentation for targeted marketing | Hard |
| Topic modeling with mixed documents | Soft |
| Image segmentation with clear boundaries | Hard |
| Anomaly detection (low max probability) | Soft |
| Recommendation systems | Soft |
| Data compression | Hard |

---

[Back to Explanation](index.html) | [Home](../)
