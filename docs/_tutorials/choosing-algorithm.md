---
title: "Choosing the Right Algorithm"
---

# Choosing the Right Algorithm

**Time:** 15 minutes
**Goal:** Select the best algorithm for your use case

---

## Decision Flowchart

```
Start
  │
  ├─ Do you know how many clusters?
  │    ├─ NO → XMeans (automatic k selection)
  │    └─ YES ↓
  │
  ├─ Is your data probability distributions?
  │    └─ YES → GeneralizedKMeans with divergence="kl"
  │
  ├─ Is your data time series / sequences?
  │    └─ YES → TimeSeriesKMeans with DTW
  │
  ├─ Do you have outliers / noise?
  │    └─ YES → RobustKMeans or KMedoids
  │
  ├─ Do you need equal-sized clusters?
  │    └─ YES → BalancedKMeans
  │
  ├─ Do you have must-link/cannot-link constraints?
  │    └─ YES → ConstrainedKMeans
  │
  ├─ Do you need soft/probabilistic assignments?
  │    └─ YES → SoftKMeans
  │
  ├─ Is data arriving in streams?
  │    └─ YES → StreamingKMeans
  │
  ├─ Do you have very high dimensions + sparse data?
  │    └─ YES → SparseKMeans
  │
  ├─ Do you have non-convex cluster shapes?
  │    └─ YES → SpectralClustering
  │
  └─ DEFAULT → GeneralizedKMeans with squaredEuclidean
```

---

## Algorithm Quick Reference

### Core Algorithms

| Algorithm | When to Use | Key Parameters |
|-----------|-------------|----------------|
| **GeneralizedKMeans** | General-purpose clustering | `k`, `divergence`, `maxIter` |
| **XMeans** | Unknown number of clusters | `minK`, `maxK`, `criterion` |
| **SoftKMeans** | Overlapping/fuzzy clusters | `k`, `beta` |
| **BisectingKMeans** | Hierarchical structure needed | `k`, `minDivisibleClusterSize` |

### Specialized Algorithms

| Algorithm | When to Use | Key Parameters |
|-----------|-------------|----------------|
| **StreamingKMeans** | Real-time data streams | `k`, `decayFactor`, `halfLife` |
| **KMedoids** | Outlier-resistant, interpretable centers | `k`, `distanceFunction` |
| **BalancedKMeans** | Equal-sized clusters required | `k`, `balanceMode`, `maxClusterSize` |
| **ConstrainedKMeans** | Semi-supervised with constraints | `k`, `mustLinkCol`, `cannotLinkCol` |
| **RobustKMeans** | Noisy data with outliers | `k`, `robustMode`, `trimFraction` |

### Advanced Algorithms

| Algorithm | When to Use | Key Parameters |
|-----------|-------------|----------------|
| **SparseKMeans** | High-dimensional sparse data | `k`, `sparseMode`, `sparseThreshold` |
| **MultiViewKMeans** | Multiple feature representations | `k`, `viewSpecs` |
| **TimeSeriesKMeans** | Sequence/time-series data | `k`, `distanceType` |
| **SpectralClustering** | Non-convex shapes, graph data | `k`, `affinityType`, `laplacianType` |
| **InformationBottleneck** | Information compression | `k`, `beta`, `relevanceCol` |
| **MiniBatchKMeans** | Very large datasets | `k`, `batchSize` |

---

## Detailed Algorithm Guide

### GeneralizedKMeans

**Best for:** Most clustering tasks with well-defined feature vectors.

```scala
val kmeans = new GeneralizedKMeans()
  .setK(5)
  .setDivergence("squaredEuclidean")  // Standard k-means
  .setMaxIter(100)
  .setTol(1e-4)
```

**Divergence options:**
- `squaredEuclidean` — Standard k-means (default)
- `kl` — Probability distributions (topic modeling)
- `itakuraSaito` — Spectral/audio data
- `l1` — Manhattan distance (robust to outliers)
- `generalizedI` — Count data
- `logistic` — Binary probabilities
- `spherical` / `cosine` — Text/document vectors

---

### XMeans

**Best for:** When you don't know how many clusters to use.

```scala
val xmeans = new XMeans()
  .setMinK(2)
  .setMaxK(20)
  .setCriterion("bic")  // or "aic"
```

**How it works:** Starts with minK, splits clusters, evaluates with BIC/AIC, stops when splitting doesn't improve.

---

### SoftKMeans

**Best for:** Points that belong to multiple clusters with different degrees.

```scala
val soft = new SoftKMeans()
  .setK(3)
  .setBeta(2.0)  // Higher = more deterministic
```

**Output:** Adds `probabilities` column with membership weights.

---

### KMedoids

**Best for:** Interpretable centers (actual data points) and outlier resistance.

```scala
val kmedoids = new KMedoids()
  .setK(5)
  .setDistanceFunction("euclidean")  // or "manhattan", "cosine"
```

**Key difference:** Centers are actual data points, not computed means.

---

### RobustKMeans

**Best for:** Datasets with noise, outliers, or contamination.

```scala
val robust = new RobustKMeans()
  .setK(5)
  .setRobustMode("trim")          // or "noise_cluster", "m_estimator"
  .setTrimFraction(0.1)           // Ignore worst 10%
  .setOutlierScoreCol("outlier")  // Output outlier scores
```

**Modes:**
- `trim` — Ignore points far from centers
- `noise_cluster` — Assign outliers to special cluster
- `m_estimator` — Down-weight outliers during updates

---

### TimeSeriesKMeans

**Best for:** Sequences that may be misaligned in time.

```scala
val ts = new TimeSeriesKMeans()
  .setK(3)
  .setDistanceType("dtw")  // or "softdtw", "gak", "derivative"
  .setBandWidth(0.1)       // Sakoe-Chiba band
```

**Distance types:**
- `dtw` — Dynamic Time Warping
- `softdtw` — Differentiable DTW
- `gak` — Global Alignment Kernel
- `derivative` — Shape-based (offset/amplitude invariant)

---

### SpectralClustering

**Best for:** Non-convex cluster shapes, graph/network data.

```scala
val spectral = new SpectralClustering()
  .setK(3)
  .setAffinityType("rbf")           // or "knn", "epsilon"
  .setLaplacianType("normalized")   // or "unnormalized", "randomWalk"
  .setSigma(1.0)                    // RBF kernel width
```

**When to use:** Clusters that are non-spherical or connected by paths.

---

## Performance Comparison

| Algorithm | Complexity | Memory | Best Scale |
|-----------|------------|--------|------------|
| GeneralizedKMeans | O(n·k·d·iter) | O(k·d) | Billions |
| XMeans | O(n·k·d·iter·splits) | O(k·d) | Millions |
| SoftKMeans | O(n·k·d·iter) | O(n·k) | Millions |
| KMedoids | O(n²·k·iter) | O(n²) | Thousands |
| StreamingKMeans | O(batch·k·d) | O(k·d) | Unlimited |
| SpectralClustering | O(n²) or O(n·m²) | O(n²) | Thousands |

---

## Examples by Domain

### Text/Documents
```scala
// Use cosine similarity for TF-IDF vectors
new GeneralizedKMeans().setDivergence("cosine")
```

### Topic Modeling
```scala
// KL divergence for word distributions
new GeneralizedKMeans().setDivergence("kl")
```

### Audio/Spectral
```scala
// Itakura-Saito for power spectra
new GeneralizedKMeans().setDivergence("itakuraSaito")
```

### Customer Segmentation
```scala
// Balance clusters for equal marketing spend
new BalancedKMeans().setBalanceMode("hard")
```

### Anomaly Detection
```scala
// Robust clustering with outlier scores
new RobustKMeans()
  .setRobustMode("noise_cluster")
  .setOutlierScoreCol("anomaly_score")
```

---

## Next Steps

- [Your First Clustering](first-clustering.html) — Hands-on tutorial
- [Parameter Reference](../reference/parameters.html) — All parameters
- [Performance Tuning](../explanation/performance.html) — Scaling tips

---

[Back to Tutorials](index.html) | [Home](../)
