---
title: Algorithm Reference
---

# Algorithm Reference

Complete list of all 15 clustering algorithms.

---

## Core Algorithms

### GeneralizedKMeans

Standard k-means with pluggable Bregman divergences.

**Class:** `com.massivedatascience.clusterer.ml.GeneralizedKMeans`

**Use when:** General-purpose clustering with known k

**Complexity:** O(n × k × d × iterations)

```scala
val kmeans = new GeneralizedKMeans()
  .setK(5)
  .setDivergence("squaredEuclidean")
  .setMaxIter(100)
```

---

### XMeans

Automatic k selection using BIC/AIC.

**Class:** `com.massivedatascience.clusterer.ml.XMeans`

**Use when:** Unknown number of clusters

**Algorithm:** Iteratively splits clusters, evaluates with information criterion

```scala
val xmeans = new XMeans()
  .setMinK(2)
  .setMaxK(20)
  .setCriterion("bic")
```

---

### SoftKMeans

Probabilistic/fuzzy cluster assignments.

**Class:** `com.massivedatascience.clusterer.ml.SoftKMeans`

**Use when:** Points may belong to multiple clusters

**Output:** Adds `probabilities` column with membership weights

```scala
val soft = new SoftKMeans()
  .setK(3)
  .setBeta(2.0)
```

---

### BisectingKMeans

Hierarchical divisive clustering.

**Class:** `com.massivedatascience.clusterer.ml.BisectingKMeans`

**Use when:** Need hierarchical structure, more deterministic than random init

**Algorithm:** Recursively bisects largest cluster

```scala
val bisecting = new BisectingKMeans()
  .setK(10)
  .setMinDivisibleClusterSize(5)
```

---

## Online/Streaming

### StreamingKMeans

Online clustering with exponential decay.

**Class:** `com.massivedatascience.clusterer.ml.StreamingKMeans`

**Use when:** Data arrives in streams, concept drift

```scala
val streaming = new StreamingKMeans()
  .setK(5)
  .setDecayFactor(0.9)
```

---

### MiniBatchKMeans

Stochastic updates on mini-batches.

**Class:** `com.massivedatascience.clusterer.ml.MiniBatchKMeans`

**Use when:** Very large datasets, faster convergence

```scala
val minibatch = new MiniBatchKMeans()
  .setK(10)
  .setBatchSize(1000)
```

---

## Robust Algorithms

### KMedoids

Uses actual data points as centers (PAM algorithm).

**Class:** `com.massivedatascience.clusterer.ml.KMedoids`

**Use when:** Need interpretable centers, outlier resistance

**Complexity:** O(n² × k × iterations)

```scala
val kmedoids = new KMedoids()
  .setK(5)
  .setDistanceFunction("manhattan")
```

---

### RobustKMeans

Explicit outlier handling.

**Class:** `com.massivedatascience.clusterer.ml.RobustKMeans`

**Use when:** Noisy data, need outlier detection

**Modes:** trim, noise_cluster, m_estimator

```scala
val robust = new RobustKMeans()
  .setK(5)
  .setRobustMode("trim")
  .setTrimFraction(0.1)
```

---

## Constrained Algorithms

### BalancedKMeans

Equal-sized cluster constraints.

**Class:** `com.massivedatascience.clusterer.ml.BalancedKMeans`

**Use when:** Need approximately equal cluster sizes

```scala
val balanced = new BalancedKMeans()
  .setK(5)
  .setBalanceMode("hard")
```

---

### ConstrainedKMeans

Semi-supervised with must-link/cannot-link.

**Class:** `com.massivedatascience.clusterer.ml.ConstrainedKMeans`

**Use when:** Have prior knowledge about relationships

```scala
val constrained = new ConstrainedKMeans()
  .setK(3)
  .setMustLinkCol("must_link")
  .setCannotLinkCol("cannot_link")
```

---

## Specialized Algorithms

### SparseKMeans

Optimized for high-dimensional sparse data.

**Class:** `com.massivedatascience.clusterer.ml.SparseKMeans`

**Use when:** High sparsity ratio (> 50%)

```scala
val sparse = new SparseKMeans()
  .setK(10)
  .setSparseMode("auto")
```

---

### MultiViewKMeans

Multiple feature representations.

**Class:** `com.massivedatascience.clusterer.ml.MultiViewKMeans`

**Use when:** Data has multiple modalities/views

```scala
val multiview = new MultiViewKMeans()
  .setK(5)
  .setViewSpecs(Seq(
    ViewSpec("view1", "squaredEuclidean", 1.0),
    ViewSpec("view2", "cosine", 0.5)
  ))
```

---

### TimeSeriesKMeans

Sequence clustering with DTW.

**Class:** `com.massivedatascience.clusterer.ml.TimeSeriesKMeans`

**Use when:** Time series, sequences of varying length

```scala
val timeseries = new TimeSeriesKMeans()
  .setK(5)
  .setDistanceType("dtw")
```

---

### SpectralClustering

Graph-based via Laplacian eigenvectors.

**Class:** `com.massivedatascience.clusterer.ml.SpectralClustering`

**Use when:** Non-convex clusters, graph/network data

**Complexity:** O(n²) or O(n×m²) with Nyström

```scala
val spectral = new SpectralClustering()
  .setK(5)
  .setAffinityType("rbf")
  .setUseNystrom(true)
```

---

### InformationBottleneck

Information-theoretic clustering.

**Class:** `com.massivedatascience.clusterer.ml.InformationBottleneck`

**Use when:** Compression with relevance preservation

```scala
val ib = new InformationBottleneck()
  .setK(5)
  .setBeta(1.0)
  .setRelevanceCol("labels")
```

---

## Algorithm Comparison

| Algorithm | Complexity | Memory | Outliers | Constraints | Streaming |
|-----------|------------|--------|----------|-------------|-----------|
| GeneralizedKMeans | O(nkd) | O(kd) | — | — | — |
| XMeans | O(nkd×splits) | O(kd) | — | — | — |
| SoftKMeans | O(nkd) | O(nk) | — | — | — |
| BisectingKMeans | O(nkd) | O(kd) | — | — | — |
| StreamingKMeans | O(batchkd) | O(kd) | — | — | ✓ |
| MiniBatchKMeans | O(batchkd) | O(kd) | — | — | — |
| KMedoids | O(n²k) | O(n²) | ✓ | — | — |
| RobustKMeans | O(nkd) | O(kd) | ✓ | — | — |
| BalancedKMeans | O(nkd) | O(kd) | — | ✓ | — |
| ConstrainedKMeans | O(nkd) | O(kd) | — | ✓ | — |
| SparseKMeans | O(nnzkd) | O(kd) | — | — | — |
| TimeSeriesKMeans | O(nkL²) | O(kL) | — | — | — |
| SpectralClustering | O(n²) | O(n²) | — | — | — |
| InformationBottleneck | O(nkd) | O(kd) | — | — | — |

---

[Back to Reference](index.html) | [Home](../)
