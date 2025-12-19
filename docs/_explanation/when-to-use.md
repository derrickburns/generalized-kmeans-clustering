---
title: "When to Use What"
---

# When to Use What

A decision framework for choosing the right divergence and algorithm.

---

## Divergence Selection

| Your Data Type | Use This Divergence |
|----------------|---------------------|
| General numeric (measurements, coordinates) | `squaredEuclidean` |
| Probability distributions (sum to 1) | `kl` |
| Text vectors (TF-IDF, embeddings) | `cosine` |
| Power spectra, audio features | `itakuraSaito` |
| Raw counts (not normalized) | `generalizedI` |
| Binary probabilities (0-1) | `logistic` |
| Data with outliers | `l1` |

---

## Algorithm Selection

| Your Situation | Use This Algorithm |
|----------------|-------------------|
| Standard clustering, known k | `GeneralizedKMeans` |
| Don't know optimal k | `XMeans` |
| Points belong to multiple clusters | `SoftKMeans` |
| Need hierarchical structure | `BisectingKMeans` |
| Data arrives in streams | `StreamingKMeans` |
| Very large dataset | `MiniBatchKMeans` |
| Need outlier resistance | `KMedoids` or `RobustKMeans` |
| Need equal-sized clusters | `BalancedKMeans` |
| Have must-link/cannot-link constraints | `ConstrainedKMeans` |
| Time series data | `TimeSeriesKMeans` |
| Non-convex clusters | `SpectralClustering` |

---

## Quick Examples

### "I have customer purchase data"
```scala
new GeneralizedKMeans()
  .setK(5)
  .setDivergence("squaredEuclidean")
```

### "I have document topic distributions"
```scala
new GeneralizedKMeans()
  .setK(10)
  .setDivergence("kl")
```

### "I don't know how many clusters"
```scala
new XMeans()
  .setMinK(2)
  .setMaxK(20)
```

### "I have noisy sensor data"
```scala
new RobustKMeans()
  .setK(5)
  .setRobustMode("trim")
```

---

See [Choose the Right Divergence](../howto/choose-divergence.html) for detailed guidance.

---

[Back to Explanation](index.html) | [Home](../)
