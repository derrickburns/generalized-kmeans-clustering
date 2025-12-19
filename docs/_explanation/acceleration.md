---
title: "Acceleration Techniques"
---

# Acceleration Techniques

Making k-means faster without sacrificing quality.

---

## Overview

Standard Lloyd's algorithm is O(n × k × d × iterations). These techniques reduce the constant factor significantly.

| Technique | Speedup | Applicable To |
|-----------|---------|---------------|
| **Elkan** | 2-10x | Squared Euclidean |
| **Mini-batch** | 5-50x | Any divergence |
| **Coresets** | 10-100x | Any divergence |

---

## Elkan's Algorithm

Uses triangle inequality to skip distance computations.

### Key Insight

If we know:
- d(x, c₁) = 5 (current assignment)
- d(c₁, c₂) = 8 (center-to-center)

Then by triangle inequality:
- d(x, c₂) ≥ |d(x, c₁) - d(c₁, c₂)| = 3
- d(x, c₂) ≥ d(x, c₁) means x could be closer to c₂

But if d(c₁, c₂) = 12:
- d(x, c₂) ≥ |5 - 12| = 7 > 5 = d(x, c₁)
- x cannot be closer to c₂, **skip the computation!**

### Bounds Maintained

1. **Upper bound**: d(x, assigned_center) — always valid
2. **Lower bounds**: d(x, cᵢ) for each center — may become stale

### When It Helps

- **Early iterations**: Many points change clusters
- **Later iterations**: Most points stay put, bounds tight
- **Well-separated clusters**: Bounds eliminate most checks

### Usage

```scala
// Enabled automatically for SE with k >= 5
new GeneralizedKMeans()
  .setDivergence("squaredEuclidean")
  .setK(20)
  // Elkan is used automatically
```

---

## Mini-Batch K-Means

Updates centers using random samples instead of full data.

### Algorithm

```
1. Initialize centers
2. For each iteration:
   a. Sample a mini-batch of b points
   b. Assign batch points to nearest centers
   c. Update centers using batch points (with momentum)
3. Return centers
```

### Update Rule

```
center[j] = (1 - η) * center[j] + η * batch_mean[j]
```

Where η decreases over time (learning rate schedule).

### Trade-offs

| Aspect | Full K-Means | Mini-Batch |
|--------|--------------|------------|
| Per-iteration cost | O(n × k × d) | O(b × k × d) |
| Iterations needed | 10-50 | 100-500 |
| Total cost | O(n × k × d × 30) | O(b × k × d × 300) |
| Quality | Optimal | Near-optimal |

With b = n/100, mini-batch is ~3x faster with <1% quality loss.

### Usage

```scala
new MiniBatchKMeans()
  .setK(100)
  .setBatchSize(10000)  // Points per iteration
  .setMaxIter(200)
```

---

## Coreset Approximation

Compress data to a small weighted sample that preserves clustering structure.

### Key Idea

Instead of n points, cluster a coreset of m << n weighted points that approximate the original data's clustering cost.

### Construction

1. Run lightweight clustering (k-means++ sampling)
2. Assign each point to nearest sample
3. Weight samples by cluster sizes
4. Cluster the weighted coreset

### Theoretical Guarantee

For an ε-coreset:
```
(1-ε) × cost(P, C) ≤ cost(S, C) ≤ (1+ε) × cost(P, C)
```

For any center set C, coreset cost approximates true cost.

### Usage

```scala
new CoresetKMeans()
  .setK(20)
  .setCoresetSize(10000)  // Compress to 10K points
  .setEpsilon(0.1)        // 10% approximation
```

---

## Combining Techniques

For very large datasets, combine multiple techniques:

```scala
// 1B points → 10K coreset → mini-batch clustering
val coreset = new CoresetKMeans()
  .setCoresetSize(10000)
  .fit(massiveData)

// Or use streaming for continuous updates
val streaming = new StreamingKMeans()
  .setK(100)
  .setDecayFactor(0.9)
```

---

## Performance Guidelines

| Data Size | Recommendation |
|-----------|---------------|
| < 100K | Standard GeneralizedKMeans |
| 100K - 1M | Elkan (automatic for SE) |
| 1M - 100M | Mini-batch |
| > 100M | Coreset + Mini-batch |

---

## Benchmarks

On 10M points × 100 dimensions, k=100:

| Method | Time | Quality (vs optimal) |
|--------|------|---------------------|
| Standard | 15 min | 100% |
| Elkan | 3 min | 100% |
| Mini-batch (b=10K) | 2 min | 99.5% |
| Coreset (m=100K) | 30 sec | 98% |

---

[Back to Explanation](index.html) | [Home](../)
