---
title: Divergence Reference
---

# Divergence Reference

Mathematical details of all supported Bregman divergences.

---

## Overview

A Bregman divergence D_φ(x, y) is defined by a strictly convex function φ:

```
D_φ(x, y) = φ(x) - φ(y) - ∇φ(y)·(x - y)
```

The library implements 8 divergences, each optimal for different data types.

---

## Squared Euclidean

**Name:** `squaredEuclidean`

**Formula:**
```
D(x, y) = ½ Σᵢ (xᵢ - yᵢ)²
```

**Generator:** φ(x) = ½||x||²

**Properties:**
- Symmetric: D(x,y) = D(y,x)
- Metric (satisfies triangle inequality when taking sqrt)
- The standard k-means distance

**Best for:** General-purpose clustering, continuous features

**Domain:** All real numbers

```scala
new GeneralizedKMeans().setDivergence("squaredEuclidean")
```

---

## Kullback-Leibler (KL)

**Name:** `kl`

**Formula:**
```
D(x, y) = Σᵢ xᵢ log(xᵢ/yᵢ) - Σᵢ xᵢ + Σᵢ yᵢ
```

**Generator:** φ(x) = Σᵢ xᵢ log(xᵢ)

**Properties:**
- Not symmetric: D(x,y) ≠ D(y,x)
- Non-negative
- Undefined when yᵢ = 0 and xᵢ > 0

**Best for:** Probability distributions, topic models, document clustering

**Domain:** Strictly positive (use `smoothing` parameter)

```scala
new GeneralizedKMeans()
  .setDivergence("kl")
  .setSmoothing(1e-10)
```

---

## Itakura-Saito

**Name:** `itakuraSaito`

**Formula:**
```
D(x, y) = Σᵢ (xᵢ/yᵢ - log(xᵢ/yᵢ) - 1)
```

**Generator:** φ(x) = -Σᵢ log(xᵢ)

**Properties:**
- Scale-invariant: D(cx, cy) = D(x, y)
- Asymmetric
- Sensitive to small y values

**Best for:** Power spectra, audio signals, spectral analysis

**Domain:** Strictly positive

```scala
new GeneralizedKMeans()
  .setDivergence("itakuraSaito")
  .setSmoothing(1e-10)
```

---

## L1 (Manhattan)

**Name:** `l1`

**Formula:**
```
D(x, y) = Σᵢ |xᵢ - yᵢ|
```

**Note:** Not a true Bregman divergence, but supported for practicality.

**Properties:**
- Symmetric
- More robust to outliers than L2
- K-medians objective

**Best for:** Robust clustering, sparse features, when outliers present

**Domain:** All real numbers

```scala
new GeneralizedKMeans().setDivergence("l1")
```

---

## Generalized I-Divergence

**Name:** `generalizedI`

**Formula:**
```
D(x, y) = Σᵢ (xᵢ log(xᵢ/yᵢ) - xᵢ + yᵢ)
```

**Generator:** φ(x) = Σᵢ xᵢ log(xᵢ) - xᵢ

**Properties:**
- Related to KL but for unnormalized data
- Works with count data directly

**Best for:** Count data, word frequencies (not normalized)

**Domain:** Non-negative

```scala
new GeneralizedKMeans().setDivergence("generalizedI")
```

---

## Logistic Loss

**Name:** `logistic`

**Formula:**
```
D(x, y) = Σᵢ (xᵢ log(xᵢ/yᵢ) + (1-xᵢ) log((1-xᵢ)/(1-yᵢ)))
```

**Generator:** φ(x) = Σᵢ xᵢ log(xᵢ) + (1-xᵢ) log(1-xᵢ)

**Properties:**
- For data in [0, 1]
- Binary cross-entropy like

**Best for:** Binary/probability data, Bernoulli-distributed features

**Domain:** Open interval (0, 1)

```scala
new GeneralizedKMeans().setDivergence("logistic")
```

---

## Spherical / Cosine

**Names:** `spherical`, `cosine`

**Formula:**
```
D(x, y) = 1 - (x · y) / (||x|| ||y||)
```

**Properties:**
- Measures angle between vectors
- Invariant to vector magnitude
- Range: [0, 2]

**Best for:** Text/document vectors (TF-IDF), when magnitude doesn't matter

**Domain:** Non-zero vectors

```scala
new GeneralizedKMeans().setDivergence("cosine")
// or
new GeneralizedKMeans().setDivergence("spherical")
```

---

## Choosing a Divergence

| Data Type | Recommended | Why |
|-----------|-------------|-----|
| General continuous | `squaredEuclidean` | Standard, well-understood |
| Probability distributions | `kl` | Information-theoretic optimal |
| Word frequencies (normalized) | `kl` | Natural for distributions |
| Word frequencies (counts) | `generalizedI` | Handles unnormalized |
| Power spectra / audio | `itakuraSaito` | Scale-invariant |
| TF-IDF / embeddings | `cosine` | Magnitude-invariant |
| Binary probabilities | `logistic` | Natural for [0,1] data |
| Robust clustering | `l1` | Less outlier-sensitive |

---

## Domain Requirements Summary

| Divergence | Requirement | Smoothing Needed |
|------------|-------------|------------------|
| squaredEuclidean | Any real | No |
| kl | Strictly positive | Yes |
| itakuraSaito | Strictly positive | Yes |
| l1 | Any real | No |
| generalizedI | Non-negative | Optional |
| logistic | (0, 1) exclusive | Yes |
| spherical/cosine | Non-zero | No |

---

[Back to Reference](index.html) | [Home](../)
