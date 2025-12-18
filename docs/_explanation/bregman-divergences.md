---
title: Bregman Divergences
---

# Bregman Divergences

Understanding the mathematical foundation of generalized k-means.

---

## What is a Bregman Divergence?

A Bregman divergence is a measure of "distance" between two points, defined by a strictly convex function φ (called the generator):

```
D_φ(x, y) = φ(x) - φ(y) - ∇φ(y) · (x - y)
```

**Intuition:** The divergence measures the difference between φ(x) and its linear approximation at y.

---

## Why Bregman Divergences?

### 1. Unique Mean Property

For any Bregman divergence, the point that minimizes the sum of divergences from a set of points is the **arithmetic mean**.

```
argmin_c Σᵢ D_φ(xᵢ, c) = (1/n) Σᵢ xᵢ
```

This is why k-means (with any Bregman divergence) uses simple averaging to update centers.

### 2. Natural for Exponential Families

Each Bregman divergence corresponds to a member of the exponential family of distributions:

| Divergence | Distribution | Natural for |
|------------|--------------|-------------|
| Squared Euclidean | Gaussian | Continuous data |
| KL | Multinomial/Poisson | Counts, probabilities |
| Itakura-Saito | Gamma | Power spectra |
| Logistic | Bernoulli | Binary data |

### 3. Consistent Objective

The k-means objective (minimize within-cluster divergence) has the same form regardless of which Bregman divergence you use:

```
minimize Σᵢ Σⱼ wᵢⱼ D_φ(xᵢ, μⱼ)
```

---

## The Generator Function

Each divergence is fully specified by its generator φ:

### Squared Euclidean
```
φ(x) = ½||x||² = ½ Σᵢ xᵢ²
∇φ(x) = x
D_φ(x,y) = ½||x - y||²
```

### KL Divergence
```
φ(x) = Σᵢ xᵢ log(xᵢ)  (negative entropy)
∇φ(x) = log(x) + 1
D_φ(x,y) = Σᵢ xᵢ log(xᵢ/yᵢ) - xᵢ + yᵢ
```

### Itakura-Saito
```
φ(x) = -Σᵢ log(xᵢ)
∇φ(x) = -1/x
D_φ(x,y) = Σᵢ (xᵢ/yᵢ - log(xᵢ/yᵢ) - 1)
```

---

## Properties

### Non-negativity
```
D_φ(x, y) ≥ 0 with equality iff x = y
```

### Convexity
D_φ(x, y) is convex in x (but not necessarily in y).

### Asymmetry
In general, D_φ(x, y) ≠ D_φ(y, x).

Squared Euclidean is the **only** symmetric Bregman divergence.

### Triangle Inequality
Bregman divergences generally **do not** satisfy the triangle inequality (except squared Euclidean).

---

## Geometric Interpretation

### Squared Euclidean
- Measures actual geometric distance
- Centers are centroids (center of mass)
- Produces spherical/convex clusters

### KL Divergence
- Measures information difference
- Centers are geometric means (in log space)
- Natural for probability simplex

### Itakura-Saito
- Scale-invariant (relative error matters)
- Centers are harmonic means (in some sense)
- Natural for multiplicative noise models

---

## Choosing the Right Divergence

The "right" divergence depends on:

1. **Data type:** What does your data represent?
2. **Noise model:** What errors are expected?
3. **Domain constraints:** Positive? Sums to 1?

### Decision Guide

```
Is your data probability distributions?
  → KL divergence

Is your data power spectra or variances?
  → Itakura-Saito

Is scale important (absolute values matter)?
  → Squared Euclidean

Is angle important (direction matters)?
  → Cosine / Spherical

Do you have counts (not normalized)?
  → Generalized I-divergence

Is your data binary probabilities?
  → Logistic loss
```

---

## Mathematical Foundations

### Connection to Exponential Families

For an exponential family with natural parameter θ and log-partition A(θ):

```
p(x|θ) = h(x) exp(θ·x - A(θ))
```

The corresponding Bregman divergence uses:
```
φ = A* (convex conjugate of A)
```

This is why:
- KL matches multinomial (log-partition = log Σ exp(θᵢ))
- Squared Euclidean matches Gaussian (log-partition = ½||θ||²)

### Connection to Information Geometry

Bregman divergences define a **dually flat** geometry on the parameter space. The primal coordinates (means) and dual coordinates (natural parameters) are connected by the gradient of φ.

---

## References

1. Banerjee, A., et al. (2005). "Clustering with Bregman divergences." *JMLR*.
2. Bregman, L. M. (1967). "The relaxation method of finding the common point of convex sets."

---

[Back to Explanation](index.html) | [Home](../)
