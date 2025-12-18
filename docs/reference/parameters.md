---
title: Parameter Reference
---

# Parameter Reference

Complete documentation of all parameters across all algorithms.

---

## GeneralizedKMeans

The core k-means estimator with Bregman divergences.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | Int | 2 | Number of clusters (must be > 1) |
| `divergence` | String | "squaredEuclidean" | Distance function: squaredEuclidean, kl, itakuraSaito, l1, generalizedI, logistic, spherical, cosine |
| `maxIter` | Int | 20 | Maximum iterations (>= 0) |
| `tol` | Double | 1e-4 | Convergence tolerance for center movement |
| `seed` | Long | random | Random seed for reproducibility |
| `featuresCol` | String | "features" | Input features column |
| `predictionCol` | String | "prediction" | Output prediction column |
| `distanceCol` | String | — | Output distance column (optional) |
| `weightCol` | String | — | Point weights column (optional) |
| `smoothing` | Double | 1e-10 | Smoothing for KL/IS divergences |
| `assignmentStrategy` | String | "auto" | Strategy: auto, crossJoin, broadcastUDF |
| `emptyClusterStrategy` | String | "reseedRandom" | Empty handling: reseedRandom, drop |
| `initMode` | String | "k-means\|\|" | Initialization: random, k-means\|\| |
| `initSteps` | Int | 2 | K-means\|\| initialization steps |
| `checkpointInterval` | Int | 10 | Checkpoint interval (0 = disabled) |
| `checkpointDir` | String | — | Checkpoint directory |

---

## XMeans

Automatic k selection using information criteria.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `minK` | Int | 2 | Minimum clusters to consider |
| `maxK` | Int | 10 | Maximum clusters to consider |
| `criterion` | String | "bic" | Selection criterion: bic, aic |
| *Plus all GeneralizedKMeans parameters* |

---

## SoftKMeans

Probabilistic/fuzzy cluster assignments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | Double | 1.0 | Temperature (higher = more deterministic) |
| `probabilitiesCol` | String | "probabilities" | Output probabilities column |
| *Plus all GeneralizedKMeans parameters* |

---

## BisectingKMeans

Hierarchical divisive clustering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `minDivisibleClusterSize` | Int | 1 | Minimum size to split |
| *Plus all GeneralizedKMeans parameters* |

---

## StreamingKMeans

Online clustering with decay.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decayFactor` | Double | 1.0 | Exponential decay (0.0-1.0) |
| `halfLife` | Double | — | Alternative to decayFactor |
| `timeUnit` | String | "batches" | Decay unit: batches, points |
| *Plus all GeneralizedKMeans parameters* |

---

## KMedoids

Clustering with actual data points as centers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | Int | 2 | Number of clusters |
| `distanceFunction` | String | "euclidean" | Distance: euclidean, manhattan, cosine |
| `maxIter` | Int | 20 | Maximum iterations |
| `seed` | Long | random | Random seed |
| `featuresCol` | String | "features" | Features column |
| `predictionCol` | String | "prediction" | Prediction column |

---

## BalancedKMeans

Equal-sized cluster constraints.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `balanceMode` | String | "soft" | Mode: soft, hard |
| `maxClusterSize` | Int | auto | Maximum cluster size |
| `balancePenalty` | Double | 1.0 | Soft mode penalty weight |
| *Plus all GeneralizedKMeans parameters* |

---

## ConstrainedKMeans

Semi-supervised with constraints.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mustLinkCol` | String | — | Must-link pairs column |
| `cannotLinkCol` | String | — | Cannot-link pairs column |
| `constraintMode` | String | "soft" | Mode: soft, hard |
| `violationPenalty` | Double | 1.0 | Soft mode penalty |
| *Plus all GeneralizedKMeans parameters* |

---

## RobustKMeans

Outlier-resistant clustering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `robustMode` | String | "trim" | Mode: trim, noise_cluster, m_estimator |
| `trimFraction` | Double | 0.1 | Fraction to trim (trim mode) |
| `noiseThreshold` | Double | 2.0 | Distance threshold (noise mode) |
| `mEstimatorType` | String | "huber" | Type: huber, tukey, cauchy |
| `outlierScoreCol` | String | — | Output outlier scores |
| *Plus all GeneralizedKMeans parameters* |

---

## SparseKMeans

High-dimensional sparse data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sparseMode` | String | "auto" | Mode: auto, force, dense |
| `sparseThreshold` | Double | 0.5 | Sparsity threshold for auto |
| *Plus all GeneralizedKMeans parameters* |

---

## TimeSeriesKMeans

Sequence clustering with DTW.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distanceType` | String | "dtw" | Distance: dtw, softdtw, gak, derivative |
| `bandWidth` | Double | 0.1 | Sakoe-Chiba band width |
| `gamma` | Double | 1.0 | Soft-DTW smoothing |
| *Plus all GeneralizedKMeans parameters* |

---

## SpectralClustering

Graph-based clustering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `affinityType` | String | "rbf" | Affinity: rbf, knn, epsilon |
| `laplacianType` | String | "normalized" | Laplacian: unnormalized, normalized, randomWalk |
| `sigma` | Double | 1.0 | RBF kernel width |
| `numNeighbors` | Int | 10 | k-NN neighbors |
| `epsilon` | Double | 1.0 | Epsilon neighborhood |
| `useNystrom` | Boolean | false | Nyström approximation |
| `nystromSamples` | Int | 100 | Nyström sample size |
| *Plus k, seed, featuresCol, predictionCol* |

---

## InformationBottleneck

Information-theoretic clustering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | Double | 1.0 | Compression-relevance trade-off |
| `relevanceCol` | String | — | Relevance variable column |
| `convergenceTol` | Double | 1e-6 | Blahut-Arimoto tolerance |
| *Plus all GeneralizedKMeans parameters* |

---

## MiniBatchKMeans

Stochastic mini-batch updates.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batchSize` | Int | 100 | Points per batch |
| `reassignmentRatio` | Double | 0.01 | Reassignment threshold |
| *Plus all GeneralizedKMeans parameters* |

---

[Back to Reference](index.html) | [Home](../)
