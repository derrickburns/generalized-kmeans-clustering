# Divergence Selection Guide

Choosing the right Bregman divergence is crucial for clustering quality. This guide helps you select the best divergence for your data type and use case.

## Quick Reference

| Divergence | Best For | Domain | Sparse? |
|------------|----------|--------|---------|
| `squaredEuclidean` | General numeric data | R^n | Yes |
| `kl` | Probability distributions, TF-IDF | R+^n (non-negative) | Yes |
| `spherical` / `cosine` | Text embeddings, unit vectors | R^n (non-zero) | Yes |
| `l1` / `manhattan` | Outlier-robust clustering | R^n | Yes |
| `itakuraSaito` | Audio/spectral analysis | R+^n (strictly positive) | No |
| `generalizedI` | Count data | R+^n (non-negative) | No |
| `logistic` | Bounded probabilities | [0,1]^n | No |

## Decision Flowchart

```
What type of data do you have?
│
├─► Probability distributions / Topic proportions
│   └─► Use `kl` with setSmoothing(1e-6)
│
├─► Text embeddings / Document vectors
│   └─► Use `spherical` (cosine similarity)
│
├─► TF-IDF vectors (sparse, non-negative)
│   └─► Use `kl` or `spherical` with SparseKMeans
│
├─► Audio / Power spectra / Spectrograms
│   └─► Use `itakuraSaito` with setSmoothing(1e-6)
│
├─► Count data / Histograms
│   └─► Use `generalizedI` or `kl`
│
├─► Data with outliers
│   └─► Use `l1` (Manhattan distance)
│
├─► Bounded [0,1] values
│   └─► Use `logistic`
│
└─► General numeric data
    └─► Use `squaredEuclidean` (default)
```

## Detailed Divergence Guide

### Squared Euclidean (Default)

**Mathematical form:** D(x,y) = ||x - y||²

```scala
new GeneralizedKMeans()
  .setDivergence("squaredEuclidean")
```

**Best for:**
- General-purpose clustering
- Continuous numeric features
- Data where Euclidean distance is meaningful

**Characteristics:**
- Fastest computation (optimized code path)
- Centers are arithmetic means
- Sensitive to scale - normalize features first
- Sensitive to outliers

**Example use cases:**
- Customer segmentation (normalized features)
- Image pixel clustering
- Sensor data clustering

---

### KL Divergence (Kullback-Leibler)

**Mathematical form:** D(x,y) = Σ x_i log(x_i/y_i) - x_i + y_i

```scala
new GeneralizedKMeans()
  .setDivergence("kl")
  .setSmoothing(1e-6)  // Required for numerical stability
```

**Best for:**
- Probability distributions
- Topic models (LDA topics)
- TF-IDF document vectors
- Histogram data

**Characteristics:**
- Requires non-negative data
- Asymmetric (D(x,y) ≠ D(y,x))
- Centers are weighted geometric means
- Natural for information-theoretic interpretation

**Example use cases:**
- Document clustering (topic distributions)
- Recommendation systems (user preference distributions)
- Image histogram clustering

**Important:** Add smoothing to avoid log(0):
```scala
// If your data contains zeros, use smoothing
.setSmoothing(1e-6)
```

---

### Spherical / Cosine

**Mathematical form:** D(x,y) = 1 - cos(x,y) = 1 - (x·y)/(||x|| ||y||)

```scala
new GeneralizedKMeans()
  .setDivergence("spherical")  // or "cosine"
```

**Best for:**
- Text embeddings (Word2Vec, BERT, etc.)
- Document vectors
- Any data where direction matters more than magnitude

**Characteristics:**
- Scale-invariant (only considers direction)
- Natural for high-dimensional sparse data
- Centers are normalized
- Well-suited for similarity-based retrieval

**Example use cases:**
- Document clustering
- Semantic similarity grouping
- Image feature clustering (CNN features)

```scala
// For text embeddings
val spherical = new GeneralizedKMeans()
  .setK(20)
  .setDivergence("spherical")
  .setMaxIter(50)

// Or use SparseKMeans for efficiency
import com.massivedatascience.clusterer.ml.SparseKMeans
val sparse = new SparseKMeans()
  .setK(20)
  .setDivergence("spherical")
  .setSparseMode("auto")
```

---

### L1 / Manhattan (K-Medians)

**Mathematical form:** D(x,y) = Σ |x_i - y_i|

```scala
new GeneralizedKMeans()
  .setDivergence("l1")  // or "manhattan"
```

**Best for:**
- Data with outliers
- Robust clustering
- When median is more meaningful than mean

**Characteristics:**
- More robust to outliers than squared Euclidean
- Centers are coordinate-wise medians
- Linear penalty for distance (vs quadratic for SE)
- Slightly slower than SE

**Example use cases:**
- Financial data (outlier-prone)
- Sensor data with noise spikes
- Any data where extreme values should have less influence

---

### Itakura-Saito

**Mathematical form:** D(x,y) = Σ (x_i/y_i - log(x_i/y_i) - 1)

```scala
new GeneralizedKMeans()
  .setDivergence("itakuraSaito")
  .setSmoothing(1e-6)  // Required - data must be strictly positive
```

**Best for:**
- Audio signal processing
- Power spectra analysis
- Spectrograms

**Characteristics:**
- Scale-invariant in a multiplicative sense
- Requires strictly positive data
- Natural for audio/spectral data
- Penalizes underestimation more than overestimation

**Example use cases:**
- Speech recognition feature clustering
- Music genre classification
- Radar/sonar signal analysis

---

### Generalized I-Divergence

**Mathematical form:** D(x,y) = Σ (x_i log(x_i/y_i) - x_i + y_i)

```scala
new GeneralizedKMeans()
  .setDivergence("generalizedI")
  .setSmoothing(1e-6)
```

**Best for:**
- Count data
- Non-negative matrix factorization
- Poisson-distributed data

**Characteristics:**
- Similar to KL but with different centering
- Natural for count data
- Requires non-negative values

---

### Logistic Loss

**Mathematical form:** Based on logistic function for bounded [0,1] values

```scala
new GeneralizedKMeans()
  .setDivergence("logistic")
  .setSmoothing(1e-6)
```

**Best for:**
- Probability values in [0,1]
- Binary feature proportions
- Soft binary attributes

**Characteristics:**
- Data must be in [0,1] range
- Natural for probability-like features

---

## Practical Examples

### Example 1: Document Clustering with TF-IDF

```scala
import com.massivedatascience.clusterer.ml.SparseKMeans
import org.apache.spark.ml.feature.{HashingTF, IDF}

// Create TF-IDF vectors
val tf = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

val tfData = tf.transform(documents)
val idfModel = idf.fit(tfData)
val tfidfData = idfModel.transform(tfData)

// Cluster with KL divergence (natural for TF-IDF)
val kmeans = new SparseKMeans()
  .setK(20)
  .setDivergence("kl")
  .setSparseMode("auto")
  .setSmoothing(1e-10)

val model = kmeans.fit(tfidfData)
```

### Example 2: Customer Segmentation

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
import org.apache.spark.ml.feature.StandardScaler

// Normalize features first (important for SE)
val scaler = new StandardScaler()
  .setInputCol("rawFeatures")
  .setOutputCol("features")
  .setWithMean(true)
  .setWithStd(true)

val scaledData = scaler.fit(customerData).transform(customerData)

// Cluster with squared Euclidean
val kmeans = new GeneralizedKMeans()
  .setK(5)
  .setDivergence("squaredEuclidean")
  .setMaxIter(50)

val model = kmeans.fit(scaledData)
```

### Example 3: Embedding Clustering

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

// For BERT/Word2Vec embeddings, use spherical
val kmeans = new GeneralizedKMeans()
  .setK(100)
  .setDivergence("spherical")
  .setMaxIter(30)

val model = kmeans.fit(embeddingsDF)
```

### Example 4: Robust Clustering with Outliers

```scala
import com.massivedatascience.clusterer.ml.RobustKMeans

// Use L1 divergence with outlier handling
val robust = new RobustKMeans()
  .setK(5)
  .setDivergence("l1")
  .setOutlierMode("trim")
  .setTrimFraction(0.05)  // Trim 5% as outliers

val model = robust.fit(noisyData)
```

## Domain Validation

Some divergences require specific data domains. The library validates this automatically:

| Divergence | Requirement | Error if violated |
|------------|-------------|-------------------|
| `kl` | Non-negative values | Use `setSmoothing()` |
| `itakuraSaito` | Strictly positive | Use `setSmoothing()` |
| `generalizedI` | Non-negative values | Use `setSmoothing()` |
| `logistic` | Values in [0,1] | Transform data first |

```scala
// Handle domain issues with smoothing
new GeneralizedKMeans()
  .setDivergence("kl")
  .setSmoothing(1e-6)  // Adds small constant to avoid zeros
```

## Performance Considerations

| Divergence | Relative Speed | Sparse Optimization |
|------------|---------------|---------------------|
| `squaredEuclidean` | Fastest (1x) | Yes |
| `l1` | Fast (1.2x) | Yes |
| `spherical` | Fast (1.3x) | Yes |
| `kl` | Medium (1.5x) | Yes |
| `itakuraSaito` | Medium (1.5x) | No |
| `generalizedI` | Medium (1.5x) | No |
| `logistic` | Slower (2x) | No |

For sparse data, use `SparseKMeans` which has optimized kernels for SE, KL, L1, and Spherical divergences.

## Summary

1. **Start with `squaredEuclidean`** for general data (normalize first)
2. **Use `kl`** for probability distributions and TF-IDF
3. **Use `spherical`** for text embeddings and directional data
4. **Use `l1`** when outliers are a concern
5. **Always add `setSmoothing()`** for KL, IS, and generalizedI
6. **Use `SparseKMeans`** for high-dimensional sparse data
