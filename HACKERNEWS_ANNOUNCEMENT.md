# HackerNews Announcement

## Title

Generalized K-Means Clustering for Apache Spark with Bregman Divergences

## Body (3,982 characters)

I've built a production-ready K-Means library for Apache Spark that supports multiple distance functions beyond Euclidean.

**Why use this instead of Spark MLlib?**

MLlib's KMeans is hard-coded to Euclidean distance, which is mathematically wrong for many data types:

- **Probability distributions** (topic models, histograms): KL divergence is the natural metric. Euclidean treats [0.5, 0.3, 0.2] and [0.49, 0.31, 0.2] as similar even though they represent different distributions.
- **Audio/spectral data**: Itakura-Saito respects multiplicative power spectra. Euclidean incorrectly treats -20dB and -10dB as closer than -10dB and 0dB.
- **Count data** (traffic, sales): Generalized-I divergence for Poisson-distributed data.
- **Outlier robustness**: L1/Manhattan gives median-based clustering vs mean-based (L2).

Using the wrong divergence yields mathematically valid but semantically meaningless clusters.

**Available divergences:**
KL, Itakura-Saito, L1/Manhattan, Generalized-I, Logistic Loss, Squared Euclidean

**What's included:**
- 6 algorithms: GeneralizedKMeans, BisectingKMeans, XMeans (auto k), SoftKMeans (fuzzy), StreamingKMeans, KMedoids
- Drop-in MLlib replacement (same DataFrame API)
- 740 tests, deterministic behavior, cross-version persistence (Spark 3.4↔3.5, Scala 2.12↔2.13)
- Automatic optimization (broadcast vs crossJoin based on k×dim to avoid OOM)
- Python and Scala APIs

**Example:**

```scala
// Clustering topic distributions from LDA
val topics: DataFrame = // probability vectors

// ❌ WRONG: MLlib with Euclidean
new org.apache.spark.ml.clustering.KMeans()
  .setK(10).fit(topics)

// ✅ CORRECT: KL divergence for probabilities
new GeneralizedKMeans()
  .setK(10)
  .setDivergence("kl")
  .fit(topics)

// For standard data, drop-in replacement:
new GeneralizedKMeans()
  .setDivergence("squaredEuclidean")
  .fit(numericData)
```

**Quick comparison:**

| Use Case | MLlib | This Library |
|----------|-------|--------------|
| General numeric | ✅ L2 | ✅ L2 (compatible) |
| Probability distributions | ❌ Wrong | ✅ KL divergence |
| Outlier-robust | ❌ | ✅ L1 or KMedoids |
| Auto k selection | ❌ | ✅ XMeans (BIC/AIC) |
| Fuzzy clustering | ❌ | ✅ SoftKMeans |

**Performance:**
~870 pts/sec (SE), ~3,400 pts/sec (KL) on modest hardware. Scales to billions of points with automatic strategy selection.

**Production-ready:**
- ✅ Cross-version model persistence
- ✅ Scalability guardrails (chunked assignment)
- ✅ Determinism tests (same seed → identical results)
- ✅ Performance regression detection
- ✅ Executable documentation

GitHub: https://github.com/derrickburns/generalized-kmeans-clustering

This started as an experiment to understand Bregman divergences. Surprisingly, KL divergence is often faster than Euclidean for probability data. Open to feedback!
