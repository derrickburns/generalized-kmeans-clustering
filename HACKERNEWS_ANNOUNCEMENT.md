# HackerNews Announcement

## Title

Generalized K-Means Clustering for Apache Spark with Bregman Divergences

## Body

I've been working on a production-ready K-Means clustering library for Apache Spark that goes beyond the standard Euclidean distance. It's now at a point where others might find it useful.

**Why use this instead of Spark MLlib's KMeans?**

Spark's built-in KMeans is hard-coded to Euclidean distance (L2), which is mathematically wrong for many real-world data types:

- **Probability distributions**: KL divergence is the natural distance metric (not Euclidean). If you're clustering topic models, histograms, or normalized feature vectors, Euclidean treats them as points in space rather than probability distributions.
- **Audio/spectral data**: Itakura-Saito divergence respects the multiplicative nature of power spectra. Euclidean would treat -20dB and -10dB as "closer" than -10dB and 0dB, which doesn't match human perception.
- **Count data**: Generalized-I divergence is the natural choice for Poisson-distributed counts (web traffic, sales data).
- **Outlier robustness**: L1/Manhattan distance gives you median-based clustering, which is more robust to outliers than mean-based (L2).

Using the wrong divergence can give you mathematically valid but semantically meaningless clusters. This library lets you match the divergence to your data type.

**What makes this different from MLlib:**

Beyond divergences, this library implements Lloyd's algorithm with pluggable Bregman divergences, which means you can cluster data using distance functions that match your problem domain:

- **KL Divergence**: For probability distributions (text, topic models)
- **Itakura-Saito**: For spectral data (audio, signals)
- **L1/Manhattan**: For outlier-robust clustering
- **Generalized-I**: For count data
- **Logistic Loss**: For binary probabilities
- Plus standard Squared Euclidean

**Quick comparison:**

| Use Case | Spark MLlib | This Library |
|----------|-------------|--------------|
| General numeric data | ✅ KMeans (L2) | ✅ GeneralizedKMeans (L2) - drop-in compatible |
| Probability distributions | ❌ Wrong metric | ✅ KL divergence |
| Outlier-robust clustering | ❌ | ✅ L1/Manhattan or KMedoids |
| Automatic k selection | ❌ Manual tuning | ✅ XMeans with BIC/AIC |
| Fuzzy clustering | ❌ | ✅ SoftKMeans (soft assignments) |
| Streaming data | ✅ StreamingKMeans | ✅ StreamingKMeans + divergences |

**What's included:**

- 6 clustering algorithms: GeneralizedKMeans, BisectingKMeans, XMeans (automatic k), SoftKMeans (fuzzy), StreamingKMeans, KMedoids
- Full Spark DataFrame API with model persistence (cross-version compatible)
- 740 tests (100% passing), deterministic behavior, comprehensive docs
- Scales to billions of points with automatic assignment strategy selection
- Python and Scala APIs
- **API compatibility**: Can be used as a drop-in replacement for MLlib KMeans (same DataFrame API, same parameters)

**Performance:**

The library includes automatic optimization - it switches between broadcast and crossJoin strategies based on k×dim to avoid OOM. Current benchmarks show ~870 points/sec for SE, ~3,400 points/sec for KL on modest hardware (full details in PERFORMANCE_BENCHMARKS.md).

**Educational value:**

I've tried to make this useful for learning as well. All algorithms link to executable examples with assertions, comprehensive test coverage, and documentation that bridges theory to code. The persistence format is versioned and documented (cross-compatible Spark 3.4↔3.5, Scala 2.12↔2.13).

**Example - Why divergence matters:**

```scala
// Clustering topic distributions from LDA
// Each row is a probability distribution over 100 topics
val topicDistributions: DataFrame = // [doc_id, prob_vector]

// ❌ WRONG: MLlib KMeans with Euclidean distance
// Treats [0.5, 0.3, 0.2] and [0.49, 0.31, 0.2] as very similar
// even though they represent different topic mixtures
val mllibKMeans = new org.apache.spark.ml.clustering.KMeans()
  .setK(10)
  .fit(topicDistributions)

// ✅ CORRECT: This library with KL divergence
// Respects probability simplex geometry
val generalizedKMeans = new GeneralizedKMeans()
  .setK(10)
  .setDivergence("kl")  // Natural distance for probability distributions
  .fit(topicDistributions)

// For standard numeric data, just use "squaredEuclidean" - works identically to MLlib
val standardKMeans = new GeneralizedKMeans()
  .setK(5)
  .setDivergence("squaredEuclidean")  // Drop-in MLlib replacement
  .fit(numericData)

// Model includes training summary
println(s"Converged in ${model.summary.iterations} iterations")
println(s"Assignment strategy: ${model.summary.assignmentStrategy}")
```

**Status:**

The library is mature enough for production use. All P0 items are complete:
- ✅ Model persistence with cross-version compatibility
- ✅ Scalability guardrails (chunked assignment for large k×dim)
- ✅ Determinism testing (same seed → identical results)
- ✅ Performance regression detection in CI
- ✅ Executable documentation with assertions

GitHub: https://github.com/derrickburns/generalized-kmeans-clustering

I'm open to feedback, contributions, or questions about the implementation. The README has links to all the code, tests, and examples if you want to dig deeper.

---

**Optional closing:**

*This started as an experiment to understand Bregman divergences better. Turns out clustering probability distributions with KL divergence is actually faster than Euclidean in many cases, which surprised me. Would love to hear if others have found similar results.*
