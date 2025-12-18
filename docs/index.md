---
title: Home
---

# Generalized K-Means Clustering

**Scalable clustering with Bregman divergences on Apache Spark**

[![Build Status](https://github.com/derrickburns/generalized-kmeans-clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/derrickburns/generalized-kmeans-clustering/actions)
[![Scala 2.13](https://img.shields.io/badge/scala-2.13-blue.svg)](https://www.scala-lang.org/)
[![Spark 3.5](https://img.shields.io/badge/spark-3.5-orange.svg)](https://spark.apache.org/)

---

## What is this library?

A production-ready Spark ML library providing **15 clustering algorithms** with support for multiple distance functions (Bregman divergences), including KL divergence for probability distributions, Itakura-Saito for spectral data, and more.

**Key features:**
- **854 tests** ensuring correctness and stability
- **Full Spark ML Pipeline integration** with Estimator/Model pattern
- **Model persistence** with cross-version compatibility
- **PySpark support** with type hints and examples

---

## Quick Example

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
import org.apache.spark.ml.linalg.Vectors

val data = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.0, 0.0)),
  Tuple1(Vectors.dense(1.0, 1.0)),
  Tuple1(Vectors.dense(9.0, 8.0)),
  Tuple1(Vectors.dense(8.0, 9.0))
)).toDF("features")

val kmeans = new GeneralizedKMeans()
  .setK(2)
  .setDivergence("squaredEuclidean")  // or "kl", "itakuraSaito", etc.
  .setMaxIter(20)

val model = kmeans.fit(data)
val predictions = model.transform(data)
```

---

## Documentation

This documentation follows the [Diátaxis](https://diataxis.fr/) framework:

### [Tutorials](tutorials/index.html) — Learning-oriented
Step-by-step guides to get you started:
- [Your First Clustering](tutorials/first-clustering.html) — Cluster data in 5 minutes
- [PySpark Tutorial](tutorials/pyspark-tutorial.html) — Python users start here
- [Choosing the Right Algorithm](tutorials/choosing-algorithm.html) — Decision guide

### [How-To Guides](howto/index.html) — Task-oriented
Practical recipes for specific tasks:
- [Cluster Probability Distributions](howto/cluster-probabilities.html) — Use KL divergence
- [Handle Outliers](howto/handle-outliers.html) — Robust clustering
- [Find Optimal K](howto/find-optimal-k.html) — Elbow method and X-Means
- [Installation](howto/installation.html) — Setup instructions

### [Reference](reference/index.html) — Information-oriented
Technical specifications:
- [Algorithm Reference](reference/algorithms.html) — All 15 algorithms
- [Parameter Reference](reference/parameters.html) — Every parameter documented
- [Divergence Reference](reference/divergences.html) — Mathematical details
- [API Reference](api/) — Scaladoc

### [Explanation](explanation/index.html) — Understanding-oriented
Conceptual guides:
- [Bregman Divergences](explanation/bregman-divergences.html) — The math behind it
- [Performance Tuning](explanation/performance.html) — Scaling to billions

---

## Algorithms

| Algorithm | Use Case | Key Feature |
|-----------|----------|-------------|
| **GeneralizedKMeans** | General clustering | 7 divergence functions |
| **XMeans** | Unknown k | Automatic cluster count |
| **SoftKMeans** | Overlapping clusters | Probabilistic assignments |
| **BisectingKMeans** | Hierarchical | Top-down divisive |
| **StreamingKMeans** | Real-time | Online updates |
| **KMedoids** | Outlier-resistant | Uses actual data points |
| **BalancedKMeans** | Equal-sized clusters | Size constraints |
| **ConstrainedKMeans** | Semi-supervised | Must-link/cannot-link |
| **RobustKMeans** | Noisy data | Outlier detection |
| **SparseKMeans** | High-dimensional | Sparse optimization |
| **MultiViewKMeans** | Multiple features | Per-view divergences |
| **TimeSeriesKMeans** | Sequences | DTW distance |
| **SpectralClustering** | Non-convex | Graph Laplacian |
| **InformationBottleneck** | Compression | Information theory |
| **MiniBatchKMeans** | Large scale | Stochastic updates |

---

## Installation

### SBT
```scala
libraryDependencies += "com.massivedatascience" %% "clusterer" % "0.7.0"
```

### spark-submit
```bash
spark-submit --packages com.massivedatascience:clusterer_2.13:0.7.0 your-app.jar
```

### Databricks
```
%pip install massivedatascience-clusterer
```

See [Installation Guide](howto/installation.html) for detailed instructions.

---

## Version Compatibility

| Spark | Scala 2.12 | Scala 2.13 |
|-------|------------|------------|
| 4.0.x | —          | ✓          |
| 3.5.x | ✓          | ✓          |
| 3.4.x | ✓          | ✓          |

---

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/derrickburns/generalized-kmeans-clustering/issues)
- **Discussions**: [Ask questions](https://github.com/derrickburns/generalized-kmeans-clustering/discussions)

---

## License

Apache License 2.0 — See [LICENSE](https://github.com/derrickburns/generalized-kmeans-clustering/blob/master/LICENSE)

*Copyright © 2025 Massive Data Science, LLC*
