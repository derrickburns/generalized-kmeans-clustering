# Tutorial: Generalized K-Means Clustering

> 📚 **Type**: Learning-oriented tutorial
>
> **Time**: 30 minutes
>
> **Level**: Beginner

[← Back to Index](INDEX.md) | [How-To Guide →](HOW-TO.md)

---

This tutorial will guide you through the basics of using the Generalized K-Means Clustering library. By the end, you'll understand the core concepts and be able to perform basic clustering tasks.

> 🔍 **Looking for something else?**
> - Need to solve a specific problem? Check the [How-To Guide](HOW-TO.md)
> - Want technical details? See the [Reference](REFERENCE.md)
> - Interested in the theory? Read the [Explanation](EXPLANATION.md)

## Overview

This library extends Spark MLlib's K-Means implementation to support:
- Batch and streaming clustering
- Multiple distance functions (Bregman divergences)
- High-dimensional data processing
- Mini-batch processing
- Various initialization strategies

## Features

### 1. Multiple Distance Functions
- Standard Euclidean distance
- [Bregman divergences](http://www.cs.utexas.edu/users/inderjit/public_papers/bregmanclustering_jmlr.pdf)
- [Symmetrized Bregman divergences](https://people.clas.ufl.edu/yun/files/article-8-1.pdf)

### 2. Clustering Strategies
- [Mini-batch processing](https://arxiv.org/abs/1108.1351) for large datasets
- [Bisection-based clustering](http://www.siam.org/meetings/sdm01/pdf/sdm01_05.pdf)
- [Near-optimal clustering](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf)
- [Streaming data support](http://papers.nips.cc/paper/3812-streaming-k-means-approximation.pdf)
- [Coreset approximation](https://people.csail.mit.edu/dannyf/coresets.pdf) for massive datasets (10-100x speedup)

## Quick Start

```scala
// Import required packages
import com.massivedatascience.clusterer._

// Create a KMeans instance
val kmeans = new KMeans()
  .setK(10)         // number of clusters
  .setMaxIterations(20)

// Fit the model
val model = kmeans.fit(data)

// Make predictions
val predictions = model.predict(newData)
```

## Advanced Usage

### Custom Distance Functions
```scala
// Use Bregman divergence
val bregmanKMeans = new KMeans()
  .setDistanceFunction(BregmanPointOps)
```

### Streaming Data
```scala
// Create streaming kmeans
val streamingKMeans = new StreamingKMeans()
  .setK(10)
  .setDecayFactor(0.5)
```

### Massive Datasets with Coresets
```scala
import com.massivedatascience.clusterer.KMeans

// Automatic strategy selection based on data size
val model = KMeans.trainSmart(
  data = data,        // RDD[Vector]
  k = 10,
  maxIterations = 50
)

// Or explicit coreset control for very large data
val model = KMeans.trainWithCoreset(
  data = data,
  k = 10,
  compressionRatio = 0.01,  // Use 1% of data
  enableRefinement = true
)
```

## Performance

This implementation has been battle-tested on:
- Datasets with tens of millions of points
- 700+ dimensional feature spaces
- Various distance functions

## Contributing

We welcome novel k-means clustering variants! If you have a provably superior approach:
1. Implement it using this package
2. Include the academic paper analyzing the variant
3. Submit a pull request

## References

Key papers implemented in this library:
* [Bregman Clustering](http://www.cs.utexas.edu/users/inderjit/public_papers/bregmanclustering_jmlr.pdf)
* [Mini-batch K-means](https://arxiv.org/abs/1108.1351)
* [High-dimensional Clustering](http://www.ida.liu.se/~arnjo/papers/pakdd-ws-11.pdf)
* [Time Series Clustering](http://www.cs.gmu.edu/~jessica/publications/ikmeans_sdm_workshop03.pdf)
* [Coreset Approximation](https://people.csail.mit.edu/dannyf/coresets.pdf)
