# How-To Guide: Generalized K-Means Clustering

> 🎩 **Type**: Task-oriented guide
>
> **Format**: Problem → Solution
>
> **Level**: Intermediate

[← Back to Index](INDEX.md) | [Reference →](REFERENCE.md)

---

This guide provides practical solutions for common clustering tasks. Each section addresses a specific problem with a concrete solution.

> 🔍 **Looking for something else?**
> - New to the library? Start with the [Tutorial](TUTORIAL.md)
> - Need technical details? See the [Reference](REFERENCE.md)
> - Want to understand the theory? Read the [Explanation](EXPLANATION.md)

> ℹ️ **Note**: This guide assumes basic familiarity with the library. If you're new, please start with the [Tutorial](TUTORIAL.md).

## Table of Contents
1. [Cluster High-Dimensional Data](#cluster-high-dimensional-data)
2. [Process Streaming Data](#process-streaming-data)
3. [Use Custom Distance Functions](#use-custom-distance-functions)
4. [Handle Large Datasets with Mini-Batches](#handle-large-datasets-with-mini-batches)
5. [Optimize Clustering Performance](#optimize-clustering-performance)

## Performance

This implementation has been thoroughly tested and proven to work efficiently on:
- Large-scale datasets: Successfully processed tens of millions of data points
- High-dimensional data: Handled feature spaces with 700+ dimensions
- Various distance functions: Tested with multiple Bregman divergences and custom metrics

> **Real-world validation**: This code has been battle-tested on production datasets with tens of millions of points in 700+ dimensional space using a variety of distance functions, powered by Spark's efficient distributed computing capabilities.

## Cluster High-Dimensional Data

When working with high-dimensional data (100+ dimensions):

```scala
import com.massivedatascience.clusterer._
import com.massivedatascience.transforms.RandomIndexEmbedding

// 1. Create dimension reducer
val embedding = new RandomIndexEmbedding()
  .setInputDim(originalDimension)
  .setOutputDim(targetDimension)

// 2. Transform your data
val reducedData = embedding.transform(data)

// 3. Configure K-means for high-dimensional data
val kmeans = new KMeans()
  .setK(numClusters)
  .setMaxIterations(20)
  .setInitializationStrategy(KMeansPlusPlus) // Better for high dimensions

// 4. Fit the model
val model = kmeans.fit(reducedData)
```

## Process Streaming Data

For real-time clustering of streaming data:

```scala
import com.massivedatascience.clusterer._

// 1. Configure streaming parameters
val streamingKMeans = new StreamingKMeans()
  .setK(numClusters)
  .setDecayFactor(0.5)    // How fast old data is forgotten
  .setHalfLife(3600)      // Time in seconds for data to lose half its weight

// 2. Update model with new data batches
streamingKMeans.update(newDataBatch, weight = 1.0)

// 3. Get predictions for new points
val predictions = streamingKMeans.predict(newPoints)

// 4. Access cluster centers
val currentCenters = streamingKMeans.latestModel().clusterCenters
```

## Use Custom Distance Functions

Implement custom distance metrics for specialized clustering:

```scala
import com.massivedatascience.clusterer._
import com.massivedatascience.divergence.BregmanDivergence

// 1. Define custom Bregman divergence
class CustomDivergence extends BregmanDivergence {
  override def compute(x: Vector, y: Vector): Double = {
    // Your custom distance calculation
  }
}

// 2. Create custom point operations
val customPointOps = new BregmanPointOps(new CustomDivergence())

// 3. Use in K-means
val kmeans = new KMeans()
  .setK(numClusters)
  .setDistanceFunction(customPointOps)
```

## Handle Large Datasets with Mini-Batches

For efficient processing of large datasets:

```scala
import com.massivedatascience.clusterer._

// 1. Configure mini-batch parameters
val kmeans = new KMeans()
  .setK(numClusters)
  .setBatchSize(1000)        // Size of each mini-batch
  .setConvergenceTol(0.01)   // Tolerance for convergence
  .setMaxIterations(50)      // Maximum passes over the data

// 2. Optional: Track convergence
val tracker = new TrackingKMeans(kmeans)
  .setTrackingStats(true)

// 3. Fit model with mini-batches
val model = tracker.fit(largeDataset)

// 4. Access convergence metrics
val stats = tracker.getTrackingStats()
```

## Optimize Clustering Performance

Tips for improving clustering quality and speed:

```scala
import com.massivedatascience.clusterer._

// 1. Use parallel initialization
val kmeans = new KMeansParallel()
  .setK(numClusters)
  .setInitializationSteps(5)   // Number of steps in initialization
  .setParallelism(4)           // Number of parallel tasks

// 2. Enable caching for repeated operations
val cachingKMeans = new CachingKMeans(kmeans)
  .setCacheLevel(StorageLevel.MEMORY_AND_DISK)

// 3. Configure convergence detection
cachingKMeans
  .setConvergenceTol(0.001)
  .setMaxIterations(100)
  .setConvergenceDetector(new RelativeConvergenceDetector())

// 4. Run with optimized settings
val model = cachingKMeans.fit(data)
```

## Common Issues and Solutions

1. **Slow Convergence**
   - Reduce dimensions using `RandomIndexEmbedding`
   - Increase `batchSize` for mini-batch processing
   - Try different initialization strategies

2. **Memory Issues**
   - Use `StreamingKMeans` for large datasets
   - Enable disk caching with `CachingKMeans`
   - Adjust mini-batch size

3. **Poor Cluster Quality**
   - Try different distance functions
   - Increase `initializationSteps`
   - Adjust `convergenceTol`

4. **Scaling Issues**
   - Use `KMeansParallel` for large datasets
   - Configure proper parallelism level
   - Enable caching strategically
