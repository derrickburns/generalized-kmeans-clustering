# Technical Reference: Generalized K-Means Clustering

> 📖 **Type**: Information-oriented reference
>
> **Format**: Technical specification
>
> **Level**: Advanced

[← Back to Index](INDEX.md) | [Explanation →](EXPLANATION.md)

---

This reference provides comprehensive technical details about the API, classes, and configuration options.

> 🔍 **Looking for something else?**
> - New to the library? Start with the [Tutorial](TUTORIAL.md)
> - Need to solve a problem? Check the [How-To Guide](HOW-TO.md)
> - Want to understand concepts? Read the [Explanation](EXPLANATION.md)

> ℹ️ **Note**: This is a technical reference. For practical examples, see the [How-To Guide](HOW-TO.md).

## Core Components

### KMeans Classes

#### `KMeans`
Main implementation class for K-means clustering.

```scala
class KMeans extends Serializable with KMeansConfig {
  def setK(k: Int): this.type
  def setMaxIterations(maxIterations: Int): this.type
  def setInitializationStrategy(strategy: InitializationStrategy): this.type
  def setDistanceFunction(distance: PointOps): this.type
  def fit(data: RDD[Vector]): KMeansModel
}
```

#### `StreamingKMeans`
Implementation for streaming K-means clustering.

```scala
class StreamingKMeans extends Serializable with KMeansConfig {
  def setK(k: Int): this.type
  def setDecayFactor(factor: Double): this.type
  def setHalfLife(halfLife: Double): this.type
  def update(data: RDD[Vector], weight: Double): this.type
  def latestModel(): KMeansModel
}
```

### Distance Functions

#### `BregmanPointOps`
Implementation of Bregman divergence distance metrics.

```scala
class BregmanPointOps(divergence: BregmanDivergence) extends PointOps {
  def distance(x: Vector, y: Vector): Double
  def combine(x: Vector, y: Vector, weight: Double): Vector
}
```

### Initialization Strategies

#### `KMeansPlusPlus`
Smart initialization strategy that improves clustering quality.

```scala
object KMeansPlusPlus extends InitializationStrategy {
  def init(data: RDD[Vector], k: Int, distance: PointOps): Array[Vector]
}
```

#### `RandomInitialization`
Simple random initialization of cluster centers.

```scala
object RandomInitialization extends InitializationStrategy {
  def init(data: RDD[Vector], k: Int, distance: PointOps): Array[Vector]
}
```

## Configuration Options

### KMeans Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| k | Int | Required | Number of clusters |
| maxIterations | Int | 20 | Maximum iterations |
| convergenceTol | Double | 1e-4 | Convergence tolerance |
| initializationStrategy | InitializationStrategy | KMeansPlusPlus | Strategy for initializing centers |
| distanceFunction | PointOps | EuclideanPointOps | Distance metric |
| batchSize | Int | 1000 | Size of mini-batches |

### StreamingKMeans Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| k | Int | Required | Number of clusters |
| decayFactor | Double | 1.0 | Rate of forgetting old data |
| halfLife | Double | 0.0 | Time for data weight to halve |
| windowSize | Int | 1 | Size of sliding window |

## Data Types

### Vector Types

```scala
// Dense Vector
class DenseVector(values: Array[Double]) extends Vector {
  def size: Int
  def apply(i: Int): Double
  def toArray: Array[Double]
}

// Sparse Vector
class SparseVector(size: Int, indices: Array[Int], values: Array[Double]) extends Vector {
  def nnz: Int  // Number of non-zero elements
  def apply(i: Int): Double
  def toArray: Array[Double]
}
```

## Performance Considerations

### Memory Usage

- Dense vectors: O(n) where n is the dimension
- Sparse vectors: O(nnz) where nnz is number of non-zero elements
- Cluster centers: O(k * d) where k is number of clusters, d is dimension

### Time Complexity

| Operation | Time Complexity |
|-----------|----------------|
| Distance calculation | O(d) for dense, O(nnz) for sparse |
| Center update | O(n * k * d) per iteration |
| KMeans++ init | O(n * k * d) |
| Mini-batch update | O(b * k * d) where b is batch size |

## Error Handling

### Common Exceptions

```scala
// Invalid number of clusters
IllegalArgumentException: requirement failed: Number of clusters must be positive

// Dimension mismatch
IllegalArgumentException: requirement failed: Vector dimensions do not match

// Empty dataset
IllegalArgumentException: requirement failed: RDD is empty

// Convergence failure
SparkException: Job aborted due to stage failure
```

## Integration

### Spark Configuration

```scala
// Recommended Spark configuration for large datasets
spark.conf.set("spark.memory.fraction", "0.8")
spark.conf.set("spark.memory.storageFraction", "0.3")
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
spark.conf.set("spark.kryoserializer.buffer.max", "1024m")
```

### Hadoop Integration

```scala
// Reading/Writing models
val model = KMeansModel.load(sc, "hdfs://path/to/model")
model.save(sc, "hdfs://path/to/save")
```

## Monitoring and Metrics

### Available Metrics

- Iteration count
- Convergence measure
- Cluster sizes
- Within-cluster sum of squared distances
- Silhouette score

```scala
// Example: Accessing metrics
val stats = model.computeStats(data)
println(s"Within Set Sum of Squared Errors = ${stats.cost}")
```

## Scaladoc Documentation

Full API documentation is available in the generated Scaladoc at:
`target/scala-2.12/api/index.html`

Key packages:
- `com.massivedatascience.clusterer`
- `com.massivedatascience.divergence`
- `com.massivedatascience.transforms`
- `com.massivedatascience.linalg`

## Version Compatibility

| Library Version | Scala Version | Spark Version | MLlib Version |
|----------------|---------------|---------------|---------------|
| 2.0.x | 2.12 | 3.x | Latest |
| 1.x.x | 2.11 | 2.x | K-Means (1.1.0), Streaming K-Means (1.2.0) |
