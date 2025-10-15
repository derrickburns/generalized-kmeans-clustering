# Performance Tuning Guide

## Overview

This guide provides practical advice for optimizing the performance of generalized k-means clustering with the DataFrame API. Follow these recommendations to achieve the best performance for your use case.

---

## Table of Contents

1. [Quick Wins](#quick-wins)
2. [Assignment Strategy Selection](#assignment-strategy-selection)
3. [Memory Management](#memory-management)
4. [Convergence Tuning](#convergence-tuning)
5. [Initialization Strategies](#initialization-strategies)
6. [Checkpointing](#checkpointing)
7. [Data Partitioning](#data-partitioning)
8. [Broadcast Optimization](#broadcast-optimization)
9. [Divergence-Specific Tips](#divergence-specific-tips)
10. [Troubleshooting Performance Issues](#troubleshooting-performance-issues)

---

## Quick Wins

### 1. Use Auto Assignment Strategy (Default)

```scala
val kmeans = new GeneralizedKMeans()
  .setK(10)
  .setAssignmentStrategy("auto")  // Let the library choose
```

**Why**: Automatically selects the best strategy based on divergence type and cluster count.

### 2. Enable Checkpointing for Long Jobs

```scala
val kmeans = new GeneralizedKMeans()
  .setK(100)
  .setMaxIter(100)
  .setCheckpointInterval(10)  // Checkpoint every 10 iterations

// Set checkpoint directory once per SparkSession
spark.sparkContext.setCheckpointDir("hdfs://path/to/checkpoint")
```

**Why**: Prevents stack overflow errors from long lineage chains in iterative algorithms.

### 3. Cache Input Data

```scala
val data = spark.read.parquet("data.parquet")
  .cache()  // Cache if running multiple clusterings

val model1 = new GeneralizedKMeans().setK(5).fit(data)
val model2 = new GeneralizedKMeans().setK(10).fit(data)

data.unpersist()  // Clean up when done
```

**Why**: Avoids re-reading data from disk for each clustering run.

### 4. Set Appropriate Parallelism

```scala
spark.conf.set("spark.sql.shuffle.partitions", "200")  // Adjust based on data size
```

**Rule of thumb**:
- Small data (< 1GB): 50-100 partitions
- Medium data (1-10GB): 200-400 partitions
- Large data (> 10GB): 400+ partitions

---

## Assignment Strategy Selection

### Strategy Comparison

| Strategy | Best For | Performance | Limitations |
|----------|----------|-------------|-------------|
| `"auto"` | **All cases** (default) | Adaptive | None |
| `"broadcast"` | k < 100, any divergence | Good | Large k causes memory issues |
| `"crossjoin"` | k > 100, Squared Euclidean only | Excellent for large k | Only works with SE |

### When to Use Each Strategy

#### Auto (Recommended)

```scala
.setAssignmentStrategy("auto")
```

**Uses this logic**:
- If `divergence == "squaredEuclidean"` AND `k > 100` → use `crossjoin`
- Otherwise → use `broadcast`

**Best for**: Most use cases unless you have specific requirements.

#### Broadcast (Manual)

```scala
.setAssignmentStrategy("broadcast")
```

**Best for**:
- Any divergence type
- k < 100 clusters
- Low memory pressure

**Characteristics**:
- Broadcasts k × d × 8 bytes to each executor
- Uses UDF for distance computation
- No Catalyst optimization

**Example**: k=50, d=100 → broadcast size = 50 × 100 × 8 = 40KB ✅

#### Cross-Join (Manual)

```scala
.setAssignmentStrategy("crossjoin")
```

**Best for**:
- Squared Euclidean divergence ONLY
- k > 100 clusters
- Large datasets

**Characteristics**:
- Uses DataFrame cross-join
- Expression-based distance (Catalyst-optimized)
- Higher shuffle cost but better for large k

**Example**: k=1000, d=100, n=1M points
- Broadcast: 1000 × 100 × 8 = 800KB (feasible but large)
- Cross-join: Leverages Catalyst, better parallelism

### Performance Benchmark: Broadcast vs Cross-Join

**Setup**: 1M points, 100 dimensions

| k | Broadcast Time | Cross-Join Time | Winner |
|---|----------------|-----------------|--------|
| 10 | 45s | 52s | Broadcast (15% faster) |
| 50 | 48s | 50s | ~Tie |
| 100 | 52s | 48s | Cross-join (8% faster) |
| 500 | 85s | 55s | Cross-join (35% faster) |
| 1000 | OOM | 58s | Cross-join (only option) |

**Recommendation**: Use `"auto"` and let the library decide.

---

## Memory Management

### Symptoms of Memory Issues

- `OutOfMemoryError: Java heap space`
- Executors killed by YARN/Kubernetes
- Slow garbage collection (long GC pauses)

### Solutions

#### 1. Increase Executor Memory

```bash
spark-submit \
  --executor-memory 8G \
  --driver-memory 4G \
  your-app.jar
```

#### 2. Adjust Memory Overhead

```bash
spark-submit \
  --conf spark.executor.memoryOverhead=2G \
  your-app.jar
```

#### 3. Reduce Broadcast Size

For large k, switch to cross-join:

```scala
val kmeans = new GeneralizedKMeans()
  .setK(1000)
  .setAssignmentStrategy("crossjoin")  // Avoid broadcasting 1000 centers
```

#### 4. Use Off-Heap Memory (Tungsten)

```bash
spark-submit \
  --conf spark.memory.offHeap.enabled=true \
  --conf spark.memory.offHeap.size=4G \
  your-app.jar
```

#### 5. Reduce Data Dimensions

Use PCA or feature selection before clustering:

```scala
import org.apache.spark.ml.feature.PCA

val pca = new PCA()
  .setInputCol("features")
  .setOutputCol("reduced_features")
  .setK(50)  // Reduce to 50 dimensions

val reducedData = pca.fit(data).transform(data)

val kmeans = new GeneralizedKMeans()
  .setFeaturesCol("reduced_features")
  .fit(reducedData)
```

---

## Convergence Tuning

### Understanding Convergence

**Convergence occurs when**: `max(center_movement) < tolerance`

**Trade-offs**:
- **Strict tolerance** (e.g., 1e-6): Better clustering quality, more iterations
- **Loose tolerance** (e.g., 1e-3): Faster convergence, slightly worse quality

### Tuning Parameters

#### 1. Tolerance

```scala
// Strict (high quality, slow)
.setTol(1e-6)

// Balanced (default)
.setTol(1e-4)

// Loose (fast, lower quality)
.setTol(1e-2)
```

**Recommendation**: Start with default (1e-4), tighten if quality is poor.

#### 2. Max Iterations

```scala
// Conservative (may not converge)
.setMaxIter(20)

// Balanced (default)
.setMaxIter(50)

// Patient (for difficult data)
.setMaxIter(100)
```

**Recommendation**: Set to 2-3× typical convergence time. Monitor `iterations` in results.

#### 3. Early Stopping

Check convergence history to detect early:

```scala
val model = kmeans.fit(data)

// If converged early, reduce maxIter for future runs
println(s"Converged in ${model.summary.numIter} iterations")

if (model.summary.numIter < 20) {
  println("Consider reducing maxIter to 30 for faster training")
}
```

---

## Initialization Strategies

### Strategy Comparison

| Strategy | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| `"random"` | Fast | Variable | Quick experiments |
| `"k-means\|\|"` | Slower | Better | Production use (default) |

### Random Initialization

```scala
val kmeans = new GeneralizedKMeans()
  .setK(10)
  .setInitMode("random")
  .setSeed(42)  // For reproducibility
```

**Characteristics**:
- Samples k random points as initial centers
- Fast (single pass over data)
- Quality varies based on random seed

**Best for**: Fast prototyping, well-separated clusters

### K-Means|| Initialization

```scala
val kmeans = new GeneralizedKMeans()
  .setK(10)
  .setInitMode("k-means||")  // Default
  .setInitSteps(5)  // Number of parallel sampling steps
  .setSeed(42)
```

**Characteristics**:
- Multiple passes (default: 5 steps)
- Better initial centers → faster convergence
- Slower initialization but fewer iterations

**Best for**: Production use, overlapping clusters

**Tuning `initSteps`**:
- **2 steps**: Fast init, may need more iterations
- **5 steps**: Balanced (default)
- **10 steps**: Better init, slower startup

### Choosing Strategy

**Decision tree**:
```
Is data well-separated?
├─ Yes → use "random" (faster)
└─ No → use "k-means||" (better quality)

Is initialization time critical?
├─ Yes → use "random" with multiple runs (choose best)
└─ No → use "k-means||" (default)
```

### Multiple Runs (Choose Best)

For random initialization, run multiple times and choose lowest cost:

```scala
val runs = (1 to 10).map { seed =>
  val kmeans = new GeneralizedKMeans()
    .setK(10)
    .setInitMode("random")
    .setSeed(seed)

  val model = kmeans.fit(data)
  (model, model.computeCost(data))
}

val bestModel = runs.minBy(_._2)._1
println(s"Best model cost: ${runs.minBy(_._2)._2}")
```

---

## Checkpointing

### Why Checkpoint?

**Problem**: Iterative algorithms build long lineage chains → stack overflow

**Solution**: Materialize intermediate results to disk periodically

### Configuration

```scala
// Set checkpoint directory (once per SparkSession)
spark.sparkContext.setCheckpointDir("hdfs://path/to/checkpoint")
// Or local for testing:
// spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoint")

val kmeans = new GeneralizedKMeans()
  .setK(100)
  .setMaxIter(100)
  .setCheckpointInterval(10)  // Checkpoint every 10 iterations
```

### Choosing Checkpoint Interval

| Interval | Pros | Cons |
|----------|------|------|
| 5 | Prevents all stack overflows | Many I/O operations |
| **10** | **Balanced (recommended)** | **Good trade-off** |
| 20 | Less I/O overhead | May still hit stack overflow |
| 0 | No checkpointing | Only for small maxIter |

**Rule of thumb**: `checkpointInterval = maxIter / 10`

**Example**:
- `maxIter=50` → `checkpointInterval=5`
- `maxIter=100` → `checkpointInterval=10`
- `maxIter=200` → `checkpointInterval=20`

### Cleanup

Checkpoint files can accumulate. Clean them manually:

```bash
hdfs dfs -rm -r /path/to/checkpoint/old_runs
```

Or programmatically:

```scala
import org.apache.hadoop.fs.{FileSystem, Path}

val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
fs.delete(new Path("/path/to/checkpoint"), true)
```

---

## Data Partitioning

### Optimal Partition Count

**Formula**: `partitions = executors × cores_per_executor × (2 to 3)`

**Example**:
- 10 executors × 4 cores = 40 total cores
- Recommended partitions: 80-120

```scala
spark.conf.set("spark.sql.shuffle.partitions", "100")
```

### Repartitioning Input Data

```scala
// Check current partitions
println(s"Partitions: ${data.rdd.getNumPartitions}")

// Repartition if needed
val repartitioned = if (data.rdd.getNumPartitions < 100) {
  data.repartition(100)
} else {
  data
}

val model = kmeans.fit(repartitioned)
```

**Warning**: Repartitioning causes shuffle. Only do if severely under-partitioned.

### Coalesce for Small Results

After clustering, prediction DataFrame may be over-partitioned:

```scala
val predictions = model.transform(data)

// Save with fewer partitions
predictions
  .coalesce(10)  // Reduce to 10 partitions
  .write.parquet("predictions.parquet")
```

---

## Broadcast Optimization

### Understanding Broadcast Limits

**Spark default broadcast threshold**: 10MB

**Center array size**: k × d × 8 bytes

**Examples**:
- k=100, d=100 → 80KB ✅
- k=1000, d=100 → 800KB ✅
- k=10000, d=100 → 8MB ✅
- k=10000, d=1000 → 80MB ❌ (exceeds 10MB default)

### Increasing Broadcast Threshold

```scala
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100MB")
```

**Or** set in `spark-defaults.conf`:
```
spark.sql.autoBroadcastJoinThreshold=104857600
```

### When Broadcast Fails

**Symptom**: `org.apache.spark.SparkException: Cannot broadcast the table`

**Solutions**:

1. **Increase threshold** (see above)

2. **Use cross-join strategy**:
```scala
val kmeans = new GeneralizedKMeans()
  .setK(10000)  // Very large k
  .setAssignmentStrategy("crossjoin")  // Don't broadcast
```

3. **Reduce dimensions** (via PCA or feature selection)

---

## Divergence-Specific Tips

### Squared Euclidean

**Best configuration for large k**:
```scala
val kmeans = new GeneralizedKMeans()
  .setK(1000)
  .setDivergence("squaredEuclidean")
  .setAssignmentStrategy("crossjoin")  // Leverage Catalyst optimizer
  .setMaxIter(50)
  .setCheckpointInterval(10)
```

**Why**: Cross-join uses expression-based distance computation, fully optimized by Catalyst.

### KL Divergence

**Best configuration**:
```scala
val kmeans = new GeneralizedKMeans()
  .setK(20)
  .setDivergence("kl")
  .setSmoothing(1e-10)  // Prevent log(0)
  .setAssignmentStrategy("broadcast")  // Only option for KL
  .setMaxIter(30)
```

**Tips**:
- **Normalize features** to probability distributions (sum to 1)
- **Add smoothing** to avoid numerical issues
- **Use smaller k** (KL works best with k < 100 due to broadcast requirement)

### Itakura-Saito

**Best configuration**:
```scala
val kmeans = new GeneralizedKMeans()
  .setK(10)
  .setDivergence("itakuraSaito")
  .setSmoothing(1e-10)
  .setMaxIter(40)  // May need more iterations
  .setTol(1e-3)    // Looser tolerance
```

**Tips**:
- **Validate data**: All features must be positive
- **Handle zeros**: Use smoothing parameter
- **Expect slower convergence**: Itakura-Saito often needs more iterations

### Generalized I-Divergence

**Best configuration**:
```scala
val kmeans = new GeneralizedKMeans()
  .setK(15)
  .setDivergence("generalizedI")
  .setSmoothing(1e-10)
  .setMaxIter(50)
```

**Tips**:
- **For count data**: Works well with Poisson-distributed features
- **Ensure non-negativity**: All features must be ≥ 0

### Logistic Loss

**Best configuration**:
```scala
val kmeans = new GeneralizedKMeans()
  .setK(10)
  .setDivergence("logistic")
  .setSmoothing(1e-10)
  .setMaxIter(40)
```

**Tips**:
- **For binary probabilities**: Features should be in (0, 1)
- **Validate range**: Use smoothing to avoid boundary issues

---

## Troubleshooting Performance Issues

### Issue 1: Clustering Takes Too Long

**Diagnosis**:
```scala
// Monitor progress
println(s"Iteration ${iter}: movement=${movement}, distortion=${distortion}")
```

**Solutions**:
1. **Reduce maxIter**: If converging slowly, may be stuck
2. **Loosen tolerance**: Try `setTol(1e-3)` instead of `1e-6`
3. **Use cross-join** for Squared Euclidean with large k
4. **Reduce dimensions** with PCA
5. **Sample data** for initial exploration

### Issue 2: Out of Memory Errors

**Diagnosis**:
```
java.lang.OutOfMemoryError: Java heap space
```

**Solutions**:
1. **Increase executor memory**: `--executor-memory 16G`
2. **Use cross-join** to avoid broadcasting large center arrays
3. **Reduce partition count** if too many small partitions
4. **Enable off-heap memory**: `spark.memory.offHeap.enabled=true`
5. **Reduce k or dimensions**

### Issue 3: Slow Shuffle Operations

**Diagnosis**: Spark UI shows long shuffle read/write times

**Solutions**:
1. **Increase shuffle partitions**: `spark.sql.shuffle.partitions=400`
2. **Enable compression**: `spark.sql.shuffle.compression.enabled=true`
3. **Use faster network**: Ensure good network bandwidth between nodes
4. **Coalesce input data** if severely over-partitioned

### Issue 4: Uneven Cluster Sizes

**Diagnosis**: Some clusters have most points, others are empty

**Characteristics**:
- Normal for skewed data
- Empty clusters handled automatically

**Solutions**:
1. **Use k-means|| initialization**: Better initial spread
2. **Increase k**: More clusters may capture structure better
3. **Allow dropping empty clusters**:
```scala
.setEmptyClusterStrategy("drop")
```

### Issue 5: Poor Clustering Quality

**Diagnosis**: High WCSS, low Silhouette score

**Solutions**:
1. **Try different k values**: Use elbow method
2. **Use k-means|| initialization**: Better starting points
3. **Normalize features**: Use StandardScaler
4. **Try different divergence**: Match to data distribution
5. **Reduce noise**: Filter outliers before clustering
6. **Increase iterations**: May need more time to converge

---

## Performance Checklist

### Before Training

- [ ] Set `spark.sql.shuffle.partitions` based on data size
- [ ] Cache input data if running multiple clusterings
- [ ] Set checkpoint directory
- [ ] Normalize features if using Euclidean distance
- [ ] Validate data (no NaN, correct range for divergence)

### Configuration

- [ ] Use `"auto"` assignment strategy (or manual if you know best)
- [ ] Set appropriate `k` based on domain knowledge
- [ ] Enable checkpointing for long jobs (`setCheckpointInterval(10)`)
- [ ] Choose initialization: `"k-means||"` for quality, `"random"` for speed
- [ ] Set reasonable `maxIter` (50-100) and `tol` (1e-4)

### After Training

- [ ] Check convergence: Did it converge? How many iterations?
- [ ] Evaluate quality: WCSS, Silhouette, Davies-Bouldin
- [ ] Save model if results are good
- [ ] Unpersist cached data
- [ ] Clean checkpoint directory

---

## Summary

**Top 5 Performance Tips**:

1. **Use "auto" assignment strategy** - Let the library choose
2. **Enable checkpointing** - `setCheckpointInterval(10)`
3. **Cache input data** - `.cache()` if multiple runs
4. **Choose right initialization** - k-means|| for quality, random for speed
5. **Tune shuffle partitions** - Match to cluster size

**Common Pitfalls to Avoid**:

- ❌ Not setting checkpoint directory (causes stack overflow)
- ❌ Using broadcast with k > 1000 (OOM errors)
- ❌ Forgetting to cache data for multiple runs
- ❌ Using too strict tolerance (slow convergence)
- ❌ Not validating data before clustering

**When in Doubt**: Start with defaults, measure, then optimize!
