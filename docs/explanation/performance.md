---
title: Performance Tuning
---

# Performance Tuning

How to scale clustering to billions of points.

---

## Complexity Overview

| Factor | Impact | How to Reduce |
|--------|--------|---------------|
| n (points) | Linear | Mini-batch, sampling |
| k (clusters) | Linear | Elkan/Hamerly pruning |
| d (dimensions) | Linear | Dimensionality reduction |
| iterations | Linear | Better initialization, early stopping |

**Total:** O(n × k × d × iterations)

---

## Data Partitioning

### Optimal Partitions

```scala
// Rule of thumb: 100-200 partitions per executor core
val numPartitions = spark.sparkContext.defaultParallelism * 10
val repartitionedData = data.repartition(numPartitions)
```

### Avoid Skew

```scala
// Check partition sizes
data.rdd.mapPartitions(iter => Iterator(iter.size)).collect()
  .foreach(println)
```

---

## Assignment Strategy

The library automatically chooses the best strategy:

| Strategy | When Used | Complexity |
|----------|-----------|------------|
| **BroadcastUDF** | k < ~1000 | O(n × k) |
| **CrossJoin** | k large, SE only | O(n × k) but faster |
| **Elkan** | SE, k ≥ 5 | O(n × k) with pruning |

### Force a Strategy

```scala
new GeneralizedKMeans()
  .setAssignmentStrategy("crossJoin")  // or "broadcastUDF", "auto"
```

---

## Elkan Acceleration

For Squared Euclidean with k ≥ 5, Elkan's algorithm can skip 50-90% of distance computations.

**How it works:**
1. Track bounds on distance to assigned center
2. Track bounds on distance to other centers
3. Use triangle inequality to prove assignment unchanged
4. Only compute distances when bounds overlap

**Speedup:** 3-10x typical, more as convergence approaches

```scala
// Automatically enabled for SE with k >= 5
new GeneralizedKMeans()
  .setDivergence("squaredEuclidean")
  .setK(20)
```

---

## Mini-Batch K-Means

For very large datasets, update centers using random samples:

```scala
new MiniBatchKMeans()
  .setK(100)
  .setBatchSize(10000)  // Points per iteration
  .setMaxIter(100)
```

**Trade-off:** Faster convergence, slightly worse final quality

---

## Initialization

### K-Means||

Default initialization, parallelizable, good quality:

```scala
new GeneralizedKMeans()
  .setInitMode("k-means||")
  .setInitSteps(2)  // 2-5 is usually enough
```

### Random

Faster but lower quality:

```scala
new GeneralizedKMeans()
  .setInitMode("random")
```

---

## Checkpointing

For long-running jobs, checkpoint to avoid recomputation:

```scala
spark.sparkContext.setCheckpointDir("hdfs:///checkpoints")

new GeneralizedKMeans()
  .setCheckpointInterval(10)  // Every 10 iterations
```

---

## Memory Tuning

### Broadcast Threshold

Centers are broadcast to all executors. For very large k×d:

```scala
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100m")
```

### Caching

Cache input data for iterative algorithms:

```scala
val cachedData = data.cache()
cachedData.count()  // Force materialization

val model = new GeneralizedKMeans().fit(cachedData)

cachedData.unpersist()
```

---

## Scaling Guidelines

| Data Size | Recommendation |
|-----------|----------------|
| < 1M points | Standard GeneralizedKMeans |
| 1M - 100M | Enable checkpointing, optimize partitions |
| 100M - 1B | Mini-batch, consider sampling for init |
| > 1B | Mini-batch + streaming, hierarchical |

---

## Benchmarks

Typical performance on 100-node cluster:

| Dataset | k | Time | Notes |
|---------|---|------|-------|
| 10M × 100 | 100 | 2 min | Standard |
| 100M × 100 | 100 | 15 min | With checkpointing |
| 1B × 100 | 100 | 2 hr | Mini-batch |
| 10M × 100 | 10000 | 10 min | Elkan acceleration |

---

## Profiling

### Monitor via Spark UI

- Check stage times
- Look for skew in task durations
- Monitor shuffle read/write

### Log Iteration Details

```scala
import org.apache.log4j.{Level, Logger}
Logger.getLogger("com.massivedatascience").setLevel(Level.DEBUG)
```

---

## Common Issues

### Out of Memory

**Symptom:** Executor OOM during broadcast

**Fix:**
```scala
// Reduce broadcast size
.setAssignmentStrategy("crossJoin")
// Or increase executor memory
spark.conf.set("spark.executor.memory", "8g")
```

### Slow Convergence

**Symptom:** Many iterations, small improvements

**Fix:**
```scala
// Increase tolerance
.setTol(1e-3)
// Or use mini-batch
new MiniBatchKMeans()
```

### Skewed Clusters

**Symptom:** One cluster has most points

**Fix:**
```scala
// Use balanced k-means
new BalancedKMeans().setBalanceMode("soft")
// Or multiple random restarts
```

---

[Back to Explanation](index.html) | [Home](../)
