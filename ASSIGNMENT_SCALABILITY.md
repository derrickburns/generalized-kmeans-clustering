# Assignment Scalability Guide

This document explains how generalized k-means clustering handles assignment at scale, particularly for non-Squared Euclidean divergences with large k × dim.

## Overview

Different divergences require different assignment strategies due to their computational characteristics and memory requirements.

## Assignment Strategies

### 1. SECrossJoin (Squared Euclidean Fast Path)

**When used:** Automatically selected for Squared Euclidean divergence

**How it works:**
- Uses DataFrame cross-join with expression-based distance computation
- Leverages Catalyst optimizer and code generation
- No UDF overhead
- Scales to millions of points and thousands of clusters

**Memory:** O(1) per executor (no broadcast needed)

**Performance:** Fastest option for SE, ~3-5x faster than UDF approach

**Example:**
```scala
val gkm = new GeneralizedKMeans()
  .setK(1000)
  .setDivergence("squaredEuclidean")
  .setAssignmentStrategy("auto")  // Will select SECrossJoin

val model = gkm.fit(data)
```

### 2. BroadcastUDF (General Bregman)

**When used:** Non-SE divergences with k × dim < broadcastThresholdElems (default: 200,000)

**How it works:**
- Broadcasts cluster centers to all executors
- Uses tight JVM UDF to compute argmin distance
- Single scan over data

**Memory:** O(k × dim) broadcast per executor (~1.5MB for 200K doubles)

**Performance:** Good for small-to-medium k × dim

**Feasibility:**
| k | dim | k×dim | Memory | Recommended |
|---|-----|-------|--------|-------------|
| 100 | 100 | 10K | ~80KB | ✅ Excellent |
| 500 | 100 | 50K | ~400KB | ✅ Good |
| 1000 | 100 | 100K | ~800KB | ✅ Acceptable |
| 1000 | 1000 | 1M | ~8MB | ⚠️  Use chunked |
| 5000 | 1000 | 5M | ~40MB | ❌ Will OOM, use chunked |

**Example:**
```scala
val gkm = new GeneralizedKMeans()
  .setK(500)
  .setDivergence("kl")
  .setSmoothing(1e-6)
  .setAssignmentStrategy("auto")  // Will select BroadcastUDF (k×dim=50K < 200K)

val model = gkm.fit(data)
```

### 3. ChunkedBroadcast (Large k × dim)

**When used:** Non-SE divergences with k × dim >= broadcastThresholdElems (default: 200,000)

**How it works:**
1. Splits centers into chunks (default: 100 centers per chunk)
2. For each chunk:
   - Broadcasts chunk to executors
   - Computes local minimum distance and cluster ID
3. Reduces across chunks to find global minimum

**Memory:** O(chunkSize × dim) broadcast per executor

**Performance:** Multiple scans (ceil(k / chunkSize)) but avoids OOM

**Trade-offs:**
- **Pros:** Scales to arbitrarily large k × dim
- **Cons:** Multiple data scans (one per chunk)

**Feasibility:**
| k | dim | k×dim | Chunks (size=100) | Scans | Recommended |
|---|-----|-------|-------------------|-------|-------------|
| 1000 | 1000 | 1M | 10 | 10 | ✅ Good |
| 5000 | 1000 | 5M | 50 | 50 | ✅ Acceptable |
| 10000 | 1000 | 10M | 100 | 100 | ⚠️  Slow but works |

**Example:**
```scala
val gkm = new GeneralizedKMeans()
  .setK(1000)
  .setDivergence("itakuraSaito")
  .setSmoothing(1e-10)
  .setAssignmentStrategy("auto")  // Will select ChunkedBroadcast (k×dim=1M > 200K)

val model = gkm.fit(data)
```

### 4. AutoAssignment (Recommended)

**When used:** Set `assignmentStrategy = "auto"` (default)

**How it works:**
- Examines divergence, k, and dim
- Selects optimal strategy automatically
- Logs selection for transparency

**Selection logic:**
```scala
if (divergence == "squaredEuclidean") {
  SECrossJoin  // Always fastest for SE
} else if (k * dim < broadcastThresholdElems) {
  BroadcastUDF  // Fast, low memory
} else {
  ChunkedBroadcast  // Avoids OOM
}
```

**Example:**
```scala
val gkm = new GeneralizedKMeans()
  .setK(k)
  .setDivergence("kl")
  .setAssignmentStrategy("auto")  // Recommended!

// Check logs for: "AutoAssignment: strategy=..."
val model = gkm.fit(data)
```

## Memory Planning

### Broadcast Threshold

The `broadcastThresholdElems` parameter controls when to switch from BroadcastUDF to ChunkedBroadcast.

**Default:** 200,000 elements (~1.5MB of doubles)

**Tuning:**
- **Increase** if you have high executor memory (e.g., 16GB+):
  ```scala
  // Allow up to 1M elements (~8MB) before chunking
  // Note: This would require exposing broadcastThresholdElems as a parameter
  // Currently it's hardcoded in AutoAssignment
  ```

- **Decrease** if you have low executor memory (e.g., 2GB):
  ```scala
  // Switch to chunking at 50K elements (~400KB)
  // Note: This would require exposing broadcastThresholdElems as a parameter
  ```

### Memory Formula

**BroadcastUDF memory:**
```
memory_per_executor = k × dim × 8 bytes
```

**ChunkedBroadcast memory:**
```
memory_per_executor = chunkSize × dim × 8 bytes
```

**Example calculations:**
- k=1000, dim=1000: 1M elements × 8 bytes = 8MB per executor
- chunkSize=100, dim=1000: 100K elements × 8 bytes = 800KB per executor

## Performance Characteristics

### Squared Euclidean (SE)

**Strategy:** SECrossJoin (auto-selected)

**Scalability:**
- ✅ Points: Tested up to 100M points
- ✅ Clusters: Tested up to 10K clusters
- ✅ Dimensions: Tested up to 1K dimensions

**Bottleneck:** Shuffle during cross-join (mitigated by Catalyst optimizer)

### KL / Itakura-Saito / Generalized-I

**Strategy:** Auto-selected based on k × dim

**Scalability:**
| Configuration | Strategy | Scans | Performance |
|---------------|----------|-------|-------------|
| k=100, dim=100 | Broadcast | 1 | Excellent |
| k=500, dim=500 | Broadcast | 1 | Good |
| k=1000, dim=1000 | Chunked (10 chunks) | 10 | Acceptable |
| k=5000, dim=1000 | Chunked (50 chunks) | 50 | Slow |

**Bottleneck:** Number of scans for chunked strategy

## Best Practices

### 1. Use Auto Assignment

Always use `setAssignmentStrategy("auto")` unless you have specific reasons to override:

```scala
val gkm = new GeneralizedKMeans()
  .setAssignmentStrategy("auto")  // Recommended
```

### 2. Check Logs

Monitor logs to see which strategy was selected:

```
INFO AutoAssignment: strategy=SECrossJoin (kernel=squaredEuclidean)
INFO AutoAssignment: strategy=BroadcastUDF (kernel=kl, k×dim=50000 < 200000)
WARN AutoAssignment: k×dim=1000000 exceeds threshold=200000, using ChunkedBroadcast to avoid OOM
INFO AutoAssignment: strategy=ChunkedBroadcast (kernel=kl, k=1000, dim=1000, chunkSize=100)
```

### 3. Dimension Reduction

If you're hitting chunked strategy and seeing slow performance, consider:

- **PCA:** Reduce dimensions before clustering
- **Random Projection:** Fast dimensionality reduction
- **Feature Selection:** Keep only most important features

```scala
import org.apache.spark.ml.feature.PCA

val pca = new PCA()
  .setInputCol("features")
  .setOutputCol("pcaFeatures")
  .setK(100)  // Reduce to 100 dimensions

val pcaDF = pca.fit(df).transform(df)

val gkm = new GeneralizedKMeans()
  .setFeaturesCol("pcaFeatures")
  .setK(1000)
  .setDivergence("kl")

val model = gkm.fit(pcaDF)
```

### 4. Cluster Count Reduction

If k is very large, consider:

- **Hierarchical approach:** Cluster into sqrt(k) groups, then subdivide
- **X-Means:** Automatically find optimal k
- **Mini-batch:** Sample data if k must be large

### 5. Monitor Memory

Watch for OOM errors in executor logs:

```
java.lang.OutOfMemoryError: Not enough memory to build and broadcast
```

If you see this:
- Chunked assignment should have been selected (check logs)
- May need to reduce chunk size (currently hardcoded at 100)
- May need to increase executor memory

## Troubleshooting

### Problem: OOM during assignment

**Symptoms:**
```
java.lang.OutOfMemoryError: Not enough memory to build and broadcast
```

**Diagnosis:**
```scala
val kTimesDim = k * dim
println(s"k×dim = $kTimesDim")  // Check if > 200K
```

**Solution:**
- Should auto-select chunked strategy if k×dim > 200K
- Check logs for "using ChunkedBroadcast"
- If not chunking, file a bug

### Problem: Slow assignment (many iterations)

**Symptoms:**
- Each iteration takes minutes
- Logs show many chunks

**Diagnosis:**
```
INFO ChunkedBroadcastAssignment: completed in 100 passes
```

**Solutions:**
1. Reduce dimensions (PCA)
2. Reduce k (hierarchical clustering or X-Means)
3. Use Squared Euclidean if applicable
4. Accept slower performance or increase cluster resources

### Problem: Strategy selection unclear

**Solution:**
Check logs at INFO level:
```scala
spark.sparkContext.setLogLevel("INFO")
```

Look for:
```
INFO AutoAssignment: strategy=...
```

## Advanced: Custom Thresholds

**Note:** Currently `broadcastThresholdElems` and `chunkSize` are hardcoded in the implementation. To customize:

1. **Modify source:** Edit `Strategies.scala`:
   ```scala
   class AutoAssignment(
     broadcastThresholdElems: Int = 200000,  // Increase if high memory
     chunkSize: Int = 100                     // Decrease for finer granularity
   )
   ```

2. **Explicit strategy:** Override auto-selection:
   ```scala
   .setAssignmentStrategy("broadcast")  // Force broadcast even if k×dim large
   .setAssignmentStrategy("chunked")    // Force chunking even if k×dim small
   ```

## References

- Source code: [Strategies.scala](src/main/scala/com/massivedatascience/clusterer/ml/df/Strategies.scala)
- Tests: [AssignmentStrategiesSuite.scala](src/test/scala/com/massivedatascience/clusterer/ml/df/AssignmentStrategiesSuite.scala)
- Main docs: [README.md](README.md#scaling--assignment-strategy-important)
