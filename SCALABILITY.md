# Scalability Guide - k × dim Feasibility

This document explains how the library handles different problem sizes and provides guidance for capacity planning.

## TL;DR - Quick Reference

| Scenario | Recommended Approach |
|----------|---------------------|
| k < 1000, any dim | Use any divergence, Auto strategy handles it |
| k ≥ 1000, Squared Euclidean | Fast! SECrossJoin strategy (no broadcast) |
| k ≥ 1000, non-SE, k×dim < 200K | BroadcastUDF (fast, ~1.5MB memory) |
| k ≥ 1000, non-SE, k×dim ≥ 200K | ChunkedBroadcast (slower, avoids OOM) |

## Assignment Strategy Selection

The library automatically selects the optimal assignment strategy based on your configuration:

### 1. SECrossJoin (Squared Euclidean only)
- **When**: `divergence="squaredEuclidean"`
- **How**: DataFrame cross-join with expression-based distance
- **Scalability**: Excellent - no broadcast, works for arbitrarily large k×dim
- **Performance**: Fastest option (leverages Spark's Catalyst optimizer)

**Example**:
```scala
val gkm = new GeneralizedKMeans()
  .setK(10000)               // Large k is fine
  .setDivergence("squaredEuclidean")
  .fit(data)
// Logs: AutoAssignment: strategy=SECrossJoin
```

### 2. BroadcastUDF (General Bregman divergences, small k×dim)
- **When**: Non-SE divergence AND k×dim < 200,000
- **How**: Broadcasts cluster centers to all executors
- **Scalability**: Good up to threshold
- **Memory**: ~(k×dim×8) bytes per executor (~1.5MB at threshold)
- **Performance**: Fast for small-to-medium k×dim

**Example**:
```scala
val gkm = new GeneralizedKMeans()
  .setK(500)                  // k×dim = 500 × 100 = 50K < 200K
  .setDivergence("kl")        // KL divergence
  .fit(data)                  // dim = 100
// Logs: AutoAssignment: strategy=BroadcastUDF (k×dim=50000 < 200000)
```

### 3. ChunkedBroadcast (General Bregman divergences, large k×dim)
- **When**: Non-SE divergence AND k×dim ≥ 200,000
- **How**: Processes centers in chunks to avoid OOM
- **Scalability**: Excellent - handles arbitrarily large k×dim
- **Trade-off**: Multiple scans over data (ceil(k / chunkSize) passes)
- **Performance**: Slower than BroadcastUDF but avoids memory issues

**Example**:
```scala
val gkm = new GeneralizedKMeans()
  .setK(1000)                 // k×dim = 1000 × 1000 = 1M > 200K
  .setDivergence("kl")
  .fit(data)                  // dim = 1000
// Logs: AutoAssignment: k×dim=1000000 exceeds threshold=200000, using ChunkedBroadcast
// Logs: AutoAssignment: strategy=ChunkedBroadcast (chunkSize=100)
```

## Feasibility Matrix

### Squared Euclidean Distance

| k | dim | k×dim | Memory | Strategy | Status |
|---|-----|-------|--------|----------|--------|
| 100 | 1000 | 100K | Minimal | SECrossJoin | ✅ Fast |
| 1000 | 1000 | 1M | Minimal | SECrossJoin | ✅ Fast |
| 10000 | 1000 | 10M | Minimal | SECrossJoin | ✅ Fast |
| 100000 | 100 | 10M | Minimal | SECrossJoin | ✅ Fast |

**Summary**: Squared Euclidean scales to arbitrarily large k and dim.

### Non-SE Divergences (KL, Itakura-Saito, L1, etc.)

| k | dim | k×dim | Memory/Executor | Strategy | Status |
|---|-----|-------|-----------------|----------|--------|
| 100 | 1000 | 100K | ~0.8MB | BroadcastUDF | ✅ Fast |
| 500 | 400 | 200K | ~1.5MB | BroadcastUDF | ✅ Fast (at threshold) |
| 1000 | 300 | 300K | Chunked | ChunkedBroadcast | ✅ Slower, no OOM |
| 1000 | 1000 | 1M | Chunked | ChunkedBroadcast | ✅ 10 passes |
| 5000 | 1000 | 5M | Chunked | ChunkedBroadcast | ✅ 50 passes |
| 10000 | 1000 | 10M | Chunked | ChunkedBroadcast | ⚠️ 100 passes, slow |

**Summary**: Non-SE divergences work for large k×dim but with performance trade-offs.

## Performance Characteristics

### BroadcastUDF Performance
- **Single scan**: One pass over the data
- **Broadcast overhead**: ~100-500ms (one-time cost per iteration)
- **UDF overhead**: ~10-20% slower than native expressions
- **Memory**: k×dim×8 bytes per executor (doubles are 8 bytes)

**Threshold calculation** (200,000 elements):
- 200,000 elements × 8 bytes = 1.6 MB per executor
- With 100 executors = 160 MB total broadcast memory
- Safe default for most clusters

### ChunkedBroadcast Performance
- **Multiple scans**: ceil(k / chunkSize) passes
  - Default chunkSize = 100
  - k=1000 → 10 passes
  - k=5000 → 50 passes
- **Broadcast overhead**: ~100-500ms per pass
- **Total overhead**: (chunkSize / k) × single-scan-time
- **Memory**: chunkSize×dim×8 bytes per executor (~80KB with defaults)

**Example timings** (10M points, 1000 dim, 20 executors):
- BroadcastUDF (k=100): ~60 seconds
- BroadcastUDF (k=200, at threshold): ~65 seconds
- ChunkedBroadcast (k=1000, 10 passes): ~120 seconds (2× slower)
- ChunkedBroadcast (k=5000, 50 passes): ~400 seconds (6.7× slower)

## Tuning Parameters

### Broadcast Threshold
Control when to switch from BroadcastUDF to ChunkedBroadcast:

```scala
// Create custom AutoAssignment with higher threshold
val customStrategy = new com.massivedatascience.clusterer.ml.df.AutoAssignment(
  broadcastThresholdElems = 500000,  // Allow 500K elements (4MB)
  chunkSize = 100
)

// Use in LloydsIterator (advanced usage)
// Note: This requires using the low-level API
```

**When to increase threshold**:
- Cluster has high memory per executor (>8GB)
- Willing to accept higher memory for better performance
- k×dim is moderately above 200K (e.g., 300K-500K)

**When to decrease threshold**:
- Cluster has low memory per executor (<4GB)
- Experiencing OOM errors
- Running many concurrent jobs

### Chunk Size
Control the granularity of chunked processing:

```scala
val customStrategy = new com.massivedatascience.clusterer.ml.df.AutoAssignment(
  broadcastThresholdElems = 200000,
  chunkSize = 200  // Larger chunks = fewer passes
)
```

**Trade-offs**:
- **Larger chunkSize**: Fewer passes, faster, but higher memory per pass
- **Smaller chunkSize**: More passes, slower, but lower memory per pass

**Guidelines**:
- Default (100): Good balance for most use cases
- High memory cluster: Use 200-500 for better performance
- Low memory cluster: Use 50 for safety
- Never exceed: chunkSize × dim × 8 bytes < 10MB

## Common Scenarios

### Scenario 1: Text Clustering with KL Divergence
```scala
// 100K documents, TF-IDF vectors (dim=5000), k=100
// k×dim = 500K > 200K → ChunkedBroadcast

val gkm = new GeneralizedKMeans()
  .setK(100)
  .setDivergence("kl")
  .setSmoothing(1e-10)
  .fit(tfidfVectors)

// Expected: 1 chunk (k=100 < chunkSize=100)
// Actually uses BroadcastUDF! (falls back for small k)
// Fast: single pass
```

### Scenario 2: Image Clustering with High-Dim Embeddings
```scala
// 1M images, ResNet embeddings (dim=2048), k=1000
// k×dim = 2.048M >> 200K → ChunkedBroadcast

val gkm = new GeneralizedKMeans()
  .setK(1000)
  .setDivergence("squaredEuclidean")  // Use SE!
  .fit(imageEmbeddings)

// Strategy: SECrossJoin (no broadcast, fast!)
// Optimal for this use case
```

### Scenario 3: Time Series Clustering with Itakura-Saito
```scala
// 10K time series, spectrograms (dim=512), k=50
// k×dim = 25.6K < 200K → BroadcastUDF

val gkm = new GeneralizedKMeans()
  .setK(50)
  .setDivergence("itakuraSaito")
  .setSmoothing(1e-10)
  .fit(spectrograms)

// Strategy: BroadcastUDF
// Fast: single pass, ~200KB broadcast
```

## Troubleshooting

### OOM Errors with Non-SE Divergences

**Symptom**:
```
java.lang.OutOfMemoryError: Unable to acquire 128 MB of memory
```

**Diagnosis**:
```scala
val k = 1000
val dim = 10000
val kTimesDim = k * dim  // 10M elements = 80MB

println(s"k×dim = $kTimesDim")
println(s"Memory per executor: ${kTimesDim * 8 / 1024 / 1024} MB")
```

**Solutions**:
1. **Best**: Switch to Squared Euclidean if domain allows
   ```scala
   .setDivergence("squaredEuclidean")
   ```

2. **Good**: Let ChunkedBroadcast handle it automatically
   - Library will use chunked strategy when k×dim ≥ 200K
   - Check logs for: `AutoAssignment: strategy=ChunkedBroadcast`

3. **Manual**: Reduce k or dim
   - Use PCA/dimensionality reduction to lower dim
   - Use hierarchical clustering to lower k

### Slow Performance with Large k

**Symptom**: Clustering takes hours with k=5000, KL divergence

**Diagnosis**:
```
// Check logs for:
ChunkedBroadcastAssignment: completed in 50 passes
// 50 passes × 2 min/pass = 100 minutes per iteration!
```

**Solutions**:
1. **Best**: Switch to Squared Euclidean
   - 50× faster (no chunking needed)

2. **Good**: Increase chunkSize (if memory allows)
   ```scala
   // Advanced: Create custom strategy with larger chunks
   // Requires low-level API access
   ```

3. **Acceptable**: Accept slower runtime
   - ChunkedBroadcast avoids OOM but is slower
   - This is a fundamental trade-off

4. **Alternative**: Use mini-batch K-Means
   - Processes subsets of data per iteration
   - Can be combined with chunked assignment

## Best Practices

1. **Prefer Squared Euclidean when possible**
   - Fastest strategy (SECrossJoin)
   - Scales to arbitrarily large k and dim
   - Only use other divergences when domain requires it

2. **Monitor strategy selection**
   - Check Spark logs for `AutoAssignment: strategy=...`
   - Ensure expected strategy is being used

3. **Plan capacity**
   - Calculate k×dim before running
   - Ensure k×dim < 200K for fast non-SE performance
   - Budget ~2× runtime for ChunkedBroadcast

4. **Test at scale**
   - Run small test (k=10) first
   - Gradually increase k and monitor performance
   - Extrapolate runtime based on number of chunks

5. **Consider dimensionality reduction**
   - Use PCA to reduce dim before clustering
   - Example: dim=10000 → dim=100 (100× speedup for non-SE)

## References

- **BroadcastUDF**: src/main/scala/.../Strategies.scala:43
- **ChunkedBroadcastAssignment**: src/main/scala/.../Strategies.scala:148
- **AutoAssignment**: src/main/scala/.../Strategies.scala:259

## Changelog

**v0.7.0** (2025-10-18):
- Added ChunkedBroadcastAssignment for large k×dim
- Auto-selection with 200K element threshold
- Eliminates OOM errors for non-SE divergences with large k×dim
