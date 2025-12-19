---
title: "Assignment Strategies"
---

# Assignment Strategies

How points are assigned to clusters at scale.

---

## Overview

The assignment step computes the distance from each point to each center and picks the minimum. With n points and k centers, this is O(n × k) distance computations.

This library provides three strategies:

| Strategy | Best For | How It Works |
|----------|----------|--------------|
| **auto** | Most cases | Automatically selects best strategy |
| **broadcastUDF** | General Bregman, k < 1000 | Broadcasts centers, UDF computes distances |
| **crossJoin** | Squared Euclidean, large k | SQL join with vectorized distance |

---

## BroadcastUDF Strategy

**Default for general Bregman divergences.**

```
1. Broadcast centers to all executors (small data replicated)
2. Apply UDF to each row computing distances to all centers
3. Select minimum distance center
```

```scala
// Pseudocode
val centersBC = spark.sparkContext.broadcast(centers)
data.withColumn("prediction",
  udf((features: Vector) => {
    centersBC.value.zipWithIndex.minBy { case (c, _) =>
      divergence.distance(features, c)
    }._2
  })
)
```

**Pros:**
- Works with any divergence
- Efficient for small-medium k

**Cons:**
- Broadcast overhead grows with k
- Single-threaded UDF per row

---

## CrossJoin Strategy

**Optimized for Squared Euclidean with large k.**

```
1. Explode each point to k rows (one per center)
2. Compute distances using vectorized SQL
3. Group by point, select minimum
```

```scala
// Pseudocode
data
  .crossJoin(centersDF)
  .withColumn("distance", squaredDistance(col("features"), col("center")))
  .groupBy("id")
  .agg(min_by(col("centerId"), col("distance")).as("prediction"))
```

**Pros:**
- Fully vectorized (no UDF overhead)
- Scales to very large k
- Benefits from Spark SQL optimizations

**Cons:**
- Only works for Squared Euclidean
- Memory overhead from cross join

---

## Auto Strategy

**Recommended for most users.**

```scala
new GeneralizedKMeans()
  .setAssignmentStrategy("auto")  // Default
```

Selection logic:
```
if (divergence == "squaredEuclidean" && k >= threshold)
  use CrossJoin
else
  use BroadcastUDF
```

---

## When to Override

### Force CrossJoin for Large k

```scala
// k = 10,000 clusters with Squared Euclidean
new GeneralizedKMeans()
  .setK(10000)
  .setDivergence("squaredEuclidean")
  .setAssignmentStrategy("crossJoin")  // Faster for large k
```

### Force BroadcastUDF for Small k

```scala
// Small k, any divergence
new GeneralizedKMeans()
  .setK(5)
  .setDivergence("kl")
  .setAssignmentStrategy("broadcastUDF")  // Required for non-SE
```

---

## Performance Comparison

Benchmarks on 1M points × 100 dimensions:

| k | BroadcastUDF | CrossJoin |
|---|--------------|-----------|
| 10 | 15s | 20s |
| 100 | 18s | 18s |
| 1,000 | 45s | 25s |
| 10,000 | 180s | 40s |

CrossJoin wins for large k due to vectorization.

---

## Elkan Acceleration

For Squared Euclidean, Elkan's algorithm can skip 50-90% of distance computations using the triangle inequality:

```
If d(x, old_center) + d(old_center, new_center) < d(x, other_center)
Then x cannot be closer to other_center
Skip the distance computation
```

Enabled automatically for SE with k ≥ 5.

---

## Implementation Details

Strategies are implemented in `clusterer.ml.df.strategies.impl`:

- `BroadcastUDFAssignment` — General Bregman
- `CrossJoinSEAssignment` — Squared Euclidean fast path
- `AcceleratedSEAssignment` — Elkan acceleration

---

[Back to Explanation](index.html) | [Home](../)
