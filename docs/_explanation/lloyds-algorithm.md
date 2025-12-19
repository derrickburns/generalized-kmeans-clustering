---
title: "How Lloyd's Algorithm Works"
---

# How Lloyd's Algorithm Works

The core iteration behind all k-means variants.

---

## The Algorithm

Lloyd's algorithm is elegantly simple:

```
1. Initialize k cluster centers
2. Repeat until convergence:
   a. ASSIGN: Each point to its nearest center
   b. UPDATE: Each center to the mean of its assigned points
3. Return final centers and assignments
```

---

## Visual Intuition

```
Iteration 0 (Random init):     Iteration 1:              Iteration 2 (Converged):

    x   x                          x   x                      x   x
  x   ●   x                      x   ●   x                  x   ●   x
    x   x                          x   x                      x   x
                        →                          →
    o   o                          o   o                      o   o
  o   ●   o                      o       o                  o   ●   o
    o   o                            ●                        o   o

● = center, x/o = points
```

---

## Why It Works

### Unique Mean Property

For any Bregman divergence D_φ, the point that minimizes the sum of divergences from a set is the **arithmetic mean**:

```
argmin_c Σᵢ D_φ(xᵢ, c) = (1/n) Σᵢ xᵢ
```

This is why k-means always uses simple averaging to update centers, regardless of which divergence you use.

### Monotonic Convergence

Each iteration decreases the total cost:

1. **Assignment step**: For fixed centers, assigning each point to its nearest center minimizes cost
2. **Update step**: For fixed assignments, the mean minimizes within-cluster cost

Since cost decreases monotonically and is bounded below by 0, the algorithm must converge.

---

## In This Library

The core iteration is implemented in `LloydsIterator`:

```scala
// Simplified view of one iteration
def iterate(data: DataFrame, centers: Array[Vector]): (DataFrame, Array[Vector]) = {
  // ASSIGN: Add prediction column with nearest center
  val assigned = assignmentStrategy.assign(data, centers)

  // UPDATE: Compute new centers as mean of assigned points
  val newCenters = updateStrategy.computeCenters(assigned, k)

  (assigned, newCenters)
}
```

---

## Convergence Criteria

The algorithm stops when:

1. **Max iterations reached**: `setMaxIter(20)`
2. **Centers stabilize**: Movement < tolerance `setTol(1e-4)`
3. **Cost stabilizes**: Improvement < threshold

```scala
val kmeans = new GeneralizedKMeans()
  .setMaxIter(100)    // At most 100 iterations
  .setTol(1e-6)       // Stop if centers move less than this
```

---

## Complexity

| Phase | Complexity |
|-------|-----------|
| Assignment | O(n × k × d) |
| Update | O(n × d) |
| Per iteration | O(n × k × d) |
| Total | O(n × k × d × iterations) |

Where:
- n = number of points
- k = number of clusters
- d = dimensionality
- iterations = typically 10-50

---

## Variants

| Variant | Modification |
|---------|-------------|
| **Mini-batch** | Update on random sample each iteration |
| **Elkan** | Use triangle inequality to skip distance computations |
| **Soft k-means** | Assign fractional membership instead of hard assignment |
| **Online** | Update centers incrementally as points arrive |

---

[Back to Explanation](index.html) | [Home](../)
