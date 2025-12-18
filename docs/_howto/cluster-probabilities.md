---
title: Cluster Probability Distributions
---

# Cluster Probability Distributions

How to cluster data that represents probability distributions using KL divergence.

---

## When to Use KL Divergence

Use KL divergence when your data:
- Represents probability distributions (values sum to 1)
- Contains word frequencies or topic distributions
- Represents discrete probability mass functions

**Examples:** Document topic vectors, user behavior profiles, mixture model outputs

---

## Basic Example

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
import org.apache.spark.ml.linalg.Vectors

// Documents as word probability distributions (must sum to 1)
val documents = spark.createDataFrame(Seq(
  // Tech documents - high prob on first terms
  Tuple1(Vectors.dense(0.5, 0.3, 0.1, 0.05, 0.05)),
  Tuple1(Vectors.dense(0.6, 0.2, 0.1, 0.05, 0.05)),
  // Sports documents - high prob on middle terms
  Tuple1(Vectors.dense(0.1, 0.1, 0.5, 0.2, 0.1)),
  Tuple1(Vectors.dense(0.05, 0.15, 0.6, 0.15, 0.05)),
  // Food documents - high prob on last terms
  Tuple1(Vectors.dense(0.05, 0.1, 0.1, 0.35, 0.4)),
  Tuple1(Vectors.dense(0.1, 0.05, 0.15, 0.3, 0.4))
)).toDF("features")

val kmeans = new GeneralizedKMeans()
  .setK(3)
  .setDivergence("kl")      // Kullback-Leibler divergence
  .setSmoothing(1e-10)      // Prevent log(0)
  .setMaxIter(30)
  .setSeed(42)

val model = kmeans.fit(documents)

// Cluster centers are also probability distributions
model.clusterCentersAsVectors.zipWithIndex.foreach { case (center, i) =>
  val sum = center.toArray.sum
  println(f"Cluster $i: $center (sum=$sum%.4f)")
}
```

---

## Data Preparation

### Ensure Valid Probabilities

```scala
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector, Vectors}

// Normalize rows to sum to 1
val normalizeUDF = udf { (v: Vector) =>
  val arr = v.toArray
  val sum = arr.sum
  if (sum > 0) Vectors.dense(arr.map(_ / sum))
  else v
}

val normalizedData = rawData.withColumn(
  "features",
  normalizeUDF(col("features"))
)
```

### Handle Zeros (Add Smoothing)

```scala
// Add small epsilon to prevent zeros
val smoothUDF = udf { (v: Vector) =>
  val epsilon = 1e-10
  val arr = v.toArray.map(_ + epsilon)
  val sum = arr.sum
  Vectors.dense(arr.map(_ / sum))
}

val smoothedData = data.withColumn(
  "features",
  smoothUDF(col("features"))
)
```

---

## Understanding KL Divergence

KL divergence D_KL(P || Q) measures how different distribution P is from Q:

```
D_KL(P || Q) = Σ P(i) * log(P(i) / Q(i))
```

**Properties:**
- Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
- Always non-negative
- Zero only when P = Q
- Undefined if Q(i) = 0 where P(i) > 0

---

## Complete Example: Topic Clustering

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._

// Simulate LDA topic distributions (10 topics per document)
val documents = spark.createDataFrame(
  (1 to 100).map { i =>
    val topicGroup = i % 3  // 3 underlying categories
    val distribution = (0 until 10).map { t =>
      val base = if (t >= topicGroup * 3 && t < (topicGroup + 1) * 3) 0.25 else 0.025
      base + scala.util.Random.nextDouble() * 0.05
    }
    val sum = distribution.sum
    Tuple1(Vectors.dense(distribution.map(_ / sum).toArray))
  }
).toDF("features")

// Cluster with KL divergence
val kmeans = new GeneralizedKMeans()
  .setK(3)
  .setDivergence("kl")
  .setSmoothing(1e-10)
  .setMaxIter(50)

val model = kmeans.fit(documents)
val predictions = model.transform(documents)

// Evaluate
import com.massivedatascience.clusterer.ml.ClusteringMetrics
val metrics = ClusteringMetrics(predictions)
println(s"Cluster sizes: ${metrics.clusterSizes}")
println(f"Balance ratio: ${metrics.balanceRatio}%.3f")
```

---

## Alternatives to KL

| Divergence | When to Use |
|------------|-------------|
| `kl` | Standard for probability distributions |
| `generalizedI` | When you have count data (not normalized) |
| `itakuraSaito` | For spectral/power data |

---

## Troubleshooting

### NaN or Infinity Values

**Problem:** Model produces NaN cluster centers.

**Solution:** Increase smoothing parameter:
```scala
.setSmoothing(1e-8)  // Try larger values if needed
```

### Poor Cluster Quality

**Problem:** Clusters don't make semantic sense.

**Solutions:**
1. Verify data sums to 1: `data.select(sum_of_vector_udf(col("features"))).show()`
2. Try different k values
3. Use multiple random seeds and pick best

---

[Back to How-To](index.html) | [Home](../)
