---
title: Handle Outliers
---

# Handle Outliers

How to perform robust clustering in the presence of noise and outliers.

---

## Method 1: RobustKMeans

Purpose-built for outlier handling with three modes.

### Trim Mode

Ignore points far from any cluster during updates.

```scala
import com.massivedatascience.clusterer.ml.RobustKMeans

val robust = new RobustKMeans()
  .setK(3)
  .setRobustMode("trim")
  .setTrimFraction(0.1)      // Ignore worst 10%
  .setOutlierScoreCol("outlier_score")

val model = robust.fit(data)
val predictions = model.transform(data)

// High outlier_score = likely outlier
predictions.orderBy(desc("outlier_score")).show(10)
```

### Noise Cluster Mode

Assign outliers to a special "noise" cluster.

```scala
val robust = new RobustKMeans()
  .setK(3)
  .setRobustMode("noise_cluster")
  .setNoiseThreshold(2.0)    // Distance threshold

val model = robust.fit(data)
// Cluster -1 = noise points
```

### M-Estimator Mode

Down-weight outliers during center updates.

```scala
val robust = new RobustKMeans()
  .setK(3)
  .setRobustMode("m_estimator")
  .setMEstimatorType("huber")  // or "tukey", "cauchy"
```

---

## Method 2: KMedoids

Uses actual data points as centers, inherently more robust.

```scala
import com.massivedatascience.clusterer.ml.KMedoids

val kmedoids = new KMedoids()
  .setK(3)
  .setDistanceFunction("manhattan")  // L1 is more robust than L2

val model = kmedoids.fit(data)

// Centers are actual data points
model.medoids.foreach(println)
```

---

## Method 3: L1 Divergence

Manhattan distance is less sensitive to outliers than Euclidean.

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

val kmeans = new GeneralizedKMeans()
  .setK(3)
  .setDivergence("l1")  // Manhattan distance

val model = kmeans.fit(data)
```

---

## Method 4: Post-Processing

Identify outliers after clustering using distance to center.

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

val kmeans = new GeneralizedKMeans()
  .setK(3)
  .setDistanceCol("distance")  // Output distance to center

val model = kmeans.fit(data)
val predictions = model.transform(data)

// Flag outliers (distance > threshold)
import org.apache.spark.sql.functions._

val threshold = predictions.stat.approxQuantile("distance", Array(0.95), 0.01)(0)
val withOutliers = predictions.withColumn(
  "is_outlier",
  col("distance") > threshold
)

println(s"Found ${withOutliers.filter(col("is_outlier")).count()} outliers")
```

---

## Comparison

| Method | Pros | Cons |
|--------|------|------|
| **RobustKMeans (trim)** | Simple, effective | Need to choose trim fraction |
| **RobustKMeans (noise)** | Explicit outlier cluster | Need threshold tuning |
| **KMedoids** | Interpretable centers | O(n²) complexity |
| **L1 divergence** | No extra parameters | Less robust than dedicated methods |
| **Post-processing** | Works with any algorithm | Outliers still affect training |

---

## Complete Example

```scala
import com.massivedatascience.clusterer.ml.RobustKMeans
import org.apache.spark.ml.linalg.Vectors

// Data with outliers
val data = spark.createDataFrame(
  // Normal cluster 1
  (1 to 30).map(_ => Tuple1(Vectors.dense(
    scala.util.Random.nextGaussian(),
    scala.util.Random.nextGaussian()
  ))) ++
  // Normal cluster 2
  (1 to 30).map(_ => Tuple1(Vectors.dense(
    5 + scala.util.Random.nextGaussian(),
    5 + scala.util.Random.nextGaussian()
  ))) ++
  // Outliers
  Seq(
    Tuple1(Vectors.dense(100.0, 100.0)),
    Tuple1(Vectors.dense(-50.0, 50.0)),
    Tuple1(Vectors.dense(25.0, -25.0))
  )
).toDF("features")

// Robust clustering
val robust = new RobustKMeans()
  .setK(2)
  .setRobustMode("trim")
  .setTrimFraction(0.05)
  .setOutlierScoreCol("outlier_score")
  .setSeed(42)

val model = robust.fit(data)
val predictions = model.transform(data)

// Show outliers
println("Top outliers by score:")
predictions.orderBy(desc("outlier_score")).show(5)

// Cluster centers (not affected by outliers)
println("\nCluster centers:")
model.clusterCentersAsVectors.foreach(println)
```

---

[Back to How-To](index.html) | [Home](../)
