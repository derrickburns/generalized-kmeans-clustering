---
title: "Your First Clustering"
---

# Your First Clustering

**Time:** 5 minutes
**Goal:** Cluster a small dataset using GeneralizedKMeans

---

## What You'll Build

You'll cluster a simple 2D dataset into two groups and visualize the results.

## Prerequisites

- Spark 3.4+ with Scala 2.13
- A Spark session (local mode is fine)

---

## Step 1: Create Sample Data

First, let's create a DataFrame with two well-separated clusters:

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors

val spark = SparkSession.builder()
  .appName("FirstClustering")
  .master("local[*]")
  .getOrCreate()

// Two clusters: one near origin, one near (10, 10)
val data = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.0, 0.0)),
  Tuple1(Vectors.dense(0.5, 0.5)),
  Tuple1(Vectors.dense(1.0, 0.0)),
  Tuple1(Vectors.dense(0.0, 1.0)),
  Tuple1(Vectors.dense(9.0, 9.0)),
  Tuple1(Vectors.dense(10.0, 10.0)),
  Tuple1(Vectors.dense(9.5, 10.5)),
  Tuple1(Vectors.dense(10.5, 9.5))
)).toDF("features")

data.show()
```

Output:
```
+----------+
|  features|
+----------+
| [0.0,0.0]|
| [0.5,0.5]|
| [1.0,0.0]|
| [0.0,1.0]|
| [9.0,9.0]|
|[10.0,10.0]|
|[9.5,10.5]|
|[10.5,9.5]|
+----------+
```

---

## Step 2: Create and Train the Model

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

val kmeans = new GeneralizedKMeans()
  .setK(2)                           // Two clusters
  .setDivergence("squaredEuclidean") // Standard k-means distance
  .setMaxIter(20)                    // Maximum iterations
  .setSeed(42L)                      // For reproducibility

val model = kmeans.fit(data)
```

---

## Step 3: Examine the Results

### Cluster Centers

```scala
println(s"Number of clusters: ${model.numClusters}")
println(s"Number of features: ${model.numFeatures}")
println("\nCluster centers:")
model.clusterCentersAsVectors.zipWithIndex.foreach { case (center, i) =>
  println(s"  Cluster $i: $center")
}
```

Output:
```
Number of clusters: 2
Number of features: 2

Cluster centers:
  Cluster 0: [0.375, 0.375]
  Cluster 1: [9.75, 9.75]
```

### Make Predictions

```scala
val predictions = model.transform(data)
predictions.select("features", "prediction").show()
```

Output:
```
+----------+----------+
|  features|prediction|
+----------+----------+
| [0.0,0.0]|         0|
| [0.5,0.5]|         0|
| [1.0,0.0]|         0|
| [0.0,1.0]|         0|
| [9.0,9.0]|         1|
|[10.0,10.0]|        1|
|[9.5,10.5]|         1|
|[10.5,9.5]|         1|
+----------+----------+
```

---

## Step 4: Evaluate the Clustering

### Compute Cost (WCSS)

```scala
val cost = model.computeCost(data)
println(f"Within-cluster sum of squares: $cost%.4f")
```

Output:
```
Within-cluster sum of squares: 2.5000
```

### Check Training Summary

```scala
if (model.hasSummary) {
  val summary = model.summary
  println(s"Algorithm: ${summary.algorithm}")
  println(s"Iterations: ${summary.iterations}")
  println(s"Converged: ${summary.converged}")
  println(s"Final distortion: ${summary.finalDistortion}")
}
```

---

## Step 5: Save the Model

```scala
model.write.overwrite().save("/tmp/my-kmeans-model")

// Load it back
import com.massivedatascience.clusterer.ml.GeneralizedKMeansModel
val loadedModel = GeneralizedKMeansModel.load("/tmp/my-kmeans-model")
```

---

## Complete Code

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.{GeneralizedKMeans, GeneralizedKMeansModel}

object FirstClustering {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("FirstClustering")
      .master("local[*]")
      .getOrCreate()

    // Create data
    val data = spark.createDataFrame(Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.5, 0.5)),
      Tuple1(Vectors.dense(1.0, 0.0)),
      Tuple1(Vectors.dense(0.0, 1.0)),
      Tuple1(Vectors.dense(9.0, 9.0)),
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(9.5, 10.5)),
      Tuple1(Vectors.dense(10.5, 9.5))
    )).toDF("features")

    // Train model
    val kmeans = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setMaxIter(20)
      .setSeed(42L)

    val model = kmeans.fit(data)

    // Show results
    println(s"Found ${model.numClusters} clusters")
    model.transform(data).select("features", "prediction").show()
    println(f"WCSS: ${model.computeCost(data)}%.4f")

    spark.stop()
  }
}
```

---

## Next Steps

- [PySpark Tutorial](pyspark-tutorial.html) — Same example in Python
- [Choosing the Right Algorithm](choosing-algorithm.html) — Which algorithm for your use case
- [Cluster Probability Distributions](../howto/cluster-probabilities.html) — Use KL divergence

---

[Back to Tutorials](index.html) | [Home](../)
