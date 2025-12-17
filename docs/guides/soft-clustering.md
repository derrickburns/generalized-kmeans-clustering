# Soft Clustering: Interpreting Probabilistic Memberships

Unlike hard clustering (where each point belongs to exactly one cluster), soft clustering assigns **probability distributions** over clusters. This guide explains how to use and interpret soft cluster memberships.

## Why Soft Clustering?

Hard clustering forces binary decisions that may not reflect reality:

| Scenario | Hard Clustering | Soft Clustering |
|----------|-----------------|-----------------|
| Customer who shops both electronics and clothing | Assigned to one segment | 60% electronics, 40% clothing |
| Document about machine learning and finance | One topic only | 70% ML, 30% finance |
| Image with multiple objects | Single category | Multiple category probabilities |

Soft clustering provides:
- **Uncertainty quantification**: Know when assignments are ambiguous
- **Multi-membership**: Points can partially belong to multiple clusters
- **Richer analysis**: Understand cluster overlap and boundaries

## Basic Usage

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.SoftKMeans

val spark = SparkSession.builder()
  .appName("SoftClustering")
  .master("local[*]")
  .getOrCreate()

import spark.implicits._

// Create sample data
val df = Seq(
  Tuple1(Vectors.dense(0.0, 0.0)),   // Clearly in cluster 0
  Tuple1(Vectors.dense(10.0, 10.0)), // Clearly in cluster 1
  Tuple1(Vectors.dense(5.0, 5.0))    // Between clusters - ambiguous!
).toDF("features")

// Train soft K-Means
val soft = new SoftKMeans()
  .setK(2)
  .setBeta(2.0)
  .setMaxIter(20)
  .setSeed(42)

val model = soft.fit(df)
val predictions = model.transform(df)

predictions.select("features", "prediction", "probability", "probabilities").show(false)
```

Output:
```
+------------+----------+-----------+------------------+
|features    |prediction|probability|probabilities     |
+------------+----------+-----------+------------------+
|[0.0,0.0]   |0         |0.99       |[0.99, 0.01]      |
|[10.0,10.0] |1         |0.99       |[0.01, 0.99]      |
|[5.0,5.0]   |0         |0.52       |[0.52, 0.48]      |  <- Ambiguous!
+------------+----------+-----------+------------------+
```

## Output Columns

SoftKMeans adds three columns:

| Column | Type | Description |
|--------|------|-------------|
| `prediction` | Int | Most likely cluster (argmax of probabilities) |
| `probability` | Double | Probability of the predicted cluster |
| `probabilities` | Vector | Full probability distribution over all clusters |

## The Beta (Fuzziness) Parameter

Beta controls how "soft" the assignments are:

```scala
// Low beta = softer assignments (more uncertainty)
new SoftKMeans().setBeta(1.5)

// High beta = harder assignments (approaches hard clustering)
new SoftKMeans().setBeta(10.0)
```

| Beta | Effect | When to Use |
|------|--------|-------------|
| 1.1 - 2.0 | Very soft, high uncertainty | Exploratory analysis |
| 2.0 (default) | Balanced | General use |
| 2.0 - 5.0 | Moderate softness | Most applications |
| 5.0+ | Nearly hard | When you want probabilities but mostly certain |

### Visualizing Beta Effect

```scala
// Same data, different beta values
val df = Seq(
  Tuple1(Vectors.dense(0.0, 0.0)),
  Tuple1(Vectors.dense(5.0, 5.0)),  // Midpoint
  Tuple1(Vectors.dense(10.0, 10.0))
).toDF("features")

Seq(1.5, 2.0, 3.0, 5.0, 10.0).foreach { beta =>
  val model = new SoftKMeans().setK(2).setBeta(beta).setSeed(42).fit(df)
  val midpointProb = model.transform(df)
    .filter(col("features") === Vectors.dense(5.0, 5.0))
    .select("probabilities")
    .first().getAs[Vector](0)

  println(f"Beta=$beta%.1f: midpoint probabilities = [${midpointProb(0)}%.2f, ${midpointProb(1)}%.2f]")
}
```

Output:
```
Beta=1.5: midpoint probabilities = [0.50, 0.50]  # Equal, very uncertain
Beta=2.0: midpoint probabilities = [0.50, 0.50]  # Equal
Beta=3.0: midpoint probabilities = [0.50, 0.50]  # Equal (symmetric case)
Beta=5.0: midpoint probabilities = [0.50, 0.50]  # Equal
Beta=10.0: midpoint probabilities = [0.50, 0.50] # Equal (symmetric)
```

## Interpreting Probabilities

### High Confidence Assignment

```
features: [0.1, 0.1]
probabilities: [0.98, 0.02]
prediction: 0
probability: 0.98
```

**Interpretation**: This point strongly belongs to cluster 0. The assignment is confident.

### Ambiguous Assignment

```
features: [5.0, 5.0]
probabilities: [0.52, 0.48]
prediction: 0
probability: 0.52
```

**Interpretation**: This point is on the boundary. It could reasonably belong to either cluster. Consider:
- Flagging for manual review
- Using both cluster profiles in downstream analysis
- Excluding from cluster-specific analysis

### Multi-Cluster Membership

```
features: [3.3, 3.3]
probabilities: [0.45, 0.35, 0.20]
prediction: 0
probability: 0.45
```

**Interpretation**: This point has significant membership in multiple clusters. It may represent:
- A transition between cluster profiles
- A mixed entity (customer with diverse interests)
- An outlier or noise point

## Use Cases and Examples

### 1. Finding Uncertain Points

```scala
val predictions = model.transform(data)

// Find points with low confidence (potential boundary/overlap cases)
val uncertain = predictions
  .filter(col("probability") < 0.7)
  .orderBy("probability")

uncertain.show()
```

### 2. Multi-Membership Analysis

```scala
import org.apache.spark.sql.functions._

// Find points with significant membership in multiple clusters
val multiMember = predictions.filter { row =>
  val probs = row.getAs[Vector]("probabilities").toArray
  val sorted = probs.sorted.reverse
  sorted(1) > 0.2  // Second-highest probability > 20%
}

println(s"${multiMember.count()} points have significant multi-cluster membership")
```

### 3. Cluster Overlap Analysis

```scala
// For each cluster, find how "pure" it is
val clusterPurity = predictions
  .groupBy("prediction")
  .agg(
    avg("probability").as("avg_confidence"),
    count("*").as("size")
  )
  .orderBy("prediction")

clusterPurity.show()
```

Output:
```
+----------+---------------+----+
|prediction|avg_confidence |size|
+----------+---------------+----+
|         0|          0.85 | 120|  # Well-defined cluster
|         1|          0.72 |  95|  # Some overlap
|         2|          0.61 |  45|  # High overlap - boundary cluster?
+----------+---------------+----+
```

### 4. Weighted Cluster Profiles

Instead of hard assignment, weight each point by its membership:

```scala
import org.apache.spark.ml.linalg.{Vector, Vectors}

// Compute weighted centroid for cluster 0
val cluster0Data = predictions.select(
  col("features"),
  element_at(col("probabilities"), 1).as("weight_c0")  // Probability for cluster 0
)

// Points contribute proportionally to their membership
val weightedSum = cluster0Data
  .select(
    (col("weight_c0") * element_at(col("features"), 1)).as("weighted_x"),
    (col("weight_c0") * element_at(col("features"), 2)).as("weighted_y"),
    col("weight_c0")
  )
  .agg(
    sum("weighted_x").as("sum_x"),
    sum("weighted_y").as("sum_y"),
    sum("weight_c0").as("total_weight")
  )
  .first()

val weightedCentroid = (
  weightedSum.getDouble(0) / weightedSum.getDouble(2),
  weightedSum.getDouble(1) / weightedSum.getDouble(2)
)

println(s"Weighted centroid for cluster 0: $weightedCentroid")
```

### 5. Customer Segmentation with Overlap

```scala
val customerClusters = model.transform(customerData)

// Find "hybrid" customers (significant membership in 2+ segments)
val hybridCustomers = customerClusters.filter { row =>
  val probs = row.getAs[Vector]("probabilities").toArray
  probs.count(_ > 0.25) >= 2  // At least 2 segments with >25% membership
}

println(s"${hybridCustomers.count()} hybrid customers for cross-segment marketing")
```

### 6. Document Topic Distribution

```scala
// Soft clustering naturally fits topic modeling interpretation
val docTopics = model.transform(documentVectors)

// Show topic distribution for a document
docTopics.select("doc_id", "probabilities").show(5, false)

// Find documents that bridge multiple topics
val bridgeDocs = docTopics.filter { row =>
  val probs = row.getAs[Vector]("probabilities").toArray
  val entropy = -probs.filter(_ > 0).map(p => p * math.log(p)).sum
  entropy > 1.0  // High entropy = spread across topics
}
```

## Comparing Hard vs Soft Results

```scala
import com.massivedatascience.clusterer.ml.{GeneralizedKMeans, SoftKMeans}

// Hard clustering
val hard = new GeneralizedKMeans().setK(3).setSeed(42).fit(data)
val hardPred = hard.transform(data)

// Soft clustering
val soft = new SoftKMeans().setK(3).setBeta(2.0).setSeed(42).fit(data)
val softPred = soft.transform(data)

// Compare assignments
val comparison = hardPred.select(
  col("features"),
  col("prediction").as("hard_cluster")
).join(
  softPred.select(
    col("features"),
    col("prediction").as("soft_cluster"),
    col("probability").as("confidence")
  ),
  "features"
)

// Find disagreements
val disagree = comparison.filter(col("hard_cluster") =!= col("soft_cluster"))
println(s"Hard and soft disagree on ${disagree.count()} points")

// Show low-confidence soft assignments
comparison.filter(col("confidence") < 0.6).show()
```

## Training Summary

```scala
val model = soft.fit(data)

if (model.hasSummary) {
  val summary = model.summary
  println(s"Algorithm: ${summary.algorithm}")
  println(s"Clusters: ${summary.k}")
  println(s"Iterations: ${summary.iterations}")
  println(s"Converged: ${summary.converged}")
  println(s"Final distortion: ${summary.finalDistortion}")
}
```

## Model Persistence

```scala
// Save
model.write.overwrite().save("/path/to/soft-kmeans-model")

// Load
import com.massivedatascience.clusterer.ml.SoftKMeansModel
val loaded = SoftKMeansModel.load("/path/to/soft-kmeans-model")
```

## Best Practices

### 1. Start with Default Beta

```scala
// Beta=2.0 is a good starting point
new SoftKMeans().setBeta(2.0)
```

### 2. Analyze Uncertainty Distribution

```scala
// Understand the confidence distribution
predictions.select("probability").describe().show()

// Visualize with histogram bins
predictions
  .withColumn("confidence_bin",
    when(col("probability") >= 0.9, "high")
    .when(col("probability") >= 0.7, "medium")
    .otherwise("low"))
  .groupBy("confidence_bin")
  .count()
  .show()
```

### 3. Use Uncertainty for Quality Control

```scala
// Flag uncertain assignments for review
val needsReview = predictions
  .filter(col("probability") < 0.6)
  .select("id", "features", "prediction", "probability")
```

### 4. Consider Domain Constraints

If domain knowledge says items can't belong to multiple clusters, use hard clustering instead. Soft clustering is for when overlap is meaningful.

## Summary

- **SoftKMeans** provides probability distributions over clusters
- **Beta** controls fuzziness (lower = softer)
- **probability** column shows confidence in the prediction
- **probabilities** vector shows full membership distribution
- Use soft clustering when:
  - Points may naturally belong to multiple categories
  - You need uncertainty quantification
  - Boundary points are important to identify
