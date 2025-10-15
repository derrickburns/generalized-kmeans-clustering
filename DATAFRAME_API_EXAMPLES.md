# DataFrame API Usage Examples

The new DataFrame API provides a clean, Spark ML-native interface for generalized k-means clustering with pluggable Bregman divergences.

## Basic Usage - Squared Euclidean

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
import org.apache.spark.ml.linalg.Vectors

// Create training data
val data = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.0, 0.0)),
  Tuple1(Vectors.dense(1.0, 1.0)),
  Tuple1(Vectors.dense(9.0, 8.0)),
  Tuple1(Vectors.dense(8.0, 9.0))
)).toDF("features")

// Train model with Squared Euclidean distance (default)
val kmeans = new GeneralizedKMeans()
  .setK(2)
  .setMaxIter(20)
  .setFeaturesCol("features")
  .setPredictionCol("cluster")

val model = kmeans.fit(data)

// Make predictions
val predictions = model.transform(data)
predictions.show()

// Evaluate
val cost = model.computeCost(data)
println(s"Within Set Sum of Squared Errors = $cost")
```

## KL Divergence for Probability Distributions

```scala
// Create probability distribution data (components sum to 1)
val probData = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.7, 0.2, 0.1)),
  Tuple1(Vectors.dense(0.6, 0.3, 0.1)),
  Tuple1(Vectors.dense(0.1, 0.2, 0.7)),
  Tuple1(Vectors.dense(0.1, 0.3, 0.6))
)).toDF("features")

// Use KL divergence
val klKMeans = new GeneralizedKMeans()
  .setK(2)
  .setDivergence("kl")
  .setSmoothing(1e-10)
  .setMaxIter(20)

val klModel = klKMeans.fit(probData)
val klPredictions = klModel.transform(probData)
```

## Itakura-Saito for Spectral Data

```scala
// For audio/spectral data
val spectralKMeans = new GeneralizedKMeans()
  .setK(5)
  .setDivergence("itakuraSaito")
  .setSmoothing(1e-10)
  .setMaxIter(30)

val spectralModel = spectralKMeans.fit(spectralData)
```

## Weighted Clustering

```scala
// Points with importance weights
val weightedData = spark.createDataFrame(Seq(
  (Vectors.dense(0.0, 0.0), 1.0),
  (Vectors.dense(0.1, 0.1), 1.0),
  (Vectors.dense(5.0, 5.0), 10.0),  // High importance
  (Vectors.dense(5.1, 5.1), 10.0)
)).toDF("features", "weight")

val weightedKMeans = new GeneralizedKMeans()
  .setK(2)
  .setWeightCol("weight")
  .setMaxIter(20)

val weightedModel = weightedKMeans.fit(weightedData)
```

## Advanced Configuration

```scala
val advancedKMeans = new GeneralizedKMeans()
  .setK(10)
  .setDivergence("squaredEuclidean")
  .setMaxIter(50)
  .setTol(1e-6)
  .setSeed(42)
  // Initialization
  .setInitMode("k-means||")          // or "random"
  .setInitSteps(2)
  // Assignment strategy
  .setAssignmentStrategy("auto")     // "broadcast", "crossJoin", or "auto"
  // Empty cluster handling
  .setEmptyClusterStrategy("reseedRandom")  // or "drop"
  // Checkpointing
  .setCheckpointInterval(10)
  // Optional distance output
  .setDistanceCol("distance")

val advancedModel = advancedKMeans.fit(data)
val results = advancedModel.transform(data)
results.select("features", "cluster", "distance").show()
```

## Pipeline Integration

```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler

// Feature preparation
val assembler = new VectorAssembler()
  .setInputCols(Array("feature1", "feature2", "feature3"))
  .setOutputCol("features")

val kmeans = new GeneralizedKMeans()
  .setK(5)
  .setFeaturesCol("features")
  .setPredictionCol("cluster")

// Create pipeline
val pipeline = new Pipeline()
  .setStages(Array(assembler, kmeans))

val pipelineModel = pipeline.fit(rawData)
val clusteredData = pipelineModel.transform(rawData)
```

## Model Persistence

```scala
// Save model
model.write.overwrite().save("path/to/model")

// Load model
import com.massivedatascience.clusterer.ml.GeneralizedKMeansModel
val loadedModel = GeneralizedKMeansModel.load("path/to/model")

// Use loaded model
val predictions = loadedModel.transform(newData)
```

## Single Point Prediction

```scala
val point = Vectors.dense(2.0, 3.0)
val cluster = model.predict(point)
println(s"Point $point belongs to cluster $cluster")
```

## Available Divergences

| Divergence | Parameter Value | Use Case |
|------------|----------------|----------|
| Squared Euclidean | `"squaredEuclidean"` | General-purpose, Gaussian data |
| KL Divergence | `"kl"` | Probability distributions, text |
| Itakura-Saito | `"itakuraSaito"` | Audio, spectral data |
| Generalized I | `"generalizedI"` | Count data, non-negative |
| Logistic Loss | `"logistic"` | Binary probabilities |

## Performance Tips

1. **Assignment Strategy:**
   - Use `"auto"` (default) - automatically selects best strategy
   - For Squared Euclidean on large data: `"crossJoin"` may be faster
   - For other divergences: `"broadcast"` is the only option

2. **Initialization:**
   - `"k-means||"` (default) usually produces better results
   - `"random"` is faster but may need more iterations

3. **Checkpointing:**
   - Enable for long-running jobs: `setCheckpointInterval(10)`
   - Set checkpoint directory: `setCheckpointDir("/path/to/checkpoint")`

4. **Convergence:**
   - Adjust tolerance: `setTol(1e-6)` (stricter) or `setTol(1e-3)` (looser)
   - Set max iterations: `setMaxIter(100)` for difficult datasets

## Comparison with MLlib KMeans

| Feature | MLlib KMeans | GeneralizedKMeans |
|---------|--------------|-------------------|
| Distance Functions | Euclidean only | 5 Bregman divergences |
| Weighted Clustering | ✗ | ✓ |
| Custom Divergences | ✗ | ✓ (extensible) |
| Assignment Strategies | Fixed | Pluggable |
| Empty Cluster Handling | Fixed | Configurable |
| API | Spark ML | Spark ML |

## Migration from RDD API

Old RDD-based API:
```scala
import com.massivedatascience.clusterer.KMeans
val model = KMeans.train(rdd, runs, k, maxIterations, ...)
```

New DataFrame API:
```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
val model = new GeneralizedKMeans()
  .setK(k)
  .setMaxIter(maxIterations)
  .fit(dataframe)
```

Benefits:
- Native Spark ML integration (pipelines, persistence, cross-validation)
- Better performance through Catalyst optimizer
- Cleaner, more maintainable code
- Type-safe parameters
- Automatic schema validation

## Real-World Examples

### Example 1: Customer Segmentation

```scala
import org.apache.spark.sql.functions._

// Load customer data
val customers = spark.read.parquet("customers.parquet")
  // Features: age, annual_income, spending_score, recency_days
  .select($"customer_id",
    array($"age", $"annual_income", $"spending_score", $"recency_days").as("features_array"))
  .withColumn("features", vec_to_vector($"features_array"))

// Normalize features first
import org.apache.spark.ml.feature.StandardScaler

val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaled_features")
  .setWithMean(true)
  .setWithStd(true)

val scalerModel = scaler.fit(customers)
val normalizedCustomers = scalerModel.transform(customers)

// Cluster customers
val kmeans = new GeneralizedKMeans()
  .setK(5)  // 5 customer segments
  .setFeaturesCol("scaled_features")
  .setPredictionCol("segment")
  .setMaxIter(50)
  .setSeed(42)

val model = kmeans.fit(normalizedCustomers)

// Analyze segments
val segments = model.transform(normalizedCustomers)

segments.groupBy("segment")
  .agg(
    count("*").as("count"),
    avg("age").as("avg_age"),
    avg("annual_income").as("avg_income"),
    avg("spending_score").as("avg_spending")
  )
  .orderBy("segment")
  .show()

// Save model for production use
model.write.overwrite().save("models/customer_segmentation")
```

### Example 2: Document Clustering with KL Divergence

```scala
// Load TF-IDF vectors representing documents
val documents = spark.read.parquet("documents_tfidf.parquet")

// TF-IDF vectors are non-negative and can be treated as probability distributions
// after normalization

import org.apache.spark.sql.functions._

// Normalize to probability distribution (sum to 1)
val normalize_udf = udf { (vec: Vector) =>
  val sum = vec.toArray.sum
  if (sum > 0) Vectors.dense(vec.toArray.map(_ / sum))
  else vec
}

val probDocuments = documents
  .withColumn("prob_features", normalize_udf($"tfidf_features"))

// Use KL divergence for clustering
val docKMeans = new GeneralizedKMeans()
  .setK(20)  // 20 topic clusters
  .setDivergence("kl")
  .setSmoothing(1e-10)  // Avoid log(0)
  .setFeaturesCol("prob_features")
  .setPredictionCol("topic")
  .setMaxIter(30)

val docModel = docKMeans.fit(probDocuments)

// Get document topics
val topicAssignments = docModel.transform(probDocuments)
  .select("doc_id", "topic", "title")

// Find representative documents per topic
topicAssignments.groupBy("topic")
  .agg(
    count("*").as("num_docs"),
    collect_list("title").as("sample_titles")
  )
  .show(false)
```

### Example 3: Audio Clustering with Itakura-Saito

```scala
// Load audio spectral features (power spectrogram)
val audioData = spark.read.parquet("audio_spectra.parquet")
  // Features: power spectral density values (non-negative)

// Itakura-Saito divergence is ideal for spectral data
val audioKMeans = new GeneralizedKMeans()
  .setK(10)  // 10 acoustic patterns
  .setDivergence("itakuraSaito")
  .setSmoothing(1e-10)
  .setFeaturesCol("spectrum")
  .setPredictionCol("acoustic_class")
  .setMaxIter(40)
  .setSeed(42)

val audioModel = audioKMeans.fit(audioData)

// Classify new audio segments
val predictions = audioModel.transform(testAudio)
  .select("audio_id", "timestamp", "acoustic_class")

predictions.write.parquet("audio_classifications.parquet")
```

### Example 4: Finding Optimal K (Elbow Method)

```scala
// Evaluate different values of k
val kValues = (2 to 20).toArray
val results = kValues.map { k =>
  val kmeans = new GeneralizedKMeans()
    .setK(k)
    .setMaxIter(30)
    .setSeed(42)

  val model = kmeans.fit(data)
  val cost = model.computeCost(data)

  (k, cost)
}

// Print results
results.foreach { case (k, cost) =>
  println(f"k=$k%2d: WCSS=$cost%.2f")
}

// Visualize elbow curve (export to CSV)
import spark.implicits._
results.toSeq.toDF("k", "wcss")
  .coalesce(1)
  .write.option("header", "true")
  .csv("elbow_curve.csv")
```

### Example 5: Quality Metrics

```scala
// Train model
val kmeans = new GeneralizedKMeans()
  .setK(5)
  .setMaxIter(30)
  .setSeed(42)

val model = kmeans.fit(data)

// Get comprehensive quality metrics
val summary = new GeneralizedKMeansSummary(
  predictions = model.transform(data),
  predictionCol = "prediction",
  featuresCol = "features",
  clusterCenters = model.clusterCenters,
  kernel = new SquaredEuclideanKernel(),
  numClusters = model.numClusters,
  numFeatures = model.numFeatures,
  numIter = 10,
  converged = true,
  distortionHistory = Array(),
  movementHistory = Array()
)

// Print quality metrics
println(s"Within-Cluster Sum of Squares (WCSS): ${summary.wcss}")
println(s"Between-Cluster Sum of Squares (BCSS): ${summary.bcss}")
println(s"Calinski-Harabasz Index: ${summary.calinskiHarabaszIndex}")
println(s"Davies-Bouldin Index: ${summary.daviesBouldinIndex}")
println(s"Dunn Index: ${summary.dunnIndex}")

// Compute silhouette (expensive, uses sampling)
val silhouette = summary.silhouette(sampleFraction = 0.1)
println(s"Mean Silhouette Coefficient: $silhouette")
```

### Example 6: Weighted Time-Series Clustering

```scala
// Give more weight to recent observations
val timeSeriesData = spark.read.parquet("time_series.parquet")

// Calculate decay weights (more recent = higher weight)
val withWeights = timeSeriesData
  .withColumn("days_ago", datediff(current_date(), $"timestamp"))
  .withColumn("weight", exp(-$"days_ago" / 30.0))  // 30-day decay

val tsKMeans = new GeneralizedKMeans()
  .setK(8)
  .setWeightCol("weight")
  .setFeaturesCol("ts_features")
  .setMaxIter(40)

val tsModel = tsKMeans.fit(withWeights)

// Recent patterns will have more influence on cluster centers
val patterns = tsModel.transform(withWeights)
```

### Example 7: Cross-Validation for Hyperparameters

```scala
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.evaluation.ClusteringEvaluator

val kmeans = new GeneralizedKMeans()
  .setFeaturesCol("features")
  .setPredictionCol("prediction")

// Build parameter grid
val paramGrid = new ParamGridBuilder()
  .addGrid(kmeans.k, Array(3, 5, 7, 10))
  .addGrid(kmeans.maxIter, Array(20, 40))
  .addGrid(kmeans.divergence, Array("squaredEuclidean", "kl"))
  .build()

// Define evaluator (use silhouette score)
val evaluator = new ClusteringEvaluator()
  .setFeaturesCol("features")
  .setPredictionCol("prediction")
  .setMetricName("silhouette")

// Cross-validation
val cv = new CrossValidator()
  .setEstimator(kmeans)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)

val cvModel = cv.fit(trainingData)
val bestModel = cvModel.bestModel.asInstanceOf[GeneralizedKMeansModel]

println(s"Best k: ${bestModel.numClusters}")
```

### Example 8: Incremental/Mini-Batch Learning

```scala
// For very large datasets, train on samples and refine
val largeDat = spark.read.parquet("huge_dataset.parquet")

// Initial model on sample
val sample = largeData.sample(withReplacement = false, fraction = 0.1, seed = 42)

val initialModel = new GeneralizedKMeans()
  .setK(100)
  .setMaxIter(20)
  .fit(sample)

// Use initial centers for full dataset (one pass)
val fullModel = new GeneralizedKMeans()
  .setK(100)
  .setMaxIter(5)  // Just a few iterations
  .setInitMode("custom")  // if supported, or just use same seed
  .fit(largeData)

// Or: process in batches
var currentCenters = initialModel.clusterCenters
val batches = largeData.randomSplit(Array.fill(10)(0.1))

batches.foreach { batch =>
  // Train on batch starting from current centers
  // (requires custom initialization support)
  println(s"Processing batch with ${batch.count()} points")
}
```

## Bisecting K-Means (Hierarchical Divisive Clustering)

Bisecting K-Means is a hierarchical divisive clustering algorithm that offers several advantages over standard k-means:

- **More deterministic**: Less sensitive to initialization than random k-means
- **Better for imbalanced clusters**: Naturally adapts to varying cluster sizes
- **Often faster for large k**: Only splits locally, avoiding global recomputation
- **Higher quality**: Generally produces better clusters than random initialization

### Basic Bisecting K-Means

```scala
import com.massivedatascience.clusterer.ml.BisectingKMeans
import org.apache.spark.ml.linalg.Vectors

// Create data with natural hierarchical structure
val data = spark.createDataFrame(Seq(
  // Group A
  Vectors.dense(0.0, 0.0),
  Vectors.dense(0.1, 0.1),
  Vectors.dense(0.2, 0.2),
  // Group B
  Vectors.dense(5.0, 5.0),
  Vectors.dense(5.1, 5.1),
  Vectors.dense(5.2, 5.2),
  // Group C
  Vectors.dense(10.0, 10.0),
  Vectors.dense(10.1, 10.1),
  Vectors.dense(10.2, 10.2)
).map(Tuple1.apply)).toDF("features")

// Configure bisecting k-means
val bisecting = new BisectingKMeans()
  .setK(3)
  .setDivergence("squaredEuclidean")
  .setMaxIter(20)
  .setMinDivisibleClusterSize(1) // Minimum cluster size to allow splitting
  .setSeed(42)

val model = bisecting.fit(data)
val predictions = model.transform(data)

println(s"Number of clusters: ${model.numClusters}")
predictions.groupBy("prediction").count().show()
```

### Bisecting K-Means with KL Divergence

Works with all Bregman divergences, not just Euclidean:

```scala
// Probability distribution data
val probData = spark.createDataFrame(Seq(
  Vectors.dense(0.7, 0.2, 0.1),
  Vectors.dense(0.6, 0.3, 0.1),
  Vectors.dense(0.1, 0.2, 0.7),
  Vectors.dense(0.1, 0.3, 0.6),
  Vectors.dense(0.3, 0.4, 0.3),
  Vectors.dense(0.3, 0.5, 0.2)
).map(Tuple1.apply)).toDF("features")

val bisectingKL = new BisectingKMeans()
  .setK(3)
  .setDivergence("kl")
  .setSmoothing(1e-10)
  .setMaxIter(20)

val klModel = bisectingKL.fit(probData)
```

### Bisecting K-Means for Outlier-Robust Clustering

Using L1 (Manhattan) distance for robustness:

```scala
// Data with outliers
val dataWithOutliers = spark.createDataFrame(Seq(
  Vectors.dense(1.0, 1.0),
  Vectors.dense(1.1, 0.9),
  Vectors.dense(0.9, 1.1),
  Vectors.dense(100.0, 100.0), // Outlier
  Vectors.dense(10.0, 10.0),
  Vectors.dense(10.1, 9.9),
  Vectors.dense(9.9, 10.1)
).map(Tuple1.apply)).toDF("features")

val bisectingL1 = new BisectingKMeans()
  .setK(2)
  .setDivergence("l1") // or "manhattan"
  .setMaxIter(20)

val robustModel = bisectingL1.fit(dataWithOutliers)
// Centers are less affected by the outlier than standard k-means
```

### Weighted Bisecting K-Means

```scala
// Data with importance weights
val weightedData = spark.createDataFrame(Seq(
  (Vectors.dense(1.0, 1.0), 10.0),  // High importance
  (Vectors.dense(2.0, 2.0), 10.0),
  (Vectors.dense(10.0, 10.0), 1.0), // Low importance outlier
  (Vectors.dense(3.0, 3.0), 10.0),
  (Vectors.dense(4.0, 4.0), 10.0)
)).toDF("features", "weight")

val weightedBisecting = new BisectingKMeans()
  .setK(2)
  .setDivergence("squaredEuclidean")
  .setWeightCol("weight")
  .setMaxIter(20)

val weightedModel = weightedBisecting.fit(weightedData)
// Centers favor high-weight points
```

### Controlling Cluster Splits

The `minDivisibleClusterSize` parameter controls when clusters can be split:

```scala
// Only split clusters with at least 10 points
val bisecting = new BisectingKMeans()
  .setK(10)
  .setMinDivisibleClusterSize(10) // Won't split clusters smaller than this
  .setMaxIter(20)

val model = bisecting.fit(data)
// May produce fewer than 10 clusters if small clusters can't be split
println(s"Actual clusters: ${model.numClusters} (requested ${bisecting.getK})")
```

### When to Use Bisecting K-Means

**Use Bisecting K-Means when:**
- You want more deterministic results (less variation across runs)
- Your clusters have varying sizes (not all equal-sized)
- You need hierarchical structure (dendrogram-like splitting)
- You have a large value of k (> 10)
- Initialization quality matters more than per-iteration speed

**Use Standard K-Means when:**
- Clusters are roughly equal-sized
- You have very large datasets (billions of points)
- You need the absolute fastest iterations
- You're using mini-batch or streaming variants

### Comparing with Standard K-Means

```scala
// Same data, different algorithms
val testData = generateTestData()

// Standard k-means (random initialization)
val standardKM = new GeneralizedKMeans()
  .setK(5)
  .setMaxIter(20)
  .setSeed(42)

val standardModel = standardKM.fit(testData)
val standardCost = standardModel.computeCost(testData)

// Bisecting k-means
val bisectingKM = new BisectingKMeans()
  .setK(5)
  .setMaxIter(20)
  .setSeed(42)

val bisectingModel = bisectingKM.fit(testData)
val bisectingCost = bisectingModel.computeCost(testData)

println(s"Standard K-Means cost: $standardCost")
println(s"Bisecting K-Means cost: $bisectingCost")
// Bisecting often has lower cost due to better initialization
```

## X-Means (Automatic K Selection)

X-Means automatically determines the optimal number of clusters using information criteria (BIC or AIC), eliminating the need to specify k in advance.

### Basic X-Means - Finding Optimal K

```scala
import com.massivedatascience.clusterer.ml.XMeans
import org.apache.spark.ml.linalg.Vectors

// Create data with unknown number of natural clusters
val data = spark.createDataFrame(Seq(
  // Cluster 1
  Vectors.dense(0.0, 0.0),
  Vectors.dense(0.1, 0.1),
  Vectors.dense(0.2, 0.2),
  // Cluster 2
  Vectors.dense(5.0, 5.0),
  Vectors.dense(5.1, 5.1),
  Vectors.dense(5.2, 5.2),
  // Cluster 3
  Vectors.dense(10.0, 10.0),
  Vectors.dense(10.1, 10.1),
  Vectors.dense(10.2, 10.2)
).map(Tuple1.apply)).toDF("features")

// X-means will try k from 2 to 10 and pick the best
val xmeans = new XMeans()
  .setMinK(2)
  .setMaxK(10)
  .setCriterion("bic") // or "aic"
  .setDivergence("squaredEuclidean")
  .setMaxIter(20)
  .setSeed(42)

val model = xmeans.fit(data)

println(s"Optimal k = ${model.numClusters}")
// Output: Optimal k = 3

val predictions = model.transform(data)
predictions.groupBy("prediction").count().show()
```

### X-Means with BIC vs AIC

BIC (Bayesian Information Criterion) penalizes complexity more than AIC, preferring simpler models:

```scala
// BIC - prefers fewer clusters
val xmeansBIC = new XMeans()
  .setMinK(2)
  .setMaxK(10)
  .setCriterion("bic")

val modelBIC = xmeansBIC.fit(data)
println(s"BIC selected k = ${modelBIC.numClusters}")

// AIC - allows more complex models
val xmeansAIC = new XMeans()
  .setMinK(2)
  .setMaxK(10)
  .setCriterion("aic")

val modelAIC = xmeansAIC.fit(data)
println(s"AIC selected k = ${modelAIC.numClusters}")

// AIC often selects k >= BIC's k
```

### X-Means with Different Divergences

Works with all Bregman divergences:

```scala
// X-Means with KL divergence for probability distributions
val probData = spark.createDataFrame(Seq(
  Vectors.dense(0.7, 0.2, 0.1),
  Vectors.dense(0.6, 0.3, 0.1),
  Vectors.dense(0.1, 0.2, 0.7),
  Vectors.dense(0.1, 0.3, 0.6)
).map(Tuple1.apply)).toDF("features")

val xmeansKL = new XMeans()
  .setMinK(2)
  .setMaxK(4)
  .setDivergence("kl")
  .setSmoothing(1e-10)

val klModel = xmeansKL.fit(probData)
println(s"KL divergence selected k = ${klModel.numClusters}")

// X-Means with L1 for outlier robustness
val xmeansL1 = new XMeans()
  .setMinK(2)
  .setMaxK(5)
  .setDivergence("l1")

val l1Model = xmeansL1.fit(data)
println(s"L1 divergence selected k = ${l1Model.numClusters}")
```

### X-Means with Weighted Data

```scala
val weightedData = spark.createDataFrame(Seq(
  (Vectors.dense(1.0, 1.0), 10.0),  // High importance
  (Vectors.dense(1.1, 0.9), 10.0),
  (Vectors.dense(9.0, 9.0), 1.0),   // Low importance
  (Vectors.dense(9.1, 8.9), 1.0)
)).toDF("features", "weight")

val xmeans = new XMeans()
  .setMinK(2)
  .setMaxK(4)
  .setWeightCol("weight")

val model = xmeans.fit(weightedData)
// High-weight points have more influence on cluster selection
```

### Controlling the Search Range

```scala
// Narrow search for faster execution
val fastXMeans = new XMeans()
  .setMinK(3)
  .setMaxK(6)
  .setMaxIter(10) // Fewer iterations per k

val model = fastXMeans.fit(data)

// Wide search for thorough exploration
val thoroughXMeans = new XMeans()
  .setMinK(2)
  .setMaxK(50)
  .setMaxIter(30)
  .setImprovementThreshold(-0.1) // Stop if improvement < 0.1

val thoroughModel = thoroughXMeans.fit(largeData)
```

### When to Use X-Means

**Use X-Means when:**
- You don't know the optimal number of clusters
- You want principled k selection (not just elbow method)
- You're doing exploratory data analysis
- You need reproducible k selection
- You want to avoid grid search over k

**Use Standard K-Means when:**
- You know the target k (e.g., business requirement)
- You need maximum speed (X-Means tries multiple k values)
- You're using streaming/mini-batch variants
- k is very large (> 50)

### Information Criteria Explained

**BIC (Bayesian Information Criterion)**:
```
BIC = -2 * log-likelihood + p * log(n)
where p = k*d + 1 (parameters)
      n = number of points
```
- Stronger complexity penalty
- Prefers simpler models (fewer clusters)
- Better for avoiding overfitting

**AIC (Akaike Information Criterion)**:
```
AIC = -2 * log-likelihood + 2 * p
```
- Weaker complexity penalty
- Allows more complex models
- Better for capturing subtle patterns

**In practice**:
- Start with BIC for conservative estimates
- Try AIC if BIC seems too simple
- Compare both to understand data structure

### Comparing X-Means with Manual K Selection

```scala
// Manual approach - try many k values
val kValues = (2 to 10).map { k =>
  val kmeans = new GeneralizedKMeans().setK(k).setMaxIter(20)
  val model = kmeans.fit(data)
  val cost = model.computeCost(data)
  (k, cost)
}

// Find elbow manually (subjective)
kValues.foreach { case (k, cost) =>
  println(s"k=$k, cost=$cost")
}

// X-Means approach - automatic with statistical criterion
val xmeans = new XMeans()
  .setMinK(2)
  .setMaxK(10)
  .setCriterion("bic")

val model = xmeans.fit(data)
println(s"X-Means automatically selected k=${model.numClusters}")
// More principled and reproducible than visual elbow method
```

## Soft K-Means (Fuzzy C-Means)

Soft K-Means provides probabilistic cluster memberships instead of hard assignments. Each point belongs to multiple clusters with varying probabilities, making it ideal for overlapping clusters and mixture model estimation.

### Basic Soft K-Means - Probabilistic Memberships

```scala
import com.massivedatascience.clusterer.ml.SoftKMeans
import org.apache.spark.ml.linalg.Vectors

// Create sample data with some overlap between clusters
val data = Seq(
  Vectors.dense(1.0, 1.0),
  Vectors.dense(1.2, 0.9),
  Vectors.dense(2.0, 2.0),  // Ambiguous point between clusters
  Vectors.dense(5.0, 5.0),
  Vectors.dense(5.1, 4.9)
)

val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

val softKMeans = new SoftKMeans()
  .setK(2)
  .setBeta(2.0)          // Control assignment sharpness
  .setMaxIter(50)
  .setSeed(42)

val model = softKMeans.fit(df)

// Transform adds both hard prediction and soft probabilities
val result = model.transform(df)
result.select("features", "prediction", "probabilities").show(false)

// Output:
// +----------+----------+----------------------------------------+
// |features  |prediction|probabilities                           |
// +----------+----------+----------------------------------------+
// |[1.0,1.0] |0         |[0.9523,0.0477]   # Strongly cluster 0 |
// |[1.2,0.9] |0         |[0.9312,0.0688]   # Strongly cluster 0 |
// |[2.0,2.0] |0         |[0.6234,0.3766]   # Mixed membership!   |
// |[5.0,5.0] |1         |[0.0412,0.9588]   # Strongly cluster 1 |
// |[5.1,4.9] |1         |[0.0389,0.9611]   # Strongly cluster 1 |
// +----------+----------+----------------------------------------+
```

### Controlling Soft vs Sharp Assignments with Beta

The `beta` parameter controls how "fuzzy" the clustering is:
- **Low beta (0.1-1.0)**: Very soft, fuzzy assignments
- **Medium beta (1.0-5.0)**: Balanced soft clustering
- **High beta (5.0-100.0)**: Sharp assignments (approaches hard clustering)

```scala
// Very Soft Clustering (high uncertainty)
val verySoftKMeans = new SoftKMeans()
  .setK(2)
  .setBeta(0.5)  // Low beta = very soft
  .setMaxIter(50)

val softModel = verySoftKMeans.fit(df)
val softPreds = softModel.transform(df)

// Check entropy (higher = more uncertainty)
import org.apache.spark.sql.functions._

val avgEntropy = softPreds.select(
  avg(
    expr("aggregate(probabilities.values, 0.0, (acc, p) -> acc - p * ln(p))")
  ).as("avgEntropy")
).head().getDouble(0)

println(s"Soft clustering average entropy: $avgEntropy")
// Output: ~0.45 (higher uncertainty)

// Sharp Clustering (low uncertainty)
val sharpKMeans = new SoftKMeans()
  .setK(2)
  .setBeta(10.0)  // High beta = sharp
  .setMaxIter(50)

val sharpModel = sharpKMeans.fit(df)
val sharpPreds = sharpModel.transform(df)

val sharpEntropy = sharpPreds.select(
  avg(
    expr("aggregate(probabilities.values, 0.0, (acc, p) -> acc - p * ln(p))")
  ).as("avgEntropy")
).head().getDouble(0)

println(s"Sharp clustering average entropy: $sharpEntropy")
// Output: ~0.12 (lower uncertainty, more decisive)
```

### Soft K-Means with Different Divergences

Like other clustering algorithms, Soft K-Means supports all Bregman divergences:

```scala
// L1 divergence (Manhattan distance) - robust to outliers
val l1SoftKMeans = new SoftKMeans()
  .setK(3)
  .setDivergence("l1")
  .setBeta(2.0)
  .setMaxIter(50)

val l1Model = l1SoftKMeans.fit(df)

// KL divergence - for probability distributions
val klSoftKMeans = new SoftKMeans()
  .setK(3)
  .setDivergence("kl")
  .setSmoothing(1e-5)
  .setBeta(1.5)
  .setMaxIter(50)

val klModel = klSoftKMeans.fit(df)
```

### Soft K-Means with Weighted Data

Weight important points more heavily during clustering:

```scala
val weightedData = Seq(
  (Vectors.dense(1.0, 1.0), 10.0),  // High weight
  (Vectors.dense(1.1, 0.9), 10.0),
  (Vectors.dense(5.0, 5.0), 1.0),   // Low weight
  (Vectors.dense(5.1, 4.9), 1.0)
)

val weightedDF = spark.createDataFrame(weightedData).toDF("features", "weight")

val weightedSoftKMeans = new SoftKMeans()
  .setK(2)
  .setWeightCol("weight")
  .setBeta(2.0)
  .setMaxIter(50)

val weightedModel = weightedSoftKMeans.fit(weightedDF)

// Centers will be pulled toward high-weight points
println("Cluster centers:")
weightedModel.clusterCenters.foreach(c => println(s"  $c"))
```

### Single Point Predictions

Get soft probabilities for individual points:

```scala
// Hard prediction (most likely cluster)
val testPoint = Vectors.dense(2.5, 2.5)
val hardPrediction = model.predict(testPoint)
println(s"Hard prediction: $hardPrediction")

// Soft prediction (probability distribution)
val softPrediction = model.predictSoft(testPoint)
println(s"Soft probabilities: ${softPrediction.toArray.mkString("[", ", ", "]")}")
// Output: [0.4123, 0.5877] - point has mixed membership
```

### Mixture Model Estimation

Soft K-Means can estimate mixture model parameters:

```scala
// Generate data from a mixture of Gaussians
import breeze.stats.distributions.Gaussian

val mixture1 = Gaussian(0.0, 1.0)
val mixture2 = Gaussian(5.0, 1.0)

val mixtureData = (1 to 100).map { i =>
  val value = if (i % 2 == 0) mixture1.sample() else mixture2.sample()
  Vectors.dense(value, value)
}

val mixtureDF = spark.createDataFrame(mixtureData.map(Tuple1.apply)).toDF("features")

val mixtureKMeans = new SoftKMeans()
  .setK(2)
  .setBeta(1.0)  // Moderate softness for mixture models
  .setMaxIter(100)

val mixtureModel = mixtureKMeans.fit(mixtureDF)

// Cluster centers approximate mixture means
println("Estimated mixture centers:")
mixtureModel.clusterCenters.foreach(c => println(s"  $c"))

// Probabilities represent mixture weights
val result = mixtureModel.transform(mixtureDF)
result.select("probabilities").show(5, false)
```

### Cost Functions

Soft K-Means provides both hard and soft cost functions:

```scala
// Hard cost (traditional K-means objective)
val hardCost = model.computeCost(df)
println(s"Hard clustering cost: $hardCost")

// Soft cost (weighted by membership probabilities)
val softCost = model.computeSoftCost(df)
println(s"Soft clustering cost: $softCost")

// Soft cost is always <= hard cost
assert(softCost <= hardCost)
```

### Effective Number of Clusters

Measure the "effective" number of clusters based on entropy:

```scala
val effectiveClusters = model.effectiveNumberOfClusters(df)
println(s"Effective number of clusters: $effectiveClusters")

// With k=3 and soft assignments:
// - If all points are evenly distributed: ~3.0
// - If assignments are very sharp: ~1.0-2.0
// - With mixed memberships: somewhere in between

// This metric helps you understand if you're over/under-clustering
```

### When to Use Soft K-Means

**Use Soft K-Means when:**
- Clusters naturally overlap
- You need uncertainty estimates (e.g., "this customer is 70% segment A, 30% segment B")
- Building mixture models or generative models
- Outliers shouldn't be forced into a single cluster
- Modeling fuzzy categories (e.g., "somewhat happy", "very happy")

**Use Hard K-Means when:**
- Clusters are well-separated
- You need deterministic, crisp assignments
- Performance is critical (soft clustering is slower)
- Downstream tasks require single cluster labels

### Soft K-Means vs Hard K-Means Comparison

```scala
// Same data, both algorithms
val hardKMeans = new GeneralizedKMeans()
  .setK(2)
  .setMaxIter(50)
  .setSeed(42)

val hardModel = hardKMeans.fit(df)
val hardResult = hardModel.transform(df)

val softKMeans = new SoftKMeans()
  .setK(2)
  .setBeta(2.0)
  .setMaxIter(50)
  .setSeed(42)

val softModel = softKMeans.fit(df)
val softResult = softModel.transform(df)

// Compare assignments for ambiguous points
val comparison = hardResult.select("features", "prediction")
  .join(softResult.select("features", "probabilities"), "features")

comparison.show(false)

// Hard K-Means: Forces every point into exactly one cluster
// Soft K-Means: Allows mixed membership - more informative for borderline cases
```

### Model Persistence

Save and load Soft K-Means models:

```scala
// Save model
model.write.overwrite().save("path/to/soft-kmeans-model")

// Load model
import com.massivedatascience.clusterer.ml.SoftKMeansModel

val loadedModel = SoftKMeansModel.load("path/to/soft-kmeans-model")

// Use loaded model
val predictions = loadedModel.transform(df)
```

### Advanced: Annealing Schedule

Start with soft clustering and gradually sharpen assignments:

```scala
var beta = 0.5  // Start soft
val betaMax = 10.0
val annealingSteps = 5

var currentModel = new SoftKMeans()
  .setK(3)
  .setBeta(beta)
  .setMaxIter(20)
  .fit(df)

(1 to annealingSteps).foreach { step =>
  beta = beta * 2.0
  if (beta > betaMax) beta = betaMax

  println(s"Step $step: beta = $beta")

  currentModel = new SoftKMeans()
    .setK(3)
    .setBeta(beta)
    .setMaxIter(20)
    .fit(df)

  val effectiveClusters = currentModel.effectiveNumberOfClusters(df)
  println(s"  Effective clusters: $effectiveClusters")
}

// Gradual sharpening can help avoid poor local optima
```

---

## Streaming K-Means (Real-Time Clustering)

Streaming K-Means enables **incremental online clustering** with exponential forgetting for streaming data. The model updates incrementally as new batches arrive, making it ideal for real-time applications where data continuously evolves.

### Basic Streaming K-Means

```scala
import com.massivedatascience.clusterer.ml.StreamingKMeans
import org.apache.spark.ml.linalg.Vectors

// Initialize with batch data
val initialData = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.0, 0.0)),
  Tuple1(Vectors.dense(0.1, 0.1)),
  Tuple1(Vectors.dense(5.0, 5.0)),
  Tuple1(Vectors.dense(5.1, 5.1))
)).toDF("features")

// Create streaming K-Means estimator
val streamingKMeans = new StreamingKMeans()
  .setK(2)
  .setMaxIter(10)
  .setDecayFactor(0.9)  // 0.0 = complete forgetting, 1.0 = no forgetting
  .setTimeUnit("batches")  // or "points"
  .setSeed(42)

// Fit initial model
val model = streamingKMeans.fit(initialData)

println(s"Initial cluster centers:")
model.clusterCentersAsVectors.foreach(c => println(s"  $c"))

// Update with new batch
val newBatch = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.2, 0.2)),
  Tuple1(Vectors.dense(5.2, 5.2))
)).toDF("features")

model.update(newBatch)

println(s"\nUpdated cluster centers:")
model.clusterCentersAsVectors.foreach(c => println(s"  $c"))

// Cluster weights track data density
println(s"\nCluster weights: ${model.currentWeights.mkString(", ")}")
```

**Output:**
```
Initial cluster centers:
  [0.05, 0.05]
  [5.05, 5.05]

Updated cluster centers:
  [0.065, 0.065]
  [5.065, 5.065]

Cluster weights: 11.8, 11.8
```

### Decay Factor: Forgetting Old Data

The **decay factor** (α) controls how quickly old data is forgotten:

- **α = 1.0**: No forgetting (all batches weighted equally)
- **α = 0.5**: Moderate forgetting (half-life decay)
- **α = 0.0**: Complete forgetting (only current batch matters)

```scala
// High forgetting (decay = 0.1) - adapts quickly to new data
val highForgetting = new StreamingKMeans()
  .setK(2)
  .setMaxIter(10)
  .setDecayFactor(0.1)  // 90% discount on old data
  .setSeed(42)
  .fit(initialData)

// Low forgetting (decay = 0.9) - retains historical data
val lowForgetting = new StreamingKMeans()
  .setK(2)
  .setMaxIter(10)
  .setDecayFactor(0.9)  // 10% discount on old data
  .setSeed(42)
  .fit(initialData)

// Update both with data from different distribution
val driftedBatch = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(10.0, 10.0)),
  Tuple1(Vectors.dense(10.1, 10.1)),
  Tuple1(Vectors.dense(15.0, 15.0)),
  Tuple1(Vectors.dense(15.1, 15.1))
)).toDF("features")

highForgetting.update(driftedBatch)
lowForgetting.update(driftedBatch)

println("High forgetting centers (adapted to new data):")
highForgetting.clusterCentersAsVectors.foreach(c => println(s"  $c"))

println("\nLow forgetting centers (still influenced by old data):")
lowForgetting.clusterCentersAsVectors.foreach(c => println(s"  $c"))

// High forgetting shifts centers more toward new data
```

**Use Cases:**
- **High decay (0.1-0.3)**: Concept drift detection, rapidly changing data
- **Medium decay (0.5-0.7)**: Gradual adaptation, seasonal patterns
- **Low decay (0.9-0.99)**: Stable environments, noise filtering

### Half-Life Parameter

Instead of specifying decay factor directly, use **half-life** for intuitive control:

```scala
// Half-life: time for data weight to decay to 50%
val model = new StreamingKMeans()
  .setK(3)
  .setMaxIter(10)
  .setHalfLife(5.0)  // Weight halves every 5 batches (or points)
  .setTimeUnit("batches")

// Equivalent decay factor: 0.5^(1/5) ≈ 0.871
println(s"Computed decay factor: ${model.fit(initialData).decayFactorValue}")
```

**Half-Life Examples:**
- `halfLife = 1.0`: Weights halve every batch (very fast decay)
- `halfLife = 10.0`: Weights halve every 10 batches (moderate decay)
- `halfLife = 100.0`: Weights halve every 100 batches (slow decay)

### Time Units: Batches vs Points

The **time unit** affects how decay is applied:

```scala
// Decay per batch (default)
val batchDecay = new StreamingKMeans()
  .setK(2)
  .setMaxIter(10)
  .setDecayFactor(0.5)
  .setTimeUnit("batches")  // Discount once per batch
  .fit(initialData)

// Decay per point (scaled by batch size)
val pointDecay = new StreamingKMeans()
  .setK(2)
  .setMaxIter(10)
  .setDecayFactor(0.5)
  .setTimeUnit("points")  // Discount = 0.5^(num_points)
  .fit(initialData)

val smallBatch = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(1.0, 1.0))
)).toDF("features")

batchDecay.update(smallBatch)
pointDecay.update(smallBatch)

println("Batch decay (discount = 0.5):")
println(s"  Weights: ${batchDecay.currentWeights.mkString(", ")}")

println("\nPoint decay (discount = 0.5^1 = 0.5):")
println(s"  Weights: ${pointDecay.currentWeights.mkString(", ")}")

// Batch decay: independent of batch size
// Point decay: scales with batch size (smoother for variable-size batches)
```

### Streaming K-Means with Different Divergences

All Bregman divergences work with streaming updates:

```scala
// L1 divergence (Manhattan distance) - robust to outliers
val l1Streaming = new StreamingKMeans()
  .setK(2)
  .setMaxIter(10)
  .setDivergence("l1")
  .setDecayFactor(0.8)
  .fit(initialData)

// KL divergence for probability distributions
val probData = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.7, 0.2, 0.1)),
  Tuple1(Vectors.dense(0.6, 0.3, 0.1))
)).toDF("features")

val klStreaming = new StreamingKMeans()
  .setK(2)
  .setMaxIter(10)
  .setDivergence("kl")
  .setSmoothing(1e-10)
  .setDecayFactor(0.85)
  .fit(probData)

// Update with new probability distributions
val newProbs = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.65, 0.25, 0.1))
)).toDF("features")

klStreaming.update(newProbs)
```

### Streaming K-Means with Weighted Data

Handle importance-weighted data streams:

```scala
val weightedData = spark.createDataFrame(Seq(
  (Vectors.dense(0.0, 0.0), 10.0),  // High importance
  (Vectors.dense(0.1, 0.1), 10.0),
  (Vectors.dense(5.0, 5.0), 1.0),   // Low importance
  (Vectors.dense(5.1, 5.1), 1.0)
)).toDF("features", "weight")

val weightedStreaming = new StreamingKMeans()
  .setK(2)
  .setMaxIter(10)
  .setWeightCol("weight")
  .setDecayFactor(0.9)
  .fit(weightedData)

// Centers favor high-weight points
println("Weighted cluster centers:")
weightedStreaming.clusterCentersAsVectors.foreach(c => println(s"  $c"))

// Update with weighted batch
val weightedBatch = spark.createDataFrame(Seq(
  (Vectors.dense(0.2, 0.2), 5.0),
  (Vectors.dense(5.2, 5.2), 1.0)
)).toDF("features", "weight")

weightedStreaming.update(weightedBatch)
```

### Integration with Structured Streaming

Use `StreamingKMeansUpdater` for Spark Structured Streaming:

```scala
import org.apache.spark.sql.streaming.Trigger
import scala.concurrent.duration._

// Create streaming source (example: rate source)
val streamingData = spark.readStream
  .format("rate")
  .option("rowsPerSecond", "10")
  .load()
  .selectExpr("CAST(value AS DOUBLE) AS x", "CAST(timestamp AS DOUBLE) / 1e9 AS y")
  .selectExpr("array(x, y) AS features")
  .selectExpr("struct(features) AS features")  // Convert to Vector

// Initialize model with batch data
val model = new StreamingKMeans()
  .setK(3)
  .setMaxIter(10)
  .setDecayFactor(0.8)
  .setTimeUnit("batches")
  .fit(initialData)

// Create streaming updater
val updater = model.createStreamingUpdater()

// Start streaming query with foreachBatch
val query = updater.updateOn(
  streamingDF = streamingData,
  checkpointLocation = Some("/tmp/streaming-kmeans-checkpoint")
)

// Monitor streaming updates
Thread.sleep(5000)

println(s"Batches processed: ${updater.batchesProcessed}")
println("Current cluster centers:")
updater.currentModel.clusterCentersAsVectors.foreach(c => println(s"  $c"))

// Make predictions on current model
val predictions = updater.transform(initialData)
predictions.show()

// Stop query
query.stop()
```

### Dying Cluster Handling

Streaming K-Means automatically handles **dying clusters** (clusters that receive no data):

```scala
val model = new StreamingKMeans()
  .setK(3)
  .setMaxIter(10)
  .setDecayFactor(0.5)  // Fast decay
  .fit(initialData)

// Repeatedly update with data only in 2 clusters
(1 to 20).foreach { _ =>
  val batch = spark.createDataFrame(Seq(
    Tuple1(Vectors.dense(0.0, 0.0)),
    Tuple1(Vectors.dense(0.1, 0.1))
  )).toDF("features")

  model.update(batch)
}

// Third cluster was dying, so it gets split from largest cluster
println("Cluster weights after 20 updates:")
println(s"  ${model.currentWeights.mkString(", ")}")

// All clusters still have reasonable weights due to splitting
model.currentWeights.foreach { w =>
  assert(w > 0.01, "No cluster should be completely dead")
}
```

**Splitting Logic:**
- Detects when `min_weight < 1e-8 * max_weight`
- Splits largest cluster into two slightly perturbed copies
- Redistributes weight equally between largest and smallest

### Multiple Sequential Updates

Process multiple batches sequentially:

```scala
val model = new StreamingKMeans()
  .setK(2)
  .setMaxIter(10)
  .setDecayFactor(0.9)
  .setSeed(42)
  .fit(initialData)

val initialCenters = model.currentCenters.map(_.copy)

// Simulate 10 batches of streaming data
(1 to 10).foreach { i =>
  val batch = spark.createDataFrame(Seq(
    Tuple1(Vectors.dense(i * 0.1, i * 0.1)),
    Tuple1(Vectors.dense(5.0 + i * 0.1, 5.0 + i * 0.1))
  )).toDF("features")

  model.update(batch)

  if (i % 3 == 0) {
    println(s"After batch $i:")
    model.clusterCentersAsVectors.foreach(c => println(s"  $c"))
  }
}

val finalCenters = model.currentCenters

// Centers have evolved gradually
println(s"\nTotal shift: ${
  initialCenters.zip(finalCenters).map { case (init, final_) =>
    math.sqrt((0 until init.size).map { i =>
      val d = init(i) - final_(i)
      d * d
    }.sum)
  }.sum
}")
```

### Cost Computation for Streaming Models

Evaluate clustering quality on any dataset:

```scala
val model = new StreamingKMeans()
  .setK(2)
  .setMaxIter(10)
  .setDecayFactor(0.9)
  .fit(initialData)

// Compute cost on training data
val trainCost = model.computeCost(initialData)
println(s"Training cost: $trainCost")

// Update model
model.update(newBatch)

// Compute cost on new data
val newCost = model.computeCost(newBatch)
println(s"New batch cost: $newCost")

// Lower cost indicates better fit
```

### When to Use Streaming K-Means

**✅ Use Streaming K-Means when:**
- Data arrives continuously (sensors, logs, user events)
- Concept drift is expected (changing patterns over time)
- Memory constraints prevent storing all historical data
- Real-time clustering results are needed
- Data distribution evolves gradually

**❌ Use Batch K-Means when:**
- Complete dataset fits in memory
- Data distribution is stationary
- Retraining from scratch is acceptable
- Highest accuracy is critical (no approximation)

### Streaming K-Means Update Rule

The mini-batch update formula with exponential forgetting:

```
For cluster c at time t:
  c_{t+1} = [(c_t * n_t * α) + (x_t * m_t)] / [n_t * α + m_t]
  n_{t+1} = n_t * α + m_t

Where:
  c_t    = current center
  n_t    = current weight (cumulative count)
  x_t    = mean of new points assigned to cluster c
  m_t    = number of new points assigned to cluster c
  α      = decay factor (discounts old data)
```

**Example:**
```scala
// After 100 points with decay=0.9:
//   Effective weight of original data ≈ 100 * 0.9^10 ≈ 35 points
//   Recent 10 points have cumulative weight ≈ 10 points
//   Total weight ≈ 45 points (weighted toward recent data)
```

### Advanced: Custom Decay Schedules

Combine multiple models with different decay rates:

```scala
// Fast-adapting model (recent trends)
val fastModel = new StreamingKMeans()
  .setK(2)
  .setDecayFactor(0.3)
  .fit(initialData)

// Slow-adapting model (long-term patterns)
val slowModel = new StreamingKMeans()
  .setK(2)
  .setDecayFactor(0.95)
  .fit(initialData)

// Update both
(1 to 10).foreach { _ =>
  val batch = generateBatch()  // Your data generation logic
  fastModel.update(batch)
  slowModel.update(batch)
}

// Compare divergence between fast and slow models
val fastCenters = fastModel.clusterCentersAsVectors
val slowCenters = slowModel.clusterCentersAsVectors

val divergence = fastCenters.zip(slowCenters).map { case (f, s) =>
  math.sqrt((0 until f.size).map { i =>
    val d = f(i) - s(i)
    d * d
  }.sum)
}.sum

println(s"Model divergence: $divergence")

// High divergence → concept drift detected!
if (divergence > 1.0) {
  println("⚠️ Significant concept drift detected!")
}
```

---

## Next Steps

- **Architecture Guide**: [ARCHITECTURE.md](ARCHITECTURE.md) - Deep dive into design patterns
- **Migration Guide**: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migrate from RDD API
- **Performance Tuning**: [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) - Optimization tips
- **Test Suites**:
  - `src/test/scala/com/massivedatascience/clusterer/ml/GeneralizedKMeansSuite.scala`
  - `src/test/scala/com/massivedatascience/clusterer/BisectingKMeansSuite.scala`
  - `src/test/scala/com/massivedatascience/clusterer/XMeansSuite.scala`
  - `src/test/scala/com/massivedatascience/clusterer/SoftKMeansSuite.scala`
  - `src/test/scala/com/massivedatascience/clusterer/StreamingKMeansSuite.scala`
- **Legacy RDD API**: Still available for backward compatibility
