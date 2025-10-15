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

## Next Steps

- **Architecture Guide**: [ARCHITECTURE.md](ARCHITECTURE.md) - Deep dive into design patterns
- **Migration Guide**: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migrate from RDD API
- **Performance Tuning**: [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) - Optimization tips
- **Test Suite**: `src/test/scala/com/massivedatascience/clusterer/ml/GeneralizedKMeansSuite.scala`
- **Bisecting K-Means Tests**: `src/test/scala/com/massivedatascience/clusterer/BisectingKMeansSuite.scala`
- **Legacy RDD API**: Still available for backward compatibility
