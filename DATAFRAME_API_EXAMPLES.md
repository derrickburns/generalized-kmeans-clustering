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

## Next Steps

- See test suite: `src/test/scala/com/massivedatascience/clusterer/ml/GeneralizedKMeansSuite.scala`
- Architecture details: `DF_ML_REFACTORING_PLAN.md`
- Legacy RDD API: Still available for backward compatibility
