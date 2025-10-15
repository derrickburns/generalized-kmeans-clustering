# PySpark Examples

This directory contains example scripts and notebooks demonstrating the massivedatascience-clusterer library.

## Running the Examples

### Prerequisites

```bash
# Install the package
pip install massivedatascience-clusterer

# Or install from source
cd python
pip install -e .
```

### Running Scripts

Each example is a standalone Python script that can be run directly:

```bash
python basic_clustering.py
python kl_divergence_clustering.py
python weighted_clustering.py
python finding_optimal_k.py
python model_persistence.py
```

### Running the Jupyter Notebook

```bash
# Install Jupyter if needed
pip install jupyter matplotlib

# Start Jupyter
jupyter notebook clustering_tutorial.ipynb
```

## Example Descriptions

### 1. `basic_clustering.py`
Demonstrates basic K-means clustering with Squared Euclidean distance.

**Covers:**
- Creating a Spark session
- Training a clustering model
- Making predictions
- Computing quality metrics (WCSS, BCSS, Calinski-Harabasz, Davies-Bouldin)
- Predicting cluster for single points

**Run time:** ~10 seconds

### 2. `kl_divergence_clustering.py`
Clustering probability distributions using KL divergence.

**Covers:**
- Working with probability distribution data (document word frequencies)
- Using KL divergence for clustering
- Importance of the smoothing parameter
- Analyzing cluster centers in probability space

**Use cases:** Document clustering, topic modeling, categorical data

**Run time:** ~10 seconds

### 3. `weighted_clustering.py`
Demonstrates weighted clustering where points have different importance.

**Covers:**
- Using the `weightCol` parameter
- How weights influence cluster centers
- Comparison between weighted and unweighted clustering

**Use cases:** User behavior (frequent vs. infrequent users), fraud detection (flag suspicious transactions with higher weights)

**Run time:** ~15 seconds

### 4. `finding_optimal_k.py`
Finding the optimal number of clusters using multiple quality metrics.

**Covers:**
- Testing multiple values of k
- Elbow method with WCSS
- Calinski-Harabasz Index (higher is better)
- Davies-Bouldin Index (lower is better)
- Visualizing quality metrics

**Output:** Creates visualization at `/tmp/optimal_k_analysis.png`

**Run time:** ~20 seconds

### 5. `model_persistence.py`
Saving and loading trained models.

**Covers:**
- Saving models to disk
- Loading models from disk
- Verifying loaded models produce identical predictions
- Using loaded models for new predictions

**Use cases:** Production deployments, model versioning, batch vs. real-time prediction

**Run time:** ~10 seconds

### 6. `clustering_tutorial.ipynb`
Comprehensive Jupyter notebook tutorial covering all major features.

**Covers:**
- All examples above in an interactive format
- Visualizations of clusters
- Step-by-step explanations
- Quality metric comparisons

**Run time:** ~2 minutes (running all cells)

## Advanced Usage

### Using Different Divergences

The library supports multiple Bregman divergences:

```python
# Squared Euclidean (default)
kmeans = GeneralizedKMeans(k=3, divergence="squaredEuclidean")

# KL divergence (for probability distributions)
kmeans = GeneralizedKMeans(k=3, divergence="kl", smoothing=1e-10)

# Itakura-Saito (for audio/spectral data)
kmeans = GeneralizedKMeans(k=3, divergence="itakuraSaito")

# Generalized I-divergence
kmeans = GeneralizedKMeans(k=3, divergence="generalizedI")

# Logistic Loss
kmeans = GeneralizedKMeans(k=3, divergence="logisticLoss")
```

### Initialization Modes

```python
# Random initialization
kmeans = GeneralizedKMeans(k=3, initMode="random")

# K-means|| parallel initialization (default)
kmeans = GeneralizedKMeans(k=3, initMode="k-means||")
```

### Assignment Strategies

```python
# Auto-select strategy based on data size (default)
kmeans = GeneralizedKMeans(k=3, assignmentStrategy="auto")

# Always use broadcast strategy (good for small k)
kmeans = GeneralizedKMeans(k=3, assignmentStrategy="broadcast")
```

### Integration with Spark ML Pipelines

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

# Assemble features
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3"],
    outputCol="features"
)

# Create clustering stage
kmeans = GeneralizedKMeans(k=5, maxIter=20)

# Build pipeline
pipeline = Pipeline(stages=[assembler, kmeans])

# Fit pipeline
model = pipeline.fit(data)

# Make predictions
predictions = model.transform(data)
```

## Performance Tips

1. **Cache data** when fitting multiple models:
   ```python
   data_cached = data.cache()
   for k in range(2, 10):
       model = GeneralizedKMeans(k=k).fit(data_cached)
   ```

2. **Set appropriate partition count**:
   ```python
   data_repartitioned = data.repartition(100)  # Adjust based on cluster size
   ```

3. **Use broadcast strategy** for small k (< 100):
   ```python
   kmeans = GeneralizedKMeans(k=10, assignmentStrategy="broadcast")
   ```

4. **Enable checkpointing** for large datasets:
   ```python
   spark.sparkContext.setCheckpointDir("/tmp/checkpoints")
   kmeans = GeneralizedKMeans(k=5, checkpointInterval=10)
   ```

## Troubleshooting

### Issue: Slow convergence
**Solution:**
- Increase `tol` parameter (e.g., `tol=1e-3` instead of `1e-4`)
- Reduce `maxIter` if acceptable
- Try different `initMode`

### Issue: Empty clusters
**Solution:**
- Use `k-means||` initialization (default)
- Reduce k
- Ensure data is properly scaled

### Issue: Out of memory
**Solution:**
- Reduce `k`
- Use `broadcast` assignment strategy
- Increase executor memory
- Repartition data to more partitions

### Issue: NaN values in results
**Solution:**
- For KL divergence, increase `smoothing` parameter
- Check for zero or negative values in input data
- Verify probability distributions sum to 1.0

## Additional Resources

- **Main Documentation**: [GitHub README](https://github.com/massivedatascience/generalized-kmeans-clustering)
- **Architecture Guide**: See `ARCHITECTURE.md` in the repository
- **Performance Tuning**: See `PERFORMANCE_TUNING.md` in the repository
- **Migration Guide**: See `MIGRATION_GUIDE.md` for upgrading from older versions

## Questions or Issues?

Please file issues at: https://github.com/massivedatascience/generalized-kmeans-clustering/issues
