# PySpark Wrapper for Generalized K-Means Clustering

Python bindings for the generalized-kmeans-clustering Scala library, providing native PySpark integration for clustering with Bregman divergences.

## Features

- **Full Spark ML Integration**: Native Estimator/Model pattern with Pipeline support
- **Multiple Distance Functions**:
  - Squared Euclidean (default)
  - KL Divergence
  - Itakura-Saito
  - Generalized I-divergence
  - Logistic Loss
- **Quality Metrics**: WCSS, BCSS, Calinski-Harabasz, Davies-Bouldin, Dunn Index, Silhouette
- **Weighted Clustering**: Support for point weights
- **Model Persistence**: Save/load trained models
- **Configurable Strategies**: Multiple initialization and assignment strategies
- **Production Ready**: Checkpointing, optimization, and fault tolerance

## Installation

### From PyPI (when published)

```bash
pip install massivedatascience-clusterer
```

### From Source

```bash
# Clone repository
git clone https://github.com/massivedatascience/generalized-kmeans-clustering.git
cd generalized-kmeans-clustering/python

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e .[dev]
```

## Requirements

- Python 3.7+
- PySpark 3.4.0+
- NumPy 1.20.0+

## Quick Start

```python
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from massivedatascience.clusterer import GeneralizedKMeans

# Create Spark session
spark = SparkSession.builder.appName("clustering").getOrCreate()

# Create sample data
data = spark.createDataFrame([
    (Vectors.dense([0.0, 0.0]),),
    (Vectors.dense([1.0, 1.0]),),
    (Vectors.dense([9.0, 8.0]),),
    (Vectors.dense([8.0, 9.0]),)
], ["features"])

# Train model
kmeans = GeneralizedKMeans(k=2, maxIter=20, seed=42)
model = kmeans.fit(data)

# Make predictions
predictions = model.transform(data)
predictions.select("features", "prediction").show()

# Evaluate clustering
cost = model.computeCost(data)
print(f"Within-cluster sum of squares: {cost}")

# Access quality metrics
summary = model.summary
print(f"Calinski-Harabasz Index: {summary.calinskiHarabaszIndex}")
print(f"Davies-Bouldin Index: {summary.daviesBouldinIndex}")
```

## API Overview

### GeneralizedKMeans (Estimator)

Main clustering estimator class.

**Parameters:**
- `k` (int): Number of clusters (default: 2)
- `divergence` (str): Distance function - "squaredEuclidean", "kl", "itakuraSaito", "generalizedI", "logisticLoss" (default: "squaredEuclidean")
- `smoothing` (float): Smoothing parameter for numerical stability (default: 1e-10)
- `maxIter` (int): Maximum iterations (default: 20)
- `tol` (float): Convergence tolerance (default: 1e-4)
- `seed` (int): Random seed for reproducibility (default: None)
- `initMode` (str): Initialization mode - "random", "k-means||" (default: "k-means||")
- `assignmentStrategy` (str): Assignment strategy - "auto", "broadcast" (default: "auto")
- `featuresCol` (str): Features column name (default: "features")
- `predictionCol` (str): Prediction column name (default: "prediction")
- `distanceCol` (str): Distance column name (default: None)
- `weightCol` (str): Weight column name (default: None)
- `checkpointInterval` (int): Checkpointing interval (default: 10)

**Methods:**
- `fit(dataset)`: Train model and return GeneralizedKMeansModel
- `setK(value)`, `getK()`: Set/get number of clusters
- `setDivergence(value)`, `getDivergence()`: Set/get divergence function
- `setMaxIter(value)`, `getMaxIter()`: Set/get max iterations
- All standard PySpark Estimator methods

### GeneralizedKMeansModel (Model)

Trained clustering model.

**Properties:**
- `numClusters`: Number of clusters in the model
- `numFeatures`: Number of features
- `summary`: GeneralizedKMeansSummary with quality metrics
- `clusterCenters()`: Returns numpy array of cluster centers (shape: [k, numFeatures])

**Methods:**
- `transform(dataset)`: Apply model to make predictions
- `predict(features)`: Predict cluster for a single point
- `computeCost(dataset)`: Compute total clustering cost (WCSS)
- `write()`: Get MLWriter for model persistence
- `load(path)`: Static method to load saved model

### GeneralizedKMeansSummary

Quality metrics for the clustering.

**Properties:**
- `wcss`: Within-cluster sum of squares
- `bcss`: Between-cluster sum of squares
- `calinskiHarabaszIndex`: Calinski-Harabasz index (higher is better)
- `daviesBouldinIndex`: Davies-Bouldin index (lower is better)
- `dunnIndex`: Dunn index (higher is better)
- `silhouetteCoefficient`: Silhouette coefficient (higher is better)
- `numClusters`: Number of clusters
- `numPoints`: Total number of points

## Usage Examples

### Using Different Divergences

```python
# KL divergence for probability distributions
kmeans = GeneralizedKMeans(
    k=3,
    divergence="kl",
    smoothing=1e-10,
    maxIter=30
)

# Itakura-Saito for audio/spectral data
kmeans = GeneralizedKMeans(
    k=5,
    divergence="itakuraSaito"
)

# Generalized I-divergence
kmeans = GeneralizedKMeans(
    k=4,
    divergence="generalizedI"
)
```

### Weighted Clustering

```python
# Create data with weights
data = spark.createDataFrame([
    (Vectors.dense([0.0, 0.0]), 1.0),
    (Vectors.dense([5.0, 5.0]), 10.0),  # Higher weight
], ["features", "weight"])

# Train with weights
kmeans = GeneralizedKMeans(k=2, weightCol="weight")
model = kmeans.fit(data)
```

### Model Persistence

```python
# Save model
model.write().overwrite().save("/path/to/model")

# Load model
from massivedatascience.clusterer import GeneralizedKMeansModel
loaded_model = GeneralizedKMeansModel.load("/path/to/model")

# Use loaded model
predictions = loaded_model.transform(new_data)
```

### Pipeline Integration

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler

# Build pipeline
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3"],
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features"
)

kmeans = GeneralizedKMeans(k=5, maxIter=20)

pipeline = Pipeline(stages=[assembler, scaler, kmeans])

# Fit pipeline
model = pipeline.fit(data)

# Make predictions
predictions = model.transform(data)
```

### Finding Optimal K

```python
# Test multiple k values
results = []
for k in range(2, 11):
    kmeans = GeneralizedKMeans(k=k, maxIter=20, seed=42)
    model = kmeans.fit(data)
    summary = model.summary

    results.append({
        'k': k,
        'wcss': summary.wcss,
        'ch_index': summary.calinskiHarabaszIndex,
        'db_index': summary.daviesBouldinIndex
    })

# Find best k by Calinski-Harabasz
best_k = max(results, key=lambda x: x['ch_index'])['k']
print(f"Optimal k: {best_k}")
```

## Examples

See the `examples/` directory for detailed examples:

- `basic_clustering.py` - Basic usage with Squared Euclidean distance
- `kl_divergence_clustering.py` - Clustering probability distributions
- `weighted_clustering.py` - Weighted clustering example
- `finding_optimal_k.py` - Finding optimal number of clusters
- `model_persistence.py` - Saving and loading models
- `clustering_tutorial.ipynb` - Comprehensive Jupyter notebook tutorial

## Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
cd python
pytest tests/

# Run with coverage
pytest --cov=massivedatascience tests/
```

See `TESTING.md` for detailed testing instructions.

## Performance Tips

1. **Cache data** when fitting multiple models
2. **Use broadcast strategy** for small k (< 100)
3. **Enable checkpointing** for large datasets
4. **Repartition data** for optimal parallelism
5. **Adjust tolerance** for faster convergence

See `examples/README.md` for more performance tips.

## Comparison with Spark MLlib

| Feature | GeneralizedKMeans | Spark MLlib KMeans |
|---------|-------------------|--------------------|
| Distance Functions | 5 (Bregman divergences) | 1 (Euclidean only) |
| Weighted Clustering | ✓ | ✗ |
| Quality Metrics | 6 metrics | 1 metric (WCSS) |
| Model Persistence | ✓ | ✓ |
| Pipeline Integration | ✓ | ✓ |
| Initialization | Random, K-means\|\| | Random, K-means\|\| |
| Assignment Strategy | Auto, Broadcast | Auto only |

## Documentation

- **Repository**: https://github.com/massivedatascience/generalized-kmeans-clustering
- **Examples**: See `examples/` directory
- **Architecture**: See `../ARCHITECTURE.md` in repository
- **Performance**: See `../PERFORMANCE_TUNING.md` in repository
- **Migration**: See `../MIGRATION_GUIDE.md` in repository

## Contributing

Contributions welcome! Please see the main repository for contribution guidelines.

## License

Apache License 2.0 - see LICENSE file in repository root.

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{generalized_kmeans,
  title = {Generalized K-Means Clustering with Bregman Divergences},
  author = {MassiveDataScience},
  year = {2025},
  url = {https://github.com/massivedatascience/generalized-kmeans-clustering}
}
```

## Support

- **Issues**: https://github.com/massivedatascience/generalized-kmeans-clustering/issues
- **Discussions**: https://github.com/massivedatascience/generalized-kmeans-clustering/discussions

## Version History

### 0.6.0 (2025)
- Initial PySpark wrapper release
- Full Spark ML Pipeline integration
- 6 quality metrics
- Model persistence
- Comprehensive examples and documentation

## Related Projects

- **Spark MLlib**: Standard Spark clustering algorithms
- **scikit-learn**: Python machine learning library (non-distributed)
- **Mahout**: Apache Mahout machine learning library
