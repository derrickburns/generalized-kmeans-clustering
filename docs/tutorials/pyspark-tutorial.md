---
title: "PySpark Tutorial"
---

# PySpark Tutorial

**Time:** 10 minutes
**Goal:** Use the Python API for clustering

---

## Prerequisites

- PySpark 3.4+
- Python 3.8+
- The JAR file on your classpath

---

## Step 1: Setup

```python
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

# Create Spark session with the library JAR
spark = (SparkSession.builder
    .appName("PySparkClustering")
    .config("spark.jars.packages",
            "com.massivedatascience:clusterer_2.13:0.7.0")
    .getOrCreate())

spark.sparkContext.setLogLevel("WARN")
```

---

## Step 2: Import the Library

```python
from massivedatascience.clusterer import GeneralizedKMeans
```

---

## Step 3: Create Sample Data

```python
# Two well-separated clusters
data = spark.createDataFrame([
    (Vectors.dense([0.0, 0.0]),),
    (Vectors.dense([0.5, 0.5]),),
    (Vectors.dense([1.0, 0.0]),),
    (Vectors.dense([0.0, 1.0]),),
    (Vectors.dense([9.0, 9.0]),),
    (Vectors.dense([10.0, 10.0]),),
    (Vectors.dense([9.5, 10.5]),),
    (Vectors.dense([10.5, 9.5]),)
], ["features"])

data.show()
```

---

## Step 4: Train the Model

```python
kmeans = GeneralizedKMeans(
    k=2,
    divergence="squaredEuclidean",
    maxIter=20,
    seed=42
)

model = kmeans.fit(data)
```

---

## Step 5: Examine Results

### Cluster Centers

```python
import numpy as np

print(f"Number of clusters: {model.numClusters}")
print(f"Number of features: {model.numFeatures}")

centers = model.clusterCenters()
print("\nCluster centers:")
for i, center in enumerate(centers):
    print(f"  Cluster {i}: {center}")
```

### Make Predictions

```python
predictions = model.transform(data)
predictions.select("features", "prediction").show()
```

---

## Step 6: Evaluate

```python
# Compute WCSS
cost = model.computeCost(data)
print(f"Within-cluster sum of squares: {cost:.4f}")

# Training summary (if available)
if model.hasSummary():
    summary = model.summary
    print(f"\nTraining Summary:")
    print(f"  Algorithm: {summary.algorithm}")
    print(f"  Iterations: {summary.iterations}")
    print(f"  Converged: {summary.converged}")
    print(f"  Final distortion: {summary.finalDistortion:.4f}")
```

---

## Step 7: Save and Load

```python
# Save model
model.write().overwrite().save("/tmp/pyspark-kmeans-model")

# Load model
from massivedatascience.clusterer import GeneralizedKMeansModel
loaded_model = GeneralizedKMeansModel.load("/tmp/pyspark-kmeans-model")
```

---

## Using Different Divergences

### KL Divergence (for probability distributions)

```python
# Data must be probability distributions (positive, sum to 1)
prob_data = spark.createDataFrame([
    (Vectors.dense([0.7, 0.2, 0.1]),),
    (Vectors.dense([0.6, 0.3, 0.1]),),
    (Vectors.dense([0.1, 0.2, 0.7]),),
    (Vectors.dense([0.1, 0.1, 0.8]),),
], ["features"])

kl_kmeans = GeneralizedKMeans(
    k=2,
    divergence="kl",
    smoothing=1e-10  # Numerical stability
)

kl_model = kl_kmeans.fit(prob_data)
```

---

## Complete Example

```python
#!/usr/bin/env python
"""Complete PySpark clustering example."""

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from massivedatascience.clusterer import GeneralizedKMeans, GeneralizedKMeansModel

def main():
    # Initialize Spark
    spark = (SparkSession.builder
        .appName("PySparkClustering")
        .config("spark.ui.enabled", "false")
        .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Create data
    data = spark.createDataFrame([
        (Vectors.dense([0.0, 0.0]),),
        (Vectors.dense([0.5, 0.5]),),
        (Vectors.dense([1.0, 0.0]),),
        (Vectors.dense([9.0, 9.0]),),
        (Vectors.dense([10.0, 10.0]),),
        (Vectors.dense([9.5, 10.5]),),
    ], ["features"])

    # Train model
    kmeans = GeneralizedKMeans(k=2, maxIter=20, seed=42)
    model = kmeans.fit(data)

    # Show results
    print(f"Found {model.numClusters} clusters")
    model.transform(data).select("features", "prediction").show()
    print(f"WCSS: {model.computeCost(data):.4f}")

    spark.stop()

if __name__ == "__main__":
    main()
```

---

## Available Algorithms

All algorithms are available in PySpark:

```python
from massivedatascience.clusterer import (
    GeneralizedKMeans,    # Standard k-means with divergences
    XMeans,               # Automatic k selection
    SoftKMeans,           # Probabilistic assignments
    BisectingKMeans,      # Hierarchical divisive
    StreamingKMeans,      # Online updates
    KMedoids,             # Uses actual data points
)
```

---

## Next Steps

- [Choosing the Right Algorithm](choosing-algorithm.html) — Decision guide
- [Find Optimal K](../howto/find-optimal-k.html) — Elbow method
- [Handle Outliers](../howto/handle-outliers.html) — Robust clustering

---

[Back to Tutorials](index.html) | [Home](../)
