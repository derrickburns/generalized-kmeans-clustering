#!/usr/bin/env python
# Copyright (c) 2025 massivedatascience
# Licensed under the Apache License, Version 2.0

"""
Setup configuration for massivedatascience-clusterer PySpark package.
"""

from setuptools import setup, find_packages
import os

# Read version from package
with open(os.path.join("massivedatascience", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

# Read long description from README
long_description = """
# MassiveDataScience Clusterer

Generalized K-Means clustering for PySpark with pluggable Bregman divergences.

## Features

- **Multiple Distance Functions**: Squared Euclidean, KL divergence, Itakura-Saito,
  Generalized I-divergence, and Logistic Loss
- **Spark ML Integration**: Native Estimator/Model pattern with Pipeline support
- **Quality Metrics**: WCSS, BCSS, Calinski-Harabasz, Davies-Bouldin, Dunn Index, Silhouette
- **Production Ready**: Built-in model persistence, checkpointing, and optimization
- **Weighted Clustering**: Support for point weights
- **Configurable Strategies**: Pluggable assignment and initialization strategies

## Installation

```bash
pip install massivedatascience-clusterer
```

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
```

## Documentation

- GitHub: https://github.com/massivedatascience/generalized-kmeans-clustering
- Examples: See `examples/` directory
- API Docs: https://massivedatascience.github.io/generalized-kmeans-clustering/
"""

setup(
    name="massivedatascience-clusterer",
    version=version,
    description="Generalized K-Means clustering for PySpark with Bregman divergences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MassiveDataScience",
    author_email="support@massivedatascience.com",
    url="https://github.com/massivedatascience/generalized-kmeans-clustering",
    license="Apache License 2.0",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "pyspark>=3.4.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
    keywords="pyspark clustering kmeans machine-learning bregman-divergence",
    project_urls={
        "Bug Reports": "https://github.com/massivedatascience/generalized-kmeans-clustering/issues",
        "Source": "https://github.com/massivedatascience/generalized-kmeans-clustering",
        "Documentation": "https://github.com/massivedatascience/generalized-kmeans-clustering#readme",
    },
)
