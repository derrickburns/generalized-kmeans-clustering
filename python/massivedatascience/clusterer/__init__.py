# Copyright (c) 2025 massivedatascience
# Licensed under the Apache License, Version 2.0

"""
Generalized K-Means Clustering
===============================

This module provides PySpark wrappers for the generalized k-means clustering
implementation with support for multiple Bregman divergences.

Classes:
    GeneralizedKMeans: Estimator for k-means clustering with pluggable divergences
    GeneralizedKMeansModel: Fitted clustering model
    GeneralizedKMeansSummary: Training summary with quality metrics

Example:
    >>> from massivedatascience.clusterer import GeneralizedKMeans
    >>> from pyspark.ml.linalg import Vectors
    >>>
    >>> # Create sample data
    >>> data = spark.createDataFrame([
    ...     (Vectors.dense([0.0, 0.0]),),
    ...     (Vectors.dense([1.0, 1.0]),),
    ...     (Vectors.dense([9.0, 8.0]),),
    ...     (Vectors.dense([8.0, 9.0]),)
    ... ], ["features"])
    >>>
    >>> # Train model
    >>> kmeans = GeneralizedKMeans(k=2, maxIter=20)
    >>> model = kmeans.fit(data)
    >>>
    >>> # Make predictions
    >>> predictions = model.transform(data)
    >>> predictions.show()
"""

from .kmeans import GeneralizedKMeans, GeneralizedKMeansModel, GeneralizedKMeansSummary

__all__ = ["GeneralizedKMeans", "GeneralizedKMeansModel", "GeneralizedKMeansSummary"]
