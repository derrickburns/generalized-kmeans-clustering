#!/usr/bin/env python
# Copyright (c) 2025 massivedatascience
# Licensed under the Apache License, Version 2.0

"""
Clustering probability distributions using KL divergence.
Demonstrates clustering of documents represented as word probability distributions.
"""

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
import numpy as np

from massivedatascience.clusterer import GeneralizedKMeans


def main():
    # Create Spark session
    spark = (
        SparkSession.builder.appName("KLDivergenceClustering")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Create probability distribution data
    # Simulate document word distributions (must sum to 1)
    # Group 1: Technical documents (high probability on technical terms)
    # Group 2: Sports documents (high probability on sports terms)
    # Group 3: Food documents (high probability on food terms)

    data = spark.createDataFrame(
        [
            # Technical documents - high prob on first few terms
            (Vectors.dense([0.5, 0.3, 0.1, 0.05, 0.05]),),
            (Vectors.dense([0.6, 0.2, 0.1, 0.05, 0.05]),),
            (Vectors.dense([0.55, 0.25, 0.1, 0.05, 0.05]),),
            # Sports documents - high prob on middle terms
            (Vectors.dense([0.1, 0.1, 0.5, 0.2, 0.1]),),
            (Vectors.dense([0.05, 0.15, 0.6, 0.15, 0.05]),),
            (Vectors.dense([0.1, 0.1, 0.55, 0.15, 0.1]),),
            # Food documents - high prob on last terms
            (Vectors.dense([0.05, 0.1, 0.1, 0.35, 0.4]),),
            (Vectors.dense([0.1, 0.05, 0.15, 0.3, 0.4]),),
            (Vectors.dense([0.05, 0.1, 0.1, 0.4, 0.35]),),
        ],
        ["features"],
    )

    print("Input probability distributions (5 terms per document):")
    data.show(truncate=False)

    # Use KL divergence for clustering probability distributions
    # Note: smoothing parameter is crucial for numerical stability
    kmeans = GeneralizedKMeans(
        k=3,
        divergence="kl",
        smoothing=1e-10,
        maxIter=30,
        seed=42,
        distanceCol="kl_distance",
    )

    print("\nTraining model with KL divergence...")
    model = kmeans.fit(data)

    # Display cluster centers (centroids in probability space)
    print(f"\nNumber of clusters: {model.numClusters}")
    print("\nCluster centers (probability distributions):")
    for i, center in enumerate(model.clusterCenters()):
        print(f"  Cluster {i}: {center}")
        print(f"    Sum: {np.sum(center):.6f} (should be ~1.0 for probabilities)")

    # Make predictions
    predictions = model.transform(data)
    print("\nPredictions:")
    predictions.select("features", "prediction", "kl_distance").show(truncate=False)

    # Compute clustering cost
    cost = model.computeCost(data)
    print(f"\nTotal KL divergence (cost): {cost:.4f}")

    # Get summary statistics
    summary = model.summary
    print(f"\nClustering quality metrics:")
    print(f"  WCSS: {summary.wcss:.4f}")
    print(f"  BCSS: {summary.bcss:.4f}")
    print(f"  Calinski-Harabasz Index: {summary.calinskiHarabaszIndex:.4f}")

    # Classify a new document
    new_doc = Vectors.dense([0.08, 0.12, 0.5, 0.2, 0.1])  # Sports-like distribution
    cluster = model.predict(new_doc)
    print(f"\nNew document {new_doc} assigned to cluster: {cluster}")

    spark.stop()


if __name__ == "__main__":
    main()
