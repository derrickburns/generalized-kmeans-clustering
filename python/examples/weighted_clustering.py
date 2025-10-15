#!/usr/bin/env python
# Copyright (c) 2025 massivedatascience
# Licensed under the Apache License, Version 2.0

"""
Weighted clustering example.
Demonstrates how to give different importance to different data points.
"""

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

from massivedatascience.clusterer import GeneralizedKMeans


def main():
    # Create Spark session
    spark = (
        SparkSession.builder.appName("WeightedClustering")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Create data with weights
    # Points with higher weights have more influence on cluster centers
    data = spark.createDataFrame(
        [
            # Cluster 1: Low-weight outliers near origin
            (Vectors.dense([0.0, 0.0]), 0.1),
            (Vectors.dense([0.5, 0.5]), 0.1),
            # Cluster 1: High-weight points around (5, 5)
            (Vectors.dense([5.0, 5.0]), 10.0),
            (Vectors.dense([5.5, 5.0]), 10.0),
            (Vectors.dense([5.0, 5.5]), 10.0),
            (Vectors.dense([5.5, 5.5]), 10.0),
            # Cluster 2: Low-weight outliers
            (Vectors.dense([10.0, 10.0]), 0.1),
            (Vectors.dense([10.5, 10.0]), 0.1),
            # Cluster 2: High-weight points around (15, 15)
            (Vectors.dense([15.0, 15.0]), 10.0),
            (Vectors.dense([15.5, 15.0]), 10.0),
            (Vectors.dense([15.0, 15.5]), 10.0),
            (Vectors.dense([15.5, 15.5]), 10.0),
        ],
        ["features", "weight"],
    )

    print("Input data with weights:")
    data.show()

    # Cluster with weights - high-weight points will pull centers toward them
    print("\n=== Clustering WITH weights ===")
    kmeans_weighted = GeneralizedKMeans(
        k=2,
        weightCol="weight",
        maxIter=20,
        seed=42,
    )

    model_weighted = kmeans_weighted.fit(data)

    print("\nCluster centers (weighted):")
    for i, center in enumerate(model_weighted.clusterCenters()):
        print(f"  Cluster {i}: {center}")

    predictions_weighted = model_weighted.transform(data)
    print("\nPredictions (weighted):")
    predictions_weighted.select("features", "weight", "prediction").show()

    cost_weighted = model_weighted.computeCost(data)
    print(f"\nWeighted cost: {cost_weighted:.4f}")

    # Compare with unweighted clustering
    print("\n=== Clustering WITHOUT weights (for comparison) ===")
    data_unweighted = data.drop("weight")

    kmeans_unweighted = GeneralizedKMeans(
        k=2,
        maxIter=20,
        seed=42,
    )

    model_unweighted = kmeans_unweighted.fit(data_unweighted)

    print("\nCluster centers (unweighted):")
    for i, center in enumerate(model_unweighted.clusterCenters()):
        print(f"  Cluster {i}: {center}")

    cost_unweighted = model_unweighted.computeCost(data_unweighted)
    print(f"\nUnweighted cost: {cost_unweighted:.4f}")

    print("\n=== Analysis ===")
    print("Notice that weighted clustering produces centers closer to the")
    print("high-weight points (around [5,5] and [15,15]), while unweighted")
    print("clustering is influenced more by the spatial distribution of all points.")

    spark.stop()


if __name__ == "__main__":
    main()
