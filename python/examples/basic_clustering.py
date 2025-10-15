#!/usr/bin/env python
# Copyright (c) 2025 massivedatascience
# Licensed under the Apache License, Version 2.0

"""
Basic clustering example using GeneralizedKMeans with Squared Euclidean distance.
"""

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

from massivedatascience.clusterer import GeneralizedKMeans


def main():
    # Create Spark session
    spark = (
        SparkSession.builder.appName("BasicClustering")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Create sample data - two well-separated clusters
    data = spark.createDataFrame(
        [
            (Vectors.dense([0.0, 0.0]),),
            (Vectors.dense([1.0, 1.0]),),
            (Vectors.dense([0.5, 0.5]),),
            (Vectors.dense([9.0, 8.0]),),
            (Vectors.dense([8.0, 9.0]),),
            (Vectors.dense([8.5, 8.5]),),
        ],
        ["features"],
    )

    print("Input data:")
    data.show()

    # Create and train clustering model
    kmeans = GeneralizedKMeans(
        k=2,
        divergence="squaredEuclidean",
        maxIter=20,
        seed=42,
        distanceCol="distance",
    )

    print("\nTraining model...")
    model = kmeans.fit(data)

    # Display cluster centers
    print(f"\nNumber of clusters: {model.numClusters}")
    print(f"Number of features: {model.numFeatures}")
    print("\nCluster centers:")
    for i, center in enumerate(model.clusterCenters()):
        print(f"  Cluster {i}: {center}")

    # Make predictions
    predictions = model.transform(data)
    print("\nPredictions:")
    predictions.select("features", "prediction", "distance").show()

    # Compute clustering cost (WCSS)
    cost = model.computeCost(data)
    print(f"\nWithin-cluster sum of squares: {cost:.4f}")

    # Get summary statistics
    summary = model.summary
    print(f"\nClustering quality metrics:")
    print(f"  WCSS: {summary.wcss:.4f}")
    print(f"  BCSS: {summary.bcss:.4f}")
    print(f"  Calinski-Harabasz Index: {summary.calinskiHarabaszIndex:.4f}")
    print(f"  Davies-Bouldin Index: {summary.daviesBouldinIndex:.4f}")

    # Predict cluster for a new point
    new_point = Vectors.dense([0.2, 0.3])
    cluster = model.predict(new_point)
    print(f"\nNew point {new_point} assigned to cluster: {cluster}")

    spark.stop()


if __name__ == "__main__":
    main()
