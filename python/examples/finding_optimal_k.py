#!/usr/bin/env python
# Copyright (c) 2025 massivedatascience
# Licensed under the Apache License, Version 2.0

"""
Finding optimal number of clusters using the Elbow method and quality metrics.
"""

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt

from massivedatascience.clusterer import GeneralizedKMeans


def main():
    # Create Spark session
    spark = (
        SparkSession.builder.appName("FindingOptimalK")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Create data with 3 natural clusters
    data = spark.createDataFrame(
        [
            # Cluster 1: around (0, 0)
            (Vectors.dense([0.0, 0.0]),),
            (Vectors.dense([0.5, 0.5]),),
            (Vectors.dense([0.5, -0.5]),),
            (Vectors.dense([-0.5, 0.5]),),
            # Cluster 2: around (5, 5)
            (Vectors.dense([5.0, 5.0]),),
            (Vectors.dense([5.5, 5.0]),),
            (Vectors.dense([5.0, 5.5]),),
            (Vectors.dense([5.5, 5.5]),),
            # Cluster 3: around (10, 0)
            (Vectors.dense([10.0, 0.0]),),
            (Vectors.dense([10.5, 0.0]),),
            (Vectors.dense([10.0, 0.5]),),
            (Vectors.dense([10.5, 0.5]),),
        ],
        ["features"],
    ).cache()

    print("Testing k from 2 to 7...\n")

    results = []

    for k in range(2, 8):
        kmeans = GeneralizedKMeans(
            k=k,
            maxIter=20,
            seed=42,
        )

        model = kmeans.fit(data)
        summary = model.summary

        results.append(
            {
                "k": k,
                "wcss": summary.wcss,
                "bcss": summary.bcss,
                "calinski_harabasz": summary.calinskiHarabaszIndex,
                "davies_bouldin": summary.daviesBouldinIndex,
            }
        )

        print(f"k={k}:")
        print(f"  WCSS: {summary.wcss:.4f}")
        print(f"  BCSS: {summary.bcss:.4f}")
        print(f"  Calinski-Harabasz: {summary.calinskiHarabaszIndex:.4f} (higher is better)")
        print(f"  Davies-Bouldin: {summary.daviesBouldinIndex:.4f} (lower is better)")
        print()

    # Create visualizations
    k_values = [r["k"] for r in results]
    wcss_values = [r["wcss"] for r in results]
    ch_values = [r["calinski_harabasz"] for r in results]
    db_values = [r["davies_bouldin"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Elbow plot (WCSS)
    axes[0].plot(k_values, wcss_values, "bo-")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("WCSS")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True)

    # Calinski-Harabasz Index (higher is better)
    axes[1].plot(k_values, ch_values, "go-")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Calinski-Harabasz Index")
    axes[1].set_title("Calinski-Harabasz Index (Higher is Better)")
    axes[1].grid(True)

    # Davies-Bouldin Index (lower is better)
    axes[2].plot(k_values, db_values, "ro-")
    axes[2].set_xlabel("Number of Clusters (k)")
    axes[2].set_ylabel("Davies-Bouldin Index")
    axes[2].set_title("Davies-Bouldin Index (Lower is Better)")
    axes[2].grid(True)

    plt.tight_layout()
    output_path = "/tmp/optimal_k_analysis.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")

    print("\n=== Recommendation ===")
    print("Based on the metrics:")
    print("1. Look for an 'elbow' in the WCSS plot")
    print("2. Choose k with high Calinski-Harabasz index")
    print("3. Choose k with low Davies-Bouldin index")
    print("\nFor this data, k=3 should be optimal (matches the 3 natural clusters)")

    # Find best k by Calinski-Harabasz
    best_k = max(results, key=lambda x: x["calinski_harabasz"])["k"]
    print(f"\nBest k by Calinski-Harabasz Index: {best_k}")

    spark.stop()


if __name__ == "__main__":
    main()
