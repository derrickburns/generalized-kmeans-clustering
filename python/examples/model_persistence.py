#!/usr/bin/env python
# Copyright (c) 2025 massivedatascience
# Licensed under the Apache License, Version 2.0

"""
Model persistence example - saving and loading trained models.
"""

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
import tempfile
import shutil

from massivedatascience.clusterer import GeneralizedKMeans, GeneralizedKMeansModel


def main():
    # Create Spark session
    spark = (
        SparkSession.builder.appName("ModelPersistence")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Create training data
    data = spark.createDataFrame(
        [
            (Vectors.dense([0.0, 0.0]),),
            (Vectors.dense([1.0, 1.0]),),
            (Vectors.dense([9.0, 8.0]),),
            (Vectors.dense([8.0, 9.0]),),
        ],
        ["features"],
    )

    print("Training model...")
    kmeans = GeneralizedKMeans(
        k=2,
        divergence="squaredEuclidean",
        maxIter=20,
        seed=42,
    )

    model = kmeans.fit(data)

    # Display original model info
    print(f"\nOriginal model:")
    print(f"  UID: {model.uid}")
    print(f"  Number of clusters: {model.numClusters}")
    print(f"  Number of features: {model.numFeatures}")
    print(f"  Cluster centers:")
    for i, center in enumerate(model.clusterCenters()):
        print(f"    Cluster {i}: {center}")

    # Make predictions with original model
    predictions_original = model.transform(data)
    print("\nOriginal predictions:")
    predictions_original.select("features", "prediction").show()

    # Save model to temporary directory
    temp_dir = tempfile.mkdtemp()
    model_path = f"{temp_dir}/kmeans_model"

    try:
        print(f"\nSaving model to: {model_path}")
        model.write().overwrite().save(model_path)
        print("Model saved successfully!")

        # Load model from disk
        print(f"\nLoading model from: {model_path}")
        loaded_model = GeneralizedKMeansModel.load(model_path)
        print("Model loaded successfully!")

        # Verify loaded model properties
        print(f"\nLoaded model:")
        print(f"  UID: {loaded_model.uid}")
        print(f"  Number of clusters: {loaded_model.numClusters}")
        print(f"  Number of features: {loaded_model.numFeatures}")
        print(f"  Cluster centers:")
        for i, center in enumerate(loaded_model.clusterCenters()):
            print(f"    Cluster {i}: {center}")

        # Make predictions with loaded model
        predictions_loaded = loaded_model.transform(data)
        print("\nLoaded model predictions:")
        predictions_loaded.select("features", "prediction").show()

        # Verify predictions match
        original_preds = [row.prediction for row in predictions_original.collect()]
        loaded_preds = [row.prediction for row in predictions_loaded.collect()]

        if original_preds == loaded_preds:
            print("\n✓ Success! Predictions from loaded model match original model.")
        else:
            print("\n✗ Warning: Predictions differ between original and loaded models.")

        # Test prediction on new data
        new_data = spark.createDataFrame(
            [
                (Vectors.dense([0.5, 0.5]),),
                (Vectors.dense([8.5, 8.5]),),
            ],
            ["features"],
        )

        print("\nPredictions on new data:")
        loaded_model.transform(new_data).select("features", "prediction").show()

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")

    spark.stop()


if __name__ == "__main__":
    main()
