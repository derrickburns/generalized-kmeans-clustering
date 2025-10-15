#!/usr/bin/env python3
"""
Smoke test for generalized-kmeans-clustering PySpark wrapper.

This test validates that the library can be imported and basic
functionality works in a PySpark environment.
"""

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

def main():
    """Run basic smoke test."""
    print("Starting smoke test...")

    # Create Spark session
    spark = SparkSession.builder \
        .appName("GeneralizedKMeans Smoke Test") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()

    try:
        # Import the wrapper
        from massivedatascience.clusterer import GeneralizedKMeans

        print("✓ Successfully imported GeneralizedKMeans")

        # Create simple test data
        data = [
            (Vectors.dense([0.0, 0.0]),),
            (Vectors.dense([1.0, 1.0]),),
            (Vectors.dense([9.0, 9.0]),),
            (Vectors.dense([10.0, 10.0]),)
        ]

        df = spark.createDataFrame(data, ["features"])
        print("✓ Created test DataFrame")

        # Create and configure estimator
        kmeans = GeneralizedKMeans() \
            .setK(2) \
            .setDivergence("squaredEuclidean") \
            .setMaxIter(10) \
            .setSeed(42)

        print("✓ Created GeneralizedKMeans estimator")

        # Fit model
        model = kmeans.fit(df)
        print(f"✓ Fitted model with {model.numClusters} clusters")

        # Transform data
        predictions = model.transform(df)
        print("✓ Transformed data")

        # Verify predictions
        pred_count = predictions.count()
        assert pred_count == 4, f"Expected 4 predictions, got {pred_count}"
        print("✓ Verified prediction count")

        # Check that prediction column exists
        assert "prediction" in predictions.columns, "Missing prediction column"
        print("✓ Verified prediction column exists")

        # Verify cluster assignments are valid (0 or 1)
        pred_values = predictions.select("prediction").rdd.map(lambda r: r[0]).collect()
        assert all(p in [0, 1] for p in pred_values), "Invalid cluster assignments"
        print("✓ Verified cluster assignments are valid")

        # Compute cost
        cost = model.computeCost(df)
        assert cost >= 0 and cost < float('inf'), f"Invalid cost: {cost}"
        print(f"✓ Computed cost: {cost:.4f}")

        print("\n✅ All smoke tests passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        spark.stop()

if __name__ == "__main__":
    exit(main())
