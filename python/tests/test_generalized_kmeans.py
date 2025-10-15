# Copyright (c) 2025 massivedatascience
# Licensed under the Apache License, Version 2.0

"""
Tests for GeneralizedKMeans PySpark wrapper.
"""

import unittest
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

from massivedatascience.clusterer import GeneralizedKMeans, GeneralizedKMeansModel


class GeneralizedKMeansTest(unittest.TestCase):
    """Test cases for GeneralizedKMeans."""

    @classmethod
    def setUpClass(cls):
        """Set up Spark session for tests."""
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("GeneralizedKMeansTest")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("WARN")

    @classmethod
    def tearDownClass(cls):
        """Tear down Spark session."""
        cls.spark.stop()

    def test_basic_clustering(self):
        """Test basic clustering with Squared Euclidean."""
        data = self.spark.createDataFrame(
            [
                (Vectors.dense([0.0, 0.0]),),
                (Vectors.dense([1.0, 1.0]),),
                (Vectors.dense([9.0, 8.0]),),
                (Vectors.dense([8.0, 9.0]),),
            ],
            ["features"],
        )

        kmeans = GeneralizedKMeans(k=2, maxIter=20, seed=42)
        model = kmeans.fit(data)

        # Check model properties
        self.assertEqual(model.numClusters, 2)
        self.assertEqual(model.numFeatures, 2)

        # Check cluster centers shape
        centers = model.clusterCenters()
        self.assertEqual(centers.shape, (2, 2))

        # Transform data
        predictions = model.transform(data)
        self.assertEqual(predictions.count(), 4)

        # Check predictions are in valid range
        pred_values = [row.prediction for row in predictions.collect()]
        self.assertTrue(all(0 <= p < 2 for p in pred_values))

    def test_kl_divergence(self):
        """Test clustering with KL divergence."""
        # Create probability distribution data
        data = self.spark.createDataFrame(
            [
                (Vectors.dense([0.7, 0.2, 0.1]),),
                (Vectors.dense([0.6, 0.3, 0.1]),),
                (Vectors.dense([0.1, 0.2, 0.7]),),
                (Vectors.dense([0.1, 0.3, 0.6]),),
            ],
            ["features"],
        )

        kmeans = GeneralizedKMeans(
            k=2,
            divergence="kl",
            smoothing=1e-10,
            maxIter=20,
            seed=42,
        )
        model = kmeans.fit(data)

        self.assertEqual(model.numClusters, 2)
        predictions = model.transform(data)
        self.assertEqual(predictions.count(), 4)

    def test_weighted_clustering(self):
        """Test weighted clustering."""
        data = self.spark.createDataFrame(
            [
                (Vectors.dense([0.0, 0.0]), 1.0),
                (Vectors.dense([0.1, 0.1]), 1.0),
                (Vectors.dense([5.0, 5.0]), 10.0),  # High weight
                (Vectors.dense([5.1, 5.1]), 10.0),  # High weight
            ],
            ["features", "weight"],
        )

        kmeans = GeneralizedKMeans(k=2, weightCol="weight", maxIter=20, seed=42)
        model = kmeans.fit(data)

        predictions = model.transform(data)
        self.assertEqual(predictions.count(), 4)

    def test_distance_column(self):
        """Test distance column output."""
        data = self.spark.createDataFrame(
            [
                (Vectors.dense([0.0, 0.0]),),
                (Vectors.dense([1.0, 1.0]),),
            ],
            ["features"],
        )

        kmeans = GeneralizedKMeans(k=2, maxIter=10, distanceCol="distance", seed=42)
        model = kmeans.fit(data)

        predictions = model.transform(data)
        self.assertTrue("distance" in predictions.columns)

        # Check distances are non-negative
        distances = [row.distance for row in predictions.collect()]
        self.assertTrue(all(d >= 0 for d in distances))

    def test_predict_single_point(self):
        """Test single point prediction."""
        data = self.spark.createDataFrame(
            [
                (Vectors.dense([0.0, 0.0]),),
                (Vectors.dense([1.0, 1.0]),),
                (Vectors.dense([9.0, 8.0]),),
            ],
            ["features"],
        )

        kmeans = GeneralizedKMeans(k=2, maxIter=20, seed=42)
        model = kmeans.fit(data)

        # Predict single points
        point1 = Vectors.dense([0.5, 0.5])
        cluster1 = model.predict(point1)
        self.assertIn(cluster1, [0, 1])

        point2 = Vectors.dense([9.5, 9.5])
        cluster2 = model.predict(point2)
        self.assertIn(cluster2, [0, 1])

    def test_compute_cost(self):
        """Test cost computation."""
        data = self.spark.createDataFrame(
            [
                (Vectors.dense([0.0, 0.0]),),
                (Vectors.dense([1.0, 1.0]),),
                (Vectors.dense([9.0, 8.0]),),
                (Vectors.dense([8.0, 9.0]),),
            ],
            ["features"],
        )

        kmeans = GeneralizedKMeans(k=2, maxIter=20, seed=42)
        model = kmeans.fit(data)

        cost = model.computeCost(data)
        self.assertGreaterEqual(cost, 0.0)
        self.assertFalse(np.isnan(cost))
        self.assertFalse(np.isinf(cost))

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        data = self.spark.createDataFrame(
            [
                (Vectors.dense([0.0, 0.0]),),
                (Vectors.dense([1.0, 1.0]),),
                (Vectors.dense([9.0, 8.0]),),
                (Vectors.dense([8.0, 9.0]),),
            ],
            ["features"],
        ).cache()

        kmeans1 = GeneralizedKMeans(k=2, maxIter=20, seed=42)
        model1 = kmeans1.fit(data)
        predictions1 = [row.prediction for row in model1.transform(data).collect()]

        kmeans2 = GeneralizedKMeans(k=2, maxIter=20, seed=42)
        model2 = kmeans2.fit(data)
        predictions2 = [row.prediction for row in model2.transform(data).collect()]

        self.assertEqual(predictions1, predictions2)

    def test_different_k_values(self):
        """Test different values of k."""
        data = self.spark.createDataFrame(
            [(Vectors.dense([float(i), float(i)]),) for i in range(10)],
            ["features"],
        )

        for k in [2, 3, 5]:
            kmeans = GeneralizedKMeans(k=k, maxIter=10, seed=42)
            model = kmeans.fit(data)
            self.assertLessEqual(model.numClusters, k)  # May be less if clusters dropped

    def test_initialization_modes(self):
        """Test different initialization modes."""
        data = self.spark.createDataFrame(
            [
                (Vectors.dense([0.0, 0.0]),),
                (Vectors.dense([1.0, 1.0]),),
                (Vectors.dense([9.0, 8.0]),),
                (Vectors.dense([8.0, 9.0]),),
            ],
            ["features"],
        )

        for init_mode in ["random", "k-means||"]:
            kmeans = GeneralizedKMeans(
                k=2,
                initMode=init_mode,
                maxIter=20,
                seed=42,
            )
            model = kmeans.fit(data)
            self.assertEqual(model.numClusters, 2)

    def test_assignment_strategies(self):
        """Test different assignment strategies."""
        data = self.spark.createDataFrame(
            [
                (Vectors.dense([0.0, 0.0]),),
                (Vectors.dense([1.0, 1.0]),),
                (Vectors.dense([9.0, 8.0]),),
            ],
            ["features"],
        )

        for strategy in ["auto", "broadcast"]:
            kmeans = GeneralizedKMeans(
                k=2,
                assignmentStrategy=strategy,
                maxIter=10,
                seed=42,
            )
            model = kmeans.fit(data)
            self.assertEqual(model.numClusters, 2)

    def test_parameter_getters(self):
        """Test parameter getters."""
        kmeans = GeneralizedKMeans(
            k=5,
            divergence="kl",
            smoothing=1e-8,
            maxIter=30,
            tol=1e-5,
        )

        self.assertEqual(kmeans.getK(), 5)
        self.assertEqual(kmeans.getDivergence(), "kl")
        self.assertAlmostEqual(kmeans.getSmoothing(), 1e-8)
        self.assertEqual(kmeans.getMaxIter(), 30)
        self.assertAlmostEqual(kmeans.getTol(), 1e-5)

    def test_parameter_setters(self):
        """Test parameter setters."""
        kmeans = GeneralizedKMeans()

        kmeans.setK(7)
        self.assertEqual(kmeans.getK(), 7)

        kmeans.setDivergence("itakuraSaito")
        self.assertEqual(kmeans.getDivergence(), "itakuraSaito")

        kmeans.setMaxIter(50)
        self.assertEqual(kmeans.getMaxIter(), 50)

    def test_model_persistence(self):
        """Test model save and load."""
        import tempfile
        import shutil

        data = self.spark.createDataFrame(
            [
                (Vectors.dense([0.0, 0.0]),),
                (Vectors.dense([1.0, 1.0]),),
                (Vectors.dense([9.0, 8.0]),),
                (Vectors.dense([8.0, 9.0]),),
            ],
            ["features"],
        )

        kmeans = GeneralizedKMeans(k=2, maxIter=20, seed=42)
        model = kmeans.fit(data)

        # Save model
        temp_dir = tempfile.mkdtemp()
        try:
            model_path = f"{temp_dir}/test_model"
            model.write().overwrite().save(model_path)

            # Load model
            loaded_model = GeneralizedKMeansModel.load(model_path)

            # Verify loaded model
            self.assertEqual(loaded_model.numClusters, model.numClusters)
            self.assertEqual(loaded_model.numFeatures, model.numFeatures)

            # Verify predictions match
            original_preds = [
                row.prediction for row in model.transform(data).collect()
            ]
            loaded_preds = [
                row.prediction for row in loaded_model.transform(data).collect()
            ]
            self.assertEqual(original_preds, loaded_preds)

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
