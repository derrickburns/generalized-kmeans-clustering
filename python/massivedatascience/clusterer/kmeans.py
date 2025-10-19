# Copyright (c) 2025 massivedatascience
# Licensed under the Apache License, Version 2.0

"""
PySpark wrapper for GeneralizedKMeans clustering.

This module provides Python bindings for the Scala implementation of
generalized k-means clustering with Bregman divergences.
"""

from typing import Optional, List
import numpy as np

from pyspark import keyword_only
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaParams
from pyspark.ml.param.shared import (
    HasFeaturesCol,
    HasPredictionCol,
    HasMaxIter,
    HasSeed,
    HasTol,
    HasWeightCol,
)
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.ml.linalg import Vector


class GeneralizedKMeansParams(
    HasFeaturesCol,
    HasPredictionCol,
    HasMaxIter,
    HasSeed,
    HasTol,
    HasWeightCol,
):
    """
    Params for GeneralizedKMeans and GeneralizedKMeansModel.

    Parameters
    ----------
    k : int, default=2
        Number of clusters to create (k > 1).

    divergence : str, default="squaredEuclidean"
        Bregman divergence to use for distance computation.
        Options: "squaredEuclidean", "kl", "itakuraSaito", "generalizedI", "logistic"

    smoothing : float, default=1e-10
        Smoothing parameter for divergences that need it (KL, Itakura-Saito, etc.)

    assignmentStrategy : str, default="auto"
        Strategy for assigning points to clusters.
        Options: "auto", "broadcast", "crossjoin"

    emptyClusterStrategy : str, default="reseedRandom"
        How to handle empty clusters.
        Options: "reseedRandom", "drop"

    checkpointInterval : int, default=10
        Checkpoint interval (0 = disabled). Set checkpoint directory using
        SparkContext.setCheckpointDir()

    initMode : str, default="k-means||"
        Initialization algorithm.
        Options: "random", "k-means||"

    initSteps : int, default=2
        Number of steps for k-means|| initialization.

    distanceCol : str, optional
        Column name for output distance to cluster center.

    featuresCol : str, default="features"
        Features column name.

    predictionCol : str, default="prediction"
        Prediction column name.

    maxIter : int, default=20
        Maximum number of iterations (>= 0).

    seed : int, optional
        Random seed.

    tol : float, default=1e-4
        Convergence tolerance for center movement.

    weightCol : str, optional
        Weight column name. If not set, all instances are treated equally.
    """

    k = Param(
        Params._dummy(),
        "k",
        "Number of clusters to create (must be > 1).",
        typeConverter=TypeConverters.toInt,
    )

    divergence = Param(
        Params._dummy(),
        "divergence",
        "Bregman divergence function: squaredEuclidean, kl, itakuraSaito, generalizedI, logistic",
        typeConverter=TypeConverters.toString,
    )

    smoothing = Param(
        Params._dummy(),
        "smoothing",
        "Smoothing parameter for numerical stability",
        typeConverter=TypeConverters.toFloat,
    )

    assignmentStrategy = Param(
        Params._dummy(),
        "assignmentStrategy",
        "Point assignment strategy: auto, broadcast, crossjoin",
        typeConverter=TypeConverters.toString,
    )

    emptyClusterStrategy = Param(
        Params._dummy(),
        "emptyClusterStrategy",
        "Empty cluster handling: reseedRandom, drop",
        typeConverter=TypeConverters.toString,
    )

    checkpointInterval = Param(
        Params._dummy(),
        "checkpointInterval",
        "Checkpoint interval (0 = disabled)",
        typeConverter=TypeConverters.toInt,
    )

    initMode = Param(
        Params._dummy(),
        "initMode",
        "Initialization mode: random, k-means||",
        typeConverter=TypeConverters.toString,
    )

    initSteps = Param(
        Params._dummy(),
        "initSteps",
        "Number of k-means|| initialization steps",
        typeConverter=TypeConverters.toInt,
    )

    distanceCol = Param(
        Params._dummy(),
        "distanceCol",
        "Column name for distance to cluster center",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self, *args):
        super(GeneralizedKMeansParams, self).__init__(*args)
        self._setDefault(
            k=2,
            divergence="squaredEuclidean",
            smoothing=1e-10,
            assignmentStrategy="auto",
            emptyClusterStrategy="reseedRandom",
            checkpointInterval=10,
            initMode="k-means||",
            initSteps=2,
            featuresCol="features",
            predictionCol="prediction",
            maxIter=20,
            tol=1e-4,
        )

    def getK(self) -> int:
        """Gets the value of k or its default value."""
        return self.getOrDefault(self.k)

    def getDivergence(self) -> str:
        """Gets the value of divergence or its default value."""
        return self.getOrDefault(self.divergence)

    def getSmoothing(self) -> float:
        """Gets the value of smoothing or its default value."""
        return self.getOrDefault(self.smoothing)

    def getAssignmentStrategy(self) -> str:
        """Gets the value of assignmentStrategy or its default value."""
        return self.getOrDefault(self.assignmentStrategy)

    def getEmptyClusterStrategy(self) -> str:
        """Gets the value of emptyClusterStrategy or its default value."""
        return self.getOrDefault(self.emptyClusterStrategy)

    def getCheckpointInterval(self) -> int:
        """Gets the value of checkpointInterval or its default value."""
        return self.getOrDefault(self.checkpointInterval)

    def getInitMode(self) -> str:
        """Gets the value of initMode or its default value."""
        return self.getOrDefault(self.initMode)

    def getInitSteps(self) -> int:
        """Gets the value of initSteps or its default value."""
        return self.getOrDefault(self.initSteps)

    def getDistanceCol(self) -> Optional[str]:
        """Gets the value of distanceCol."""
        return self.getOrDefault(self.distanceCol)


class GeneralizedKMeans(
    JavaEstimator, GeneralizedKMeansParams, JavaMLReadable, JavaMLWritable
):
    """
    Generalized K-Means clustering with pluggable Bregman divergences.

    This estimator trains a k-means clustering model using various distance functions
    (divergences) beyond standard Euclidean distance. It implements Lloyd's algorithm
    with configurable strategies for assignment, initialization, and empty cluster handling.

    Parameters
    ----------
    k : int, default=2
        Number of clusters to create.

    divergence : str, default="squaredEuclidean"
        Distance function to use. Options:
        - "squaredEuclidean": Standard k-means (L2 distance)
        - "kl": Kullback-Leibler divergence (for probability distributions)
        - "itakuraSaito": Itakura-Saito divergence (for spectral data)
        - "generalizedI": Generalized I-divergence (for count data)
        - "logistic": Logistic loss divergence (for binary probabilities)

    maxIter : int, default=20
        Maximum number of iterations.

    tol : float, default=1e-4
        Convergence tolerance (maximum center movement).

    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
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
    >>> # Train with Squared Euclidean (default)
    >>> kmeans = GeneralizedKMeans(k=2, maxIter=20, seed=42)
    >>> model = kmeans.fit(data)
    >>>
    >>> # Make predictions
    >>> predictions = model.transform(data)
    >>> predictions.select("features", "prediction").show()
    >>>
    >>> # Train with KL divergence for probability distributions
    >>> prob_data = spark.createDataFrame([
    ...     (Vectors.dense([0.7, 0.2, 0.1]),),
    ...     (Vectors.dense([0.6, 0.3, 0.1]),),
    ...     (Vectors.dense([0.1, 0.2, 0.7]),),
    ... ], ["features"])
    >>>
    >>> kl_kmeans = GeneralizedKMeans(k=2, divergence="kl", smoothing=1e-10)
    >>> kl_model = kl_kmeans.fit(prob_data)

    Notes
    -----
    - For best performance with Squared Euclidean and k > 100, the library
      automatically uses cross-join assignment strategy.
    - Set SparkContext.setCheckpointDir() before training for long jobs.
    - Use weightCol parameter for weighted clustering.

    See Also
    --------
    GeneralizedKMeansModel : The fitted model
    """

    @keyword_only
    def __init__(
        self,
        *,
        k: int = 2,
        divergence: str = "squaredEuclidean",
        smoothing: float = 1e-10,
        assignmentStrategy: str = "auto",
        emptyClusterStrategy: str = "reseedRandom",
        checkpointInterval: int = 10,
        initMode: str = "k-means||",
        initSteps: int = 2,
        distanceCol: Optional[str] = None,
        featuresCol: str = "features",
        predictionCol: str = "prediction",
        maxIter: int = 20,
        seed: Optional[int] = None,
        tol: float = 1e-4,
        weightCol: Optional[str] = None,
    ):
        """
        Initialize GeneralizedKMeans estimator.
        """
        super(GeneralizedKMeans, self).__init__()
        self._java_obj = self._new_java_obj(
            "com.massivedatascience.clusterer.ml.GeneralizedKMeans", self.uid
        )
        self._setDefault(
            k=2,
            divergence="squaredEuclidean",
            smoothing=1e-10,
            assignmentStrategy="auto",
            emptyClusterStrategy="reseedRandom",
            checkpointInterval=10,
            initMode="k-means||",
            initSteps=2,
            featuresCol="features",
            predictionCol="prediction",
            maxIter=20,
            tol=1e-4,
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        *,
        k: int = 2,
        divergence: str = "squaredEuclidean",
        smoothing: float = 1e-10,
        assignmentStrategy: str = "auto",
        emptyClusterStrategy: str = "reseedRandom",
        checkpointInterval: int = 10,
        initMode: str = "k-means||",
        initSteps: int = 2,
        distanceCol: Optional[str] = None,
        featuresCol: str = "features",
        predictionCol: str = "prediction",
        maxIter: int = 20,
        seed: Optional[int] = None,
        tol: float = 1e-4,
        weightCol: Optional[str] = None,
    ):
        """
        Set parameters for GeneralizedKMeans.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setK(self, value: int):
        """Sets the value of k."""
        return self._set(k=value)

    def setDivergence(self, value: str):
        """Sets the value of divergence."""
        return self._set(divergence=value)

    def setSmoothing(self, value: float):
        """Sets the value of smoothing."""
        return self._set(smoothing=value)

    def setAssignmentStrategy(self, value: str):
        """Sets the value of assignmentStrategy."""
        return self._set(assignmentStrategy=value)

    def setEmptyClusterStrategy(self, value: str):
        """Sets the value of emptyClusterStrategy."""
        return self._set(emptyClusterStrategy=value)

    def setCheckpointInterval(self, value: int):
        """Sets the value of checkpointInterval."""
        return self._set(checkpointInterval=value)

    def setInitMode(self, value: str):
        """Sets the value of initMode."""
        return self._set(initMode=value)

    def setInitSteps(self, value: int):
        """Sets the value of initSteps."""
        return self._set(initSteps=value)

    def setDistanceCol(self, value: str):
        """Sets the value of distanceCol."""
        return self._set(distanceCol=value)

    def setMaxIter(self, value: int):
        """Sets the value of maxIter."""
        return self._set(maxIter=value)

    def setSeed(self, value: int):
        """Sets the value of seed."""
        return self._set(seed=value)

    def setTol(self, value: float):
        """Sets the value of tol."""
        return self._set(tol=value)

    def setFeaturesCol(self, value: str):
        """Sets the value of featuresCol."""
        return self._set(featuresCol=value)

    def setPredictionCol(self, value: str):
        """Sets the value of predictionCol."""
        return self._set(predictionCol=value)

    def setWeightCol(self, value: str):
        """Sets the value of weightCol."""
        return self._set(weightCol=value)

    def _create_model(self, java_model):
        """Create Python model from Java model."""
        return GeneralizedKMeansModel(java_model)


class GeneralizedKMeansModel(JavaModel, GeneralizedKMeansParams, JavaMLReadable, JavaMLWritable):
    """
    Model fitted by GeneralizedKMeans.

    This model can transform data to add cluster predictions and optionally
    distances to cluster centers. It also provides methods for evaluating
    clustering quality.

    Attributes
    ----------
    clusterCenters : np.ndarray
        Array of cluster centers (k x d matrix where k = number of clusters,
        d = feature dimension).

    numClusters : int
        Number of clusters.

    numFeatures : int
        Number of features (dimension).

    Examples
    --------
    >>> # After fitting a model
    >>> centers = model.clusterCenters()
    >>> print(f"Cluster centers: {centers}")
    >>>
    >>> # Transform new data
    >>> predictions = model.transform(test_data)
    >>>
    >>> # Predict single point
    >>> point = Vectors.dense([2.0, 3.0])
    >>> cluster = model.predict(point)
    >>> print(f"Point belongs to cluster {cluster}")
    >>>
    >>> # Compute cost (WCSS)
    >>> cost = model.computeCost(data)
    >>> print(f"Within-cluster sum of squares: {cost}")
    >>>
    >>> # Save model
    >>> model.write().overwrite().save("path/to/model")
    >>>
    >>> # Load model
    >>> loaded_model = GeneralizedKMeansModel.load("path/to/model")
    """

    def clusterCenters(self) -> np.ndarray:
        """
        Get the cluster centers as a NumPy array.

        Returns
        -------
        np.ndarray
            Array of shape (k, d) where k is the number of clusters and
            d is the feature dimension.
        """
        java_centers = self._call_java("clusterCenters")
        # Convert Scala Array[Array[Double]] to numpy array
        return np.array([[float(x) for x in center] for center in java_centers])

    @property
    def numClusters(self) -> int:
        """
        Number of clusters.

        Returns
        -------
        int
            The number of clusters in the model.
        """
        return self._call_java("numClusters")

    @property
    def numFeatures(self) -> int:
        """
        Number of features (dimension).

        Returns
        -------
        int
            The dimensionality of the feature space.
        """
        return self._call_java("numFeatures")

    def predict(self, value: Vector) -> int:
        """
        Predict the cluster for a single data point.

        Parameters
        ----------
        value : Vector
            Feature vector to predict.

        Returns
        -------
        int
            The predicted cluster ID (0 to k-1).

        Examples
        --------
        >>> from pyspark.ml.linalg import Vectors
        >>> point = Vectors.dense([2.0, 3.0])
        >>> cluster = model.predict(point)
        >>> print(f"Cluster: {cluster}")
        """
        return self._call_java("predict", value)

    def computeCost(self, dataset: DataFrame) -> float:
        """
        Compute the within-cluster sum of squares (WCSS).

        This is the sum of squared distances from each point to its
        assigned cluster center, weighted by point weights if provided.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to evaluate (must have features column).

        Returns
        -------
        float
            The WCSS cost.

        Examples
        --------
        >>> cost = model.computeCost(test_data)
        >>> print(f"WCSS: {cost:.2f}")
        """
        return self._call_java("computeCost", dataset)

    def summary(self):
        """
        Get the training summary.

        Returns
        -------
        GeneralizedKMeansSummary
            Training summary with quality metrics.

        Examples
        --------
        >>> summary = model.summary
        >>> print(f"WCSS: {summary.wcss}")
        >>> print(f"BCSS: {summary.bcss}")
        >>> print(f"Calinski-Harabasz: {summary.calinskiHarabaszIndex}")
        """
        return GeneralizedKMeansSummary(self._call_java("summary"))


class GeneralizedKMeansSummary(JavaParams):
    """
    Summary of GeneralizedKMeans training with quality metrics.

    This class provides comprehensive clustering quality metrics including
    within-cluster and between-cluster statistics, and various clustering
    validation indices.

    Attributes
    ----------
    numClusters : int
        Number of clusters.

    numFeatures : int
        Number of features.

    numIter : int
        Number of iterations performed.

    converged : bool
        Whether the algorithm converged.

    wcss : float
        Within-cluster sum of squares.

    bcss : float
        Between-cluster sum of squares.

    calinskiHarabaszIndex : float
        Calinski-Harabasz index (variance ratio criterion).
        Higher values indicate better-defined clusters.

    daviesBouldinIndex : float
        Davies-Bouldin index (average similarity between clusters).
        Lower values indicate better cluster separation.

    dunnIndex : float
        Dunn index (ratio of min inter-cluster to max intra-cluster distance).
        Higher values indicate better cluster separation.

    Examples
    --------
    >>> summary = model.summary
    >>> print(f"Converged: {summary.converged}")
    >>> print(f"Iterations: {summary.numIter}")
    >>> print(f"WCSS: {summary.wcss:.2f}")
    >>> print(f"BCSS: {summary.bcss:.2f}")
    >>> print(f"Calinski-Harabasz: {summary.calinskiHarabaszIndex:.2f}")
    >>> print(f"Davies-Bouldin: {summary.daviesBouldinIndex:.2f}")
    >>> print(f"Dunn Index: {summary.dunnIndex:.2f}")
    >>>
    >>> # Compute silhouette (expensive, uses sampling)
    >>> silhouette = summary.silhouette(sampleFraction=0.1)
    >>> print(f"Mean Silhouette: {silhouette:.3f}")
    """

    def __init__(self, java_obj):
        super(GeneralizedKMeansSummary, self).__init__()
        self._java_obj = java_obj

    @property
    def numClusters(self) -> int:
        """Number of clusters."""
        return self._call_java("numClusters")

    @property
    def numFeatures(self) -> int:
        """Number of features."""
        return self._call_java("numFeatures")

    @property
    def numIter(self) -> int:
        """Number of iterations performed."""
        return self._call_java("numIter")

    @property
    def converged(self) -> bool:
        """Whether the algorithm converged."""
        return self._call_java("converged")

    @property
    def wcss(self) -> float:
        """Within-cluster sum of squares."""
        return self._call_java("wcss")

    @property
    def bcss(self) -> float:
        """Between-cluster sum of squares."""
        return self._call_java("bcss")

    @property
    def calinskiHarabaszIndex(self) -> float:
        """
        Calinski-Harabasz index (variance ratio criterion).
        Higher values indicate better-defined clusters.
        """
        return self._call_java("calinskiHarabaszIndex")

    @property
    def daviesBouldinIndex(self) -> float:
        """
        Davies-Bouldin index (average similarity ratio).
        Lower values indicate better cluster separation.
        """
        return self._call_java("daviesBouldinIndex")

    @property
    def dunnIndex(self) -> float:
        """
        Dunn index (min separation / max diameter).
        Higher values indicate better cluster separation.
        """
        return self._call_java("dunnIndex")

    def silhouette(self, sampleFraction: float = 0.1) -> float:
        """
        Compute mean silhouette coefficient.

        This metric measures how similar each point is to its own cluster
        compared to other clusters. Values range from -1 to 1, where:
        - 1: Point is well-matched to its cluster
        - 0: Point is on the border between clusters
        - -1: Point may be assigned to the wrong cluster

        Parameters
        ----------
        sampleFraction : float, default=0.1
            Fraction of data to sample for computation (0 to 1).
            Silhouette is expensive to compute, so sampling is recommended.

        Returns
        -------
        float
            Mean silhouette coefficient.

        Examples
        --------
        >>> # Sample 10% of data
        >>> silhouette = summary.silhouette(sampleFraction=0.1)
        >>> print(f"Mean Silhouette: {silhouette:.3f}")
        """
        return self._call_java("silhouette", sampleFraction)
