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

    def hasSummary(self) -> bool:
        """
        Check if training summary is available.

        Returns
        -------
        bool
            True if summary is available (model trained in current session).
        """
        return self._call_java("hasSummary")

    @property
    def summary(self):
        """
        Get the training summary.

        Returns
        -------
        TrainingSummary
            Training summary with detailed metrics.

        Examples
        --------
        >>> if model.hasSummary():
        ...     summary = model.summary
        ...     print(f"Algorithm: {summary.algorithm}")
        ...     print(f"Iterations: {summary.iterations}")
        ...     print(f"Converged: {summary.converged}")
        ...     print(f"Final distortion: {summary.finalDistortion}")
        """
        return TrainingSummary(self._call_java("summary"))


class TrainingSummary(JavaParams):
    """
    Training summary with detailed metrics about the clustering process.

    This class provides comprehensive information about model training including
    iterations, convergence, distortion history, and performance metrics.

    Attributes
    ----------
    algorithm : str
        Algorithm name (e.g., "GeneralizedKMeans", "XMeans", "SoftKMeans").

    k : int
        Requested number of clusters.

    effectiveK : int
        Actual number of non-empty clusters.

    dim : int
        Feature dimensionality.

    numPoints : int
        Number of training points.

    iterations : int
        Number of iterations performed.

    converged : bool
        Whether the algorithm converged.

    finalDistortion : float
        Final clustering cost/distortion.

    assignmentStrategy : str
        Assignment strategy used (e.g., "CrossJoin", "BroadcastUDF", "PAM").

    divergence : str
        Divergence function used.

    elapsedMillis : int
        Training time in milliseconds.

    Examples
    --------
    >>> if model.hasSummary():
    ...     summary = model.summary
    ...     print(f"Algorithm: {summary.algorithm}")
    ...     print(f"Converged in {summary.iterations} iterations")
    ...     print(f"Final distortion: {summary.finalDistortion:.4f}")
    ...     print(f"Training time: {summary.elapsedMillis}ms")
    ...     print(f"Average per iteration: {summary.avgIterationMillis:.1f}ms")
    """

    def __init__(self, java_obj):
        super(TrainingSummary, self).__init__()
        self._java_obj = java_obj

    @property
    def algorithm(self) -> str:
        """Algorithm name."""
        return self._call_java("algorithm")

    @property
    def k(self) -> int:
        """Requested number of clusters."""
        return self._call_java("k")

    @property
    def effectiveK(self) -> int:
        """Actual number of non-empty clusters."""
        return self._call_java("effectiveK")

    @property
    def dim(self) -> int:
        """Feature dimensionality."""
        return self._call_java("dim")

    @property
    def numPoints(self) -> int:
        """Number of training points."""
        return self._call_java("numPoints")

    @property
    def iterations(self) -> int:
        """Number of iterations performed."""
        return self._call_java("iterations")

    @property
    def converged(self) -> bool:
        """Whether the algorithm converged."""
        return self._call_java("converged")

    @property
    def finalDistortion(self) -> float:
        """Final clustering cost/distortion."""
        return self._call_java("finalDistortion")

    @property
    def assignmentStrategy(self) -> str:
        """Assignment strategy used."""
        return self._call_java("assignmentStrategy")

    @property
    def divergence(self) -> str:
        """Divergence function used."""
        return self._call_java("divergence")

    @property
    def elapsedMillis(self) -> int:
        """Training time in milliseconds."""
        return self._call_java("elapsedMillis")

    @property
    def avgIterationMillis(self) -> float:
        """Average time per iteration in milliseconds."""
        return self._call_java("avgIterationMillis")

    def convergenceReport(self) -> str:
        """Get a detailed convergence report as a string."""
        return self._call_java("convergenceReport")


# Keep old class name for backward compatibility
GeneralizedKMeansSummary = TrainingSummary


class XMeans(JavaEstimator, GeneralizedKMeansParams, JavaMLReadable, JavaMLWritable):
    """
    X-Means clustering with automatic k selection using BIC/AIC.

    X-Means automatically determines the optimal number of clusters by
    iteratively splitting clusters and evaluating model quality using
    information criteria (BIC or AIC).

    Parameters
    ----------
    minK : int, default=2
        Minimum number of clusters to consider.

    maxK : int, default=10
        Maximum number of clusters to consider.

    criterion : str, default="bic"
        Information criterion for model selection ("bic" or "aic").

    Examples
    --------
    >>> from massivedatascience.clusterer import XMeans
    >>> xmeans = XMeans(minK=2, maxK=5, criterion="bic")
    >>> model = xmeans.fit(data)
    >>> print(f"Optimal k: {model.numClusters}")
    """

    minK = Param(
        Params._dummy(),
        "minK",
        "Minimum number of clusters",
        typeConverter=TypeConverters.toInt,
    )

    maxK = Param(
        Params._dummy(),
        "maxK",
        "Maximum number of clusters",
        typeConverter=TypeConverters.toInt,
    )

    criterion = Param(
        Params._dummy(),
        "criterion",
        "Information criterion: bic or aic",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(
        self,
        *,
        minK: int = 2,
        maxK: int = 10,
        criterion: str = "bic",
        divergence: str = "squaredEuclidean",
        **kwargs
    ):
        super(XMeans, self).__init__()
        self._java_obj = self._new_java_obj(
            "com.massivedatascience.clusterer.ml.XMeans", self.uid
        )
        self._setDefault(minK=2, maxK=10, criterion="bic")
        self.setParams(minK=minK, maxK=maxK, criterion=criterion, divergence=divergence, **kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def setMinK(self, value: int):
        return self._set(minK=value)

    def setMaxK(self, value: int):
        return self._set(maxK=value)

    def setCriterion(self, value: str):
        return self._set(criterion=value)

    def _create_model(self, java_model):
        return GeneralizedKMeansModel(java_model)


class SoftKMeans(JavaEstimator, GeneralizedKMeansParams, JavaMLReadable, JavaMLWritable):
    """
    Soft (Fuzzy) K-Means clustering with probabilistic cluster assignments.

    Unlike standard k-means, Soft K-Means assigns each point to multiple clusters
    with probabilities, providing a more nuanced view of cluster membership.

    Parameters
    ----------
    k : int, default=2
        Number of clusters.

    beta : float, default=1.0
        Temperature parameter controlling soft assignment fuzziness.
        Higher values make assignments more deterministic.

    Examples
    --------
    >>> from massivedatascience.clusterer import SoftKMeans
    >>> soft = SoftKMeans(k=3, beta=2.0)
    >>> model = soft.fit(data)
    >>> predictions = model.transform(data)
    >>> # predictions DataFrame includes "probabilities" column
    """

    beta = Param(
        Params._dummy(),
        "beta",
        "Temperature parameter for soft assignments",
        typeConverter=TypeConverters.toFloat,
    )

    @keyword_only
    def __init__(self, *, k: int = 2, beta: float = 1.0, **kwargs):
        super(SoftKMeans, self).__init__()
        self._java_obj = self._new_java_obj(
            "com.massivedatascience.clusterer.ml.SoftKMeans", self.uid
        )
        self._setDefault(beta=1.0)
        self.setParams(k=k, beta=beta, **kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def setBeta(self, value: float):
        return self._set(beta=value)

    def _create_model(self, java_model):
        return GeneralizedKMeansModel(java_model)


class BisectingKMeans(JavaEstimator, GeneralizedKMeansParams, JavaMLReadable, JavaMLWritable):
    """
    Bisecting K-Means clustering using hierarchical divisive approach.

    Bisecting K-Means starts with all points in one cluster and iteratively
    splits the largest cluster until reaching k clusters. This approach is:
    - More deterministic than random initialization
    - Often faster for large k
    - Better at handling imbalanced cluster sizes

    Parameters
    ----------
    k : int, default=2
        Number of leaf clusters to create.

    minDivisibleClusterSize : int, default=1
        Minimum size for a cluster to be divisible.

    Examples
    --------
    >>> from massivedatascience.clusterer import BisectingKMeans
    >>> bisecting = BisectingKMeans(k=10, minDivisibleClusterSize=5)
    >>> model = bisecting.fit(data)
    """

    minDivisibleClusterSize = Param(
        Params._dummy(),
        "minDivisibleClusterSize",
        "Minimum divisible cluster size",
        typeConverter=TypeConverters.toInt,
    )

    @keyword_only
    def __init__(self, *, k: int = 2, minDivisibleClusterSize: int = 1, **kwargs):
        super(BisectingKMeans, self).__init__()
        self._java_obj = self._new_java_obj(
            "com.massivedatascience.clusterer.ml.BisectingKMeans", self.uid
        )
        self._setDefault(minDivisibleClusterSize=1)
        self.setParams(k=k, minDivisibleClusterSize=minDivisibleClusterSize, **kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def setMinDivisibleClusterSize(self, value: int):
        return self._set(minDivisibleClusterSize=value)

    def _create_model(self, java_model):
        return GeneralizedKMeansModel(java_model)


class KMedoids(JavaEstimator, JavaMLReadable, JavaMLWritable):
    """
    K-Medoids clustering using PAM (Partitioning Around Medoids) algorithm.

    K-Medoids uses actual data points as cluster centers (medoids) instead of
    computed centroids. This makes it:
    - More robust to outliers
    - More interpretable (medoids are real data points)
    - Works with any distance function

    Parameters
    ----------
    k : int, default=2
        Number of clusters.

    distanceFunction : str, default="euclidean"
        Distance function ("euclidean", "manhattan", "cosine").

    maxIter : int, default=20
        Maximum number of swap iterations.

    Examples
    --------
    >>> from massivedatascience.clusterer import KMedoids
    >>> kmedoids = KMedoids(k=3, distanceFunction="manhattan")
    >>> model = kmedoids.fit(data)
    >>> # model.medoidIndices contains indices of selected medoids
    """

    k = Param(
        Params._dummy(),
        "k",
        "Number of clusters",
        typeConverter=TypeConverters.toInt,
    )

    distanceFunction = Param(
        Params._dummy(),
        "distanceFunction",
        "Distance function: euclidean, manhattan, cosine",
        typeConverter=TypeConverters.toString,
    )

    maxIter = Param(
        Params._dummy(),
        "maxIter",
        "Maximum iterations",
        typeConverter=TypeConverters.toInt,
    )

    @keyword_only
    def __init__(
        self,
        *,
        k: int = 2,
        distanceFunction: str = "euclidean",
        maxIter: int = 20,
        seed: Optional[int] = None,
        featuresCol: str = "features",
        predictionCol: str = "prediction",
    ):
        super(KMedoids, self).__init__()
        self._java_obj = self._new_java_obj(
            "com.massivedatascience.clusterer.ml.KMedoids", self.uid
        )
        self._setDefault(k=2, distanceFunction="euclidean", maxIter=20)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def setK(self, value: int):
        return self._set(k=value)

    def setDistanceFunction(self, value: str):
        return self._set(distanceFunction=value)

    def setMaxIter(self, value: int):
        return self._set(maxIter=value)

    def setSeed(self, value: int):
        return self._set(seed=value)

    def setFeaturesCol(self, value: str):
        return self._set(featuresCol=value)

    def setPredictionCol(self, value: str):
        return self._set(predictionCol=value)

    def _create_model(self, java_model):
        return KMedoidsModel(java_model)


class KMedoidsModel(JavaModel, JavaMLReadable, JavaMLWritable):
    """
    Model fitted by KMedoids.

    Attributes
    ----------
    medoids : np.ndarray
        Array of medoid vectors.

    medoidIndices : List[int]
        Indices of medoids in the original dataset.
    """

    @property
    def medoids(self) -> np.ndarray:
        """Get medoid vectors as NumPy array."""
        java_medoids = self._call_java("medoids")
        return np.array([[float(x) for x in medoid] for medoid in java_medoids])

    @property
    def medoidIndices(self) -> List[int]:
        """Get indices of medoids in original dataset."""
        return list(self._call_java("medoidIndices"))

    def hasSummary(self) -> bool:
        """Check if training summary is available."""
        return self._call_java("hasSummary")

    @property
    def summary(self):
        """Get training summary."""
        return TrainingSummary(self._call_java("summary"))


class StreamingKMeans(JavaEstimator, GeneralizedKMeansParams, JavaMLReadable, JavaMLWritable):
    """
    Streaming K-Means for incremental online clustering.

    Updates cluster centers incrementally as new batches of data arrive,
    using exponential forgetting to handle concept drift.

    Parameters
    ----------
    k : int, default=2
        Number of clusters.

    decayFactor : float, default=1.0
        Exponential decay factor (0.0 to 1.0).
        - 1.0: No forgetting (all batches weighted equally)
        - 0.0: Complete forgetting (only current batch matters)

    halfLife : float, optional
        Alternative to decayFactor. Time for weight to decay to 50%.

    Examples
    --------
    >>> from massivedatascience.clusterer import StreamingKMeans
    >>> streaming = StreamingKMeans(k=3, decayFactor=0.9)
    >>> model = streaming.fit(initial_batch)
    >>> # Update with new batch
    >>> updated_model = model.update(new_batch)
    """

    decayFactor = Param(
        Params._dummy(),
        "decayFactor",
        "Exponential decay factor",
        typeConverter=TypeConverters.toFloat,
    )

    halfLife = Param(
        Params._dummy(),
        "halfLife",
        "Half-life for decay",
        typeConverter=TypeConverters.toFloat,
    )

    timeUnit = Param(
        Params._dummy(),
        "timeUnit",
        "Time unit: batches or points",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(
        self,
        *,
        k: int = 2,
        decayFactor: float = 1.0,
        halfLife: Optional[float] = None,
        timeUnit: str = "batches",
        **kwargs
    ):
        super(StreamingKMeans, self).__init__()
        self._java_obj = self._new_java_obj(
            "com.massivedatascience.clusterer.ml.StreamingKMeans", self.uid
        )
        self._setDefault(decayFactor=1.0, timeUnit="batches")
        self.setParams(k=k, decayFactor=decayFactor, halfLife=halfLife, timeUnit=timeUnit, **kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def setDecayFactor(self, value: float):
        return self._set(decayFactor=value)

    def setHalfLife(self, value: float):
        return self._set(halfLife=value)

    def setTimeUnit(self, value: str):
        return self._set(timeUnit=value)

    def _create_model(self, java_model):
        return StreamingKMeansModel(java_model)


class StreamingKMeansModel(GeneralizedKMeansModel):
    """
    Model fitted by StreamingKMeans.

    Supports incremental updates with new data batches.
    """

    def update(self, dataset: DataFrame):
        """
        Update model with new batch of data.

        Parameters
        ----------
        dataset : DataFrame
            New batch of data to incorporate.

        Returns
        -------
        StreamingKMeansModel
            Updated model (same object, mutated in place).
        """
        self._call_java("update", dataset)
        return self

    @property
    def currentWeights(self) -> List[float]:
        """Get current cluster weights."""
        return list(self._call_java("currentWeights"))
