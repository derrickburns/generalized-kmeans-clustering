# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Building and Testing
- `sbt compile` - Compile the project
- `sbt test` - Run all tests 
- `sbt scalastyle` - Run Scalastyle checks (configuration in scalastyle-config.xml)
- `sbt package` - Create JAR package

### Project Structure
This is a Scala 2.12 project built with SBT targeting Spark 3.4.0 and Java 17. The project implements generalized K-means clustering algorithms using Bregman divergences.

## Architecture Overview

### Core Components

**Bregman Divergences** (`com.massivedatascience.divergence`):
- `BregmanDivergence` - Core trait defining convex functions and gradients for distance calculations
- Implements various distance functions (Euclidean, Kullback-Leibler, Itakura-Saito, etc.)

**Point Operations** (`com.massivedatascience.clusterer.BregmanPointOps`):
- Enriches Bregman divergences with clustering operations
- Factory methods for creating `BregmanPoint` and `BregmanCenter` instances
- Handles distance calculations and cluster assignments

**Clustering Algorithms** (`com.massivedatascience.clusterer`):
- `KMeans` - Main entry point with various training methods
- `MultiKMeansClusterer` - Interface for different clustering implementations:
  - `COLUMN_TRACKING` - High-performance implementation that reduces work on later iterations
  - `MINI_BATCH_10` - Mini-batch algorithm sampling 10% of data per round
  - `RESEED` - Re-seeds empty clusters to reach target cluster count

**Models** (`com.massivedatascience.clusterer.KMeansModel`):
- Represents trained clustering models with prediction capabilities
- Support for Vector, WeightedVector, and BregmanPoint inputs
- Factory methods for creating models from various sources

**Initialization** (`com.massivedatascience.clusterer.KMeansSelector`):
- `RANDOM` - Random selection of initial centers
- `K_MEANS_PARALLEL` - 5-step K-Means|| parallel initialization

**Embeddings** (`com.massivedatascience.transforms`):
- `IDENTITY_EMBEDDING` - No transformation
- `HAAR_EMBEDDING` - Haar wavelet transform for time series
- `*_DIMENSIONAL_RI` - Random indexing for dimensionality reduction

### Key Design Patterns

**Weighted Vectors**: All operations work with `WeightedVector` objects that combine data points with weights, enabling weighted clustering.

**Iterative Training**: Supports multi-stage training where lower-dimensional embeddings initialize higher-dimensional clustering.

**Pluggable Distance Functions**: Architecture allows easy addition of new Bregman divergences through the `BregmanDivergence` trait.

### Testing Infrastructure

Tests use ScalaTest with a custom `LocalClusterSparkContext` trait that sets up local Spark contexts for testing. Test files are in `src/test/scala/com/massivedatascience/clusterer/`.

### Dependencies
- Apache Spark 3.4.0 (core, SQL, streaming, MLlib)
- Scala 2.12.18
- ScalaTest 3.2.17 for testing
- BLAS libraries for linear algebra operations