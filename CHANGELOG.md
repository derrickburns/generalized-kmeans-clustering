# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive CI validation DAG with cross-version testing
- Production quality blockers documented in ACTION_ITEMS.md
- SECURITY.md with vulnerability reporting guidelines
- CONTRIBUTING.md with development guidelines
- Test suite fixes for Scala 2.12/2.13 compatibility
- Spherical K-Means / cosine divergence support across estimators and models
- New estimators: Mini-Batch K-Means, DP-Means, Balanced K-Means, Constrained (semi-supervised) K-Means, Kernel K-Means, Agglomerative Bregman, Bregman mixture models (EM), and CoClustering following the Spark ML Estimator/Model pattern
- Bregman-native k-means++ seeding plus executable examples for spherical k-means
- Outlier detection scaffolding with distance- and trim-based detectors
- Property-based kernel accuracy suites and a performance benchmark suite with JSON outputs
- **RobustKMeans estimator** for outlier-resistant clustering with trim, noise_cluster, and m_estimator modes
- **SparseKMeans estimator** for high-dimensional sparse data with auto-sparsity detection (21 tests)
  - Automatic sparse kernel selection based on data sparsity ratio
  - Support for SE, KL, L1, and Spherical divergences with sparse optimization
  - `sparseMode` parameter: "auto", "force", or "dense"
  - `sparseThreshold` parameter for auto-mode sparsity cutoff
- **KernelFactory** for unified dense/sparse kernel creation with clear API
  - Single entry point for all 8 Bregman divergences
  - Auto-selection based on data sparsity with `forSparsity()` method
  - Canonical divergence name constants in `KernelFactory.Divergence`
- **MultiViewKMeans estimator** for clustering data with multiple feature representations (21 tests)
  - Per-view divergences (different distance measures for each view)
  - Per-view weights (importance weighting)
  - Combine strategies: "weighted" (default), "max", "min"
  - `ViewSpec` case class for view configuration
  - Full persistence support (save/load)
- **Test suites for new components** (150 new tests, 758 total):
  - OutlierDetectionSuite: 16 tests for distance-based and trimmed outlier detection
  - SparseBregmanKernelSuite: 28 tests for sparse-optimized SE, KL, L1, Spherical kernels
  - ConstraintsSuite: 30 tests for must-link/cannot-link constraints and penalty computation
  - ConstrainedKMeansSuite: 17 tests for semi-supervised clustering with soft/hard constraints
  - RobustKMeansSuite: 17 tests for robust clustering with outlier handling and persistence
  - SparseKMeansSuite: 21 tests for sparse clustering with auto-detection and persistence
  - MultiViewKMeansSuite: 21 tests for multi-view clustering with persistence

### Architecture
- Moved AcceleratedSEAssignment to `strategies/impl/` subpackage for better organization
- Added type aliases in package objects for backward compatibility
- Models now use KernelFactory for kernel creation (reduces code duplication)

### Fixed
- Package name conflicts in StreamingKMeans and XMeans test suites
- Scala 2.12 compatibility issues with `isFinite` method
- SparseSEKernel divergenceSparse missing 0.5 factor (now matches SquaredEuclideanKernel)
- AgglomerativeBregmanModel persistence serializing IntParam object instead of value
- Spark 3.4 compatibility issues with `model.summary` API
- CollectionConverters imports for cross-version support
- BLAS `doMax` comparison, division-by-zero guards in strategies and co-clustering initializer, and invalid javac option in `build.sbt`

### Changed
- Unified divergence math via `BregmanFunction` and refactored kernel factory for consistency
- Added Bregman-native initialization path and enriched Scaladoc across major estimators
- Enhanced clustering iterator and constraint frameworks to support new variants

### Performance
- Accelerated squared-Euclidean assignment and Elkan-style cross-iteration bounds for Lloyd's iterations
- Vectorized BLAS helpers for common linear algebra operations

### Removed
- Legacy RDD API and associated coreset/transform modules (DataFrame/ML API is now the sole surface)

## [0.6.0] - 2025-10-18

### Added
- **New Algorithms**:
  - Bisecting K-Means with DataFrame API (10/10 tests passing)
  - X-Means for automatic cluster count selection (BIC/AIC, 12/12 tests)
  - Soft K-Means for probabilistic assignments (15/15 tests)
  - Streaming K-Means for online learning (16/16 tests)
  - K-Medoids clustering (PAM/CLARA, 26/26 tests)
  - K-Medians (L1/Manhattan distance)

- **Core Abstractions**:
  - Feature Transform system
  - CenterStore for persistence
  - AssignmentPlan for strategy selection
  - KernelOps for type-safe operations
  - ReseedPolicy for empty cluster handling
  - SummarySink for telemetry
  - Typed error handling with GKMError

- **Cross-Version Support**:
  - Scala 2.12.18 and 2.13.14 support
  - Spark 3.4.x and 3.5.x compatibility
  - Cross-version test matrix in CI

- **CI/CD**:
  - GitHub Actions workflows
  - Comprehensive test matrix (Scala 2.12/2.13 × Spark 3.4.x/3.5.x)
  - Example execution validation
  - Cross-version persistence tests
  - Performance sanity checks
  - Python smoke tests
  - Scalastyle linting

- **Testing**:
  - 290/290 tests passing
  - Property-based tests
  - Edge case coverage
  - Performance regression tests

### Changed
- Migrated from Scala 2.11 to 2.12/2.13
- Updated from Spark 1.x to 3.4.x/3.5.x
- Modernized build system
- Improved parallel collections compatibility

### Fixed
- KMeans++ weighted selection correctness
- K-means|| initialization issues
- Numerical stability improvements
- Memory efficiency optimizations

### Security
- Updated all dependencies to latest secure versions
- Removed Travis CI configuration

## [0.5.x] - Historical

Earlier versions (0.1.0 - 0.5.x) were developed between 2014-2020 with:
- Initial Bregman K-Means implementation
- Support for multiple divergences (Euclidean, KL, Itakura-Saito, etc.)
- RDD-based API
- Spark 1.x compatibility
- Basic testing infrastructure

[Unreleased]: https://github.com/derrickburns/generalized-kmeans-clustering/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/derrickburns/generalized-kmeans-clustering/releases/tag/v0.6.0
