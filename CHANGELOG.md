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

### Fixed
- Package name conflicts in StreamingKMeans and XMeans test suites
- Scala 2.12 compatibility issues with `isFinite` method
- Spark 3.4 compatibility issues with `model.summary` API
- CollectionConverters imports for cross-version support

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
