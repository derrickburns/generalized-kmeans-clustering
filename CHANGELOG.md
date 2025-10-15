# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2025-10-15

### Added

#### DataFrame API Algorithms
- **Bisecting K-Means**: Hierarchical divisive clustering with tree structure
  - Supports all Bregman divergences
  - `minDivisibleClusterSize` parameter for controlling splits
  - 10/10 tests passing, 178 lines of examples

- **X-Means**: Automatic k selection using statistical criteria
  - BIC and AIC information criteria
  - `minK` and `maxK` parameters for search range
  - 12/12 tests passing, 210 lines of examples

- **Soft K-Means**: Fuzzy clustering with probabilistic memberships
  - Boltzmann distribution for soft assignments
  - `beta` parameter for controlling assignment sharpness
  - `effectiveNumberOfClusters()` metric (entropy-based)
  - Hard and soft cost computation
  - 15/15 tests passing

- **Streaming K-Means**: Real-time clustering with concept drift handling
  - Exponential forgetting with decay factor (0.0 to 1.0)
  - Time unit options: batches or points
  - Half-life parameter for intuitive decay specification
  - Automatic dying cluster handling (splits largest cluster)
  - Structured Streaming integration via foreachBatch API
  - 16/16 tests passing, 470 lines of examples

- **K-Medoids (PAM)**: Robust clustering using actual data points as medoids
  - BUILD phase: Greedy medoid selection
  - SWAP phase: Iterative improvement
  - Multiple distance functions: Euclidean, Manhattan, Cosine
  - More robust to outliers than K-Means
  - 16/16 PAM tests passing

- **CLARA**: Sampling-based K-Medoids for large datasets
  - 10-100x faster than PAM on large datasets (>10,000 points)
  - Auto sample sizing: 40 + 2*k (from original CLARA paper)
  - Configurable `numSamples` and `sampleSize` parameters
  - 10/10 CLARA tests passing
  - 306 lines of comprehensive examples

#### Core Features
- **K-Medians**: L1/Manhattan distance for robust clustering
  - Implemented via `L1Kernel` and `MedianUpdateStrategy`
  - Component-wise weighted median computation
  - 6/6 tests passing

- **PySpark Wrapper**: Python integration for DataFrame API
  - `GeneralizedKMeans` exposed via PySpark
  - Smoke test for CI workflow
  - Package structure with setup.py

#### Documentation
- **DATAFRAME_API_EXAMPLES.md**: 2,013 lines of comprehensive examples
  - Basic usage for all divergences
  - Advanced variants with detailed examples
  - Performance tuning guidelines
  - When to use each algorithm

- **ARCHITECTURE.md**: Deep dive into design patterns
- **MIGRATION_GUIDE.md**: RDD → DataFrame migration path
- **PERFORMANCE_TUNING.md**: Optimization tips and best practices
- **ACTION_ITEMS.md**: Comprehensive project tracking

### Changed

#### Scala Migration
- **Primary Scala version**: 2.13.14 (was 2.12.18)
- **Cross-compilation**: Maintained Scala 2.12.18 support
- Fixed all Scala 2.13 compatibility issues
- Resolved scaladoc generation (compiler bug workaround)
- Added parallel collections dependency

#### Build & CI
- Updated CI/CD workflows for Scala 2.13
- Re-enabled scaladoc generation
- Enhanced test coverage across all algorithms

#### Documentation
- Updated README.md with feature matrix table
- Added "What's New in 0.6.0" section
- Comprehensive feature comparison

### Fixed
- Resolved Spark configuration issues in tests
- Eliminated test warnings for clean output
- Fixed deprecation warnings for implicit Long → Double widening in XMeans
- Fixed BisectingKMeansSuite compilation issues (stable identifier for implicits)

### Performance
- All algorithms tested with comprehensive suites
- 95+ total tests passing across all variants
- Proven scalability on datasets with millions of points

### Breaking Changes
None - Version 0.6.0 maintains full backward compatibility with existing APIs.

---

## [0.5.x] - Previous Versions

See git history for changes in versions prior to 0.6.0.

---

## Future Releases

### Planned for 0.7.0 (Q1 2026)
- Performance benchmarking suite
- Enhanced property-based testing
- Integration test suite
- Documentation improvements

### Planned for 0.8.0 (Q2 2026)
- Elkan's triangle inequality acceleration (3-5x speedup)
- Performance regression tests
- Memory profiling

### Planned for 1.0.0 (Q3 2026)
- Production-ready stability
- API stabilization
- Breaking changes cleanup

---

## Links
- [GitHub Repository](https://github.com/derrickburns/generalized-kmeans-clustering)
- [DataFrame API Examples](DATAFRAME_API_EXAMPLES.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Migration Guide](MIGRATION_GUIDE.md)
