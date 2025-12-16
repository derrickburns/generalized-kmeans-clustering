# Roadmap: Generalized K-Means Clustering

> **Last Updated:** 2025-12-15 (Vectorized BLAS)
> **Status:** Active planning document
> **Maintainer Note:** Claude should inspect and update this file as changes are made.

This document tracks planned improvements, technical debt, and future directions for the generalized-kmeans-clustering library.

---

## Priority Legend

| Priority | Meaning | Typical Timeline |
|----------|---------|------------------|
| P0 | Critical / Blocking | Immediate |
| P1 | High value, low effort | Next release |
| P2 | Medium value or effort | Future release |
| P3 | Nice to have | Backlog |

---

## 1. Bug Fixes (P0)

### 1.1 ~~BLAS.doMax computes minimum~~ ✅ FIXED
- **File:** `src/main/scala/com/massivedatascience/linalg/BLAS.scala:348`
- **Issue:** Comparison operator was `<` instead of `>`
- **Status:** Fixed 2025-12-15

### 1.2 ~~Division by zero in GradMeanUDAFUpdate~~ ✅ FIXED
- **File:** `src/main/scala/com/massivedatascience/clusterer/ml/df/Strategies.scala:482`
- **Issue:** No guard for `weightSum == 0` when user provides zero-weight data
- **Status:** Fixed 2025-12-15

### 1.3 ~~Division by zero in CoClusteringInitializer~~ ✅ FIXED
- **File:** `src/main/scala/com/massivedatascience/clusterer/CoClusteringInitializer.scala:223,251`
- **Issue:** No guard for `weights.sum == 0` in marginal computation
- **Status:** Fixed 2025-12-15

### 1.4 ~~Invalid javac version in build.sbt~~ ✅ FIXED
- **File:** `build.sbt:33`
- **Issue:** Used `"17.0"` instead of valid `"17"` for javac options
- **Status:** Fixed 2025-12-15

---

## 2. Architecture Improvements (P1-P2)

### 2.1 ~~Unify BregmanDivergence and BregmanKernel~~ ✅ COMPLETED
- **Problem:** Duplicate divergence implementations between RDD API (`BregmanDivergence`) and DataFrame API (`BregmanKernel`)
- **Solution implemented:**
  - Created `BregmanFunction` trait as single source of truth
  - Provides `F(x)`, `gradF(x)`, `invGradF(θ)`, `divergence(x,y)`, `validate(x)`
  - `BregmanFunctionAdapter` bridges to both APIs:
    - `asKernel(func)` → BregmanKernel for DataFrame API
    - `asDivergence(func)` → BregmanDivergence for RDD API
  - `BregmanFunctions` factory with all 7 divergences
- **Files created:**
  - `src/main/scala/com/massivedatascience/divergence/BregmanFunction.scala`
  - `src/main/scala/com/massivedatascience/divergence/BregmanFunctionAdapter.scala`
  - `src/test/scala/com/massivedatascience/divergence/BregmanFunctionSuite.scala`
- **Benefits:** Single source of mathematical correctness, reduced duplication
- **Backwards compatible:** Existing APIs unchanged; adapters enable gradual migration
- **Status:** Completed 2025-12-15

### 2.2 ~~Migrate Co-clustering to ML Estimator Pattern~~ ✅ COMPLETED
- **Problem:** `BregmanCoClustering` didn't follow Spark ML Estimator/Model pattern
- **Solution implemented:**
  - Created `CoClustering` Estimator class following Spark ML pattern
  - Created `CoClusteringModel` with transform, persistence, and prediction support
  - Created `CoClusteringParams` trait with all configuration parameters
  - Created `CoClusteringTrainingSummary` for training metrics
  - Supports `squaredEuclidean`, `kl`, `itakuraSaito` divergences
  - Full MLWritable/MLReadable persistence support
- **Files created:**
  - `src/main/scala/com/massivedatascience/clusterer/ml/CoClustering.scala`
  - `src/test/scala/com/massivedatascience/clusterer/ml/CoClusteringSuite.scala`
- **Features:**
  - Alternating minimization algorithm for simultaneous row/column clustering
  - Custom column names for row/col indices and predictions
  - Convergence tolerance and regularization parameters
  - Block center prediction for matrix completion
- **Benefits:** Consistent API, persistence support, pipeline integration
- **Legacy:** Original `BregmanCoClustering` remains for RDD API users
- **Status:** Completed 2025-12-15

### 2.3 Deprecate RDD API (P3)
- **Problem:** Maintaining two parallel APIs increases complexity
- **Current state:** RDD API is "archived" but still present
- **Proposed timeline:**
  1. Add deprecation warnings to RDD entry points
  2. Move RDD code to separate module
  3. Eventually remove from main artifact
- **Status:** Not started

---

## 3. Algorithm Additions (P1-P2)

### 3.1 ~~Add Spherical K-Means / Cosine Similarity~~ ✅ COMPLETED
- **Motivation:** Cosine similarity is standard for text embeddings, NLP, recommendation
- **Implementation:**
  - Added `SphericalKernel` class to `BregmanKernel.scala`
  - Divergence: `spherical` or `cosine` in all estimators
  - 17 comprehensive unit tests added
- **Files modified:**
  - `src/main/scala/com/massivedatascience/clusterer/ml/df/BregmanKernel.scala`
  - `src/main/scala/com/massivedatascience/clusterer/ml/GeneralizedKMeans.scala`
  - `src/main/scala/com/massivedatascience/clusterer/ml/BisectingKMeans.scala`
  - `src/main/scala/com/massivedatascience/clusterer/ml/SoftKMeans.scala`
  - `src/main/scala/com/massivedatascience/clusterer/ml/StreamingKMeans.scala`
  - `src/main/scala/com/massivedatascience/clusterer/ml/GeneralizedKMeansModel.scala`
  - `src/main/scala/com/massivedatascience/clusterer/ml/SoftKMeansModel.scala`
  - `src/test/scala/com/massivedatascience/clusterer/ml/df/BregmanKernelAccuracySuite.scala`
- **Status:** Completed 2025-12-15

### 3.2 Add Elkan/Hamerly Acceleration for Squared Euclidean (P2) ✅ COMPLETED
- **Motivation:** 3-10x speedup for squared Euclidean via triangle inequality bounds
- **Scope:** Only applicable to squared Euclidean (not general Bregman)
- **Phase 1 (single-iteration pruning):**
  - Created `AcceleratedSEAssignment` with center-distance pruning
  - Precomputes k² center-to-center distances
  - Uses triangle inequality to skip distance computations
  - 2-5x speedup for well-separated clusters
  - Falls back to standard for k < 5 (overhead not worthwhile)
  - Factory method: `AcceleratedAssignment.forKernel(kernel, k)`
- **Files created:**
  - `src/main/scala/com/massivedatascience/clusterer/ml/df/AcceleratedSEAssignment.scala`
  - `src/test/scala/com/massivedatascience/clusterer/ml/df/AcceleratedSEAssignmentSuite.scala`
- **Phase 2 (cross-iteration bounds):**
  - Created `ElkanLloydsIterator` with full Elkan algorithm
  - Tracks per-point upper/lower bounds across iterations as DataFrame columns
  - `_elkan_upper`: upper bound on distance to assigned center
  - `_elkan_lower`: lower bound on distance to second-closest center
  - Updates bounds based on center movements each iteration
  - Prunes most points when clusters are converged (10-50x speedup in late iterations)
  - Falls back to DefaultLloydsIterator for k < 5 or non-SE kernels
  - Factory method: `LloydsIteratorFactory.create(kernel, k, useAcceleration)`
- **Files created (Phase 2):**
  - `src/main/scala/com/massivedatascience/clusterer/ml/df/ElkanLloydsIterator.scala`
  - `src/test/scala/com/massivedatascience/clusterer/ml/df/ElkanLloydsIteratorSuite.scala`
- **References:**
  - Elkan (2003): "Using the Triangle Inequality to Accelerate k-Means"
  - Hamerly (2010): "Making k-means Even Faster"
- **Status:** Completed 2025-12-15

### 3.3 Add DP-Means (Bayesian Nonparametric) (P3) ✅ COMPLETED
- **Motivation:** Automatic k selection via distance threshold
- **Algorithm:** Create new cluster when point is > lambda from all centers
- **Implementation:**
  - Created `DPMeans` estimator following Spark ML pattern
  - Created `DPMeansModel` with transform and prediction support
  - Key parameters:
    - `lambda`: Distance threshold for creating new clusters
    - `maxK`: Upper bound on cluster count (prevents runaway)
  - Supports all divergences (squaredEuclidean, kl, spherical, etc.)
  - Algorithm: iteratively adds one cluster per iteration when furthest point > lambda
- **Files created:**
  - `src/main/scala/com/massivedatascience/clusterer/ml/DPMeans.scala`
  - `src/test/scala/com/massivedatascience/clusterer/ml/DPMeansSuite.scala`
- **Reference:** Kulis & Jordan (2012): "Revisiting k-means: New Algorithms via Bayesian Nonparametrics"
- **Status:** Completed 2025-12-15

---

## 4. Performance Improvements (P2)

### 4.1 Vectorized BLAS Operations ✅ COMPLETED
- **Problem:** Some BLAS operations use scalar loops instead of native BLAS
- **Solution:** Added native BLAS functions for common vector operations
- **Implementation:**
  - `nrm2(v)`: L2 norm using native `dnrm2`
  - `squaredNorm(v)`: ||x||² using native `ddot(x, x)`
  - `asum(v)`: L1 norm using native `dasum`
  - `normalize(v)`: Unit L2 normalization using `nrm2` + `scal`
- **Files modified:**
  - `src/main/scala/com/massivedatascience/linalg/BLAS.scala`
  - `src/main/scala/com/massivedatascience/transforms/VectorizedTransforms.scala`
  - `src/main/scala/com/massivedatascience/clusterer/ml/df/FeatureTransform.scala`
- **Files created:**
  - `src/test/scala/com/massivedatascience/linalg/BLASSuite.scala` (24 tests)
- **Benefits:**
  - Native BLAS operations are SIMD-vectorized for better performance
  - Consistent API for norm operations across codebase
  - Supports both dense and sparse vectors
- **Status:** Completed 2025-12-15

### 4.2 Improve Broadcast Chunking Strategy ✅ COMPLETED
- **Problem:** Current chunking uses simple fixed-size chunks
- **Solution:** Created `AdaptiveBroadcastAssignment` with memory-aware chunking
- **Implementation:**
  - Queries Spark configuration for executor memory settings
  - Calculates optimal chunk size: `available_memory / (dim × 8 × safety_factor)`
  - Parameters: `broadcastFraction`, `safetyFactor`, `minChunkSize`, `maxChunkSize`
  - Falls back to sensible defaults if configuration unavailable
  - Comprehensive logging of memory calculations and strategy selection
- **Files modified:**
  - `src/main/scala/com/massivedatascience/clusterer/ml/df/Strategies.scala`
  - `src/test/scala/com/massivedatascience/clusterer/ml/df/AdaptiveBroadcastAssignmentSuite.scala`
- **Status:** Completed 2025-12-15

---

## 5. Documentation & Examples (P2)

### 5.1 Add Jupyter Notebook Examples
- **Content needed:**
  - Quick start notebook
  - Divergence selection guide with visualizations
  - X-Means auto-k demonstration
  - Soft clustering interpretation
- **Status:** Not started

### 5.2 ~~API Documentation (Scaladoc)~~ ✅ COMPLETED
- **Problem:** Some public APIs lacked comprehensive Scaladoc
- **Files enhanced:**
  - `GeneralizedKMeans.scala` - Full algorithm docs, divergence table, examples
  - `BregmanKernel.scala` - Mathematical formulas, domain requirements
  - `BisectingKMeans.scala` - Algorithm steps, use cases, advantages
  - `SoftKMeans.scala` - Beta parameter guide, fuzzy clustering examples
  - `StreamingKMeans.scala` - Decay factor guide, streaming examples
- **Status:** Completed 2025-12-15

---

## 6. Testing Improvements (P2)

### 6.1 ~~Add Property-Based Tests for All Divergences~~ ✅ COMPLETED
- **Properties tested:**
  - Non-negativity: D(x,y) >= 0
  - Identity: D(x,x) = 0
  - Gradient consistency: invGrad(grad(x)) ≈ x
  - Divergence accuracy (known analytical values)
- **Coverage:** 79 tests covering all 7 kernels (SE, KL, IS, GenI, Logistic, L1, Spherical)
- **Files:** `BregmanKernelAccuracySuite.scala`, `PropertyBasedTestSuite.scala`
- **Status:** Completed 2025-12-15

### 6.2 ~~Add Benchmark Suite~~ ✅ COMPLETED
- **Purpose:** Catch performance regressions, compare strategies
- **Benchmarks implemented:**
  - Lloyd's iteration throughput (points/sec, points/iter/sec)
  - Assignment strategy comparison (auto vs crossJoin vs broadcastUDF)
  - Divergence comparison (SE, KL, Spherical, L1)
  - Scaling with k (2, 5, 10, 20 clusters)
  - Scaling with dimension (5, 20, 50, 100 dims)
  - Spherical vs SE on normalized data
- **Framework:** ScalaTest with JSON output to `target/perf-reports/`
- **File:** `BenchmarkSuite.scala`
- **Status:** Completed 2025-12-15

---

## 7. Completed Items

| Item | Completed | Notes |
|------|-----------|-------|
| DataFrame API migration | 2024 | Primary API is now DF-based |
| Domain validation helpers | 2024 | Actionable error messages |
| Broadcast threshold diagnostics | 2024 | Warns on large k×dim |
| Model persistence | 2024 | Cross-version compatible |
| K-Medoids (PAM/CLARA) | 2024 | DataFrame API |
| K-Medians (L1) | 2024 | Via L1Kernel |
| Streaming K-Means | 2024 | DataFrame API |
| Bug: BLAS.doMax | 2025-12-15 | Fixed comparison operator |
| Bug: Zero-weight division | 2025-12-15 | Fixed in Strategies.scala |
| Bug: CoClusteringInitializer div/0 | 2025-12-15 | Fixed marginal computation |
| Bug: build.sbt javac version | 2025-12-15 | Fixed "17.0" → "17" |
| Spherical K-Means kernel | 2025-12-15 | Added `SphericalKernel` for cosine similarity |
| Property tests for kernels | 2025-12-15 | 79 tests covering all 7 kernels (6.1 complete) |
| Comprehensive Scaladoc | 2025-12-15 | All 5 estimators documented with examples |
| README Spherical K-Means docs | 2025-12-15 | Added section, examples, domain requirements |
| SphericalKMeansExample | 2025-12-15 | Executable example with assertions in CI |
| BenchmarkSuite | 2025-12-15 | Strategy comparison, scaling, throughput metrics |
| BregmanFunction unification | 2025-12-15 | Single source of truth for divergences (2.1) |
| Co-clustering ML migration | 2025-12-15 | CoClustering Estimator/Model following ML pattern (2.2) |
| AcceleratedSEAssignment | 2025-12-15 | Center-distance pruning for SE (3.2 Phase 1) |
| ElkanLloydsIterator | 2025-12-15 | Cross-iteration bounds tracking (3.2 Phase 2) |
| DPMeans | 2025-12-15 | Automatic k selection via distance threshold (3.3) |
| AdaptiveBroadcastAssignment | 2025-12-15 | Memory-aware broadcast chunking (4.2) |
| Vectorized BLAS | 2025-12-15 | Native nrm2, squaredNorm, asum, normalize (4.1) |

---

## 8. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2024 | DataFrame API as primary | Better Spark integration, SQL optimization |
| 2024 | Archive RDD API | Maintenance burden, DF covers use cases |
| 2024 | Keep L1 as "kernel" despite not being Bregman | Practical utility outweighs theoretical purity |
| 2025-12-15 | Prioritize spherical k-means | High demand for embedding clustering |
| 2025-12-15 | SphericalKernel in BregmanKernel.scala | Keep all kernels in one file for consistency |
| 2025-12-15 | BregmanFunction as unified trait | Single source of truth with adapters for backwards compatibility |
| 2025-12-15 | CoClustering as separate ML Estimator | Keep legacy RDD API intact, provide clean DF-based alternative |
| 2025-12-15 | Phased approach for Elkan acceleration | Start with single-iteration pruning (fits current architecture), defer cross-iteration bounds |
| 2025-12-15 | ElkanLloydsIterator as specialized iterator | Encapsulates full Elkan algorithm with DataFrame-based bounds tracking, keeps AssignmentStrategy interface unchanged |
| 2025-12-15 | DPMeans adds one cluster per iteration | Conservative approach prevents runaway cluster creation; furthest outlier gets new cluster |

---

## How to Contribute

1. Pick an item from sections 2-6
2. Create a branch: `feature/<item-id>` or `fix/<item-id>`
3. Implement with tests
4. Update this ROADMAP.md to mark progress
5. Submit PR referencing this document

## Updating This Document

When making changes to the codebase:
1. Mark completed items with date
2. Add new issues/opportunities as discovered
3. Update priority based on user feedback
4. Keep decision log current
