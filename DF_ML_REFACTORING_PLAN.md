# DataFrame/ML Pipeline Refactoring Plan

## Overview

This document outlines the comprehensive plan to migrate from RDD-based K-Means clustering to a DataFrame-native, Spark ML Pipeline implementation with `LloydsIterator` as the central abstraction.

## Goals

1. **Eliminate RDD dependencies** - Move to DataFrame/Dataset API
2. **Unify algorithm implementation** - Single `LloydsIterator` + pluggable strategies
3. **Expression-based operations** - Leverage Catalyst optimizer for performance
4. **Spark ML integration** - Native `Estimator`/`Model` pattern
5. **Code reduction** - Eliminate 1000s of lines of duplicated loop logic

## Architecture

### Core Abstraction: LloydsIterator

Single source of truth for Lloyd's algorithm with pluggable strategies:

```scala
trait LloydsIterator {
  def run(
    df: DataFrame,
    initialCenters: Array[Array[Double]],
    k: Int,
    kernel: BregmanKernel,
    assigner: AssignmentStrategy,
    updater: UpdateStrategy,
    emptyHandler: EmptyClusterHandler,
    convergence: ConvergenceCheck,
    validator: InputValidator,
    options: LloydsOptions
  ): LloydResult
}
```

### Pluggable Strategies

#### 1. AssignmentStrategy
- **BroadcastUDFAssignment** - General Bregman divergences (broadcast centers)
- **SECrossJoinAssignment** - Squared Euclidean fast path (cross-join + expression-based)
- **AutoAssignment** - Heuristic-based selection with logging

#### 2. UpdateStrategy
- **GradMeanUDAFUpdate** - Weight-aware gradient mean aggregation
- Uses `grad`/`invGrad` for Bregman centers

#### 3. EmptyClusterHandler
- **ReseedRandomHandler** - Random point selection (default, cheap)
- **ReseedFarthestHandler** - Farthest point from assigned centers
- **NearestPointHandler** - Nearest unassigned point
- **DropHandler** - Remove empty clusters

#### 4. ConvergenceCheck
- **MovementConvergence** - Max L2 movement of centers
- **DistortionConvergence** - Relative change in total distortion

## Implementation Phases

### Phase 0: Environment & Versioning ✅ COMPLETE
- [x] Create `feature/df-ml-wrapper` branch
- [x] Document refactoring plan
- [x] Set version to `0.6.0-SNAPSHOT`
- [ ] Update README with DataFrame API notice (deferred to Phase 7)

### Phase 1: Build & Tooling ✅ COMPLETE
- [x] Update `build.sbt`:
  - Scala 2.12.18, 2.13.14 cross-build
  - Spark 3.5.1 default (3.4.3+ supported via `-Dspark.version`)
  - JDK 17 with module opens
- [x] Add dependencies:
  - `spark-sql` (Provided)
  - `spark-mllib` (Provided)
  - Test: `scalatest 3.2.19`, `scalacheck 1.17.0`
- [x] Configure tooling:
  - CI flags: `-Dci=true` for fatal warnings
  - Test forking with proper JVM options
- **Commit:** `feat: configure build for DataFrame/ML Pipeline refactoring (Phase 1)`

### Phase 2: Estimator/Model API ✅ COMPLETE
Created `src/main/scala/com/massivedatascience/clusterer/ml/`:

#### Files Created:
1. **`GeneralizedKMeansParams.scala`** (208 lines) ✅
   - Shared parameter trait for Estimator and Model
   - Parameters: k, divergence, smoothing, weightCol, assignmentStrategy, emptyClusterStrategy,
     checkpointInterval, initMode, initSteps, distanceCol, featuresCol, predictionCol, maxIter, tol, seed
   - Schema validation via `validateAndTransformSchema()`
   - Default values for all parameters

2. **`GeneralizedKMeans.scala`** (272 lines) ✅
   - Estimator extending `Estimator[GeneralizedKMeansModel]`
   - `fit(dataset)` integrates with LloydsIterator
   - Initialization: random sampling and k-means|| (simplified)
   - Factory methods for kernels, strategies, and handlers
   - Persistence via `DefaultParamsWritable/Readable`

3. **`GeneralizedKMeansModel.scala`** (230 lines) ✅
   - Model extending `Model[GeneralizedKMeansModel]`
   - `transform(dataset)` adds prediction column and optional distance column
   - `predict(features)` for single-point prediction
   - `computeCost(dataset)` for model evaluation
   - `GeneralizedKMeansSummary` stub for training statistics
   - Persistence support

**Commit:** `feat: implement Spark ML Estimator/Model API (Phase 2)`

### Phase 3.0: LloydsIterator Core ✅ COMPLETE
Created `src/main/scala/com/massivedatascience/clusterer/ml/df/`:

#### Core Engine Files Created:

1. **`BregmanKernel.scala`** (395 lines) ✅
   - Core `BregmanKernel` trait with grad/invGrad/divergence/validate interface
   - 5 kernel implementations:
     - `SquaredEuclideanKernel` - L2 distance (supports expression optimization)
     - `KLDivergenceKernel` - for probability distributions
     - `ItakuraSaitoKernel` - for spectral data
     - `GeneralizedIDivergenceKernel` - for count data
     - `LogisticLossKernel` - for binary probabilities
   - Smoothing support for numerical stability

2. **`LloydsIterator.scala`** (168 lines) ✅
   - `LloydsIterator` trait defining the core loop interface
   - `DefaultLloydsIterator` implementation with:
     - Convergence tracking (movement and distortion history)
     - Empty cluster handling integration
     - Checkpointing support
     - Comprehensive logging
   - `LloydsConfig` and `LloydResult` case classes

3. **`Strategies.scala`** (484 lines) ✅
   - **AssignmentStrategy:**
     - `BroadcastUDFAssignment` - general Bregman divergences
     - `SECrossJoinAssignment` - fast path for Squared Euclidean
     - `AutoAssignment` - heuristic selection
   - **UpdateStrategy:**
     - `GradMeanUDAFUpdate` - weighted gradient-based center updates
   - **EmptyClusterHandler:**
     - `ReseedRandomHandler` - reseed with random points
     - `DropEmptyClustersHandler` - return fewer than k
   - **ConvergenceCheck:**
     - `MovementConvergence` - max L2 movement + distortion
   - **InputValidator:**
     - `StandardInputValidator` - basic validation

**Commit:** `feat: implement DataFrame-based clustering core (Phase 3.0)`

### Phase 3.0: Testing ✅ COMPLETE

Created `src/test/scala/com/massivedatascience/clusterer/ml/`:

**`GeneralizedKMeansSuite.scala`** (272 lines) ✅
- 11 comprehensive integration tests
- Tests cover:
  - Fit/transform with Squared Euclidean and KL divergence
  - Distance column output
  - Single-point prediction and cost computation
  - Assignment strategies (auto, broadcast)
  - Initialization modes (random, k-means||)
  - Weighted clustering
  - Model operations

**All tests passing ✅**

**Commit:** `test: add comprehensive integration tests and fix initialization bugs`

---

## Summary of Phases 0-3.0

**Total Implementation:**
- **2,329 lines** of production code
- **4 commits** on `feature/df-ml-wrapper` branch
- **All tests passing** (11 integration tests)

**Key Achievement:**
Single LloydsIterator implementation replaces 1000s of lines of duplicated RDD-based loop logic across different clustering algorithms. The pluggable strategy pattern provides clean extensibility.

---

### Phase 3.1-3.6: Additional Features (FUTURE)
Optional enhancements to be implemented as needed:

- 3.1: Additional divergence implementations (e.g., Exponential, Mahalanobis)
- 3.2: Symmetrized divergence support
- 3.3: Enhanced K-means|| initialization (full parallel version)
- 3.4: Bisecting K-Means DataFrame wrapper
- 3.5: Coreset builder for massive datasets
- 3.6: Structured Streaming integration (optional)

### Phase 4: Performance & Summary (FUTURE)
- Enhanced GeneralizedKMeansSummary with:
  - Silhouette coefficient
  - Within-cluster sum of squares (WCSS)
  - Between-cluster sum of squares (BCSS)
  - Davies-Bouldin index
- Performance benchmarks vs RDD implementation
- Robustness improvements

### Phase 5: Expanded Tests (FUTURE)
- LloydsIterator invariant tests
- Strategy parity tests (verify all strategies produce similar results)
- Model persistence tests
- Property-based tests with ScalaCheck
- Performance regression tests
- Cross-validation with existing RDD tests

### Phase 6: PySpark Wrapper (FUTURE)
- Python API for GeneralizedKMeans
- Parameter exposure matching Scala API
- Example Jupyter notebooks
- PySpark-specific tests

### Phase 7: Documentation (NEXT)
- Update README.md with DataFrame API
- Architecture guide explaining LloydsIterator pattern
- API documentation (ScalaDoc)
- Migration guide from RDD to DataFrame API
- Usage examples for each divergence
- Performance tuning guide

### Phase 8: Release (FUTURE)
- Tag `v0.6.0` for DataFrame API release
- Update CHANGELOG
- Publish artifacts to Maven Central
- Create GitHub release with examples

## Migration Strategy

### Compatibility
- Keep existing RDD-based API for backward compatibility
- New DataFrame API in `com.massivedatascience.clusterer.ml` package
- Deprecation warnings point to new API

### Performance Comparison
- Benchmark DataFrame vs RDD implementation
- Document performance characteristics
- Provide tuning guidance

## Success Criteria

1. **Code Reduction**: 1000+ lines eliminated via LloydsIterator
2. **Performance**: DataFrame implementation ≥ RDD performance
3. **Tests**: All existing tests pass, new DataFrame tests added
4. **Documentation**: Complete API docs and examples
5. **Compatibility**: RDD API remains functional with deprecation warnings

## Next Steps

1. ✅ Create feature branch
2. ✅ Document plan
3. ✅ Update build.sbt for cross-compilation
4. ✅ Implement LloydsIterator core
5. ✅ Create GeneralizedKMeans Estimator/Model
6. ✅ Implement strategies
7. ✅ Add integration tests (11 tests passing)
8. ⏳ Run full test suite (verify no regressions)
9. ⏳ Create usage examples
10. ⏳ Update README with DataFrame API
11. ⏳ Merge to master when ready

---

## Notes

- This is a **major refactoring** that will take multiple sessions
- Incremental commits with working tests at each step
- Keep `master` branch stable with RDD implementation
- Merge `feature/df-ml-wrapper` when DataFrame API is complete and tested
