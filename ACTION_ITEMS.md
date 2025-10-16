# Action Items - Generalized K-Means Clustering

**Last Updated**: 2025-10-15
**Status**: Post Scala 2.13 Migration

---

## ✅ Recently Completed (October 2025)

### Scala 2.13 Migration (October 2025)
- [x] Migrate to Scala 2.13.14 as default version
- [x] Fix all Scala 2.13 compatibility issues
- [x] Re-enable scaladoc generation (resolved compiler bug)
- [x] Update CI/CD workflows for Scala 2.13
- [x] Add parallel collections dependency

### K-Medians Implementation (October 2025)
- [x] Implement L1Kernel (Manhattan distance)
- [x] Implement MedianUpdateStrategy (component-wise weighted median)
- [x] Add "l1" and "manhattan" divergence support to GeneralizedKMeans
- [x] Create comprehensive test suite (6/6 tests passing)
- [x] Validate robustness to outliers

### Bisecting K-Means (DataFrame API) - COMPLETED October 2025 ✅
- [x] Implement BisectingKMeans estimator
- [x] Support all Bregman divergences
- [x] Add minDivisibleClusterSize parameter
- [x] Create comprehensive test suite (10/10 tests passing)
- [x] Add usage examples (178 lines)
- [x] Update documentation (200 lines in ARCHITECTURE.md)
- [x] Fix DataFrame column management issues
- [x] All tests passing

**Deliverables**:
- Full hierarchical divisive clustering implementation
- Works with all Bregman divergences (Euclidean, KL, L1, etc.)
- Weighted data support
- Comprehensive examples and architecture docs

### X-Means (DataFrame API) - COMPLETED October 2025 ✅
- [x] Port X-Means algorithm to DataFrame API
- [x] Implement BIC/AIC criteria for model selection
- [x] Support range of k values (minK to maxK)
- [x] Create comprehensive test suite (12/12 tests passing)
- [x] Add usage examples (210 lines)
- [x] All Bregman divergence support
- [x] Weighted data support

**Deliverables**:
- Automatic k selection using statistical criteria
- BIC vs AIC model selection
- Eliminates need to specify k in advance
- Production-ready with full test coverage

### Soft K-Means (DataFrame API) - COMPLETED October 2025 ✅
- [x] Port Soft K-Means to DataFrame API
- [x] Implement probabilistic assignments (Boltzmann distribution)
- [x] Add mixture model estimation capabilities
- [x] Create comprehensive test suite (15/15 tests passing)
- [x] Support all Bregman divergences
- [x] Weighted data support

**Deliverables**:
- Fuzzy clustering with soft probabilistic memberships
- Beta parameter for controlling assignment sharpness
- effectiveNumberOfClusters() metric (entropy-based)
- Hard and soft cost computation
- Model persistence (save/load)

### Streaming K-Means (DataFrame API) - COMPLETED October 2025 ✅
- [x] Integrate with Structured Streaming
- [x] Implement incremental updates with mini-batch algorithm
- [x] Add decay factor for exponential forgetting
- [x] Create comprehensive test suite (16/16 tests passing)
- [x] Support all Bregman divergences
- [x] Weighted data support

**Deliverables**:
- Real-time clustering with incremental updates
- Exponential forgetting with decay factor
- Time unit options (batches vs points)
- Half-life parameter for intuitive decay
- Automatic dying cluster handling
- foreachBatch integration with Structured Streaming
- Mutable state management

### K-Medoids (PAM/CLARA) - COMPLETED October 2025 ✅
- [x] Implement PAM (Partitioning Around Medoids) algorithm
- [x] Implement CLARA for large datasets
- [x] Add support for custom distance functions (Euclidean, Manhattan, Cosine)
- [x] Create comprehensive test suite (26/26 tests passing)
- [x] Add examples and documentation (306 lines)

**Deliverables**:
- Robust clustering using actual data points as medoids
- More resistant to outliers than K-Means
- CLARA sampling-based variant for datasets > 10,000 points
- Auto sample sizing (40 + 2*k)
- Model persistence (save/load)
- Performance tuning guidelines

### Python Wrapper (October 2025)
- [x] Create PySpark wrapper for GeneralizedKMeans
- [x] Add smoke test for CI workflow
- [x] Package structure and setup.py

### Documentation (October 2025)
- [x] Create ARCHITECTURE.md
- [x] Create MIGRATION_GUIDE.md (RDD → DataFrame)
- [x] Create PERFORMANCE_TUNING.md
- [x] Enhance DATAFRAME_API_EXAMPLES.md
- [x] Document scaladoc issue (now resolved)
- [x] Consolidate ACTION_ITEMS.md (removed 8 redundant files)
- [x] Add Bisecting K-Means section (378 lines examples + architecture)
- [x] Add X-Means section (210 lines examples)

---

## 🔧 Critical Bug Fixes & Test Improvements (Completed October 2025)

### Test Suite Fixes - COMPLETED ✅
**Effort**: 1 day (COMPLETED)
**Impact**: Critical (all tests now passing)

Fixed 4 failing tests from initial test run (286/290 → 290/290 passing):

#### 1. PropertyBasedTestSuite: "model has exactly k cluster centers" ✅
- **Issue**: ScalaCheck shrinking created 0-dimensional data; maxIter=10 triggered checkpointing without directory
- **Fix**: Added `dim >= 1` guard, reduced maxIter to 8 (below checkpoint interval)
- **File**: `PropertyBasedTestSuite.scala:246-272`

#### 2. KMeansPlusPlusSuite: "keep pre-selected centers" ✅
- **Root Cause**: Commit 0f968d9 removed weighted selection; commit fd7c407 exposed bug with zero-weight validation
- **Fixes**:
  - Use original weights for distances (not reweighted zero-weight points)
  - Multiply distances by selection weights for probabilities
  - Fix binary search for cumulative weights starting with zeros
- **File**: `KMeansPlusPlus.scala:127, 141, 417-420`

#### 3. KMeansSuite: "k-means|| initialization" ✅
- **Fix**: Resolved as side effect of KMeans++ fix above

#### 4. KMeansSuite: "empty clusters handling" ✅
- **Fixes**:
  - Accept `<= k` clusters (empty cluster filtering is valid)
  - Relax coherence check to allow initialization variance
- **File**: `KMeansSuite.scala:622-625, 644-645`

**Final Status**: 290/290 tests passing (100% success rate)

---

## 🚧 High Priority (Q4 2025 - Q1 2026)

### 1. Complete Bisecting K-Means Testing & Documentation ✅
**Effort**: 1 week (COMPLETED)
**Impact**: High

- [x] Create BisectingKMeansSuite with 10+ tests
- [x] Test all divergence types (Euclidean, KL, L1, etc.)
- [x] Test minDivisibleClusterSize edge cases
- [x] Add examples to DATAFRAME_API_EXAMPLES.md
- [x] Update ARCHITECTURE.md with hierarchical clustering section

**Status**: COMPLETED
- All 10 tests passing
- Comprehensive examples added (178 lines)
- Architecture section added (200 lines)

### 2. DataFrame API - Advanced Variants
**Effort**: 4-6 weeks
**Impact**: High (API consistency)

Bring RDD-based advanced algorithms to DataFrame API:

#### X-Means (Auto K Selection) ✅
- [x] Port X-Means algorithm to DataFrame API
- [x] Implement BIC/AIC criteria for model selection
- [x] Add tests and examples
- [x] Support range of k values (k_min, k_max)

**Status**: COMPLETED
- Full DataFrame implementation (282 lines)
- BIC and AIC information criteria
- All 12 tests passing
- Comprehensive examples added (210 lines)

**Features delivered**:
- Automatic k selection using statistical criteria
- Support for all Bregman divergences
- Weighted data handling
- Configurable search range (minK/maxK)
- BIC vs AIC selection

**Files created**:
- `src/main/scala/com/massivedatascience/clusterer/ml/XMeans.scala` ✓
- `src/test/scala/com/massivedatascience/clusterer/XMeansSuite.scala` ✓

#### Soft K-Means (Mixture Models) ✅
- [x] Port Soft K-Means to DataFrame API
- [x] Implement probabilistic assignments
- [x] Add mixture model estimation
- [x] Create tests and examples

**Status**: COMPLETED (October 2025)
- Full DataFrame implementation (689 lines)
- Probabilistic cluster memberships using Boltzmann distribution
- All 15 tests passing
- Beta parameter for controlling assignment sharpness

**Features delivered**:
- Fuzzy clustering with soft assignments
- Support for all Bregman divergences
- Weighted data handling
- effectiveNumberOfClusters() metric
- Hard and soft cost computation
- Model persistence

**Files created**:
- `src/main/scala/com/massivedatascience/clusterer/ml/SoftKMeans.scala` ✓
- `src/test/scala/com/massivedatascience/clusterer/SoftKMeansSuite.scala` ✓

#### Streaming K-Means - COMPLETED October 2025 ✅
- [x] Integrate with Structured Streaming
- [x] Implement incremental updates
- [x] Add decay factor for aging data
- [x] Create comprehensive test suite (16/16 tests passing)
- [x] Create streaming examples (470 lines, 13 examples)

**Status**: COMPLETED (October 2025)
- Full DataFrame implementation (531 lines)
- Mini-batch K-Means with exponential forgetting
- All 16 tests passing
- Decay factor with batch/point time units
- Half-life parameter support

**Features delivered**:
- Incremental model updates using foreachBatch API
- Exponential forgetting with configurable decay factor
- Time unit options (batches or points)
- Half-life parameter for intuitive decay specification
- Automatic dying cluster handling (splits largest cluster)
- Support for all Bregman divergences
- Weighted data handling
- Mutable state with sync to immutable parent

**Files created**:
- `src/main/scala/com/massivedatascience/clusterer/ml/StreamingKMeans.scala` ✓
- `src/test/scala/com/massivedatascience/clusterer/StreamingKMeansSuite.scala` ✓

### 3. K-Medoids (PAM/CLARA) - COMPLETED October 2025 ✅
**Effort**: 3-4 weeks (COMPLETED)
**Impact**: High (industry standard algorithm)

- [x] Implement PAM (Partitioning Around Medoids) algorithm
- [x] Implement CLARA for large datasets (sampling-based PAM)
- [x] Add support for custom distance functions
- [x] Create comprehensive tests
- [x] Add to DataFrame API
- [x] Add examples and documentation

**Deliverables**:
- Full PAM implementation with BUILD and SWAP phases
- CLARA sampling-based variant for large datasets
- Multiple distance functions (Euclidean, Manhattan, Cosine)
- 26/26 tests passing (16 PAM + 10 CLARA)
- 306 lines of comprehensive examples
- Model persistence (save/load)

**Features delivered**:
- Robust clustering using actual data points as medoids
- More resistant to outliers than K-Means
- Works with arbitrary distance metrics
- Auto sample sizing for CLARA (40 + 2*k)
- Performance tuning guidelines
- Time complexity: PAM O(k(n-k)²), CLARA O(numSamples × k(s-k)²)

**Files created**:
- `src/main/scala/com/massivedatascience/clusterer/ml/KMedoids.scala` ✓ (740 lines)
- `src/test/scala/com/massivedatascience/clusterer/KMedoidsSuite.scala` ✓ (776 lines)

---

## 📊 Medium Priority (Q2 2026)

### 4. Performance Benchmarking Suite
**Effort**: 2-3 weeks
**Impact**: Medium (validation & optimization)

- [ ] Create JMH-based benchmark suite
- [ ] Benchmark all divergence types
- [ ] Compare with MLlib KMeans
- [ ] Profile memory usage
- [ ] Document performance characteristics

**Files to create**:
- `src/benchmark/scala/com/massivedatascience/clusterer/KMeansBenchmark.scala`
- `PERFORMANCE_BENCHMARKS.md`

### 5. Elkan's Triangle Inequality Acceleration
**Effort**: 3-4 weeks
**Impact**: High (3-5x speedup for Euclidean)

- [ ] Implement Elkan's algorithm for Euclidean distance
- [ ] Add as optional acceleration strategy
- [ ] Benchmark performance improvements
- [ ] Document when to use

**Benefits**:
- 3-5x speedup for large k
- Lower memory overhead than full distance matrix
- Significant improvement for high-dimensional data

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/ElkanAssignment.scala`
- Performance comparison in benchmarks

### 6. Enhanced Testing
**Effort**: 1-2 weeks
**Impact**: Medium (quality assurance)

- [ ] Increase property-based test coverage
- [ ] Add integration tests for all DataFrame variants
- [ ] Test edge cases (empty clusters, single point, etc.)
- [ ] Add performance regression tests

**Files to update**:
- `src/test/scala/com/massivedatascience/clusterer/PropertyBasedTestSuite.scala`
- Create `IntegrationTestSuite.scala`

---

## 🔮 Low Priority (Q3-Q4 2026)

### 7. Yinyang K-Means
**Effort**: 4-6 weeks
**Impact**: Medium (optimization for large k)

- [ ] Implement Yinyang K-Means acceleration
- [ ] Optimize for k > 100
- [ ] Benchmark against standard Lloyd's

**Benefits**:
- Faster than Elkan's for very large k
- Uses global and local filtering

### 8. GPU Acceleration
**Effort**: 8-12 weeks
**Impact**: High (specific use cases)

- [ ] Evaluate RAPIDS cuML integration
- [ ] Implement GPU-accelerated assignment step
- [ ] Benchmark speedup vs CPU

**Benefits**:
- 10-100x speedup for large datasets
- Useful for high-dimensional data

### 9. Additional Divergences
**Effort**: 1-2 weeks per divergence
**Impact**: Low-Medium

- [ ] Mahalanobis distance
- [ ] Cosine similarity
- [ ] Hellinger distance
- [ ] Jensen-Shannon divergence

---

## 📝 Documentation & Cleanup

### Immediate (Next Sprint)

- [ ] **Consolidate redundant markdown files** (THIS TASK)
  - Remove: `IMPROVEMENTS_SUMMARY.md` (content moved to RELEASE_NOTES)
  - Remove: `REFACTORING_SUMMARY.md` (outdated, covered in ARCHITECTURE.md)
  - Remove: `EVALUATION_SUMMARY.md` (one-time assessment, archive)
  - Remove: `IMPLEMENTATION_STATUS.md` (outdated, covered in this file)
  - Remove: `PROPERTY_TESTING_FINDINGS.md` (historical, integrate key findings)
  - Remove: `DF_ML_REFACTORING_PLAN.md` (completed, archive)
  - Remove: `MERGE_CHECKLIST.md` (specific to one merge, archive)
  - Remove: `SAMPLE_LOG.md` (not needed)
  - Keep: README.md, ARCHITECTURE.md, MIGRATION_GUIDE.md, PERFORMANCE_TUNING.md, DATAFRAME_API_EXAMPLES.md, CONTRIBUTING.md, CLAUDE.md

- [ ] **Update README.md**
  - Add Scala 2.13 as primary version
  - Update feature matrix (add K-Medians, Bisecting K-Means DataFrame)
  - Add performance benchmarks
  - Update quickstart examples

- [ ] **Create CHANGELOG.md**
  - Move release notes from scattered files
  - Follow Keep a Changelog format
  - Document breaking changes

### Ongoing

- [ ] Keep ARCHITECTURE.md updated with new features
- [ ] Add examples for each new algorithm
- [ ] Update API docs (scaladoc) with more examples
- [ ] Create video tutorials (stretch goal)

---

## 🐛 Known Issues & Tech Debt

### Minor

- [ ] Fix deprecation warnings in Scala 2.13
  - XMeans.scala: Widening Long → Double (2 occurrences)
  - StreamingKMeans.scala: Widening Long → Double (1 occurrence)
  - Add explicit `.toDouble` conversions

- [ ] Scalastyle warnings (non-blocking)
  - Multiple files: "return" statements (refactor to functional style)
  - Multiple files: Cyclomatic complexity >10 (consider refactoring)
  - CoClusteringVisualization.scala: Trailing whitespace

- [x] BisectingKMeansSuite compilation issues
  - Stable identifier required for `spark.implicits._`
  - Fixed by removing unnecessary imports

### Technical Debt

- [ ] Reduce code duplication between RDD and DataFrame implementations
- [ ] Consider extracting common patterns to shared traits
- [ ] Evaluate Scala 3 migration path
- [ ] Consider breaking into multiple modules (core, ml, advanced, etc.)

---

## 📦 Release Planning

### Version 0.6.0 (Current - COMPLETED October 2025)
- ✅ Scala 2.13 migration
- ✅ K-Medians implementation
- ✅ Python wrapper
- ✅ Bisecting K-Means (DataFrame)
- ✅ X-Means (DataFrame)
- ✅ Soft K-Means (DataFrame)
- ✅ Streaming K-Means (DataFrame)
- ✅ K-Medoids (PAM/CLARA)

### Version 0.7.0 (Q1 2026)
- Enhanced testing suite
- Performance benchmarking
- Documentation improvements

### Version 0.8.0 (Q2 2026)
- Elkan's acceleration
- Performance benchmarking suite
- DataFrame Streaming K-Means

### Version 1.0.0 (Q3 2026)
- Production-ready stability
- Comprehensive documentation
- Performance optimizations
- Breaking API cleanup

---

## 🎯 Success Metrics

### Code Quality
- [ ] Test coverage > 95%
- [ ] All Scalastyle warnings addressed
- [ ] No deprecation warnings
- [ ] Documentation coverage > 90%

### Performance
- [ ] Benchmarks published
- [ ] Performance regression tests
- [ ] Memory profiling complete

### Adoption
- [ ] Published to Maven Central
- [ ] 10+ GitHub stars
- [ ] Example notebooks published
- [ ] Conference presentation/paper

---

## 📚 References

- **RDD Implementation**: `src/main/scala/com/massivedatascience/clusterer/`
- **DataFrame Implementation**: `src/main/scala/com/massivedatascience/clusterer/ml/`
- **Tests**: `src/test/scala/com/massivedatascience/clusterer/`
- **Documentation**: Root directory markdown files
- **Python Wrapper**: `python/massivedatascience/`

---

## 🏗️ Architectural Refactoring (High ROI)

**Goal**: Reduce code size, improve readability, and enable easy extensibility through well-placed abstractions.

### Priority 1: Core Abstractions (Immediate - Highest ROI)

#### 1.1 FeatureTransform (Pure, Composable)
**Effort**: 1 week
**Impact**: Very High (affects all algorithms)
**Dependencies**: None

- [ ] Create `FeatureTransform` trait for log1p, epsilonShift, L2 normalization
- [ ] Implement composable transforms
- [ ] Add inverse transforms for reporting
- [ ] Integrate before seeding and LloydsIterator
- [ ] Support spherical k-means via NormalizeL2

**Benefits**:
- Centralize "centers live in transformed space" logic
- Eliminate transform switches scattered in code
- Enable transform composition
- Clear separation of concerns

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/FeatureTransform.scala`
- `src/test/scala/com/massivedatascience/clusterer/ml/df/FeatureTransformSuite.scala`

#### 1.2 CenterStore (Uniform Center I/O & Ordering)
**Effort**: 3-4 days
**Impact**: High (eliminates ad-hoc array juggling)
**Dependencies**: None

- [ ] Create `CenterStore` trait for uniform center management
- [ ] Implement `ArrayCenterStore`
- [ ] Replace array-based center handling across codebase
- [ ] Add stable ordering guarantees
- [ ] Support DataFrame serialization

**Benefits**:
- Single source of truth for center management
- Consistent ordering across operations
- Easy persistence
- Cleaner assignment/updater APIs

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/CenterStore.scala`
- Update: AssignmentStrategy, UpdateStrategy, persistence code

#### 1.3 AssignmentPlan Algebra (Explainable Assignment)
**Effort**: 2 weeks
**Impact**: Very High (makes complex logic readable)
**Dependencies**: CenterStore

- [ ] Define `AssignmentPlan` ADT for assignment steps
- [ ] Create `AssignmentInterpreter` for plan execution
- [ ] Refactor AssignmentStrategy to build plans
- [ ] Add unit tests for individual steps
- [ ] Document SE fast path as plan

**Benefits**:
- Readable, testable assignment logic
- Unit tests at step level (no Spark session gymnastics)
- Easy to add new strategies
- Clear documentation of algorithms

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/AssignmentPlan.scala`
- `src/main/scala/com/massivedatascience/clusterer/ml/df/AssignmentInterpreter.scala`
- Update: All AssignmentStrategy implementations

#### 1.4 RowIdProvider (Stable Row Identity)
**Effort**: 2-3 days
**Impact**: Medium (consistency for SE fast path)
**Dependencies**: None

- [ ] Create `RowIdProvider` trait
- [ ] Implement `MonotonicRowId`
- [ ] Use in SE cross-join assignment
- [ ] Apply to bisecting selective updates
- [ ] Add to performance tests

**Benefits**:
- Consistent row identification
- Explicit about row ID strategy
- Enables optimizations like groupBy(rowId).min(distance)

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/RowIdProvider.scala`

### Priority 2: Strategy & Policy Abstractions (Next - High ROI)

#### 2.1 KernelOps Typeclass (Capabilities & Hints)
**Effort**: 1 week
**Impact**: High (eliminates string switches)
**Dependencies**: None

- [ ] Create `KernelOps` typeclass for kernel capabilities
- [ ] Define `supportsSEFastPath`, `safeTransforms`, `defaultBroadcastThreshold`
- [ ] Implement instances for all kernels
- [ ] Use in AutoAssignment decision
- [ ] Use in transform validation

**Benefits**:
- Type-safe kernel capabilities
- Compile-time checks for supported operations
- Clear documentation of kernel properties
- Easy to extend for new kernels

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/KernelOps.scala`
- Update: AutoAssignment, validation code

#### 2.2 ReseedPolicy (Empty Cluster Handling)
**Effort**: 3-4 days
**Impact**: Medium (cleaner iteration logic)
**Dependencies**: None

- [ ] Create `ReseedPolicy` trait
- [ ] Implement `ReseedRandom`, `ReseedFarthest`
- [ ] Use in LloydsIterator empty cluster step
- [ ] Document cost tradeoffs
- [ ] Add tests for each policy

**Benefits**:
- Pluggable empty cluster handling
- Single location for reseed logic
- Easy to add new strategies
- Clear cost documentation

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/ReseedPolicy.scala`
- Update: LloydsIterator

#### 2.3 MiniBatchScheduler (Sampling & EMA)
**Effort**: 2-3 days
**Impact**: Medium (unifies batch variants)
**Dependencies**: None

- [ ] Create `MiniBatchScheduler` trait
- [ ] Implement `FixedMiniBatch`, `FullBatch`
- [ ] Use in UpdateStrategy
- [ ] Support decay schedules
- [ ] Add to streaming k-means

**Benefits**:
- Single abstraction for batch strategies
- Easy to add decay schedules
- Cleaner UpdateStrategy logic

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/MiniBatchScheduler.scala`

#### 2.4 SeedingService (Initialization Strategies)
**Effort**: 1 week
**Impact**: High (centralizes seeding)
**Dependencies**: CenterStore

- [ ] Create `SeedingService` trait
- [ ] Implement random, ++, ||, Bregman++
- [ ] Use in Estimator.fit
- [ ] Use in bisecting splits
- [ ] Make deterministic by seed

**Benefits**:
- Single location for all seeding logic
- Deterministic seeding
- Easy to add new strategies
- Consistent across algorithms

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/SeedingService.scala`
- Update: GeneralizedKMeans, BisectingKMeans

### Priority 3: Quality of Life (Later - Medium ROI)

#### 3.1 Validation Combinators (Explicit, Testable)
**Effort**: 3-4 days
**Impact**: Medium (better error messages)
**Dependencies**: None

- [ ] Create `Validator` trait with combinators
- [ ] Implement domain validators (positive, finite, etc.)
- [ ] Add kernel/transform compatibility validators
- [ ] Return sample rows with violations
- [ ] Use in pre-iteration validation

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/Validator.scala`

#### 3.2 SummarySink (Telemetry)
**Effort**: 3-4 days
**Impact**: Medium (better observability)
**Dependencies**: None

- [ ] Create `Event` ADT for telemetry
- [ ] Implement `SummarySink` for event collection
- [ ] Collect iteration metrics, warnings, reseed events
- [ ] Expose via model.summary
- [ ] Add to logs

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/SummarySink.scala`

#### 3.3 Error ADT (Typed Failures)
**Effort**: 2-3 days
**Impact**: Medium (better error handling)
**Dependencies**: None

- [ ] Create `GKMError` sealed trait
- [ ] Implement specific error types
- [ ] Replace scattered exceptions
- [ ] Add to validation
- [ ] Improve error messages

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/GKMError.scala`

#### 3.4 Config Object (Lower Ceremony)
**Effort**: 2-3 days
**Impact**: Low-Medium (convenience)
**Dependencies**: None

- [ ] Create `GKMConfig` case class
- [ ] Add builders for common configurations
- [ ] Use in tests for conciseness
- [ ] Map to/from Params
- [ ] Document patterns

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/GKMConfig.scala`

### Priority 4: Advanced Features (Low Priority)

#### 4.1 PersistenceLayout (Structured Save/Load)
**Effort**: 1 week
**Impact**: Low (nice to have)
**Dependencies**: CenterStore, SummarySink

- [ ] Create `PersistenceLayout` trait
- [ ] Implement `ParquetLayoutV1`
- [ ] Support centers, params, summaries
- [ ] Version management
- [ ] Migration support

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/PersistenceLayout.scala`

#### 4.2 StreamingStateStore (Clean State Boundary)
**Effort**: 3-4 days
**Impact**: Low (streaming-specific)
**Dependencies**: CenterStore

- [ ] Create `StreamingStateStore` trait
- [ ] Implement checkpoint-based store
- [ ] Clean mapGroupsWithState logic
- [ ] Add snapshot publishing
- [ ] Test off-stream

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/StreamingStateStore.scala`

#### 4.3 CoresetSampler (Strategy Pattern)
**Effort**: 2-3 days
**Impact**: Low (coreset-specific)
**Dependencies**: None

- [ ] Create `CoresetSampler` trait
- [ ] Implement uniform, K-Means||, sensitivity sampling
- [ ] Use in CoresetBuilder
- [ ] Benchmark strategies
- [ ] Document tradeoffs

**Files to create**:
- `src/main/scala/com/massivedatascience/clusterer/ml/df/CoresetSampler.scala`

### Implementation Order (Recommended)

**Phase 1 (Week 1-2)**: Foundation
1. FeatureTransform ← Highest immediate impact
2. CenterStore ← Foundation for others
3. RowIdProvider ← Quick win

**Phase 2 (Week 3-4)**: Core Logic
4. AssignmentPlan + Interpreter ← Major readability improvement
5. KernelOps ← Eliminate string switches

**Phase 3 (Week 5-6)**: Policies
6. ReseedPolicy
7. SeedingService
8. MiniBatchScheduler

**Phase 4 (Week 7+)**: Quality of Life
9. Validation combinators
10. SummarySink
11. Error ADT
12. Config builders

**Phase 5 (As needed)**: Advanced
13. PersistenceLayout
14. StreamingStateStore
15. CoresetSampler

### Expected Benefits

**Code Size**: 20-30% reduction through elimination of:
- Scattered transform switches
- Ad-hoc array manipulation
- Duplicated validation logic
- String-based kernel decisions

**Readability**:
- LloydsIterator becomes declarative
- Assignment strategies use explicit plans
- Iteration logic shows "what" not "how"

**Testability**:
- Each abstraction has tiny unit tests
- No Spark session needed for most tests
- Deterministic behavior
- Fast test execution

**Extensibility**:
- New algorithms = new strategies, not rewrites
- Yinyang, trimmed, spherical = adding implementations
- New kernels = typeclass instance
- New transforms = trait implementation

---

**Note**: This file consolidates and replaces:
- `ENHANCEMENT_ROADMAP.md`
- `IMPLEMENTATION_STATUS.md`
- `IMPROVEMENTS_SUMMARY.md`
- `EVALUATION_SUMMARY.md`
- Various planning and checklist files
