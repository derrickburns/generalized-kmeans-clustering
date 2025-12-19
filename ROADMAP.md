# Roadmap: Generalized K-Means Clustering

> **Last Updated:** 2025-12-19
> **Status:** Forward-looking (completed work now lives in `CHANGELOG.md`)
> **Maintainer Note:** Keep this document for upcoming work only; ship-ready or finished items belong in the changelog.

This roadmap tracks upcoming improvements. It is organized by time horizon and priority so the next release is easy to see at a glance.

---

## Priority Legend

| Priority | Meaning | Typical Timeline |
|----------|---------|------------------|
| P0 | Critical / Blocking | Immediate |
| P1 | High value, low/medium effort | Next release |
| P2 | Medium value or effort | Following releases |
| P3 | Nice to have | Backlog |

---

## Adoption & Distribution (P0) — Highest ROI

> **Insight:** More algorithms won't drive adoption if users can't easily install or trust the library. These items reduce friction and increase real-world usage more than any algorithm tweak.

### 1. Frictionless Distribution

**Problem:** README shows SBT coordinates implying "just add a dependency," but Spark Packages listing says no releases/no Maven coordinates — users must build from source.

**Actions:**
- [ ] Publish to Maven Central (or Spark Packages) with clear `--packages` string
- [ ] Version matrix spelled out: Spark 3.4/3.5/4.0 × Scala 2.12/2.13
- [ ] "Works on Databricks" cookbook (cluster library install, init script)
- [ ] EMR / Kubernetes deployment examples
- [ ] Copy-paste install paths for `spark-submit`, notebooks, and build tools

### 2. Performance Accelerations (SE + large k)

**Problem:** Squared-Euclidean at scale is the #1 use case; users care about cost and wall-clock.

**Actions:**
- [x] **Fast exact:** Hamerly/Elkan/Yinyang pruning for Lloyd's iterations — **DONE**: `ElkanLloydsIterator` with cross-iteration bounds, `AcceleratedSEAssignment` with triangle inequality pruning (13 tests)
- [ ] **Fast approximate:** ANN-assisted assignment (LSH, KD-tree, ball tree)
- [ ] Benchmark suite with published numbers (iterations/sec, speedup vs. baseline)

### 3. PySpark Parity

**Problem:** Python wrapper exists but isn't first-class; PySpark users are the majority.

**Actions:**
- [ ] pip-installable wheel with pinned Spark/Scala compatibility
- [x] Type hints and docstrings for IDE support — **DONE**: Full type hints in `kmeans.py`, comprehensive Google-style docstrings
- [x] Native-feeling PySpark examples (not just Scala translations) — **DONE**: 5 example files (basic, KL divergence, weighted, finding optimal k, persistence)
- [x] Model save/load and full param support matching Scala API — **DONE**: Full persistence and TrainingSummary support
- [ ] CI smoke tests for Python API

### 4. Model Selection & Diagnostics

**Problem:** Users ask "Is this clustering any good?" after fitting.

**Actions:**
- [x] Scalable silhouette score (Spark-native) — **DONE**: `ClusteringMetrics` with exact and approximate silhouette (12 tests)
- [x] Elbow method helper (cost vs. k curve) — **DONE**: `ClusteringMetrics.elbowCurve()` method
- [ ] Stability/bootstrap metrics
- [x] Iteration history: objective per iter, convergence reason, cluster sizes — **DONE**: `TrainingSummary` with distortionHistory, movementHistory, convergenceReport (8 tests)
- [x] First-class `ModelSummary` with JSON persistence — **DONE**: `TrainingSummary.toDF()` for DataFrame/JSON export

### 5. Production Features (Surface Existing Work)

**Problem:** Features like constraints and balanced clustering exist in master but aren't in the documented/released API.

**Actions:**
- [ ] Sample weights support across estimators
- [ ] Document `ConstrainedKMeans` (must-link / cannot-link) in README/guides
- [ ] Document `BalancedKMeans` (size constraints) in README/guides
- [ ] Add these to the quick-start and divergence selection guides

---

## Next Release Focus (P1)

Goal: land the highest-demand capabilities and supporting docs.

- ~~**Robust Bregman clustering + outlier handling** (3.11 / 5.8)~~ — **DONE**: `RobustKMeans` with trim/noise_cluster/m_estimator modes, outlier scoring, persistence.
- ~~**Sparse Bregman clustering** (3.12)~~ — **DONE**: `SparseKMeans` estimator with auto-sparsity detection, `KernelFactory` for unified kernel creation.
- ~~**Multi-view clustering** (3.13 / 5.9)~~ — **DONE**: `MultiViewKMeans` estimator with per-view weights/divergences, combine strategies (weighted/max/min), `ViewSpec` configuration.
- ~~**Docs & notebooks** (6.1)~~ — **DONE**: Quick-start guide, divergence selection guide, X-Means auto-k demo, soft-clustering interpretation examples in `docs/guides/`.

---

## Mid-Term (P2)

- ~~**Time-series Bregman clustering** (3.15 / 5.10)~~ — **DONE**: `TimeSeriesKMeans` with DTW, Soft-DTW, GAK, and Derivative-DTW kernels; DBA barycenter computation; full persistence support.
- ~~**Spectral/graph-based clustering** (3.18 / 5.13)~~ — **DONE**: `SpectralClustering` with affinity builders (full, k-NN, ε-neighborhood), Laplacian types (unnormalized, symmetric normalized, random walk), Mercer kernels (RBF, Laplacian, polynomial, linear), Nyström approximation for O(nm²) scalability; 25 tests.
- **Co-clustering extensions** (3.14) — sparse input support, streaming/incremental updates, improved initialization, block-center refinement.
- **Consensus/ensemble clustering** (3.16 / 5.11) — base clustering generator + co-association aggregation with target-k selection.
- **Federated/distributed aggregation** (3.17 / 5.12) — sufficient statistics exchange, optional differential privacy noise, secure aggregation hooks.
- **Subspace clustering** (3.19 / 5.14) — cluster-specific projections or feature weights with alternating optimization.
- **Model type hierarchy & soft-assignment refactor** (5.7 / 5.3) — shared clustering model traits, common persistence hooks, reusable soft-assignment iterator.
- **Multi-objective convergence support** (5.6) — Pareto tracking and configurable objective combination for multi-criterion algorithms.

---

## Long-Term (P3)

- ~~**Information-theoretic clustering** (5.15)~~ — **DONE**: `InformationBottleneck` estimator with Blahut-Arimoto algorithm, `MutualInformation` utilities, β trade-off parameter, discrete/continuous relevance support, 28 tests.
- **Future variants/backlog:** additional experimental divergences, alternative initialization schemes, and advanced evaluation metrics as demand warrants.

---

## Architecture Enablers (cross-cutting)

These frameworks unblock multiple roadmap items; prefer delivering them before dependent algorithms.

| Component | Priority | Enables | Notes |
|-----------|----------|---------|-------|
| ~~Outlier Detection (5.8)~~ | ~~P1~~ | ~~Robust Bregman clustering (3.11)~~ | **DONE**: Trim/noise-cluster strategies, scoring column |
| ~~Multi-View (5.9)~~ | ~~P1~~ | ~~Multi-view clustering (3.13)~~ | **DONE**: ViewSpec, per-view weights/divergences, combine strategies |
| ~~Sequence Kernels (5.10)~~ | ~~P2~~ | ~~Time-series clustering (3.15)~~ | **DONE**: DTW, Soft-DTW, GAK, Derivative-DTW kernels; DBA barycenters |
| Consensus (5.11) | P2 | Ensemble clustering (3.16) | Base generator + co-association |
| Federated (5.12) | P2 | Federated Bregman clustering (3.17) | Secure aggregation, optional DP |
| ~~Spectral (5.13)~~ | ~~P2~~ | ~~Spectral/graph clustering (3.18)~~ | **DONE**: Affinity builders, Laplacians, Nyström, 25 tests |
| Subspace (5.14) | P2 | Subspace clustering (3.19) | Projections/feature weights |
| ~~IB (5.15)~~ | ~~P3~~ | ~~Information bottleneck clustering~~ | **DONE**: Blahut-Arimoto, MI utilities, 28 tests |
| Multi-Objective (5.6) | P3 | Multi-criterion algorithms | Pareto tracking, objective combination |

---

## Testing & Operational Readiness

- Add deterministic seeds and small-partition fixtures to new suites; align with Spark local[2] defaults.
- Extend benchmark coverage to new kernels/strategies; keep JSON outputs in `target/perf-reports/`.
- Ensure persistence round-trips for any new model or param (Spark 3.4 ↔ 3.5 ↔ 4.0, Scala 2.12 ↔ 2.13).
- Keep Spark UI disabled in tests to avoid flakiness.

---

## Decision Log (context for upcoming work)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2024 | DataFrame API remains primary surface; RDD API retired | Reduce maintenance, align with Spark ML patterns |
| 2024 | Keep L1 listed alongside Bregman divergences | Practical utility outweighs theoretical purity |
| 2025-12-15 | Prioritize robust/sparse/multi-view work next | Highest user demand and unlocks downstream variants |
| 2025-12-15 | Maintain kernels in a single module (`BregmanKernel.scala`) | Consistency and discoverability |
| 2025-12-15 | Use phased delivery for accelerations and new iterators | Keep CI stable while iterating |
| 2025-12-16 | Created `KernelFactory` for unified kernel creation | Single API for dense/sparse kernels, reduces duplication |
| 2025-12-16 | Moved assignment strategies to `impl/` subpackage | Better organization, backward-compatible via type aliases |
| 2025-12-16 | Implemented `MultiViewKMeans` with ViewSpec configuration | Per-view divergences/weights, weighted/max/min combine strategies |
| 2025-12-16 | Created initial documentation guides | Quick-start, divergence selection, X-Means auto-k, soft clustering (later restructured to Diátaxis) |
| 2025-12-16 | Implemented `TimeSeriesKMeans` with sequence kernels | DTW, Soft-DTW, GAK, Derivative-DTW; DBA barycenters; 31 tests |
| 2025-12-16 | Implemented `InformationBottleneck` estimator | Blahut-Arimoto algorithm, MutualInformation utilities, 28 tests |
| 2025-12-16 | Implemented `SpectralClustering` estimator | Graph Laplacian eigenvectors (Ng-Jordan-Weiss), Nyström approximation, 25 tests |
| 2025-12-19 | Restructured documentation to Diátaxis framework | Tutorials, how-to, reference, explanation categories; Jekyll deployment; custom domain |
| 2025-12-19 | Created `SPECIFICATION.md` | Compressed reconstruction spec for AI code generation |
| 2025-12-19 | Consolidated obsolete files | Removed redundant guides, release notes; single source of truth in CHANGELOG.md |

---

## Updating This Document

- Keep only **upcoming** work here. When an item is completed or shipped, move it to `CHANGELOG.md` and drop it from this roadmap.
- Add new issues/opportunities as discovered and adjust priorities based on user feedback.
- Record significant architectural decisions above to preserve context.
