# Roadmap: Generalized K-Means Clustering

> **Last Updated:** 2025-12-16
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

## Next Release Focus (P1)

Goal: land the highest-demand capabilities and supporting docs.

- ~~**Robust Bregman clustering + outlier handling** (3.11 / 5.8)~~ — **DONE**: `RobustKMeans` with trim/noise_cluster/m_estimator modes, outlier scoring, persistence.
- ~~**Sparse Bregman clustering** (3.12)~~ — **DONE**: `SparseKMeans` estimator with auto-sparsity detection, `KernelFactory` for unified kernel creation.
- **Multi-view clustering** (3.13 / 5.9) — implement `MultiViewKMeans` with shared `MultiViewAssignment`, per-view weights/divergences.
- **Docs & notebooks** (6.1) — quick-start notebook, divergence selection guide, X-Means auto-k demo, soft-clustering interpretation examples.

---

## Mid-Term (P2)

- **Co-clustering extensions** (3.14) — sparse input support, streaming/incremental updates, improved initialization, block-center refinement.
- **Time-series Bregman clustering** (3.15 / 5.10) — `TimeSeriesKMeans` with `DTWKernel`/shape-based kernels and sequence barycenter support.
- **Consensus/ensemble clustering** (3.16 / 5.11) — base clustering generator + co-association aggregation with target-k selection.
- **Federated/distributed aggregation** (3.17 / 5.12) — sufficient statistics exchange, optional differential privacy noise, secure aggregation hooks.
- **Spectral/graph-based clustering** (3.18 / 5.13) — affinity builders, Laplacian embeddings, Nyström approximation for large n.
- **Subspace clustering** (3.19 / 5.14) — cluster-specific projections or feature weights with alternating optimization.
- **Model type hierarchy & soft-assignment refactor** (5.7 / 5.3) — shared clustering model traits, common persistence hooks, reusable soft-assignment iterator.
- **Multi-objective convergence support** (5.6) — Pareto tracking and configurable objective combination for multi-criterion algorithms.

---

## Long-Term (P3)

- **Information-theoretic clustering** (5.15) — mutual-information estimators and `IBIterator` for information bottleneck objectives.
- **Future variants/backlog:** additional experimental divergences, alternative initialization schemes, and advanced evaluation metrics as demand warrants.

---

## Architecture Enablers (cross-cutting)

These frameworks unblock multiple roadmap items; prefer delivering them before dependent algorithms.

| Component | Priority | Enables | Notes |
|-----------|----------|---------|-------|
| ~~Outlier Detection (5.8)~~ | ~~P1~~ | ~~Robust Bregman clustering (3.11)~~ | **DONE**: Trim/noise-cluster strategies, scoring column |
| Multi-View (5.9) | P1 | Multi-view clustering (3.13) | View specs, weights, divergences |
| Sequence Kernels (5.10) | P2 | Time-series clustering (3.15) | DTW/shape kernels, barycenters |
| Consensus (5.11) | P2 | Ensemble clustering (3.16) | Base generator + co-association |
| Federated (5.12) | P2 | Federated Bregman clustering (3.17) | Secure aggregation, optional DP |
| Spectral (5.13) | P2 | Spectral/graph clustering (3.18) | Affinity builders, embeddings |
| Subspace (5.14) | P2 | Subspace clustering (3.19) | Projections/feature weights |
| Multi-Objective + IB (5.6/5.15) | P3 | Information bottleneck variants | Pareto tracking, IB iterator |

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

---

## Updating This Document

- Keep only **upcoming** work here. When an item is completed or shipped, move it to `CHANGELOG.md` and drop it from this roadmap.
- Add new issues/opportunities as discovered and adjust priorities based on user feedback.
- Record significant architectural decisions above to preserve context.
