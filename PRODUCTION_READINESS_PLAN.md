# Production Readiness Plan

This document tracks the critical gaps that block "production-quality" and "maximum educational value" for the generalized-kmeans-clustering library.

## Status Legend
- ✅ **Complete** - Implemented and tested
- 🚧 **In Progress** - Work started
- ⏳ **Planned** - Prioritized, not started
- 📋 **Backlog** - Lower priority

---

## Top Blockers (Fix First)

### 1. Persistence Contract & Cross-Version Compatibility ✅→🚧

**Status:** Core infrastructure ✅ Complete, Extensions 🚧 In Progress

**What's Done:**
- ✅ PersistenceLayoutV1 with versioned schema
- ✅ Deterministic center ordering via center_id (0..k-1)
- ✅ SHA-256 checksums for integrity
- ✅ GeneralizedKMeansModel full persistence
- ✅ PERSISTENCE_COMPATIBILITY.md documentation

**What Remains:**
- ⏳ Extend to all models (Bisecting, XMeans, Soft, Streaming, KMedoids)
- ⏳ Cross-version CI job (Spark 3.4↔3.5, Scala 2.12↔2.13)
- ⏳ PersistenceRoundTrip executable examples

**Action Items:**
```scala
// For each model:
1. Implement MLWritable with PersistenceLayoutV1 helpers
2. Implement MLReadable with version validation
3. Add model-specific metadata fields
4. Create roundtrip test
5. Create runMain example
```

**Estimated:** 2-3 hours per model × 5 models = 10-15 hours

---

### 2. Assignment Scalability for Non-SE Divergences 🚧

**Status:** Partially addressed, needs chunking

**Gap:** Broadcast + UDF fails when k × dim is large (e.g., k=1000, dim=10000)

**Current Implementation:**
- ✅ broadcastThreshold parameter exists (element count)
- ✅ Auto-selection between crossJoin (SE) and broadcastUDF (non-SE)
- ❌ No chunked fallback for large k × dim

**Fix Plan:**

#### 2a. Implement Chunked-Centers Evaluator ⏳
```scala
// src/main/scala/com/massivedatascience/clusterer/ml/df/ChunkedAssignment.scala
object ChunkedAssignment {
  def assignToNearest(
    df: DataFrame,
    centers: Array[Array[Double]],
    kernel: BregmanKernel,
    chunkSize: Int = 100
  ): DataFrame = {
    // Split centers into chunks
    val chunks = centers.grouped(chunkSize).zipWithIndex.toSeq

    // For each chunk, compute min distance + argmin
    val chunkResults = chunks.map { case (chunk, chunkIdx) =>
      // Broadcast small chunk
      // Compute distances for this chunk only
      // Return (point_id, local_min_dist, local_argmin)
    }

    // Reduce: find global min across chunks
    // Join back to get final assignment
  }
}
```

#### 2b. Add Auto-Guardrails ⏳
```scala
// In GeneralizedKMeansParams
val broadcastThresholdElems = new IntParam(
  this, "broadcastThresholdElems",
  "Max k×dim for broadcast (element count, not bytes)",
  ParamValidators.gt(0)
)
setDefault(broadcastThresholdElems -> 200000) // ~1.5MB for doubles

// In fit() logic:
val kTimesDim = k * dim
if (divergence != "squaredEuclidean" && kTimesDim > $(broadcastThresholdElems)) {
  logWarning(s"k×dim=$kTimesDim exceeds threshold, using chunked assignment")
  strategy = "chunked"
}
```

#### 2c. Document Rules of Thumb ⏳

Add to README:
```markdown
### Scalability: k × dim Feasibility

| Divergence | Strategy | Max k×dim | Notes |
|------------|----------|-----------|-------|
| Squared Euclidean | crossJoin | ~1M | Expression-based, no broadcast |
| KL, IS, L1, etc. | broadcastUDF | ~200K | Broadcast limited by executor memory |
| KL, IS, L1, etc. | chunked | ~10M | Multiple scans, no broadcast |

**Automatic selection:**
- SE: Always uses crossJoin (fast path)
- Non-SE with k×dim ≤ 200K: broadcastUDF
- Non-SE with k×dim > 200K: chunked (auto-selected, logged)

**Manual override:**
```scala
gkm.setAssignmentStrategy("chunked")  // Force chunking
```
```

**Estimated:** 8-10 hours

---

### 3. Determinism, Numeric Hygiene, Domain Validation 🚧

**Status:** Partially addressed, needs property tests and epsilon persistence

#### 3a. Fix shiftValue/Epsilon Persistence 🚧 IN PROGRESS

**Current Gap:** `shiftValue` is a param but not prominently documented or validated

**Fix:**
```scala
// Already in GeneralizedKMeansParams:
val smoothing = new DoubleParam(...) // This is epsilon for transforms

// Need to add to docs:
/**
  * Epsilon shift for domain constraints.
  *
  * - KL divergence: Requires P > 0, Q > 0. Use smoothing=1e-10.
  * - Itakura-Saito: Requires P > 0, Q > 0. Use smoothing=1e-10.
  * - Logistic loss: Requires P ∈ (0,1). Use smoothing=1e-10.
  *
  * This value is persisted with the model.
  */
```

**Action:** ✅ Already persisted! Just needs better docs.

#### 3b. Add Property Tests for Determinism ⏳

```scala
// Add to PropertyBasedTestSuite
test("Property: same seed produces identical centers") {
  forAll(dimGen, kGen, numPointsGen) { (dim, k, n) =>
    val data = generateRandomData(n, dim, seed = 42)

    val model1 = new GeneralizedKMeans()
      .setK(k).setDivergence("squaredEuclidean")
      .setSeed(1234).fit(data)

    val model2 = new GeneralizedKMeans()
      .setK(k).setDivergence("squaredEuclidean")
      .setSeed(1234).fit(data)

    // Centers should be identical (or within epsilon for floating point)
    model1.clusterCenters.zip(model2.clusterCenters).foreach { case (c1, c2) =>
      c1.zip(c2).foreach { case (x1, x2) =>
        math.abs(x1 - x2) should be < 1e-10
      }
    }
  }
}

// Repeat for each estimator: Bisecting, XMeans, Soft, Streaming
```

**Estimated:** 2-3 hours

#### 3c. Add NaN/Inf Guards ⏳

```scala
// src/main/scala/com/massivedatascience/clusterer/ml/df/NumericGuards.scala
object NumericGuards {
  def checkFinite(v: Vector, context: String): Unit = {
    if (v.toArray.exists(x => x.isNaN || x.isInfinite)) {
      throw new GKMNumericException(
        s"$context: Vector contains NaN or Inf: ${v.toArray.take(10).mkString(",")}"
      )
    }
  }

  def checkPositive(v: Vector, context: String, epsilon: Double): Unit = {
    if (v.toArray.exists(_ < -epsilon)) {
      throw new GKMDomainException(
        s"$context: Vector contains negative values (KL/IS require positivity)"
      )
    }
  }
}

// Use in update logic:
def updateCenters(points: RDD[BregmanPoint], assignments: RDD[(Int, BregmanPoint)]): Array[BregmanCenter] = {
  val newCenters = // ... compute ...

  newCenters.foreach { c =>
    NumericGuards.checkFinite(c.vector, "Center update")
  }

  newCenters
}
```

**Estimated:** 4-6 hours

---

### 4. Executable Documentation (Examples as Tests) ⏳

**Gap:** Examples drift; not continuously verified

**Fix Plan:**

#### 4a. Create Runnable Examples
```scala
// src/main/scala/examples/GeneralizedKMeansExample.scala
object GeneralizedKMeansExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("GKM Example").getOrCreate()
    import spark.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(9.0, 9.0)),
      Tuple1(Vectors.dense(10.0, 10.0))
    ).toDF("features")

    val gkm = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setSeed(42)

    val model = gkm.fit(data)
    val predictions = model.transform(data)

    // Assertions
    assert(predictions.count() == 4, "Should have 4 predictions")
    assert(model.numClusters == 2, "Should have 2 clusters")

    println("✓ GeneralizedKMeans example passed")
  }
}
```

#### 4b. Add Examples CI Job
```yaml
# .github/workflows/ci.yml
examples-run:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-java@v4
    - uses: sbt/setup-sbt@v1
    - name: Run all examples
      run: |
        sbt "runMain examples.GeneralizedKMeansExample"
        sbt "runMain examples.BisectingExample"
        sbt "runMain examples.XMeansExample"
        sbt "runMain examples.SoftKMeansExample"
        sbt "runMain examples.StreamingExample"
        sbt "runMain examples.KMedoidsExample"
```

**Estimated:** 1 hour per example × 6 = 6 hours

---

### 5. Uniform model.summary and Telemetry ⏳

**Gap:** Iteration metrics not uniformly available

**Fix Plan:**

#### 5a. Standardize TrainingSummary
```scala
// src/main/scala/com/massivedatascience/clusterer/ml/TrainingSummary.scala
case class TrainingSummary(
  algorithm: String,
  k: Int,
  dim: Int,
  numPoints: Long,
  iterations: Int,
  converged: Boolean,

  // Per-iteration metrics
  distortionHistory: Array[Double],
  movementHistory: Array[Double],
  pointsMovedHistory: Array[Int],
  reseedEvents: Seq[ReseedEvent],

  // Strategy & performance
  assignmentStrategy: String,  // "crossJoin" | "broadcastUDF" | "chunked"
  elapsedMillis: Long,
  iterationTimings: Array[Long],

  // Quality metrics
  finalDistortion: Double,
  effectiveK: Int,  // Actual non-empty clusters

  // Metadata
  trainedAt: java.time.Instant
)

case class ReseedEvent(iteration: Int, emptyClusterIds: Seq[Int], strategy: String)
```

#### 5b. Add to Every Model
```scala
class GeneralizedKMeansModel(...) {
  private[ml] var trainingSummary: Option[TrainingSummary] = None

  def summary: TrainingSummary = trainingSummary.getOrElse(
    throw new NoSuchElementException("summary not available (model was loaded, not trained)")
  )

  def hasSummary: Boolean = trainingSummary.isDefined
}
```

#### 5c. Persist Summary Snapshot
```scala
// In PersistenceLayoutV1
def writeSummary(path: String, summary: TrainingSummary): Unit = {
  val json = Serialization.write(Map(
    "iterations" -> summary.iterations,
    "converged" -> summary.converged,
    "distortionHistory" -> summary.distortionHistory,
    "assignmentStrategy" -> summary.assignmentStrategy,
    "elapsedMillis" -> summary.elapsedMillis
  ))
  writeJsonFile(s"$path/summary.json", json)
}
```

**Estimated:** 6-8 hours

---

### 6. Python UX & Packaging ⏳

**Gap:** PySpark wrapper exists but no pip package

**Fix Plan:**

#### 6a. Create PyPI Package Structure
```
python/
  gkm_clustering/
    __init__.py
    generalized_kmeans.py
    bisecting.py
    xmeans.py
    version.py
  setup.py
  README.md
  requirements.txt
```

#### 6b. setup.py
```python
from setuptools import setup, find_packages

setup(
    name="gkm-clustering",
    version="0.6.0",
    packages=find_packages(),
    install_requires=[
        "pyspark>=3.4.0,<3.6.0",
    ],
    python_requires=">=3.8",
    author="Massive Data Science, LLC",
    description="Generalized K-Means clustering with Bregman divergences",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/derrickburns/generalized-kmeans-clustering",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
```

#### 6c. Publish Workflow
```yaml
# .github/workflows/publish-python.yml
name: Publish Python Package
on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - name: Build and publish
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: |
          cd python
          python setup.py sdist bdist_wheel
          twine upload dist/*
```

**Estimated:** 6-8 hours

---

## High-Value Gaps (Next)

### 7. Performance Truth & Regressions ⏳

**Current:** perf-sanity job exists, logs time

**Fix:** Add structured output
```yaml
- name: Performance sanity check
  run: |
    TIME=$(sbt "test:runMain PerfSanityCheck" | grep "perf_sanity_seconds" | awk '{print $2}')
    echo "perf_sanity_seconds=$TIME" >> $GITHUB_STEP_SUMMARY
    if [ $(echo "$TIME > 60" | bc) -eq 1 ]; then
      echo "::error::Performance regression: ${TIME}s exceeds 60s budget"
      exit 1
    fi
```

**Estimated:** 2-3 hours

### 8. Security & Supply-Chain Hygiene 🚧

**Current:**
- ✅ CodeQL workflow exists
- ✅ GitHub Actions pinned by SHA
- ⏳ Dependabot needed
- ⏳ SBOM needed
- ⏳ SECURITY.md needed

**Fix:**
```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "sbt"
    directory: "/"
    schedule:
      interval: "weekly"
```

```markdown
# SECURITY.md
## Reporting Security Issues

Please report security vulnerabilities to security@massivedatascience.com

Do not open public GitHub issues for security vulnerabilities.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.6.x   | ✅        |
| < 0.6   | ❌        |
```

**Estimated:** 2 hours

### 9. API Clarity & Parameter Semantics ⏳

**Fix:** Add sealed traits internally
```scala
sealed trait Divergence
case object SquaredEuclidean extends Divergence
case object KL extends Divergence
case object ItakuraSaito extends Divergence

object Divergence {
  def fromString(s: String): Divergence = s.toLowerCase match {
    case "squaredeuclidean" | "se" => SquaredEuclidean
    case "kl" | "kullbackleibler" => KL
    case "itakurasaito" | "is" => ItakuraSaito
    case _ => throw new IllegalArgumentException(s"Unknown divergence: $s")
  }
}
```

**Estimated:** 4-6 hours

### 10-12. Streaming, Bisecting, Educational Docs ⏳

See detailed plans in sections above.

---

## Acceptance Criteria (Release Gate)

Before marking v0.6.0 production-ready:

### Must Have ✅/🚧
- [✅] Persistence: Cross-version contract defined
- [🚧] Persistence: All models implement save/load
- [⏳] Persistence: CI validates cross-version roundtrips
- [⏳] Scalability: Chunked assignment for large k×dim
- [⏳] Determinism: Property tests pass
- [⏳] Docs: Examples run in CI
- [⏳] Docs: Feature matrix links to code/tests/examples
- [🚧] Summaries: Available on all models

### Should Have ⏳
- [⏳] Python: PyPI package published
- [⏳] Performance: Regression detection in CI
- [🚧] Security: CodeQL + Dependabot + SECURITY.md
- [⏳] Educational: Divergences 101 doc

### Nice to Have 📋
- [📋] Streaming: Snapshot/export for batch
- [📋] Bisecting: Cluster-id filtering optimization
- [📋] Notebooks: Interactive visualizations

---

## Timeline Estimate

| Priority | Item | Estimated Hours | Status |
|----------|------|----------------|--------|
| P0 | Persistence (remaining models) | 10-15 | ⏳ |
| P0 | Chunked assignment + guardrails | 8-10 | ⏳ |
| P0 | Determinism property tests | 2-3 | ⏳ |
| P0 | NaN/Inf guards | 4-6 | ⏳ |
| P0 | Executable examples | 6 | ⏳ |
| P1 | model.summary implementation | 6-8 | ⏳ |
| P1 | Python PyPI package | 6-8 | ⏳ |
| P1 | Security hardening | 2 | 🚧 |
| P2 | Performance regression detection | 2-3 | ⏳ |
| P2 | Educational docs | 4-6 | ⏳ |

**Total Estimated:** 50-75 hours for production-ready v0.6.0

---

## Progress Tracking

Last Updated: 2025-10-18

- ✅ PersistenceLayoutV1 implemented (commits 9a8334f, c08d0c1)
- ✅ PERSISTENCE_COMPATIBILITY.md documented
- 🚧 Working on: shiftValue/epsilon documentation
- ⏳ Next up: Property tests for determinism
