# Generalized K-Means Clustering

[![CI](https://github.com/derrickburns/generalized-kmeans-clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/derrickburns/generalized-kmeans-clustering/actions/workflows/ci.yml)
[![CodeQL](https://github.com/derrickburns/generalized-kmeans-clustering/actions/workflows/codeql.yml/badge.svg)](https://github.com/derrickburns/generalized-kmeans-clustering/actions/workflows/codeql.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Scala 2.13](https://img.shields.io/badge/scala-2.13.14-red.svg)](https://www.scala-lang.org/)
[![Scala 2.12](https://img.shields.io/badge/scala-2.12.18-red.svg)](https://www.scala-lang.org/)
[![Spark 4.0](https://img.shields.io/badge/spark-4.0.1-orange.svg)](https://spark.apache.org/)
[![Spark 3.5](https://img.shields.io/badge/spark-3.5.1-orange.svg)](https://spark.apache.org/)

> **Security**: This project follows security best practices. See [SECURITY.md](SECURITY.md) for vulnerability reporting and [dependabot.yml](.github/dependabot.yml) for automated dependency updates.

**DataFrame-only API** — Version 0.7.0 removes the legacy RDD API entirely.
The library is now 100% DataFrame/Spark ML native with a clean, modern architecture.

This project generalizes K-Means to multiple Bregman divergences and advanced variants (Bisecting, X-Means, Soft/Fuzzy, Streaming, K-Medians, K-Medoids). It provides a pure DataFrame/ML API following Spark's Estimator/Model pattern.

## What's in here

- Multiple divergences: Squared Euclidean, KL, Itakura–Saito, L1/Manhattan (K-Medians), Generalized-I, Logistic-loss, Spherical/Cosine
- Variants: Bisecting, X-Means (BIC/AIC), Soft K-Means, Structured-Streaming K-Means, K-Medoids (PAM/CLARA)
- Scale: Tested on tens of millions of points in 700+ dimensions
- Tooling: Scala 2.13 (primary) / 2.12, Spark 4.0.x / 3.5.x / 3.4.x
  - **Spark 4.0.x**: Scala 2.13 only (Scala 2.12 support dropped in Spark 4.0)
  - **Spark 3.x**: Both Scala 2.13 and 2.12 supported

---

## Quick Start (DataFrame API)

Recommended for all new projects. The DataFrame API follows the Spark ML Estimator/Model pattern.

```scala
import org.apache.spark.ml.linalg.Vectors
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

val df = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.0, 0.0)),
  Tuple1(Vectors.dense(1.0, 1.0)),
  Tuple1(Vectors.dense(9.0, 8.0)),
  Tuple1(Vectors.dense(8.0, 9.0))
)).toDF("features")

val gkm = new GeneralizedKMeans()
  .setK(2)
  .setDivergence("kl")              // "squaredEuclidean", "itakuraSaito", "l1", "generalizedI", "logistic", "spherical"
  .setAssignmentStrategy("auto")    // "auto" | "crossJoin" (SE fast path) | "broadcastUDF" (general Bregman)
  .setMaxIter(20)

val model = gkm.fit(df)
val pred  = model.transform(df)
pred.show(false)
```

More recipes: see DataFrame API Examples.

---

## What CI Validates

Our comprehensive CI pipeline ensures quality across multiple dimensions:

| **Validation** | **What It Checks** | **Badge** |
|----------------|-------------------|-----------|
| **Lint & Style** | Scalastyle compliance, code formatting | Part of main CI |
| **Build Matrix** | Scala 2.12.18 & 2.13.14 × Spark 3.4.3 / 3.5.1 / 4.0.1 | [![CI](https://github.com/derrickburns/generalized-kmeans-clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/derrickburns/generalized-kmeans-clustering/actions/workflows/ci.yml) |
| **Test Matrix** | 576 tests across all Scala/Spark combinations<br/>• 62 kernel accuracy tests (divergence formulas, gradients, inverse gradients)<br/>• 19 Lloyd's iterator tests (core k-means loop)<br/>• Determinism, edge cases, numerical stability | Part of main CI |
| **Executable Documentation** | All examples run with assertions that verify correctness ([ExamplesSuite](src/test/scala/examples/ExamplesSuite.scala)):<br/>• [BisectingExample](src/main/scala/examples/BisectingExample.scala) - validates cluster count<br/>• [SoftKMeansExample](src/main/scala/examples/SoftKMeansExample.scala) - validates probability columns<br/>• [XMeansExample](src/main/scala/examples/XMeansExample.scala) - validates automatic k selection<br/>• [SphericalKMeansExample](src/main/scala/examples/SphericalKMeansExample.scala) - validates cosine similarity clustering<br/>• [PersistenceRoundTrip](src/main/scala/examples/PersistenceRoundTrip.scala) - validates save/load with center accuracy<br/>• [PersistenceRoundTripKMedoids](src/main/scala/examples/PersistenceRoundTripKMedoids.scala) - validates medoid preservation | Part of main CI |
| **Cross-version Persistence** | Models save/load across Scala 2.12↔2.13 and Spark 3.4↔3.5↔4.0 | Part of main CI |
| **Performance Sanity** | Basic performance regression check (30s budget) | Part of main CI |
| **Python Smoke Test** | PySpark wrapper with both SE and non-SE divergences | Part of main CI |
| **Security Scanning** | CodeQL static analysis for vulnerabilities | [![CodeQL](https://github.com/derrickburns/generalized-kmeans-clustering/actions/workflows/codeql.yml/badge.svg)](https://github.com/derrickburns/generalized-kmeans-clustering/actions/workflows/codeql.yml) |

**View live CI results:** [CI Workflow Runs](https://github.com/derrickburns/generalized-kmeans-clustering/actions/workflows/ci.yml)

---

## Feature Matrix

Truth-linked to code, tests, and examples for full transparency:

### Core Algorithms

| Algorithm | Code | Tests | Use Case |
|-----------|------|-------|----------|
| **GeneralizedKMeans** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/GeneralizedKMeans.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/ml/GeneralizedKMeansSuite.scala) | General clustering with 8 divergences |
| **Bisecting K-Means** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/BisectingKMeans.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/BisectingKMeansSuite.scala) | Hierarchical/divisive clustering |
| **X-Means** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/XMeans.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/XMeansSuite.scala) | Automatic k selection via BIC/AIC |
| **Soft K-Means** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/SoftKMeans.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/SoftKMeansSuite.scala) | Fuzzy/probabilistic memberships |
| **Streaming K-Means** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/StreamingKMeans.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/StreamingKMeansSuite.scala) | Real-time with exponential forgetting |
| **K-Medoids** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/KMedoids.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/KMedoidsSuite.scala) | Outlier-robust, custom distances |
| **Coreset K-Means** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/CoresetKMeans.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/ml/CoresetKMeansSuite.scala) | Large-scale approximation (10-100x speedup) |

### Production Features (NEW)

| Algorithm | Code | Tests | Use Case |
|-----------|------|-------|----------|
| **ConstrainedKMeans** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/ConstrainedKMeans.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/ml/ConstrainedKMeansSuite.scala) | Must-link / cannot-link constraints |
| **BalancedKMeans** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/BalancedKMeans.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/ml/BalancedKMeansSuite.scala) | Equal-sized clusters, capacity constraints |
| **RobustKMeans** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/RobustKMeans.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/ml/RobustKMeansSuite.scala) | Outlier detection (trim/noise cluster) |
| **SparseKMeans** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/SparseKMeans.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/ml/SparseKMeansSuite.scala) | High-dimensional sparse data |

### Advanced Variants (NEW)

| Algorithm | Code | Tests | Use Case |
|-----------|------|-------|----------|
| **MultiViewKMeans** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/MultiViewKMeans.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/ml/MultiViewKMeansSuite.scala) | Multiple feature representations |
| **TimeSeriesKMeans** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/TimeSeriesKMeans.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/ml/TimeSeriesKMeansSuite.scala) | DTW / Soft-DTW / GAK kernels |
| **SpectralClustering** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/SpectralClustering.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/ml/SpectralClusteringSuite.scala) | Graph Laplacian eigenvectors |
| **InformationBottleneck** | [Code](src/main/scala/com/massivedatascience/clusterer/ml/InformationBottleneck.scala) | [Tests](src/test/scala/com/massivedatascience/clusterer/ml/InformationBottleneckSuite.scala) | Information-theoretic compression |

**Divergences Available**: Squared Euclidean, KL, Itakura-Saito, L1/Manhattan, Generalized-I, Logistic Loss, Spherical/Cosine

All algorithms include:
- ✅ Model persistence (save/load across Spark 3.4↔3.5↔4.0, Scala 2.12↔2.13)
- ✅ Comprehensive test coverage (842 tests, 100% passing)
- ✅ Deterministic behavior (same seed → identical results)
- ✅ CI validation on every commit

---

## Installation / Versions

### Version Compatibility Matrix

| Spark Version | Scala 2.13 | Scala 2.12 | Notes |
|---------------|------------|------------|-------|
| **4.0.x** | ✅ | ❌ | Spark 4.0 dropped Scala 2.12 support |
| **3.5.x** | ✅ | ✅ | Recommended for production |
| **3.4.x** | ✅ | ✅ | LTS support |

**Java:** 17 (required for Spark 4.0, recommended for all)

### SBT (Scala projects)

```scala
// build.sbt
libraryDependencies += "com.massivedatascience" %% "massivedatascience-clusterer" % "0.7.0"
```

### Maven

```xml
<!-- Scala 2.13 -->
<dependency>
    <groupId>com.massivedatascience</groupId>
    <artifactId>massivedatascience-clusterer_2.13</artifactId>
    <version>0.7.0</version>
</dependency>

<!-- Scala 2.12 (Spark 3.x only) -->
<dependency>
    <groupId>com.massivedatascience</groupId>
    <artifactId>massivedatascience-clusterer_2.12</artifactId>
    <version>0.7.0</version>
</dependency>
```

### spark-submit / spark-shell

```bash
# For Spark 3.5 + Scala 2.13
spark-submit --packages com.massivedatascience:massivedatascience-clusterer_2.13:0.7.0 \
  your-app.jar

# For Spark 3.5 + Scala 2.12
spark-submit --packages com.massivedatascience:massivedatascience-clusterer_2.12:0.7.0 \
  your-app.jar
```

### Databricks

**Option 1: Cluster Library (recommended)**
1. Cluster → Libraries → Install New → Maven
2. Coordinates: `com.massivedatascience:massivedatascience-clusterer_2.12:0.7.0`
   - Use `_2.12` for DBR 13.x and earlier
   - Use `_2.13` for DBR 14.x+ (Spark 3.5+)

**Option 2: Notebook-scoped**
```python
%pip install --quiet pyspark  # if needed
```
```scala
// In Scala notebook
%scala
// Library should be attached to cluster
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
```

### EMR / AWS

Add to your EMR step or bootstrap action:
```bash
# spark-submit with packages
spark-submit \
  --packages com.massivedatascience:massivedatascience-clusterer_2.12:0.7.0 \
  --class com.example.YourApp \
  s3://your-bucket/your-app.jar
```

Or add to `spark-defaults.conf`:
```
spark.jars.packages com.massivedatascience:massivedatascience-clusterer_2.12:0.7.0
```

### Build from Source

```bash
git clone https://github.com/derrickburns/generalized-kmeans-clustering.git
cd generalized-kmeans-clustering

# Build for Scala 2.13 (default)
sbt ++2.13.14 package

# Build for Scala 2.12
sbt ++2.12.18 package

# Run tests
sbt test

# Cross-build for all Scala versions
sbt +package
```

The JAR will be in `target/scala-2.1x/massivedatascience-clusterer_2.1x-0.7.0.jar`

## What's New in 0.7.0

**Breaking Change: RDD API Removed**
- Legacy RDD API completely removed (53% code reduction)
- Library is now 100% DataFrame/Spark ML native
- Cleaner architecture with modular package structure

**Architecture Improvements**
- Split large files into focused modules:
  - `kernels/` subpackage: 8 Bregman kernel implementations
  - `strategies/impl/` subpackage: 5 assignment strategy implementations
- Added compiler warning flags for dead code detection
- Zero compiler warnings across Scala 2.12 and 2.13

**Maintained from 0.6.0**
- All DataFrame API algorithms: GeneralizedKMeans, Bisecting, X-Means, Soft, Streaming, K-Medoids, Coreset
- All divergences: Squared Euclidean, KL, Itakura-Saito, L1, Generalized-I, Logistic, Spherical
- Cross-version persistence (Spark 3.4↔3.5↔4.0, Scala 2.12↔2.13)
- PySpark wrapper + smoke test

---

## Scaling & Assignment Strategy (important)

Different divergences require different assignment mechanics at scale:
-	**Squared Euclidean (SE) fast path** — expression/codegen route:
	1.	Cross-join points with centers
	2.	Compute squared distance column
	3.	Prefer groupBy(rowId).min(distance) → join to pick argmin (scales better than window sorts)
	4.	Requires a stable rowId; we provide a RowIdProvider.
-	**General Bregman** — broadcast + UDF route:
	-	Broadcast the centers; compute argmin via a tight JVM UDF.
	-	Broadcast ceiling: you'll hit executor/memory limits if k × dim is too large to broadcast.

**Parameters**
-	`assignmentStrategy: StringParam = auto | crossJoin | broadcastUDF | chunked`
	-	`auto` (recommended): Chooses SE fast path when divergence == SE; otherwise selects between broadcastUDF and chunked based on k×dim size
	-	`crossJoin`: Forces SE expression-based path (only works with Squared Euclidean)
	-	`broadcastUDF`: Forces broadcast + UDF (works with any divergence, but may OOM on large k×dim)
	-	`chunked`: Processes centers in chunks to avoid OOM (multiple data scans, but safe for large k×dim)
-	`broadcastThreshold: IntParam` (elements, not bytes)
	-	Default: 200,000 elements (~1.5MB)
	-	Heuristic ceiling for k × dim. If exceeded for non-SE divergences, AutoAssignment switches to chunked broadcast.
-	`chunkSize: IntParam` (for chunked strategy)
	-	Default: 100 clusters per chunk
	-	Controls how many centers are processed in each scan when using chunked broadcast

**Broadcast Diagnostics**

The library provides detailed diagnostics to help you tune performance and avoid OOM errors:

```scala
// Example: Large cluster configuration
val gkm = new GeneralizedKMeans()
  .setK(500)          // 500 clusters
  .setDivergence("kl") // Non-SE divergence
  // If your data has dim=1000, then k×dim = 500,000 elements

// AutoAssignment will log:
// [WARN] AutoAssignment: Broadcast size exceeds threshold
//   Current: k=500 × dim=1000 = 500000 elements ≈ 3.8MB
//   Threshold: 200000 elements ≈ 1.5MB
//   Overage: +150%
//
//   Using ChunkedBroadcast (chunkSize=100) to avoid OOM.
//   This will scan the data 5 times.
//
//   To avoid chunking overhead, consider:
//     1. Reduce k (number of clusters)
//     2. Reduce dimensionality (current: 1000 dimensions)
//     3. Increase broadcastThreshold (suggested: k=500 would need ~500000 elements)
//     4. Use Squared Euclidean divergence if appropriate (enables fast SE path)
```

**When you see these warnings:**
-	**Chunked broadcast selected**: Your configuration will work but may be slower due to multiple data scans. Follow the suggestions to improve performance.
-	**Large broadcast warning** (>100MB): Risk of executor OOM errors. Consider reducing k or dimensionality, or increasing executor memory.
-	**No warning**: Your configuration is well-sized for broadcasting.

---

## Input Transforms & Interpretation

Some divergences (KL, IS) require positivity or benefit from stabilized domains.
-	inputTransform: StringParam = none | log1p | epsilonShift
-	shiftValue: DoubleParam (e.g., 1e-6) when epsilonShift is used.

Note: Cluster centers are learned in the transformed space. If you need original-space interpretation, apply the appropriate inverse (e.g., expm1) for reporting, understanding that this is an interpretive mapping, not a different optimum.

---

## Domain Requirements & Validation

**Automatic validation at fit time** — Different divergences have different input domain requirements. The library automatically validates your data and provides actionable error messages if violations are found:

| Divergence | Domain Requirement | Example Fix |
|------------|-------------------|-------------|
| **squaredEuclidean** | Any finite values (x ∈ ℝ) | None needed |
| **l1** / **manhattan** | Any finite values (x ∈ ℝ) | None needed |
| **spherical** / **cosine** | Non-zero vectors (‖x‖ > 0) | None needed (auto-normalized) |
| **kl** | Strictly positive (x > 0) | Use `log1p` or `epsilonShift` transform |
| **itakuraSaito** | Strictly positive (x > 0) | Use `log1p` or `epsilonShift` transform |
| **generalizedI** | Non-negative (x ≥ 0) | Take absolute values or shift data |
| **logistic** | Open interval (0 < x < 1) | Normalize to [0,1] then use `epsilonShift` |

**What happens on validation failure:**

When you call `fit()`, the library samples your data (first 1000 rows by default) and checks domain requirements. If violations are found, you'll see an **actionable error message** with:
- The specific invalid value and its location (feature index)
- Suggested fixes with example code
- Transform options to map your data into the valid domain

**Example validation error:**

```scala
// This will fail for KL divergence (contains zero)
val df = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(1.0, 0.0)),  // Zero at index 1!
  Tuple1(Vectors.dense(2.0, 3.0))
)).toDF("features")

val kmeans = new GeneralizedKMeans()
  .setK(2)
  .setDivergence("kl")

kmeans.fit(df)  // ❌ Throws with actionable message
```

**Error message you'll see:**

```
kl divergence requires strictly positive values, but found: 0.0

The kl divergence is only defined for positive data.

Suggested fixes:
  - Use .setInputTransform("log1p") to transform data using log(1 + x), which maps [0, ∞) → [0, ∞)
  - Use .setInputTransform("epsilonShift") with .setShiftValue(1e-6) to add a small constant
  - Pre-process your data to ensure all values are positive
  - Consider using Squared Euclidean divergence (.setDivergence("squaredEuclidean")) which has no domain restrictions

Example:
  new GeneralizedKMeans()
    .setDivergence("kl")
    .setInputTransform("log1p")  // Transform to valid domain
    .setMaxIter(20)
```

**How to fix domain violations:**

1. **For KL/Itakura-Saito (requires x > 0):**
   ```scala
   val kmeans = new GeneralizedKMeans()
     .setK(2)
     .setDivergence("kl")
     .setInputTransform("log1p")  // Maps [0, ∞) → [0, ∞) via log(1+x)
     .setMaxIter(20)
   ```

2. **For Logistic Loss (requires 0 < x < 1):**
   ```scala
   // First normalize your data to [0, 1], then:
   val kmeans = new GeneralizedKMeans()
     .setK(2)
     .setDivergence("logistic")
     .setInputTransform("epsilonShift")
     .setShiftValue(1e-6)  // Shifts to (ε, 1-ε)
     .setMaxIter(20)
   ```

3. **For Generalized-I (requires x ≥ 0):**
   ```scala
   // Pre-process to ensure non-negative values
   val df = originalDF.withColumn("features",
     udf((v: Vector) => Vectors.dense(v.toArray.map(math.abs)))
       .apply(col("features")))

   val kmeans = new GeneralizedKMeans()
     .setK(2)
     .setDivergence("generalizedI")
     .setMaxIter(20)
   ```

**Validation scope:**

- Validates first 1000 rows by default (configurable in code)
- Checks for NaN/Infinity in all divergences
- Provides early failure with clear guidance before expensive computation
- All DataFrame API estimators include validation: `GeneralizedKMeans`, `BisectingKMeans`, `XMeans`, `SoftKMeans`, `CoresetKMeans`

---

## Spherical K-Means (Cosine Similarity)

Spherical K-Means clusters data on the unit hypersphere using cosine similarity. This is ideal for:
- **Text/document clustering** (TF-IDF vectors, word embeddings)
- **Image feature clustering** (CNN embeddings)
- **Recommendation systems** (user/item embeddings)
- **Any high-dimensional sparse data** where direction matters more than magnitude

**How it works:**
1. All vectors are automatically L2-normalized to unit length
2. Distance: `D(x, μ) = 1 - cos(x, μ) = 1 - (x · μ)` for unit vectors
3. Centers are computed as normalized mean of assigned points

**Example:**

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

// Example: Clustering text embeddings
val embeddings = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.8, 0.6, 0.0)),   // Document about topic A
  Tuple1(Vectors.dense(0.9, 0.5, 0.1)),   // Also topic A (similar direction)
  Tuple1(Vectors.dense(0.1, 0.2, 0.95)),  // Document about topic B
  Tuple1(Vectors.dense(0.0, 0.3, 0.9))    // Also topic B
)).toDF("features")

val sphericalKMeans = new GeneralizedKMeans()
  .setK(2)
  .setDivergence("spherical")  // or "cosine"
  .setMaxIter(20)

val model = sphericalKMeans.fit(embeddings)
val predictions = model.transform(embeddings)
predictions.show()
```

**Key properties:**
- Distance range: `[0, 2]` (0 = identical direction, 2 = opposite direction)
- Equivalent to squared Euclidean on normalized data: `‖x - μ‖² = 2(1 - x·μ)`
- No domain restrictions except non-zero vectors
- Available in all estimators: `GeneralizedKMeans`, `BisectingKMeans`, `SoftKMeans`, `StreamingKMeans`

---

## Bisecting K-Means — efficiency note

The driver maintains a cluster_id column. For each split:
	1.	Filter only the target cluster: df.where(col("cluster_id") === id)
	2.	Run the base learner on that subset (k=2)
	3.	Join back predictions to update only the touched rows

This avoids reshuffling the full dataset at every split.

---

## Structured Streaming K-Means

Estimator/Model for micro-batch streams using the same core update logic.
-	initStrategy = pretrained | randomFirstBatch
-	pretrained: provide setInitialModel / setInitialCenters
-	randomFirstBatch: seed from the first micro-batch
-	State & snapshots: Each micro-batch writes centers to
${checkpointDir}/centers/latest.parquet for batch reuse.
-	StreamingGeneralizedKMeansModel.read(path) reconstructs a batch model from snapshots.

---

## Persistence (Spark ML)

Models implement DefaultParamsWritable/Readable.

**Layout**

```
<path>/
  ├─ metadata/params.json
  ├─ centers/*.parquet          # (center_id, vector[, weight])
  └─ summary/*.json             # events, metrics (optional)
```

**Compatibility**
-	Save/Load verified across Spark 3.4.x ↔ 3.5.x in CI.
-	New params default safely on older loads; unknown params are ignored.

---

## Python (PySpark) wrapper
-	Package exposes GeneralizedKMeans, BisectingGeneralizedKMeans, SoftGeneralizedKMeans, StreamingGeneralizedKMeans, KMedoids, etc.
-	CI runs a spark-submit smoke test on local[*] with a non-SE divergence.

---

## Production Features

### ConstrainedKMeans — Semi-supervised Clustering

Use when you have prior knowledge about which points should (or shouldn't) be in the same cluster.

```scala
import com.massivedatascience.clusterer.ml.ConstrainedKMeans

val df = spark.createDataFrame(Seq(
  (0L, Vectors.dense(0.0, 0.0)),
  (1L, Vectors.dense(0.1, 0.1)),
  (2L, Vectors.dense(5.0, 5.0)),
  (3L, Vectors.dense(5.1, 5.1))
)).toDF("id", "features")

// Define constraints
val mustLink = Seq((0L, 1L))      // Points 0 and 1 must be in the same cluster
val cannotLink = Seq((0L, 2L))    // Points 0 and 2 must be in different clusters

val ckm = new ConstrainedKMeans()
  .setK(2)
  .setMustLinkPairs(mustLink)
  .setCannotLinkPairs(cannotLink)
  .setConstraintMode("soft")      // "soft" (penalty) or "hard" (strict enforcement)
  .setPenaltyWeight(1.0)          // Weight for constraint violations (soft mode)
  .setMaxIter(20)

val model = ckm.fit(df)
model.transform(df).show()
```

**Parameters:**
- `mustLinkPairs`: Pairs of point IDs that should be in the same cluster
- `cannotLinkPairs`: Pairs of point IDs that should be in different clusters
- `constraintMode`: `"soft"` (penalize violations) or `"hard"` (strictly enforce)
- `penaltyWeight`: Weight for constraint violations in soft mode

### BalancedKMeans — Equal-sized Clusters

Use when you need clusters of similar size (e.g., load balancing, fair allocation).

```scala
import com.massivedatascience.clusterer.ml.BalancedKMeans

val df = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.0, 0.0)),
  Tuple1(Vectors.dense(0.1, 0.1)),
  Tuple1(Vectors.dense(5.0, 5.0)),
  Tuple1(Vectors.dense(5.1, 5.1)),
  Tuple1(Vectors.dense(10.0, 10.0)),
  Tuple1(Vectors.dense(10.1, 10.1))
)).toDF("features")

val bkm = new BalancedKMeans()
  .setK(3)
  .setBalanceMode("soft")         // "soft" (penalty) or "hard" (exact balance)
  .setBalancePenalty(1.0)         // Penalty for size imbalance (soft mode)
  .setMaxClusterSize(0)           // 0 = auto (n/k), or set explicit limit
  .setMaxIter(20)

val model = bkm.fit(df)
model.transform(df).show()

// Check cluster sizes
model.summary.clusterSizes.foreach { case (k, size) =>
  println(s"Cluster $k: $size points")
}
```

**Parameters:**
- `balanceMode`: `"soft"` (penalize imbalance) or `"hard"` (enforce exact sizes)
- `balancePenalty`: Weight for size imbalance penalty (soft mode)
- `maxClusterSize`: Maximum points per cluster (0 = automatic, n/k)

### RobustKMeans — Outlier Detection

Use when your data contains outliers that would skew cluster centers.

```scala
import com.massivedatascience.clusterer.ml.RobustKMeans

val rkm = new RobustKMeans()
  .setK(3)
  .setRobustMode("trim")          // "trim", "noise_cluster", or "m_estimator"
  .setTrimFraction(0.1)           // Ignore 10% most distant points (trim mode)
  .setMaxIter(20)

val model = rkm.fit(df)
val predictions = model.transform(df)

// Check outlier scores
predictions.select("features", "prediction", "outlierScore").show()
```

**Robust modes:**
- `"trim"`: Ignore the most distant points (controlled by `trimFraction`)
- `"noise_cluster"`: Assign outliers to a special noise cluster (-1)
- `"m_estimator"`: Use robust M-estimator for center computation

---

## Migration from RDD API (v0.6.x and earlier)

The RDD API was removed in v0.7.0. If migrating from an older version:

**Before (RDD API):**
```scala
import com.massivedatascience.clusterer.KMeans
import org.apache.spark.mllib.linalg.Vectors

val data = sc.parallelize(Array(
  Vectors.dense(0.0, 0.0),
  Vectors.dense(1.0, 1.0)
))
val model = KMeans.train(data, runs = 1, k = 2, maxIterations = 20)
```

**After (DataFrame API):**
```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
import org.apache.spark.ml.linalg.Vectors

val df = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.0, 0.0)),
  Tuple1(Vectors.dense(1.0, 1.0))
)).toDF("features")

val model = new GeneralizedKMeans()
  .setK(2)
  .setMaxIter(20)
  .fit(df)
```

Key differences:
- Use `org.apache.spark.ml.linalg.Vectors` (not `mllib.linalg`)
- Data is a DataFrame with a features column
- Use Estimator/Model pattern (`.fit()` / `.transform()`)



---

## Table of Contents
- Generalized K-Means Clustering
- Quick Start (DataFrame API)
- Feature Matrix
- Installation / Versions
- What's New in 0.7.0
- Scaling & Assignment Strategy
- Input Transforms & Interpretation
- Domain Requirements & Validation
- Spherical K-Means (Cosine Similarity)
- Bisecting K-Means — efficiency note
- Structured Streaming K-Means
- Persistence (Spark ML)
- Python (PySpark) wrapper
- Migration from RDD API

---

## Contributing
-	Please prefer PRs that target the DataFrame/ML path.
-	Add tests (including property-based where sensible) and update examples.
-	Follow Conventional Commits (feat:, fix:, docs:, refactor:, test:).

---

## License

Apache 2.0

---

## Notes for maintainers
- Keep the "Scaling & Assignment Strategy" section up-to-date when adding SE accelerations (Hamerly/Elkan/Yinyang) or ANN-assisted paths—mark SE-only and exact/approximate as appropriate.
- Update test counts in Feature Matrix when adding new test suites.
