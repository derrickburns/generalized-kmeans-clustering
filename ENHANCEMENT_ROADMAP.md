# Enhancement Roadmap

This document outlines the implementation plan for adding missing K-means variants and improving the library based on the comprehensive evaluation conducted on October 14, 2025.

## Current Status

- **Maintainability Score**: 8.5/10
- **Extensibility Score**: 9.0/10
- **Test Coverage**: 205 tests (99.5% passing)
- **Documentation**: Excellent (25+ markdown files, ARCHITECTURE.md)

## Priority Matrix

| Feature | Priority | Effort | Impact | Timeline |
|---------|----------|--------|--------|----------|
| K-Medians | HIGH | 2-3 weeks | High (robustness) | Q1 2026 |
| K-Medoids | HIGH | 3-4 weeks | High (industry standard) | Q1 2026 |
| Elkan's Acceleration | HIGH | 3-4 weeks | High (3-5x speedup) | Q1 2026 |
| DataFrame API - Advanced Variants | HIGH | 4-6 weeks | High (API consistency) | Q1-Q2 2026 |
| Performance Benchmarking | MEDIUM | 2-3 weeks | Medium (validation) | Q2 2026 |
| Scaladoc API Reference | MEDIUM | 1 week | Medium (discoverability) | Q2 2026 |
| Yinyang K-Means | LOW | 4-6 weeks | Medium (large k) | Q3 2026 |
| GPU Acceleration | LOW | 8-12 weeks | High (specific use cases) | Q4 2026 |

---

## Phase 1: K-Medians Implementation (2-3 weeks)

### Overview
K-Medians uses the L1 norm (Manhattan distance) and computes cluster centers as the component-wise median instead of mean. This provides robustness to outliers.

### Implementation Strategy

**1. Create MedianPointOps Trait** (New file: `MedianPointOps.scala`)

```scala
package com.massivedatascience.clusterer

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/**
 * Point operations for K-Medians clustering.
 *
 * Unlike Bregman divergences which use gradient-based means,
 * K-Medians computes component-wise medians for robustness.
 */
trait MedianPointOps extends Serializable with ClusterFactory {
  type P = MedianPoint
  type C = MedianCenter

  /**
   * Distance function: L1 norm (Manhattan distance)
   */
  def distance(p: Vector, c: Vector): Double = {
    val pArr = p.toArray
    val cArr = c.toArray
    var sum = 0.0
    var i = 0
    while (i < pArr.length) {
      sum += math.abs(pArr(i) - cArr(i))
      i += 1
    }
    sum
  }

  /**
   * Compute component-wise weighted median.
   *
   * @param points RDD of points assigned to this cluster
   * @return median center
   */
  def computeMedian(points: RDD[WeightedVector]): Vector = {
    val numFeatures = points.first().homogeneous.size

    // Compute median for each dimension independently
    val medians = (0 until numFeatures).map { dim =>
      val values = points.map { p =>
        (p.homogeneous(dim), p.weight)
      }.collect()

      weightedMedian(values)
    }

    Vectors.dense(medians.toArray)
  }

  /**
   * Compute weighted median of a 1D array.
   */
  private def weightedMedian(values: Array[(Double, Double)]): Double = {
    if (values.isEmpty) return 0.0

    // Sort by value
    val sorted = values.sortBy(_._1)
    val totalWeight = sorted.map(_._2).sum
    val halfWeight = totalWeight / 2.0

    // Find median
    var cumWeight = 0.0
    var i = 0
    while (i < sorted.length && cumWeight < halfWeight) {
      cumWeight += sorted(i)._2
      i += 1
    }

    sorted(math.max(0, i - 1))._1
  }
}

case class MedianPoint(homogeneous: Vector, weight: Double) extends WeightedVector {
  lazy val inhomogeneous = asInhomogeneous(homogeneous, weight)
}

case class MedianCenter(homogeneous: Vector, weight: Double) extends WeightedVector {
  lazy val inhomogeneous = asInhomogeneous(homogeneous, weight)
}
```

**2. Create KMedians Clusterer** (New file: `KMedians.scala`)

```scala
package com.massivedatascience.clusterer

import org.apache.spark.rdd.RDD

/**
 * K-Medians clustering using L1 distance and component-wise medians.
 *
 * More robust to outliers than K-Means (which uses L2 distance).
 * Especially useful for skewed distributions.
 */
class KMedians(
    k: Int,
    maxIterations: Int = 20,
    tolerance: Double = 1e-4)
  extends MultiKMeansClusterer with MedianPointOps {

  override def cluster(
      data: RDD[WeightedVector],
      centers: Array[WeightedVector]): (Double, Array[WeightedVector]) = {

    var currentCenters = centers.map(toCenter)
    var iteration = 0
    var converged = false
    var totalCost = 0.0

    while (iteration < maxIterations && !converged) {
      // Assignment step
      val assignments = data.map { point =>
        val p = toPoint(point)
        val closestIdx = findClosestCenter(p, currentCenters)
        (closestIdx, p)
      }

      // Update step: compute medians
      val newCenters = assignments
        .groupByKey()
        .map { case (clusterId, points) =>
          val pointsRDD = data.sparkContext.parallelize(points.toSeq)
          (clusterId, computeMedian(pointsRDD))
        }
        .collect()
        .sortBy(_._1)
        .map { case (_, median) => toCenter(WeightedVector(median, 1.0)) }

      // Check convergence
      val movements = currentCenters.zip(newCenters).map { case (old, new_) =>
        distance(old.homogeneous, new_.homogeneous)
      }

      converged = movements.max < tolerance
      currentCenters = newCenters
      iteration += 1

      // Compute cost
      totalCost = assignments.map { case (clusterId, point) =>
        distance(point.homogeneous, currentCenters(clusterId).homogeneous)
      }.sum()
    }

    (totalCost, currentCenters.map(c => WeightedVector(c.homogeneous, c.weight)))
  }

  private def findClosestCenter(point: MedianPoint, centers: Array[MedianCenter]): Int = {
    var minDist = Double.PositiveInfinity
    var minIdx = 0
    var i = 0
    while (i < centers.length) {
      val dist = distance(point.homogeneous, centers(i).homogeneous)
      if (dist < minDist) {
        minDist = dist
        minIdx = i
      }
      i += 1
    }
    minIdx
  }
}
```

**3. DataFrame API Integration** (Modify: `Strategies.scala`)

Add new `MedianUpdateStrategy`:

```scala
/**
 * Median update strategy for K-Medians clustering.
 *
 * Computes component-wise medians instead of means.
 */
class MedianUDAFUpdate extends UpdateStrategy with Logging {

  override def update(
      assigned: DataFrame,
      featuresCol: String,
      weightCol: Option[String],
      k: Int,
      kernel: BregmanKernel): Array[Array[Double]] = {

    require(kernel.name == "l1", "MedianUpdate requires L1 kernel")

    val numFeatures = assigned.select(featuresCol).first().getAs[Vector](0).size

    val grouped = assigned.groupBy("cluster")

    // For each cluster, compute median of each dimension
    val centers = (0 until k).map { clusterId =>
      val clusterData = assigned.filter(col("cluster") === clusterId)

      if (clusterData.count() == 0) {
        // Empty cluster
        Array.fill(numFeatures)(0.0)
      } else {
        // Compute median for each dimension
        val medians = (0 until numFeatures).map { dim =>
          val dimValues = clusterData.select(featuresCol)
            .rdd
            .map { row =>
              val vec = row.getAs[Vector](0)
              vec(dim)
            }
            .collect()
            .sorted

          // Simple median (unweighted for now)
          val mid = dimValues.length / 2
          if (dimValues.length % 2 == 0) {
            (dimValues(mid - 1) + dimValues(mid)) / 2.0
          } else {
            dimValues(mid)
          }
        }

        medians.toArray
      }
    }

    centers.filter(_.nonEmpty).toArray
  }
}
```

**4. Add L1 Kernel** (Modify: `BregmanKernel.scala`)

```scala
/**
 * L1 (Manhattan distance) kernel for K-Medians.
 */
class L1Kernel extends BregmanKernel {
  override val name: String = "l1"
  override val supportsExpressionOptimization: Boolean = false

  override def divergence(p: Vector, q: Vector): Double = {
    val pArr = p.toArray
    val qArr = q.toArray
    var sum = 0.0
    var i = 0
    while (i < pArr.length) {
      sum += math.abs(pArr(i) - qArr(i))
      i += 1
    }
    sum
  }

  override def gradient(v: Vector): Vector = {
    // Subgradient: sign function
    Vectors.dense(v.toArray.map(x => if (x > 0) 1.0 else if (x < 0) -1.0 else 0.0))
  }

  override def inverseGradient(g: Vector): Vector = {
    // Not well-defined for L1, use identity
    g
  }
}
```

**5. Tests** (New file: `KMediansSuite.scala`)

```scala
class KMediansSuite extends FunSuite with LocalSparkContext {

  test("K-Medians should be robust to outliers") {
    val data = Seq(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(1.1, 0.9),
      Vectors.dense(0.9, 1.1),
      Vectors.dense(100.0, 100.0),  // Outlier
      Vectors.dense(10.0, 10.0),
      Vectors.dense(10.1, 9.9),
      Vectors.dense(9.9, 10.1)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val kmedians = new GeneralizedKMeans()
      .setK(2)
      .setDivergence("l1")
      .setUpdateStrategy("median")
      .setMaxIter(20)

    val model = kmedians.fit(df)

    // Centers should be close to (1, 1) and (10, 10), not pulled by outlier
    val centers = model.clusterCenters

    assert(centers.exists(c => math.abs(c(0) - 1.0) < 1.0))
    assert(centers.exists(c => math.abs(c(0) - 10.0) < 1.0))
  }

  test("K-Medians should handle weighted points") {
    // Test weighted median computation
  }

  test("K-Medians convergence") {
    // Test that algorithm converges
  }
}
```

### Integration Steps

1. Add `MedianPointOps.scala` to `src/main/scala/com/massivedatascience/clusterer/`
2. Add `KMedians.scala` to same directory
3. Modify `BregmanKernel.scala` to add L1Kernel
4. Modify `Strategies.scala` to add MedianUDAFUpdate
5. Modify `GeneralizedKMeansParams.scala` to add `updateStrategy` parameter
6. Add `KMediansSuite.scala` to `src/test/scala/com/massivedatascience/clusterer/`
7. Update `DATAFRAME_API_EXAMPLES.md` with K-Medians example
8. Update `README.md` to mention K-Medians support

### Expected Benefits

- **Robustness**: 10-100x better performance on data with outliers
- **Use Cases**: Financial data, sensor data with noise, skewed distributions
- **Performance**: Similar to K-Means (same O(nkd) complexity)

---

## Phase 2: K-Medoids Implementation (3-4 weeks)

### Overview
K-Medoids (PAM algorithm) uses actual data points as cluster centers instead of computing means. Even more robust than K-Medians, but computationally expensive.

### Implementation Strategy

**Challenge**: K-Medoids requires O(n²) distance computations per iteration, which doesn't scale well in Spark.

**Solution**: Implement CLARA (Clustering Large Applications):
- Sample subset of data
- Run PAM on sample
- Assign remaining points to nearest medoid

**Files to Create**:
1. `KMedoids.scala` - RDD-based implementation
2. `KMedoidsSuite.scala` - Tests
3. `CLARAKMedoids.scala` - Scalable variant

**Expected Effort**: 3-4 weeks (algorithm complexity)

**Performance**: O(k(n-k)²) → not recommended for n > 10,000 without CLARA

---

## Phase 3: Elkan's Triangle Inequality Acceleration (3-4 weeks)

### Overview
Elkan's algorithm uses triangle inequality to skip distance computations, achieving 3-5x speedup.

### Implementation Strategy

**Key Idea**:
```
If |d(x, c1) - d(x, c2)| > 2 * d(c1, c2), then c2 cannot be closer to x than c1
```

**Bounds Tracking**:
- Upper bound: distance to assigned center
- Lower bounds: distances to all other centers

**Files to Modify**:
1. `ColumnTrackingKMeans.scala` - Add bounds tracking
2. `Strategies.scala` - Add `ElkanAssignmentStrategy`

**Pseudocode**:

```scala
class ElkanAssignmentStrategy extends AssignmentStrategy {

  // Precompute center-to-center distances
  val centerDistances: Array[Array[Double]] = ???

  override def assign(...): DataFrame = {
    // Track upper and lower bounds per point
    val withBounds = df.withColumn("upperBound", lit(Double.PositiveInfinity))
      .withColumn("lowerBounds", array((0 until k).map(_ => lit(Double.PositiveInfinity)): _*))

    // For each point, use bounds to skip distance computations
    val assignUDF = udf { (features: Vector, upper: Double, lower: Seq[Double], assigned: Int) =>
      // Check if current assignment is still valid
      if (canSkipReassignment(upper, lower, assigned, centerDistances)) {
        assigned  // Keep current assignment
      } else {
        // Recompute distances only to candidates
        findNewAssignment(features, upper, lower, centers, kernel)
      }
    }

    withBounds.withColumn("cluster", assignUDF(...))
  }
}
```

**Expected Speedup**: 3-5x for k > 10

---

## Phase 4: DataFrame API for Advanced Variants (4-6 weeks)

### Overview
Currently, only basic K-Means is available in DataFrame API. Advanced variants (Bisecting, X-Means, Soft K-Means, Streaming) are RDD-only.

### Implementation Tasks

**4.1 Bisecting K-Means** (1.5 weeks)
- Hierarchical divisive clustering
- New class: `BisectingKMeans` in `com.massivedatascience.clusterer.ml`
- Reuse `LloydsIterator` for each bisection
- **File**: `BisectingKMeans.scala`

**4.2 X-Means** (1.5 weeks)
- Automatic k selection via BIC/AIC
- New class: `XMeans`
- Iteratively split clusters that improve BIC
- **File**: `XMeans.scala`

**4.3 Soft K-Means** (2 weeks)
- Probabilistic cluster assignments
- Modify `AssignmentStrategy` to return probabilities
- New `SoftAssignmentStrategy`
- **Files**: `SoftKMeans.scala`, modify `Strategies.scala`

**4.4 Streaming K-Means** (1 week)
- Integrate with Structured Streaming
- New class: `StreamingKMeans`
- Exponential decay for online updates
- **File**: `StreamingKMeans.scala`

### API Example

```scala
// Bisecting K-Means
val bisecting = new BisectingKMeans()
  .setK(10)
  .setMinDivisibleClusterSize(5)
  .setMaxIter(20)

// X-Means
val xmeans = new XMeans()
  .setMinK(2)
  .setMaxK(20)
  .setCriterion("bic")

// Soft K-Means
val soft = new SoftKMeans()
  .setK(5)
  .setTemperature(1.0)
  .setProbabilityCol("probabilities")

// Streaming K-Means
val streaming = new StreamingKMeans()
  .setK(5)
  .setDecayFactor(0.9)
```

---

## Phase 5: Performance Benchmarking Suite (2-3 weeks)

### Overview
Create comprehensive benchmarks using JMH (Java Microbenchmarking Harness).

### Benchmark Dimensions

**1. Scalability Benchmarks**
- Data size: 1K, 10K, 100K, 1M, 10M points
- Dimensions: 2, 10, 50, 100, 1000
- Clusters (k): 2, 5, 10, 50, 100

**2. Algorithm Comparison**
- K-Means (Euclidean) vs. K-Medians vs. K-Medoids
- Standard vs. Mini-Batch vs. Coreset
- With/without Elkan's acceleration
- RDD API vs. DataFrame API

**3. Divergence Comparison**
- Squared Euclidean vs. KL vs. Itakura-Saito
- Performance impact of different kernels

**4. Spark Configuration**
- Partitions: 10, 50, 100, 200
- Executors: 1, 2, 4, 8
- Memory: 2GB, 4GB, 8GB per executor

### Implementation

**File**: `PerformanceBenchmarkSuite.scala`

```scala
package com.massivedatascience.clusterer

import org.openjdk.jmh.annotations._
import java.util.concurrent.TimeUnit

@State(Scope.Benchmark)
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MILLISECONDS)
class KMeansBenchmarks {

  @Param(Array("1000", "10000", "100000", "1000000"))
  var numPoints: Int = _

  @Param(Array("10", "50", "100"))
  var numDimensions: Int = _

  @Param(Array("5", "10", "20"))
  var k: Int = _

  var data: DataFrame = _

  @Setup
  def setup(): Unit = {
    // Generate synthetic data
    data = generateData(numPoints, numDimensions)
  }

  @Benchmark
  def benchmarkKMeans(): Unit = {
    val kmeans = new GeneralizedKMeans()
      .setK(k)
      .setMaxIter(20)
    val model = kmeans.fit(data)
  }

  @Benchmark
  def benchmarkKMedians(): Unit = {
    val kmedians = new GeneralizedKMeans()
      .setK(k)
      .setDivergence("l1")
      .setUpdateStrategy("median")
      .setMaxIter(20)
    val model = kmedians.fit(data)
  }
}
```

**Output Format**: Markdown table + charts

```
| Algorithm | n=10K | n=100K | n=1M |
|-----------|-------|--------|------|
| K-Means   | 45ms  | 423ms  | 4.2s |
| K-Medians | 52ms  | 487ms  | 4.9s |
| Elkan     | 18ms  | 165ms  | 1.6s |
```

---

## Phase 6: Scaladoc API Reference (1 week)

### Overview
Generate browsable API documentation and publish to GitHub Pages.

### Implementation Steps

**1. Add sbt-unidoc plugin** (`project/plugins.sbt`)

```scala
addSbtPlugin("com.github.sbt" % "sbt-unidoc" % "0.5.0")
```

**2. Configure unidoc** (`build.sbt`)

```scala
enablePlugins(ScalaUnidocPlugin)

unidocProjectFilter in (ScalaUnidoc, unidoc) := inProjects(root)

scalacOptions in (ScalaUnidoc, unidoc) ++= Seq(
  "-doc-title", "Generalized K-Means Clustering",
  "-doc-version", version.value,
  "-doc-root-content", baseDirectory.value + "/docs/root-doc.txt",
  "-diagrams",
  "-groups"
)
```

**3. Create GitHub Action** (`.github/workflows/docs.yml`)

```yaml
name: Generate API Docs

on:
  push:
    branches: [master]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-java@v3
        with:
          java-version: '17'

      - name: Generate Scaladoc
        run: sbt unidoc

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./target/scala-2.12/unidoc
```

**4. Add root documentation** (`docs/root-doc.txt`)

```
= Generalized K-Means Clustering API =

This library provides scalable k-means clustering with pluggable Bregman divergences for Apache Spark.

== Key Packages ==

 - [[com.massivedatascience.clusterer]] - Core clustering algorithms (RDD API)
 - [[com.massivedatascience.clusterer.ml]] - Spark ML integration (DataFrame API)
 - [[com.massivedatascience.divergence]] - Bregman divergence functions
 - [[com.massivedatascience.linalg]] - Linear algebra utilities

== Getting Started ==

See the [[com.massivedatascience.clusterer.ml.GeneralizedKMeans GeneralizedKMeans]] class for the main entry point.

== Examples ==

{{{
val kmeans = new GeneralizedKMeans()
  .setK(5)
  .setDivergence("kl")
  .setMaxIter(20)

val model = kmeans.fit(dataset)
val predictions = model.transform(dataset)
}}}
```

**Expected Output**: https://massivedatascience.github.io/generalized-kmeans-clustering/api/

---

## Phase 7: Additional Enhancements (Q3-Q4 2026)

### 7.1 Yinyang K-Means
- **Effort**: 4-6 weeks
- **Benefit**: Further acceleration for large k (k > 100)
- **Complexity**: High (group-based filtering)

### 7.2 GPU Acceleration (RAPIDS Integration)
- **Effort**: 8-12 weeks
- **Benefit**: 10-100x speedup for dense data
- **Requirements**: CUDA, cuML library
- **File**: `GPUKMeans.scala`

### 7.3 Spectral K-Means
- **Effort**: 6-8 weeks
- **Use Case**: Non-convex clusters, graph data
- **Dependencies**: Eigendecomposition (ARPACK)

### 7.4 Model Export (PMML/ONNX)
- **Effort**: 2-4 weeks
- **Benefit**: Interoperability with other ML frameworks
- **Files**: `PMMLExporter.scala`, `ONNXExporter.scala`

---

## Success Metrics

### Code Quality
- [ ] All new code has >80% test coverage
- [ ] All tests passing (property-based included)
- [ ] Zero Scalastyle violations
- [ ] Zero compiler warnings

### Performance
- [ ] K-Means performance within 10% of MLlib for Euclidean
- [ ] Elkan's algorithm achieves 3x+ speedup for k>10
- [ ] K-Medians handles 1M points in <10 seconds (local mode)

### Documentation
- [ ] API reference published to GitHub Pages
- [ ] Each new algorithm has usage example in DATAFRAME_API_EXAMPLES.md
- [ ] Performance benchmarks documented
- [ ] Migration guide for new features

### Adoption
- [ ] Zero critical bugs reported in first 3 months
- [ ] 100+ GitHub stars
- [ ] 10+ production deployments

---

## Implementation Guidelines

### Code Style
- Follow existing patterns (Strategy, Template Method, DI)
- Use `trait` for extensibility
- Minimize dependencies
- Prefer DataFrame API over RDD API for new features

### Testing Strategy
- Unit tests for all new classes
- Integration tests for end-to-end workflows
- Property-based tests for invariants
- Performance regression tests

### Documentation Requirements
- ScalaDoc for all public classes/methods
- Usage examples in markdown
- Architecture diagrams where complex
- Migration notes in MIGRATION_GUIDE.md

### Review Process
1. Self-review against checklist
2. Run full test suite + Scalastyle
3. Benchmark performance vs. baseline
4. Update documentation
5. Submit PR with detailed description

---

## Resources

### Reference Implementations
- **K-Medoids**: R package `cluster::pam`
- **Elkan's**: scikit-learn `KMeans(algorithm='elkan')`
- **Yinyang**: https://github.com/yanglei/yinyang-kmeans

### Academic Papers
- Elkan (2003): "Using the Triangle Inequality to Accelerate k-Means"
- Ding & He (2004): "K-means Clustering via Principal Component Analysis"
- Arthur & Vassilvitskii (2007): "k-means++: The Advantages of Careful Seeding"

### Performance Targets (Baseline)
- **MLlib KMeans**: 100K points, 10D, k=5 → ~500ms (local mode)
- **scikit-learn**: 100K points, 10D, k=5 → ~200ms (single-threaded)
- **RAPIDS cuML**: 1M points, 10D, k=5 → ~50ms (GPU)

---

## Conclusion

This roadmap addresses all critical gaps identified in the evaluation:
- ✅ K-Medians (Phase 1)
- ✅ K-Medoids (Phase 2)
- ✅ Elkan's acceleration (Phase 3)
- ✅ DataFrame API completion (Phase 4)
- ✅ Performance benchmarks (Phase 5)
- ✅ API documentation (Phase 6)

**Estimated Total Effort**: 16-24 weeks (4-6 months)

**Expected Outcome**: Industry-leading clustering library with **9.5/10** maintainability and feature-completeness.

---

*Document created: October 14, 2025*
*Last updated: October 14, 2025*
*Status: Draft - Ready for Implementation*
