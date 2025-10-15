# Architecture Guide: DataFrame API with LloydsIterator

## Overview

The DataFrame API is built around a **single source of truth** design: the `LloydsIterator`. This architectural pattern eliminates thousands of lines of duplicated clustering logic by using pluggable strategies for different behaviors.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Design Patterns](#design-patterns)
5. [Hierarchical Clustering: Bisecting K-Means](#hierarchical-clustering-bisecting-k-means)
6. [Extension Points](#extension-points)
7. [Performance Considerations](#performance-considerations)

---

## Core Concepts

### The LloydsIterator Pattern

Lloyd's algorithm (k-means iteration) follows a simple loop:

```
1. Assign each point to the nearest center
2. Update centers based on assignments
3. Handle empty clusters
4. Check convergence
5. Repeat until converged or max iterations
```

**Key Insight**: This loop is identical across all k-means variants. What differs are the *strategies* used at each step.

### Single Implementation, Multiple Behaviors

Instead of duplicating this loop for each clustering variant, we implement it **once** in `DefaultLloydsIterator` and inject strategies to customize behavior.

**Before (RDD approach):**
- `KMeans.scala`: 500+ lines with Lloyd's loop
- `StreamingKMeans.scala`: 300+ lines duplicating the loop
- `BisectingKMeans.scala`: 400+ lines duplicating the loop
- **Total**: 1200+ lines of duplicated logic

**After (DataFrame approach):**
- `DefaultLloydsIterator.scala`: 168 lines (one loop)
- `Strategies.scala`: 478 lines (reusable strategies)
- **Total**: 646 lines serving all variants

**Result**: 54% code reduction with better testability and maintainability.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     GeneralizedKMeans                        │
│                    (Spark ML Estimator)                      │
│  • Parameter validation                                      │
│  • Initialization (random, k-means||)                       │
│  • Creates LloydsConfig with strategies                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ fit(DataFrame)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    DefaultLloydsIterator                     │
│                  (Core Algorithm Engine)                     │
│  1. Assign points → AssignmentStrategy                      │
│  2. Update centers → UpdateStrategy                          │
│  3. Handle empties → EmptyClusterHandler                     │
│  4. Check convergence → ConvergenceCheck                     │
│  5. Validate input → InputValidator                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ returns LloydResult
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  GeneralizedKMeansModel                      │
│                   (Spark ML Model)                           │
│  • transform(DataFrame) → predictions                        │
│  • predict(Vector) → cluster ID                             │
│  • computeCost(DataFrame) → WCSS                            │
│  • save/load (MLWriter/MLReader)                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Pluggable Strategies                      │
├─────────────────────────────────────────────────────────────┤
│ BregmanKernel (divergence function)                         │
│  • SquaredEuclideanKernel                                   │
│  • KLDivergenceKernel                                       │
│  • ItakuraSaitoKernel                                       │
│  • GeneralizedIDivergenceKernel                             │
│  • LogisticLossKernel                                       │
├─────────────────────────────────────────────────────────────┤
│ AssignmentStrategy (point → cluster)                        │
│  • BroadcastUDFAssignment (general, UDF-based)              │
│  • SECrossJoinAssignment (fast path, expression-based)      │
│  • AutoAssignment (heuristic selection)                     │
├─────────────────────────────────────────────────────────────┤
│ UpdateStrategy (clusters → new centers)                     │
│  • GradMeanUDAFUpdate (gradient-based aggregation)          │
├─────────────────────────────────────────────────────────────┤
│ EmptyClusterHandler (handle empty clusters)                 │
│  • ReseedRandomHandler (reseed with random points)          │
│  • DropEmptyClustersHandler (return k < original)           │
├─────────────────────────────────────────────────────────────┤
│ ConvergenceCheck (detect when to stop)                      │
│  • MovementConvergence (max L2 center movement)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. BregmanKernel

**Location**: `src/main/scala/com/massivedatascience/clusterer/ml/df/BregmanKernel.scala`

**Purpose**: Defines the distance function and center computation for clustering.

**Interface**:
```scala
trait BregmanKernel {
  def name: String
  def divergence(p: Vector, q: Vector): Double
  def grad(p: Vector): Vector
  def invGrad(grad: Vector): Vector
  def validate(v: Vector): Unit
  def supportsExpressionOptimization: Boolean
}
```

**How it works**:
- `divergence(p, q)`: Computes distance from point `p` to center `q`
- `grad(p)`: Transforms point to gradient space
- `invGrad(grad)`: Transforms average gradient back to point space (center)
- Centers are computed as: `invGrad(mean(grad(points)))`

**Why Bregman divergences?**
- **Squared Euclidean**: Standard k-means (L2 distance)
- **KL Divergence**: For probability distributions (text clustering)
- **Itakura-Saito**: For spectral data (audio, signals)
- **Generalized I**: For count data (Poisson-like distributions)
- **Logistic Loss**: For binary probabilities

### 2. AssignmentStrategy

**Location**: `src/main/scala/com/massivedatascience/clusterer/ml/df/Strategies.scala`

**Purpose**: Assigns each point to the nearest cluster center.

**Implementations**:

#### BroadcastUDFAssignment
- **Use case**: Any Bregman divergence
- **Method**: Broadcasts centers to executors, uses UDF to compute distances
- **Performance**: O(n × k × d) where n=points, k=clusters, d=dimensions
- **Best for**: Small k (< 100), any divergence

```scala
val assignUDF = udf { (features: Vector) =>
  val distances = centers.map(c => kernel.divergence(features, c))
  distances.zipWithIndex.minBy(_._1)._2
}
df.withColumn("cluster", assignUDF($"features"))
```

#### SECrossJoinAssignment
- **Use case**: Squared Euclidean only
- **Method**: Cross-join DataFrame with centers, use Catalyst expressions
- **Performance**: Leverages Spark's optimizer, often faster for large k
- **Best for**: Squared Euclidean with large k (> 100)

```scala
df.crossJoin(centersDF)
  .withColumn("distance", /* expression-based computation */)
  .groupBy($"id").agg(min_by(struct($"distance", $"clusterId")))
```

#### AutoAssignment
- **Use case**: Automatic selection
- **Method**: Chooses based on kernel type and cluster count
- **Heuristic**: SE + k > 100 → cross-join, otherwise UDF

### 3. UpdateStrategy

**Location**: `src/main/scala/com/massivedatascience/clusterer/ml/df/Strategies.scala`

**Purpose**: Computes new cluster centers from assigned points.

**Implementation: GradMeanUDAFUpdate**

**Algorithm**:
```scala
for each cluster c:
  points_in_c = points where cluster == c
  gradients = points_in_c.map(p => kernel.grad(p) * weight(p))
  sum_weights = sum(weights in cluster c)

  if sum_weights > 0:
    mean_gradient = sum(gradients) / sum_weights
    new_center[c] = kernel.invGrad(mean_gradient)
  else:
    // empty cluster, will be handled by EmptyClusterHandler
    skip
```

**Key insight**: Works for all Bregman divergences via grad/invGrad abstraction.

### 4. EmptyClusterHandler

**Location**: `src/main/scala/com/massivedatascience/clusterer/ml/df/Strategies.scala`

**Purpose**: Handles clusters with no assigned points.

**Implementations**:

#### ReseedRandomHandler
- Samples random points from the dataset
- Adds them as new centers for empty clusters
- **Guarantees**: Returns exactly k centers (if possible)

#### DropEmptyClustersHandler
- Simply drops empty clusters
- **Result**: May return fewer than k centers
- **Use case**: When exact k is not critical

**Empty cluster detection**:
```scala
val clusterSizes = updateStrategy.update(...)
val emptyClusters = (0 until k).filter(i =>
  !clusterSizes.contains(i)
)
```

### 5. ConvergenceCheck

**Location**: `src/main/scala/com/massivedatascience/clusterer/ml/df/Strategies.scala`

**Purpose**: Determines when to stop iterating.

**Implementation: MovementConvergence**

**Metrics computed**:
1. **Movement**: Max L2 distance any center moved
2. **Distortion**: Total weighted distance of all points to their centers

```scala
val movement = oldCenters.zip(newCenters).map { case (old, neu) =>
  euclideanDistance(old, neu)
}.max

val distortion = df.map { point =>
  val clusterId = point.cluster
  val center = newCenters(clusterId)
  kernel.divergence(point.features, center) * point.weight
}.sum
```

**Convergence condition**: `movement < tolerance`

### 6. LloydsIterator

**Location**: `src/main/scala/com/massivedatascience/clusterer/ml/df/LloydsIterator.scala`

**Core loop implementation**:

```scala
var centers = initialCenters
var iter = 0
var converged = false

while (iter < maxIter && !converged) {
  iter += 1

  // 1. Assignment
  val assigned = assigner.assign(df, featuresCol, weightCol, centers, kernel)

  // 2. Update
  val newCenters = updater.update(assigned, featuresCol, weightCol, k, kernel)

  // 3. Handle empties
  val (finalCenters, emptyCount) = emptyHandler.handle(
    assigned, featuresCol, weightCol, newCenters, df, kernel
  )

  // 4. Check convergence
  val (movement, distortion) = convergence.check(
    centers, finalCenters, assigned, featuresCol, weightCol, kernel
  )

  converged = movement < tol
  centers = finalCenters

  // 5. Checkpoint if needed
  if (iter % checkpointInterval == 0) {
    assigned.checkpoint()
  }
}

return LloydResult(centers, iter, distortionHistory, movementHistory, converged)
```

---

## Design Patterns

### 1. Strategy Pattern

**Problem**: Different clustering variants need different behaviors at each step.

**Solution**: Define strategy interfaces, inject implementations at runtime.

**Example**:
```scala
// Different assignment strategies
val udfAssign = new BroadcastUDFAssignment()
val exprAssign = new SECrossJoinAssignment()
val autoAssign = new AutoAssignment()

// Choose at runtime based on configuration
val assigner = params.assignmentStrategy match {
  case "auto" => autoAssign
  case "broadcast" => udfAssign
  case "crossjoin" => exprAssign
}
```

### 2. Template Method Pattern

**Problem**: Core algorithm loop is the same, but steps vary.

**Solution**: `DefaultLloydsIterator` defines the template, strategies implement the steps.

```scala
// Template (in LloydsIterator)
def run(...): LloydResult = {
  while (!converged) {
    assign()    // delegated to AssignmentStrategy
    update()    // delegated to UpdateStrategy
    handleEmpties()  // delegated to EmptyClusterHandler
    checkConvergence()  // delegated to ConvergenceCheck
  }
}
```

### 3. Dependency Injection

**Problem**: LloydsIterator needs many collaborators.

**Solution**: Pass all dependencies via `LloydsConfig`.

```scala
case class LloydsConfig(
  k: Int,
  maxIter: Int,
  tol: Double,
  kernel: BregmanKernel,
  assigner: AssignmentStrategy,
  updater: UpdateStrategy,
  emptyHandler: EmptyClusterHandler,
  convergence: ConvergenceCheck,
  validator: InputValidator,
  checkpointInterval: Int
)
```

**Benefits**:
- Easy to test (inject mocks)
- Easy to extend (add new strategies)
- Clear dependencies

---

## Hierarchical Clustering: Bisecting K-Means

While `GeneralizedKMeans` uses the Lloyd's algorithm with pluggable strategies, `BisectingKMeans` implements a hierarchical divisive clustering approach that offers different tradeoffs and use cases.

### Algorithm Overview

Bisecting K-Means is a **top-down hierarchical** algorithm:

1. **Start**: All points begin in a single cluster
2. **Select**: Choose the largest cluster to split
3. **Split**: Use k=2 clustering to divide the selected cluster
4. **Repeat**: Continue until reaching the target k clusters

This contrasts with Lloyd's algorithm, which assigns all points to k clusters simultaneously.

### Architecture

```
BisectingKMeans (Estimator)
    ├─ fit(DataFrame) → GeneralizedKMeansModel
    │   ├─ bisect(): Main loop
    │   │   ├─ Find largest cluster
    │   │   ├─ splitCluster(): k=2 Lloyd's
    │   │   │   ├─ AssignmentStrategy (reused)
    │   │   │   └─ UpdateStrategy (reused)
    │   │   └─ Reassign points
    │   └─ Return cluster centers
    └─ Parameters:
        ├─ k: Target number of clusters
        ├─ minDivisibleClusterSize: Minimum size to split
        ├─ divergence: Same as GeneralizedKMeans
        └─ All Lloyd's parameters (maxIter, tol, etc.)
```

### Key Design Decisions

#### 1. Reusing Strategies

**Decision**: Reuse `AssignmentStrategy` and `UpdateStrategy` from the Lloyd's iterator for the k=2 splits.

**Benefits**:
- Code reuse and consistency
- All divergences automatically supported
- Performance optimizations (SE cross-join) work immediately
- Easier to maintain and test

**Implementation**:
```scala
private def splitCluster(
  clusterData: DataFrame,
  featuresCol: String,
  weightCol: Option[String],
  kernel: BregmanKernel
): (Array[Double], Array[Double]) = {

  // Drop "cluster" column to avoid conflicts
  val cleanData = clusterData.drop("cluster")

  // Run k=2 Lloyd's using existing strategies
  val assigner = createAssignmentStrategy("auto")
  val updater = createUpdateStrategy(divergence)

  var centers = initializeTwoCenters(cleanData)
  for (_ <- 0 until maxIter) {
    val assigned = assigner.assign(cleanData, featuresCol, weightCol, centers, kernel)
    centers = updater.update(assigned, featuresCol, weightCol, 2, kernel)
  }

  (centers(0), centers(1))
}
```

#### 2. DataFrame Column Management

**Challenge**: Avoiding duplicate "cluster" columns when reassigning points.

**Solution**:
- Drop the "cluster" column before passing to `splitCluster`
- Use `withColumn("cluster_new", ...)` → `drop("cluster")` → `rename("cluster_new", "cluster")`
- This ensures clean DataFrame schema throughout

```scala
// Update cluster assignments
val oldClusteredDF = clusteredDF
val oldClusterCol = oldClusteredDF("cluster")  // Capture before dropping
clusteredDF = oldClusteredDF
  .withColumn("cluster_new", assignUDF(col(featuresCol), oldClusterCol))
  .drop("cluster")
  .withColumnRenamed("cluster_new", "cluster")
```

#### 3. Shared Model Output

**Decision**: Both `GeneralizedKMeans` and `BisectingKMeans` return `GeneralizedKMeansModel`.

**Benefits**:
- Users can use the same model API regardless of training method
- Prediction, transformation, and cost computation work identically
- Easy to compare algorithms (same evaluation metrics)
- No code duplication for model operations

**Tradeoff**: Hierarchical structure information is lost (only final flat clustering is returned)

### Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `k` | 2 | Target number of clusters (must be > 1) |
| `minDivisibleClusterSize` | 1 | Minimum cluster size to allow splitting |
| `divergence` | "squaredEuclidean" | Same as GeneralizedKMeans |
| `maxIter` | 20 | Iterations per split (not total iterations) |
| `tol` | 1e-4 | Convergence tolerance per split |
| `seed` | Random | Random seed for initialization |

**Key difference**: `maxIter` controls iterations *per k=2 split*, not total clustering iterations.

### When to Use Bisecting vs Standard K-Means

| Aspect | Bisecting K-Means | Standard K-Means (Lloyd's) |
|--------|-------------------|----------------------------|
| **Initialization sensitivity** | Low (hierarchical structure) | High (depends on random init) |
| **Determinism** | More deterministic | Less deterministic |
| **Imbalanced clusters** | Handles well | May struggle |
| **Large k** | Faster (local splits) | Slower (global assignment) |
| **Very large data** | Similar | Similar (with optimizations) |
| **Streaming/mini-batch** | Not applicable | Well-suited |
| **Hierarchical structure** | Implicit (not exposed) | None |

### Data Flow Example

**Input**: 12 points, k=3

**Step 1**: Initial state
```
Cluster 0: [all 12 points]
```

**Step 2**: Split cluster 0 into 0 and 1
```
k=2 clustering on all points:
Cluster 0: [points 0-5]    (size=6)
Cluster 1: [points 6-11]   (size=6)
```

**Step 3**: Split largest cluster (0) into 0 and 2
```
k=2 clustering on cluster 0's points:
Cluster 0: [points 0-2]    (size=3)
Cluster 2: [points 3-5]    (size=3)
Cluster 1: [unchanged]     (size=6)
```

**Output**: 3 clusters with centers computed by the update strategy

### Performance Characteristics

**Time Complexity**:
- Worst case: O(k * maxIter * n * d)
- Often faster than Lloyd's for large k because:
  - Each split only processes subset of points
  - Fewer global reassignments

**Space Complexity**:
- O(n * d) for data (cached)
- O(k * d) for centers
- Temporary O(subset_size) for each split

**Optimization Opportunities**:
- **Caching**: The main DataFrame is cached for repeated splits
- **SE optimization**: Reuses `SECrossJoinAssignment` when applicable
- **Early stopping**: `minDivisibleClusterSize` prevents unnecessary splits

### Testing Strategy

The `BisectingKMeansSuite` validates:

1. **Correctness**: Produces valid k clusters
2. **Divergence support**: Works with all Bregman divergences
3. **Parameter handling**: `minDivisibleClusterSize` prevents small splits
4. **Determinism**: More stable than random initialization
5. **Weighted data**: Respects weight column
6. **Hierarchical structure**: Splits follow expected pattern
7. **Edge cases**: k=2, single points, etc.

### Future Enhancements

Potential improvements (not yet implemented):

1. **Expose hierarchy**: Return dendrogram structure
2. **Split selection strategies**: Beyond "largest cluster"
   - Highest variance
   - Custom user function
3. **Parallel splits**: Split multiple clusters simultaneously
4. **Incremental updates**: Add new data to existing hierarchy

---

## Extension Points

### Adding a New Divergence

1. **Implement BregmanKernel**:

```scala
class MyCustomKernel extends BregmanKernel {
  override def name: String = "MyCustom"

  override def divergence(p: Vector, q: Vector): Double = {
    // Your distance function
  }

  override def grad(p: Vector): Vector = {
    // Transform to gradient space
  }

  override def invGrad(grad: Vector): Vector = {
    // Transform back to point space
  }

  override def validate(v: Vector): Unit = {
    // Check if vector is valid for this kernel
  }
}
```

2. **Register in GeneralizedKMeans**:

```scala
def createKernel(name: String): BregmanKernel = name match {
  case "myCustom" => new MyCustomKernel()
  case _ => // existing kernels
}
```

### Adding a New Assignment Strategy

1. **Implement AssignmentStrategy**:

```scala
class MyFastAssignment extends AssignmentStrategy {
  override def assign(
    df: DataFrame,
    featuresCol: String,
    weightCol: Option[String],
    centers: Array[Array[Double]],
    kernel: BregmanKernel
  ): DataFrame = {
    // Your optimized assignment logic
    // Must return DataFrame with "cluster" column (Int)
  }
}
```

2. **Use in configuration**:

```scala
val config = LloydsConfig(
  // ... other params
  assigner = new MyFastAssignment()
)
```

### Adding a New Empty Cluster Handler

```scala
class ReseedFarthestHandler extends EmptyClusterHandler {
  override def handle(
    assigned: DataFrame,
    featuresCol: String,
    weightCol: Option[String],
    centers: Array[Array[Double]],
    originalDF: DataFrame,
    kernel: BregmanKernel
  ): (Array[Array[Double]], Int) = {
    // Find points farthest from their centers
    // Use them to reseed empty clusters
    (newCenters, numEmpty)
  }
}
```

---

## Performance Considerations

### Assignment Strategy Selection

**BroadcastUDFAssignment**:
- **Best for**: k < 100, any divergence
- **Memory**: O(k × d) per executor
- **Computation**: UDF overhead, no Catalyst optimization

**SECrossJoinAssignment**:
- **Best for**: k > 100, Squared Euclidean only
- **Memory**: O(n × k) shuffle (can be large!)
- **Computation**: Catalyst-optimized expressions

**Rule of thumb**:
```
if (kernel == SquaredEuclidean && k > 100):
  use SECrossJoinAssignment
else:
  use BroadcastUDFAssignment
```

### Checkpointing

**Why**: Breaks long lineage chains, prevents stack overflow in iterative algorithms.

**When**: Set `checkpointInterval` to 10-20 iterations.

**Cost**: Writes DataFrame to disk every N iterations.

```scala
val kmeans = new GeneralizedKMeans()
  .setK(10)
  .setCheckpointInterval(10)  // checkpoint every 10 iterations
```

### Caching

**LloydsIterator** automatically caches DataFrames during iteration:
- Input DataFrame cached once
- Assigned DataFrame cached per iteration
- Unpersisted after convergence check

**User-level**: Cache input data if running multiple clusterings:

```scala
val data = spark.read.parquet("data.parquet").cache()

// Multiple clusterings reuse cached data
val model1 = new GeneralizedKMeans().setK(3).fit(data)
val model2 = new GeneralizedKMeans().setK(5).fit(data)
```

### Broadcast Threshold

**Problem**: Spark's auto-broadcast may fail for large center arrays.

**Solution**: Configure broadcast threshold in `spark-defaults.conf`:

```
spark.sql.autoBroadcastJoinThreshold=10485760  # 10MB
```

For k=1000 clusters in d=100 dimensions:
- Array size ≈ k × d × 8 bytes = 800KB ✅ (fits in default 10MB)

For k=10000 clusters in d=1000 dimensions:
- Array size ≈ k × d × 8 bytes = 80MB ❌ (exceeds default 10MB)
- Increase threshold or use `BroadcastUDFAssignment` explicitly

---

## Data Flow Example

Let's trace a single k-means iteration for Squared Euclidean with k=3:

### Input
```
DataFrame: 1000 rows × 2 features
Features: [x, y] coordinates
Centers: [[1, 1], [5, 5], [9, 9]]
```

### Step 1: Assignment (BroadcastUDFAssignment)

```scala
// Broadcast centers to all executors
val bcCenters = sc.broadcast(Array([1,1], [5,5], [9,9]))

// UDF computes distances to all 3 centers
assignUDF([3.2, 3.8]) = {
  dist_to_c0 = sqrt((3.2-1)^2 + (3.8-1)^2) = 3.67
  dist_to_c1 = sqrt((3.2-5)^2 + (3.8-5)^2) = 2.15  ← minimum
  dist_to_c2 = sqrt((3.2-9)^2 + (3.8-9)^2) = 7.91
  return 1  // cluster ID
}

// Result: DataFrame with "cluster" column
[3.2, 3.8] → cluster 1
[0.5, 0.9] → cluster 0
[8.7, 9.1] → cluster 2
...
```

### Step 2: Update (GradMeanUDAFUpdate)

```scala
// For Squared Euclidean: grad(p) = p, invGrad(g) = g
// So new center = mean(points in cluster)

cluster_0_points = [[0.5, 0.9], [1.2, 1.3], ...]
cluster_1_points = [[3.2, 3.8], [4.8, 5.1], ...]
cluster_2_points = [[8.7, 9.1], [9.2, 8.8], ...]

new_center_0 = mean(cluster_0_points) = [1.1, 1.2]
new_center_1 = mean(cluster_1_points) = [4.9, 5.0]
new_center_2 = mean(cluster_2_points) = [9.0, 9.0]
```

### Step 3: Handle Empties

```scala
// Check if any cluster is empty
cluster_sizes = [320, 450, 230]  // all non-zero
// No empty clusters, return centers as-is
```

### Step 4: Check Convergence

```scala
// Compute movement
movement_0 = euclidean([1,1], [1.1, 1.2]) = 0.22
movement_1 = euclidean([5,5], [4.9, 5.0]) = 0.10
movement_2 = euclidean([9,9], [9.0, 9.0]) = 0.00

max_movement = 0.22

// Compute distortion
distortion = sum over all points {
  distance(point, center[point.cluster])^2
}

// Check convergence
if (max_movement < tolerance):
  converged = true
```

### Output

```scala
LloydResult(
  centers = [[1.1, 1.2], [4.9, 5.0], [9.0, 9.0]],
  iterations = 1,
  distortionHistory = [123.45],
  movementHistory = [0.22],
  converged = false
)
```

---

## Testing Strategy

### Unit Tests

Test each strategy in isolation:

```scala
test("BroadcastUDFAssignment assigns correctly") {
  val kernel = new SquaredEuclideanKernel()
  val assigner = new BroadcastUDFAssignment()
  val centers = Array([0,0], [10,10])

  val assigned = assigner.assign(df, "features", None, centers, kernel)

  // Verify points near [0,0] assigned to cluster 0
  // Verify points near [10,10] assigned to cluster 1
}
```

### Integration Tests

Test LloydsIterator with real configurations:

```scala
test("LloydsIterator converges for simple dataset") {
  val config = LloydsConfig(
    k = 3,
    maxIter = 20,
    tol = 0.01,
    kernel = new SquaredEuclideanKernel(),
    assigner = new BroadcastUDFAssignment(),
    updater = new GradMeanUDAFUpdate(),
    // ... other strategies
  )

  val result = new DefaultLloydsIterator().run(df, "features", None, initialCenters, config)

  assert(result.converged)
  assert(result.iterations < 20)
}
```

### Property-Based Tests

Verify invariants across random inputs:

```scala
test("Property: clustering is reproducible with same seed") {
  forAll(dimGen, kGen, numPointsGen) { (dim, k, n) =>
    val model1 = train(data, k, seed=42)
    val model2 = train(data, k, seed=42)

    model1.clusterCenters shouldBe model2.clusterCenters
  }
}
```

---

## Comparison: RDD vs DataFrame

| Aspect | RDD API | DataFrame API |
|--------|---------|---------------|
| **Code** | ~1200 lines duplicated | ~646 lines reusable |
| **Performance** | Good | Equal or better |
| **Optimizer** | No Catalyst | Full Catalyst support |
| **Type Safety** | Strong (Scala types) | Mixed (SQL + UDFs) |
| **Extensibility** | Subclass per variant | Plug strategies |
| **Testing** | Hard (stateful) | Easy (pure strategies) |
| **Debuggability** | Scala debugger | Spark UI + explain() |
| **Maintenance** | High (duplicated code) | Low (single loop) |

---

## Summary

The DataFrame API architecture achieves:

1. **Code Reduction**: 54% fewer lines than RDD approach
2. **Flexibility**: Easy to add new divergences, strategies
3. **Performance**: Leverages Catalyst optimizer
4. **Testability**: Strategies are pure, easy to test
5. **Maintainability**: Single loop implementation

**Key Pattern**: **Strategy + Template Method + Dependency Injection**

This design makes the library production-ready while remaining extensible for research and experimentation.
