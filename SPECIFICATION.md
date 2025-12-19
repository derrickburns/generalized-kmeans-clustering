# Generalized K-Means Clustering: Reconstruction Specification

*A compressed specification enabling AI reconstruction of the codebase.*

---

## 1. Purpose

A Spark ML library implementing **Lloyd's algorithm generalized to Bregman divergences**, with 20+ clustering variants. Given this spec, an AI should be able to produce functionally equivalent code.

---

## 2. Mathematical Foundation

### 2.1 Bregman Divergence

For strictly convex, differentiable F: S → ℝ:

```
D_F(x, y) = F(x) - F(y) - ⟨∇F(y), x - y⟩
```

**Properties:** Non-negative, zero iff x=y, generally asymmetric.

### 2.2 Lloyd's Algorithm (Generalized)

```
Input: points X, initial centers C, kernel K, maxIter, tol
Output: final centers C*

repeat:
  # E-step: Assign each point to nearest center
  for x in X:
    assign[x] = argmin_j D_F(x, C_j)

  # M-step: Update centers via Bregman centroid
  for j in 1..k:
    points_j = {x : assign[x] == j}
    gradient_sum = Σ (w_i · ∇F(x_i)) for x_i in points_j
    weight_sum = Σ w_i
    C_j = (∇F)^(-1)(gradient_sum / weight_sum)

  # Convergence check
  movement = max(||C_j - C_j_prev||)
until movement < tol or iterations >= maxIter
```

**Key insight:** Center = `invGrad(weighted_mean_of_gradients)`, not simple mean.

### 2.3 K-Means++ Initialization (Bregman-native)

```
C[0] = random point from X
for i in 1..k-1:
  for x in X:
    d[x] = min_j D_F(x, C[j])  # Distance to nearest center
  C[i] = sample x with probability ∝ d[x]
return C
```

---

## 3. Divergence Implementations

| Name | F(x) | ∇F(x) | (∇F)⁻¹(θ) | D_F(x,y) | Domain |
|------|------|-------|-----------|----------|--------|
| **SquaredEuclidean** | ½\|\|x\|\|² | x | θ | ½\|\|x-y\|\|² | ℝⁿ |
| **KL** | Σ xᵢlog(xᵢ) | log(xᵢ)+1 | exp(θᵢ-1) | Σ xᵢlog(xᵢ/yᵢ) | ℝ₊ⁿ |
| **ItakuraSaito** | -Σ log(xᵢ) | -1/xᵢ | -1/θᵢ | Σ(xᵢ/yᵢ - log(xᵢ/yᵢ) - 1) | ℝ₊ⁿ |
| **GeneralizedI** | Σ xᵢ(log(xᵢ)-1) | log(xᵢ) | exp(θᵢ) | Σ(xᵢlog(xᵢ/yᵢ) - xᵢ + yᵢ) | ℝ₊ⁿ |
| **Logistic** | Σ[xᵢlog(xᵢ)+(1-xᵢ)log(1-xᵢ)] | log(xᵢ/(1-xᵢ)) | 1/(1+exp(-θᵢ)) | binary cross-entropy | (0,1)ⁿ |
| **L1** | Σ\|xᵢ\| | sign(xᵢ) | θ (identity) | Σ\|xᵢ-yᵢ\| | ℝⁿ |
| **Spherical** | 0 | normalize(x) | normalize(θ) | 1 - cos(x,y) | ℝⁿ\{0} |

**Smoothing:** KL, IS, GenI, Logistic add ε=1e-10 to avoid log(0)/div-by-zero.

---

## 4. Architecture

### 4.1 Package Structure

```
com.massivedatascience.clusterer.ml/
├── GeneralizedKMeans.scala     # Main estimator
├── *Model.scala                # Trained models
├── df/
│   ├── LloydsIterator.scala    # Core algorithm loop
│   ├── kernels/
│   │   ├── BregmanKernel.scala # Kernel trait
│   │   └── *.scala             # Implementations
│   └── strategies/
│       └── impl/               # Assignment strategies
└── divergence/
    └── BregmanFunction.scala   # Math definitions
```

### 4.2 Core Traits

```scala
trait BregmanKernel extends Serializable {
  def divergence(x: Vector, y: Vector): Double
  def grad(x: Vector): Vector
  def invGrad(theta: Vector): Vector
  def validate(x: Vector): Boolean
  def name: String
  def supportsExpressionOptimization: Boolean  // SE only
}

trait Estimator[M <: Model] {
  def fit(dataset: DataFrame): M
  // Spark ML params pattern
}

trait Model[M] extends Transformer {
  def transform(dataset: DataFrame): DataFrame
  def clusterCenters: Array[Vector]
}
```

### 4.3 Lloyd's Iterator

```scala
class LloydsIterator(config: LloydsConfig) {
  def run(df: DataFrame, initialCenters: Array[Vector]): LloydResult = {
    var centers = initialCenters
    var iter = 0
    var converged = false

    while (iter < config.maxIter && !converged) {
      // 1. Assign: df + "cluster" column via strategy
      val assigned = config.assignmentStrategy.assign(df, centers)

      // 2. Update: compute new centers via gradient aggregation
      val newCenters = config.updateStrategy.update(assigned, config.k)

      // 3. Handle empty clusters
      newCenters = config.emptyHandler.handle(newCenters, df)

      // 4. Check convergence
      converged = maxMovement(centers, newCenters) < config.tol
      centers = newCenters
      iter += 1
    }
    LloydResult(centers, iter, converged)
  }
}
```

### 4.4 Assignment Strategies

| Strategy | Description | Use When |
|----------|-------------|----------|
| **BroadcastUDF** | Broadcast centers, UDF computes distances | Any divergence |
| **CrossJoin** | SQL cross-join + expression distance | SE only, moderate k×d |
| **Accelerated** | Elkan/Hamerly triangle inequality | SE, large datasets |
| **Auto** | Selects best based on k, d, divergence | Default |

---

## 5. Algorithm Variants

### 5.1 Core Algorithms

| Algorithm | Key Modification to Lloyd's |
|-----------|----------------------------|
| **GeneralizedKMeans** | Standard Lloyd's with pluggable divergence |
| **BisectingKMeans** | Recursive binary splits until k clusters |
| **XMeans** | Try k from minK to maxK, select by BIC/AIC |
| **SoftKMeans** | Probabilistic assignments: P(j\|x) ∝ exp(-βD(x,cⱼ)) |
| **StreamingKMeans** | Online updates with decay factor |
| **MiniBatchKMeans** | Update on random sample each iteration |
| **BalancedKMeans** | Enforce min/max cluster sizes |
| **ConstrainedKMeans** | Must-link/cannot-link constraints |
| **KMedoids** | Centers must be actual data points (PAM) |
| **CoresetKMeans** | Build weighted coreset first, cluster that |
| **RobustKMeans** | Outlier detection/trimming modes |

### 5.2 Specialized Algorithms

| Algorithm | Use Case |
|-----------|----------|
| **TimeSeriesKMeans** | DTW/SoftDTW/GAK kernels for sequences |
| **SpectralClustering** | Graph Laplacian eigenvector embedding |
| **InformationBottleneck** | Information-theoretic compression |
| **MultiViewKMeans** | Multiple feature representations with per-view divergences |
| **KernelKMeans** | Mercer kernels (RBF, polynomial) |

---

## 6. Parameters

### 6.1 Common Parameters (GeneralizedKMeansParams)

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| k | Int | 2 | Number of clusters |
| divergence | String | "squaredEuclidean" | Divergence name |
| smoothing | Double | 1e-10 | Domain safety margin |
| maxIter | Int | 20 | Max iterations |
| tol | Double | 1e-4 | Convergence threshold |
| seed | Long | random | RNG seed for determinism |
| initMode | String | "k-means\|\|" | Initialization: "random" or "k-means\|\|" |
| initSteps | Int | 2 | K-means++ oversampling steps |
| featuresCol | String | "features" | Input column name |
| predictionCol | String | "prediction" | Output column name |
| weightCol | String | None | Optional weight column |
| distanceCol | String | None | Optional distance output |
| assignmentStrategy | String | "auto" | auto/broadcast/crossJoin |
| emptyClusterStrategy | String | "reseedRandom" | reseedRandom/drop |

### 6.2 Algorithm-Specific Parameters

- **SoftKMeans:** `beta` (softness, default=2.0), `probabilityCol`
- **StreamingKMeans:** `decayFactor`, `halfLife`, `timeUnit`
- **MiniBatchKMeans:** `batchSize`, `maxNoImprovement`
- **XMeans:** `minK`, `maxK`, `criterion` (bic/aic)
- **BalancedKMeans:** `minClusterSize`, `maxClusterSize`, `balanceMode`
- **RobustKMeans:** `robustMode` (trim/noise/mestimator), `outlierThreshold`
- **TimeSeriesKMeans:** `kernelType` (dtw/softdtw/gak)
- **SpectralClustering:** `affinityType`, `laplacianType`, `numEigenvectors`

---

## 7. Persistence

### 7.1 Model Serialization

```
path/
├── metadata/
│   └── part-00000.json    # Params, version, checksums
└── centers/
    └── *.parquet          # (center_id, weight, vector)
```

### 7.2 Metadata Schema

```json
{
  "class": "com.massivedatascience.clusterer.ml.GeneralizedKMeansModel",
  "timestamp": 1703001234567,
  "sparkVersion": "3.5.1",
  "uid": "gkm_abc123",
  "paramMap": {
    "k": 5,
    "divergence": "kl",
    "smoothing": 1e-10,
    ...
  },
  "checksums": {
    "centers": "sha256:abc...",
    "metadata": "sha256:def..."
  }
}
```

### 7.3 Cross-Version Compatibility

- Scala 2.12 ↔ 2.13: Compatible via Parquet
- Spark 3.4 ↔ 3.5 ↔ 4.0: Compatible (4.0 requires Scala 2.13)

---

## 8. Training Summary

```scala
trait TrainingSummary {
  def iterations: Int
  def converged: Boolean
  def distortionHistory: Array[Double]  // WCSS per iteration
  def movementHistory: Array[Double]    // Max center movement
  def elapsedMillis: Long
}

trait GeneralizedKMeansSummary extends TrainingSummary {
  def wcss: Double                    // Within-cluster sum of squares
  def bcss: Double                    // Between-cluster sum of squares
  def calinskiHarabaszIndex: Double   // Variance ratio
  def daviesBouldinIndex: Double      // Cluster similarity
  def silhouette(sample: Double): Double
  def clusterSizes: Array[Long]
}
```

---

## 9. Key Implementation Details

### 9.1 Weighted Center Computation

```scala
def computeCenter(points: Seq[(Vector, Double)], kernel: BregmanKernel): Vector = {
  var gradSum = new Array[Double](d)
  var weightSum = 0.0

  for ((point, weight) <- points) {
    val grad = kernel.grad(point)
    for (i <- 0 until d) {
      gradSum(i) += weight * grad(i)
    }
    weightSum += weight
  }

  // Divide by total weight
  for (i <- 0 until d) {
    gradSum(i) /= weightSum
  }

  kernel.invGrad(Vectors.dense(gradSum))
}
```

### 9.2 SE Fast Path (Expression-based)

```scala
// For Squared Euclidean only, avoid UDF overhead
def seAssignment(df: DataFrame, centers: Array[Vector]): DataFrame = {
  val centroidsDf = spark.createDataFrame(centers.zipWithIndex)
    .toDF("center", "centerId")

  df.crossJoin(centroidsDf)
    .withColumn("distance",
      aggregate(
        zip_with(col("features"), col("center"), (a, b) => pow(a - b, 2)),
        lit(0.0),
        (acc, x) => acc + x
      )
    )
    .groupBy("rowId")
    .agg(min_by(col("centerId"), col("distance")).as("prediction"))
}
```

### 9.3 Domain Validation

```scala
def validateForDivergence(df: DataFrame, divergence: String): Unit = {
  divergence match {
    case "kl" | "itakuraSaito" | "generalizedI" =>
      require(allPositive(df("features")),
        s"$divergence requires strictly positive values. " +
        "Use inputTransform='log1p' or 'epsilonShift'.")
    case "logistic" =>
      require(allInZeroOne(df("features")),
        "Logistic loss requires values in (0, 1).")
    case _ => // No constraint
  }
}
```

---

## 10. Testing Requirements

### 10.1 Test Categories

1. **Kernel Tests:** Verify F, ∇F, (∇F)⁻¹, divergence formulas
2. **Convergence Tests:** Same seed → same result
3. **Persistence Tests:** Round-trip save/load across versions
4. **Algorithm Tests:** Each variant produces valid clusters
5. **Metrics Tests:** Silhouette, Davies-Bouldin, etc. correct

### 10.2 Key Invariants

- `divergence(x, x) == 0` for all x in domain
- `divergence(x, y) >= 0` for all x, y
- `invGrad(grad(x)) ≈ x` (inverse relationship)
- Same seed + same data → identical centers
- Model load produces identical transform as original

---

## 11. Performance Characteristics

| Operation | Complexity |
|-----------|------------|
| Assignment (standard) | O(n × k × d) |
| Assignment (Elkan) | O(n × k) average |
| Update | O(n × d) |
| K-means++ init | O(n × k × d) |
| Total per iteration | O(n × k × d) |
| Mini-batch per iter | O(b × k × d) where b << n |

**Spark-specific:**
- Centers broadcast to all executors (O(k × d) memory)
- Assignment is embarrassingly parallel
- Update uses UDAF or reduceByKey aggregation

---

## 12. Reconstruction Priorities

To recreate this library:

1. **Phase 1:** BregmanFunction trait + SE/KL implementations
2. **Phase 2:** LloydsIterator with broadcast assignment
3. **Phase 3:** GeneralizedKMeans Estimator/Model
4. **Phase 4:** K-means++ initialization
5. **Phase 5:** Additional divergences (IS, L1, spherical, etc.)
6. **Phase 6:** Assignment strategy variants
7. **Phase 7:** Algorithm variants (Bisecting, XMeans, Soft, etc.)
8. **Phase 8:** Persistence layer
9. **Phase 9:** Training summary and metrics
10. **Phase 10:** Specialized algorithms (TimeSeries, Spectral, IB)

---

## 13. Example Usage

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
import org.apache.spark.ml.linalg.Vectors

val data = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.0, 0.0)),
  Tuple1(Vectors.dense(1.0, 1.0)),
  Tuple1(Vectors.dense(9.0, 8.0)),
  Tuple1(Vectors.dense(8.0, 9.0))
)).toDF("features")

val kmeans = new GeneralizedKMeans()
  .setK(2)
  .setDivergence("kl")
  .setMaxIter(20)
  .setSeed(42L)

val model = kmeans.fit(data)
val predictions = model.transform(data)

println(s"Centers: ${model.clusterCenters.mkString(", ")}")
println(s"WCSS: ${model.summary.wcss}")
```

---

*This specification captures the essential behavior of the generalized-kmeans-clustering library. An AI with access to Spark ML documentation should be able to reconstruct functionally equivalent code.*
