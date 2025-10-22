# Action Items - Generalized K-Means Clustering

**Last Updated:** 2025-10-20
**Status:** DataFrame API Migration for RDD-only Algorithms

---

## 🎯 NEW PRIORITY: DataFrame API Implementation for RDD-Only Algorithms

This document now prioritizes migrating RDD-only algorithms to the modern DataFrame API based on:
- **User demand** - How often requested/used
- **Production readiness** - How stable and well-tested
- **Implementation complexity** - Effort required
- **Value proposition** - Unique capabilities vs existing algorithms

All P0 production blockers from previous roadmap are complete (persistence ✅, CI ✅, determinism ✅, docs ✅, telemetry ✅).

---

## 🚀 PRIORITY 1: Mini-Batch K-Means ⭐⭐⭐⭐⭐

**Status**: ⚠️ RDD only
**Estimated Effort**: Medium (3-4 days)
**Value**: High - massive dataset scalability
**Target Completion**: Week 1

### Why Priority 1
- **High demand**: Critical for datasets that don't fit in memory
- **Clear use case**: 10-100x speedup on massive datasets via sampling
- **Production proven**: Well-tested in RDD API with MiniBatchScheduler
- **Complements CoresetKMeans**: CoresetKMeans = one-time approximation, MiniBatch = iterative sampling
- **Growing need**: Datasets continue to grow, memory is expensive

### Technical Approach
1. **Create `MiniBatchKMeansParams.scala`**:
   ```scala
   trait MiniBatchKMeansParams extends GeneralizedKMeansParams {
     val batchFraction: DoubleParam = new DoubleParam(
       this, "batchFraction", "Fraction of data to sample per iteration (0, 1]",
       ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true)
     )

     val learningRateDecay: Param[String] = new Param[String](
       this, "learningRateDecay", "Learning rate decay strategy",
       (value: String) => Set("constant", "inverse", "exponential", "step").contains(value.toLowerCase)
     )

     val initialLearningRate: DoubleParam = new DoubleParam(
       this, "initialLearningRate", "Initial learning rate",
       ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true)
     )

     val decayRate: DoubleParam = new DoubleParam(
       this, "decayRate", "Decay rate for learning rate schedule",
       ParamValidators.gt(0.0)
     )

     setDefault(
       batchFraction -> 0.1,
       learningRateDecay -> "constant",
       initialLearningRate -> 1.0,
       decayRate -> 1.0
     )
   }
   ```

2. **Create `MiniBatchKMeans.scala`**:
   ```scala
   class MiniBatchKMeans(override val uid: String)
       extends Estimator[GeneralizedKMeansModel]
       with MiniBatchKMeansParams {

     override def fit(dataset: Dataset[_]): GeneralizedKMeansModel = {
       val df = dataset.toDF()

       // Initialize centers using standard k-means|| on full data or sample
       val initialCenters = initializeCenters(df)
       var currentCenters = initialCenters

       // Mini-batch iterations
       for (iteration <- 0 until $(maxIter)) {
         // Sample batch
         val batch = df.sample(withReplacement = false, $(batchFraction), $(seed) + iteration)

         // Compute batch centers using GeneralizedKMeans
         val batchModel = computeBatchCenters(batch, currentCenters)

         // Update global centers with learning rate
         val lr = computeLearningRate(iteration)
         currentCenters = interpolateCenters(currentCenters, batchModel.clusterCenters, lr)

         // Check convergence
         if (hasConverged(currentCenters, batchModel.clusterCenters)) {
           break
         }
       }

       // Return model with final centers
       copyValues(new GeneralizedKMeansModel(uid, currentCenters))
     }

     private def computeLearningRate(iteration: Int): Double = {
       $(learningRateDecay).toLowerCase match {
         case "constant" => $(initialLearningRate)
         case "inverse" => $(initialLearningRate) / (1.0 + iteration * $(decayRate))
         case "exponential" => $(initialLearningRate) * math.pow($(decayRate), iteration)
         case "step" => $(initialLearningRate) * math.pow(0.5, (iteration / 10).toDouble)
       }
     }

     private def interpolateCenters(
       oldCenters: Array[Vector],
       batchCenters: Array[Vector],
       lr: Double
     ): Array[Vector] = {
       oldCenters.zip(batchCenters).map { case (old, batch) =>
         Vectors.dense(old.toArray.zip(batch.toArray).map {
           case (o, b) => (1 - lr) * o + lr * b
         })
       }
     }
   }
   ```

3. **Create `MiniBatchKMeansSuite.scala`** (20+ tests):
   - Basic clustering with different batch fractions
   - Learning rate decay strategies
   - Convergence behavior
   - Comparison with full-batch GeneralizedKMeans
   - Determinism with same seed
   - All divergences (SE, KL, IS, L1)
   - Edge cases (batch fraction = 1.0, very small batches)

4. **Create `MiniBatchExample.scala`**:
   - Demonstrate speedup vs full-batch
   - Show learning rate effects
   - Compare quality metrics

5. **Update README feature matrix**

### Technical Challenges
- **Center interpolation**: Must work correctly with different divergences (not just arithmetic mean)
- **Convergence detection**: Mini-batches make convergence harder to detect
- **Learning rate tuning**: Need good defaults that work across use cases
- **Quality-speed tradeoff**: Document when to use mini-batch vs coreset

### Success Criteria
- ✅ 20+ comprehensive tests covering all parameters
- ✅ Works with all 6 divergences
- ✅ Executable example with assertions
- ✅ Model persistence support
- ✅ CI validation across Scala 2.12/2.13 and Spark 3.4/3.5
- ✅ Documentation in README
- ✅ Performance comparison showing 10-100x speedup

---

## 🚀 PRIORITY 2: Constrained K-Means ⭐⭐⭐⭐

**Status**: ⚠️ RDD only
**Estimated Effort**: High (5-7 days)
**Value**: Medium-High - semi-supervised clustering
**Target Completion**: Week 2-3

### Why Priority 2
- **Unique capability**: Only algorithm supporting pairwise constraints
- **Business value**: Enforce domain knowledge and business rules
- **Growing demand**: Semi-supervised learning increasingly popular
- **No alternatives**: Can't be replicated with other DataFrame algorithms
- **Real use cases**: Customer segmentation, document clustering, compliance clustering

### Technical Approach
1. **Create `ConstrainedKMeansParams.scala`**:
   ```scala
   trait ConstrainedKMeansParams extends GeneralizedKMeansParams {
     val mustLinkConstraints: Param[Array[(Long, Long)]] = new Param(
       this, "mustLinkConstraints", "Pairs of point IDs that must be in same cluster"
     )

     val cannotLinkConstraints: Param[Array[(Long, Long)]] = new Param(
       this, "cannotLinkConstraints", "Pairs of point IDs that cannot be in same cluster"
     )

     val violationPenalty: DoubleParam = new DoubleParam(
       this, "violationPenalty", "Penalty for constraint violations",
       ParamValidators.gt(0.0)
     )

     val maxViolations: IntParam = new IntParam(
       this, "maxViolations", "Maximum constraint violations allowed",
       ParamValidators.gtEq(0)
     )

     val hardConstraints: BooleanParam = new BooleanParam(
       this, "hardConstraints", "Whether to enforce hard constraints (Infinity penalty)"
     )

     setDefault(
       mustLinkConstraints -> Array.empty,
       cannotLinkConstraints -> Array.empty,
       violationPenalty -> Double.PositiveInfinity,
       maxViolations -> 0,
       hardConstraints -> true
     )
   }
   ```

2. **Create `ConstrainedKMeans.scala`**:
   ```scala
   class ConstrainedKMeans(override val uid: String)
       extends Estimator[GeneralizedKMeansModel]
       with ConstrainedKMeansParams {

     override def fit(dataset: Dataset[_]): GeneralizedKMeansModel = {
       val df = dataset.toDF()

       // Add stable row IDs using RowIdProvider
       val dfWithIds = new RowIdProvider().withRowId(df, "__rowId")

       // Build constraint lookup structures
       val mustLinkClosure = computeMustLinkClosure($(mustLinkConstraints))
       val cannotLinkMap = buildCannotLinkMap($(cannotLinkConstraints))

       // Broadcast constraints for assignment phase
       val bcMustLink = dfWithIds.sparkSession.sparkContext.broadcast(mustLinkClosure)
       val bcCannotLink = dfWithIds.sparkSession.sparkContext.broadcast(cannotLinkMap)

       // Initialize centers
       var centers = initializeCenters(dfWithIds)

       // Lloyd's iterations with constraint-aware assignment
       for (iteration <- 0 until $(maxIter)) {
         // Modified assignment: respect constraints
         val assignments = constrainedAssignment(dfWithIds, centers, bcMustLink, bcCannotLink)

         // Standard center update
         val newCenters = updateCenters(dfWithIds, assignments)

         // Track violations
         val violations = countViolations(assignments, bcMustLink.value, bcCannotLink.value)
         logInfo(s"Iteration $iteration: $violations violations")

         if (hasConverged(centers, newCenters)) break
         centers = newCenters
       }

       copyValues(new GeneralizedKMeansModel(uid, centers))
     }

     private def constrainedAssignment(
       df: DataFrame,
       centers: Array[Vector],
       mustLink: Broadcast[Map[Long, Set[Long]]],
       cannotLink: Broadcast[Map[Long, Set[Long]]]
     ): DataFrame = {
       // UDF that computes best cluster considering constraints
       val assignUDF = udf { (rowId: Long, features: Vector) =>
         val forbidden = cannotLink.value.getOrElse(rowId, Set.empty)

         val distances = centers.indices.map { i =>
           val baseDist = computeDistance(features, centers(i))
           val penalty = if (forbidden.contains(i)) $(violationPenalty) else 0.0
           baseDist + penalty
         }

         distances.zipWithIndex.minBy(_._1)._2
       }

       df.withColumn("prediction", assignUDF(col("__rowId"), col($(featuresCol))))
     }

     private def computeMustLinkClosure(
       mustLink: Array[(Long, Long)]
     ): Map[Long, Set[Long]] = {
       // Union-find to compute transitive closure
       val groups = scala.collection.mutable.Map[Long, Set[Long]]()
       mustLink.foreach { case (a, b) =>
         val groupA = groups.getOrElse(a, Set(a))
         val groupB = groups.getOrElse(b, Set(b))
         val merged = groupA ++ groupB
         merged.foreach { id => groups(id) = merged }
       }
       groups.toMap
     }
   }
   ```

3. **Create `ConstrainedKMeansSuite.scala`** (15+ tests):
   - Must-link constraints enforced
   - Cannot-link constraints enforced
   - Hard vs soft constraints
   - Constraint violations tracking
   - Empty constraint sets (should behave like standard k-means)
   - Large constraint sets
   - Contradictory constraints handling

4. **Create `ConstrainedKMeansExample.scala`**:
   - Business rules example (competitors must be in different clusters)
   - Semi-supervised clustering with partial labels
   - Comparison with unconstrained clustering

5. **Update README feature matrix**

### Technical Challenges
- **Row ID management**: Need stable IDs across iterations (use RowIdProvider pattern from SE path)
- **Constraint broadcasting**: May not scale to millions of constraints (document limits)
- **Assignment complexity**: Can't use standard cross-join pattern, need custom UDF
- **Convergence**: Hard constraints may prevent convergence (need max iterations safeguard)
- **Constraint validation**: Detect contradictory constraints early

### Success Criteria
- ✅ 15+ comprehensive tests
- ✅ Works with all divergences
- ✅ Handles up to 100K constraints efficiently
- ✅ Executable example
- ✅ Model persistence
- ✅ Documentation of constraint limits
- ✅ Violation tracking in training summary

---

## 🚀 PRIORITY 3: Annealed K-Means ⭐⭐⭐

**Status**: RDD only (not in README)
**Estimated Effort**: Medium (4-5 days)
**Value**: Medium - better convergence quality
**Target Completion**: Week 3-4

### Why Priority 3
- **Better optima**: Less sensitive to initialization
- **Academic interest**: Novel approach to clustering
- **Complements existing**: Works with all divergences
- **Moderate complexity**: Mostly a wrapper around SoftKMeans
- **Research applications**: Popular in machine learning research

### Technical Approach
1. **Create `AnnealedKMeansParams.scala`**:
   ```scala
   trait AnnealedKMeansParams extends GeneralizedKMeansParams {
     val initialBeta: DoubleParam = new DoubleParam(
       this, "initialBeta", "Starting inverse temperature (low = soft, high = hard)",
       ParamValidators.gt(0.0)
     )

     val finalBeta: DoubleParam = new DoubleParam(
       this, "finalBeta", "Ending inverse temperature",
       ParamValidators.gt(0.0)
     )

     val annealingSchedule: Param[String] = new Param[String](
       this, "annealingSchedule", "Strategy for increasing beta",
       (value: String) => Set("exponential", "linear", "geometric").contains(value.toLowerCase)
     )

     val annealingRate: DoubleParam = new DoubleParam(
       this, "annealingRate", "Rate at which beta increases",
       ParamValidators.gt(1.0)
     )

     val stepsPerTemperature: IntParam = new IntParam(
       this, "stepsPerTemperature", "Number of iterations at each temperature",
       ParamValidators.gt(0)
     )

     setDefault(
       initialBeta -> 0.1,
       finalBeta -> 100.0,
       annealingSchedule -> "exponential",
       annealingRate -> 1.5,
       stepsPerTemperature -> 5
     )
   }
   ```

2. **Create `AnnealedKMeans.scala`**:
   ```scala
   class AnnealedKMeans(override val uid: String)
       extends Estimator[GeneralizedKMeansModel]
       with AnnealedKMeansParams {

     override def fit(dataset: Dataset[_]): GeneralizedKMeansModel = {
       val df = dataset.toDF()

       var currentBeta = $(initialBeta)
       var centers = initializeCenters(df)

       // Annealing schedule
       while (currentBeta < $(finalBeta)) {
         // Run soft clustering at current temperature
         val softKMeans = new SoftKMeans()
           .setK($(k))
           .setDivergence($(divergence))
           .setBeta(currentBeta)
           .setMaxIter($(stepsPerTemperature))
           .setFeaturesCol($(featuresCol))

         // Warm start with current centers (need to add initialCenters param to SoftKMeans)
         val model = softKMeans.fit(df)
         centers = model.clusterCenters

         // Increase temperature
         currentBeta = updateBeta(currentBeta)
         logInfo(s"Annealing: beta=$currentBeta")
       }

       // Final hard assignment at high temperature
       val finalModel = new GeneralizedKMeans()
         .setK($(k))
         .setDivergence($(divergence))
         .setMaxIter($(stepsPerTemperature))
         .fit(df)

       copyValues(finalModel)
     }

     private def updateBeta(currentBeta: Double): Double = {
       $(annealingSchedule).toLowerCase match {
         case "exponential" => currentBeta * $(annealingRate)
         case "linear" => currentBeta + $(annealingRate)
         case "geometric" => currentBeta * math.sqrt($(annealingRate))
       }
     }
   }
   ```

3. **Create `AnnealedKMeansSuite.scala`** (12+ tests):
   - Different annealing schedules
   - Comparison with standard k-means (should get better results)
   - All divergences
   - Temperature progression
   - Convergence at each temperature level

4. **Create `AnnealedKMeansExample.scala`**:
   - Compare quality with standard k-means
   - Visualize annealing schedule
   - Show sensitivity to initialization

5. **Update README feature matrix**

### Technical Challenges
- **SoftKMeans integration**: Need to add warm-start capability to SoftKMeans
- **Temperature scheduling**: Need good defaults that work across datasets
- **Computational cost**: Multiple rounds of clustering (document tradeoff)
- **Divergence compatibility**: Some divergences may not work well with annealing

### Success Criteria
- ✅ 12+ comprehensive tests
- ✅ Works with SE, KL, IS divergences
- ✅ Demonstrates better convergence than standard k-means
- ✅ Executable example
- ✅ Model persistence
- ✅ Documentation of computational cost
- ✅ Annealing schedule visualization

---

## ⏸️ DEFERRED: Lower Priority Algorithms

### Priority 4: Bregman Co-Clustering ⭐⭐

**Status**: RDD only (not in README)
**Estimated Effort**: Very High (10+ days)
**Value**: Medium - niche use case
**Decision**: Defer to v0.8 or later

**Why Deferred:**
- **Niche**: Narrow use case (document-term, user-item matrices)
- **Complexity**: Requires matrix representation, not standard point clustering
- **Alternative approaches**: Specialized tools exist (sklearn, dedicated libraries)
- **DataFrame fit**: Awkward API - matrices don't map naturally to Spark ML patterns
- **User demand**: Very few requests compared to mini-batch and constrained

**Reconsider if**: Multiple users request it with specific use cases

---

### Priority 5: Bregman Mixture Models ⭐

**Status**: RDD only (not in README)
**Estimated Effort**: Very High (10+ days)
**Value**: Low-Medium - overlaps with SoftKMeans
**Decision**: Defer indefinitely

**Why Deferred:**
- **Overlap**: SoftKMeans already provides soft assignments and probabilities
- **Complexity**: Full EM algorithm implementation
- **Limited demand**: Most users want clustering, not statistical mixture modeling
- **Better alternatives**: Use dedicated probabilistic libraries (PyMC3, Stan)

**Reconsider if**: Strong user demand for full mixture model capabilities

---

### ❌ Not Implementing: Internal Optimizations

**Column Tracking K-Means**: Internal optimization, not user-facing
**Online K-Means**: Covered by StreamingKMeans in DataFrame API

---

## 📅 IMPLEMENTATION TIMELINE

### Week 1 (Nov 2025)
- ✅ Priority analysis complete
- ✅ ACTION_ITEMS.md updated
- 🚀 Start Mini-Batch K-Means implementation
  - Day 1-2: Params and estimator
  - Day 3-4: Tests and examples
  - Day 5: Documentation and CI validation

### Week 2 (Nov 2025)
- 🚀 Complete Mini-Batch K-Means
- 🚀 Start Constrained K-Means implementation
  - Day 1-2: Params and constraint data structures
  - Day 3-5: Constrained assignment logic

### Week 3 (Dec 2025)
- 🚀 Complete Constrained K-Means
  - Day 1-2: Tests
  - Day 3: Examples and documentation
- 🚀 Start Annealed K-Means
  - Day 4-5: Params and annealing schedule

### Week 4 (Dec 2025)
- 🚀 Complete Annealed K-Means
  - Day 1-2: Integration with SoftKMeans
  - Day 3: Tests and examples
  - Day 4-5: Documentation and benchmarks

---

## ✅ SUCCESS CRITERIA FOR EACH ALGORITHM

Every algorithm implementation must include:

1. **Code Quality**
   - ✅ Comprehensive Params trait with validation
   - ✅ Estimator extending Estimator[GeneralizedKMeansModel]
   - ✅ Proper parameter defaults
   - ✅ Clear scaladoc for all parameters

2. **Testing**
   - ✅ 15+ comprehensive tests
   - ✅ All divergences tested (where applicable)
   - ✅ Edge cases covered
   - ✅ Determinism verified
   - ✅ Comparison with RDD version (same results)

3. **Documentation**
   - ✅ Executable example with assertions
   - ✅ README feature matrix updated
   - ✅ Use cases documented
   - ✅ Performance characteristics documented
   - ✅ Parameter tuning guide

4. **CI/CD**
   - ✅ All tests pass on Scala 2.12 & 2.13
   - ✅ All tests pass on Spark 3.4 & 3.5
   - ✅ Example runs in CI (ExamplesSuite)
   - ✅ Model persistence works

5. **Production Ready**
   - ✅ Model persistence support
   - ✅ Training summary with metrics
   - ✅ Error handling and validation
   - ✅ Performance benchmarks

---

## 📊 COMPARISON: RDD vs DataFrame API

| Feature | RDD API | DataFrame API (After Migration) |
|---------|---------|----------------------------------|
| **Mini-Batch K-Means** | ✅ (MiniBatchScheduler) | 🚀 Priority 1 |
| **Constrained K-Means** | ✅ (ConstrainedKMeans) | 🚀 Priority 2 |
| **Annealed K-Means** | ✅ (AnnealedKMeans) | 🚀 Priority 3 |
| **Bregman Co-Clustering** | ✅ (BregmanCoClustering) | ⏸️ Deferred |
| **Bregman Mixture Models** | ✅ (BregmanMixtureModel) | ⏸️ Deferred |
| **Column Tracking** | ✅ (Internal optimization) | ❌ Not needed |
| **Online K-Means** | ✅ (OnlineKMeans) | ✅ StreamingKMeans |

---

## 🎯 IMMEDIATE NEXT ACTIONS

1. **Update todo list** ✅
2. **Write ACTION_ITEMS.md** ✅
3. **Start Mini-Batch K-Means implementation** ⬅️ YOU ARE HERE
   - Create MiniBatchKMeansParams.scala
   - Create MiniBatchKMeans.scala
   - Create MiniBatchKMeansSuite.scala
   - Create MiniBatchExample.scala
   - Update README

---

## 📝 NOTES FROM PREVIOUS ROADMAP

All P0 production blockers are complete:
- ✅ Persistence (all models)
- ✅ CI system (100% passing)
- ✅ Determinism tests
- ✅ Documentation with truth-links
- ✅ Telemetry and summaries
- ✅ Assignment scalability
- ✅ Performance benchmarks

The library is now production-ready for existing algorithms. This new phase focuses on **algorithm completeness** by bringing RDD-only features to the modern DataFrame API.

---

## 🔗 RELATED DOCUMENTATION

- **PERSISTENCE_COMPATIBILITY.md** - Model persistence specification
- **ASSIGNMENT_SCALABILITY.md** - Scalability guide for large k×dim
- **PERFORMANCE_BENCHMARKS.md** - Performance baselines and tuning guide
- **README.md** - Feature matrix with truth-links

This plan migrates the most valuable RDD-only algorithms to the DataFrame API, completing the library's feature set for production use.
