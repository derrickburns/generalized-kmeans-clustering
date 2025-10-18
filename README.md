# Generalized K-Means Clustering

[![CI](https://github.com/derrickburns/generalized-kmeans-clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/derrickburns/generalized-kmeans-clustering/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Scala 2.13](https://img.shields.io/badge/scala-2.13.14-red.svg)](https://www.scala-lang.org/)
[![Scala 2.12](https://img.shields.io/badge/scala-2.12.18-red.svg)](https://www.scala-lang.org/)
[![Spark 3.5](https://img.shields.io/badge/spark-3.5.1-orange.svg)](https://spark.apache.org/)

🆕 DataFrame API (Spark ML) is the default.
Version 0.6.0 introduces a modern, RDD-free DataFrame-native API with Spark ML integration.
See DataFrame API Examples for end-to-end usage.

This project generalizes K-Means to multiple Bregman divergences and advanced variants (Bisecting, X-Means, Soft/Fuzzy, Streaming, K-Medians, K-Medoids). It provides:
	•	A DataFrame/ML API (recommended), and
	•	A legacy RDD API kept for backwards compatibility (archived below).

What’s in here
	•	Multiple divergences: Squared Euclidean, KL, Itakura–Saito, L1/Manhattan (K-Medians), Generalized-I, Logistic-loss
	•	Variants: Bisecting, X-Means (BIC/AIC), Soft K-Means, Structured-Streaming K-Means, K-Medoids (PAM/CLARA)
	•	Scale: Tested on tens of millions of points in 700+ dimensions
	•	Tooling: Scala 2.13 (primary), Spark 3.5.x (default, with 3.4.x compatibility)

⸻

Quick Start (DataFrame API)

Recommended for all new projects. The DataFrame API follows the Spark ML Estimator/Model pattern.

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
  .setDivergence("kl")              // "squaredEuclidean", "itakuraSaito", "l1", "generalizedI", "logistic"
  .setAssignmentStrategy("auto")    // "auto" | "crossJoin" (SE fast path) | "broadcastUDF" (general Bregman)
  .setMaxIter(20)

val model = gkm.fit(df)
val pred  = model.transform(df)
pred.show(false)

More recipes: see DataFrame API Examples.

⸻

Feature Matrix

Algorithm	DataFrame API	Use Case	Key Benefit
GeneralizedKMeans	✅	General clustering	5+ Bregman divergences
Bisecting K-Means	✅	Hierarchical/divisive	Tree structure; no k tuning
X-Means	✅	Unknown k	Automatic k via BIC/AIC
Soft K-Means	✅	Fuzzy	Probabilistic memberships
Streaming K-Means	✅	Real-time	Exponential forgetting
K-Medoids (PAM/CLARA)	✅	Outlier-robust	Medoids; custom distances
K-Medians	✅	Robust	L1/Manhattan divergence
Constrained K-Means	⚠️ RDD only	Balance/capacity	Size constraints
Mini-Batch K-Means	⚠️ RDD only	Massive datasets	Sampling-based
Coreset K-Means	⚠️ RDD only	Massive datasets	Approximation/acceleration


⸻

Installation / Versions
	•	Spark: 3.5.1 default (override via -Dspark.version), 3.4.x tested
	•	Scala: 2.13.14 (primary), 2.12.18 (cross-compiled)
	•	Java: 17

libraryDependencies += "com.massivedatascience" %% "massivedatascience-clusterer" % "0.6.0"

What’s New in 0.6.0
	•	Scala 2.13 primary; 3.5.x Spark default
	•	DataFrame API implementations for: Bisecting, X-Means, Soft, Streaming, K-Medoids
	•	K-Medians (L1) divergence support
	•	PySpark wrapper + smoke test
	•	Expanded examples & docs

⸻

Scaling & Assignment Strategy (important)

Different divergences require different assignment mechanics at scale:
	•	Squared Euclidean (SE) fast path — expression/codegen route:
	1.	Cross-join points with centers
	2.	Compute squared distance column
	3.	Prefer groupBy(rowId).min(distance) → join to pick argmin (scales better than window sorts)
	4.	Requires a stable rowId; we provide a RowIdProvider.
	•	General Bregman — broadcast + UDF route:
	•	Broadcast the centers; compute argmin via a tight JVM UDF.
	•	Broadcast ceiling: you’ll hit executor/memory limits if k × dim is too large to broadcast.

Parameters
	•	assignmentStrategy: StringParam = auto | crossJoin | broadcastUDF
	•	auto chooses SE fast path when divergence == SE and feasible; otherwise broadcastUDF.
	•	broadcastThreshold: IntParam (elements, not bytes)
	•	Heuristic ceiling for k × dim to guard broadcasts. If exceeded for non-SE, we warn and keep the broadcastUDF path (no DF fallback exists for general Bregman).

⸻

Input Transforms & Interpretation

Some divergences (KL, IS) require positivity or benefit from stabilized domains.
	•	inputTransform: StringParam = none | log1p | epsilonShift
	•	shiftValue: DoubleParam (e.g., 1e-6) when epsilonShift is used.

Note: Cluster centers are learned in the transformed space. If you need original-space interpretation, apply the appropriate inverse (e.g., expm1) for reporting, understanding that this is an interpretive mapping, not a different optimum.

⸻

Bisecting K-Means — efficiency note

The driver maintains a cluster_id column. For each split:
	1.	Filter only the target cluster: df.where(col("cluster_id") === id)
	2.	Run the base learner on that subset (k=2)
	3.	Join back predictions to update only the touched rows

This avoids reshuffling the full dataset at every split.

⸻

Structured Streaming K-Means

Estimator/Model for micro-batch streams using the same core update logic.
	•	initStrategy = pretrained | randomFirstBatch
	•	pretrained: provide setInitialModel / setInitialCenters
	•	randomFirstBatch: seed from the first micro-batch
	•	State & snapshots: Each micro-batch writes centers to
${checkpointDir}/centers/latest.parquet for batch reuse.
	•	StreamingGeneralizedKMeansModel.read(path) reconstructs a batch model from snapshots.

⸻

Persistence (Spark ML)

Models implement DefaultParamsWritable/Readable.

Layout

<path>/
  ├─ metadata/params.json
  ├─ centers/*.parquet          # (center_id, vector[, weight])
  └─ summary/*.json             # events, metrics (optional)

Compatibility
	•	Save/Load verified across Spark 3.4.x ↔ 3.5.x in CI.
	•	New params default safely on older loads; unknown params are ignored.

⸻

Python (PySpark) wrapper
	•	Package exposes GeneralizedKMeans, BisectingGeneralizedKMeans, SoftGeneralizedKMeans, StreamingGeneralizedKMeans, KMedoids, etc.
	•	CI runs a spark-submit smoke test on local[*] with a non-SE divergence.

⸻

Legacy RDD API (Archived)

Status: Kept for backward compatibility. New development should use the DataFrame API.
The material below documents the original RDD interfaces and helper objects. Some snippets show API signatures (placeholders) rather than runnable examples.

Quick Start (Legacy RDD API)

import com.massivedatascience.clusterer.KMeans
import org.apache.spark.mllib.linalg.Vectors

val data = sc.parallelize(Array(
  Vectors.dense(0.0, 0.0),
  Vectors.dense(1.0, 1.0),
  Vectors.dense(9.0, 8.0),
  Vectors.dense(8.0, 9.0)
))

val model = KMeans.train(
  data,
  runs = 1,
  k = 2,
  maxIterations = 20
)


⸻

The remainder of this section is an archived reference for the RDD API.

It includes: Bregman divergences, BregmanPoint/BregmanCenter, KMeansModel, clusterers, seeding, embeddings, iterative training, coreset helpers, and helper object builders.
Code blocks that include ??? indicate signatures in the original design.

<details>
<summary>Open archived RDD documentation</summary>


<!-- BEGIN LEGACY CONTENT (unchanged) -->


(All of your original README RDD content goes here — exactly as provided in your message.
For brevity in this chat, I’m not duplicating it again, but in your repo, place the full section here.)

<!-- END LEGACY CONTENT -->


</details>



⸻

Table of Contents
	•	Generalized K-Means Clustering
	•	Quick Start (DataFrame API)
	•	Feature Matrix
	•	Installation / Versions
	•	Scaling & Assignment Strategy
	•	Input Transforms & Interpretation
	•	Bisecting K-Means — efficiency note
	•	Structured Streaming K-Means
	•	Persistence (Spark ML)
	•	Python (PySpark) wrapper
	•	Legacy RDD API (Archived)

⸻

Contributing
	•	Please prefer PRs that target the DataFrame/ML path.
	•	Add tests (including property-based where sensible) and update examples.
	•	Follow Conventional Commits (feat:, fix:, docs:, refactor:, test:).

⸻

License

Apache 2.0

⸻

Notes for maintainers (can be removed later)
	•	As you land more DF features, consider extracting the RDD material into LEGACY_RDD.md to keep the README short.
	•	Keep the “Scaling & Assignment Strategy” section up-to-date when adding SE accelerations (Hamerly/Elkan/Yinyang) or ANN-assisted paths—mark SE-only and exact/approximate as appropriate.
