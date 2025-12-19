---
title: "Choose the Right Divergence"
---

# Choose the Right Divergence

**The #1 question**: Which divergence should I use for my data?

This guide helps you pick the right distance measure in under 2 minutes.

---

## Quick Decision Tree

```
START HERE: What kind of data do you have?

├─ General numeric data (measurements, sensor readings, coordinates)
│   └─ Use: squaredEuclidean (default)
│
├─ Probability distributions (histograms, topic mixtures, normalized frequencies)
│   └─ Use: kl
│
├─ Power spectra, audio features, variance estimates
│   └─ Use: itakuraSaito
│
├─ Text vectors (TF-IDF, embeddings where direction matters)
│   └─ Use: cosine / spherical
│
├─ Count data (word counts, event frequencies - NOT normalized)
│   └─ Use: generalizedI
│
├─ Binary probabilities (click rates, conversion rates in [0,1])
│   └─ Use: logistic
│
└─ Data with outliers, need robustness
    └─ Use: l1
```

---

## Decision by Example

### "I have customer purchase amounts"
→ **squaredEuclidean** — Standard numeric data

### "I have document topic distributions that sum to 1"
→ **kl** — These are probability distributions

### "I have TF-IDF vectors for text documents"
→ **cosine** — You care about direction, not magnitude

### "I have audio spectrograms or FFT magnitudes"
→ **itakuraSaito** — Designed for spectral data, scale-invariant

### "I have raw word counts per document (not normalized)"
→ **generalizedI** — Handles unnormalized count data

### "I have user engagement rates between 0 and 1"
→ **logistic** — Perfect for probabilities in open interval (0,1)

### "I have sensor data with occasional outliers"
→ **l1** — More robust to outliers than squared Euclidean

---

## Divergence Properties

| Divergence | Domain | Symmetric? | Outlier Robust? | Best For |
|------------|--------|------------|-----------------|----------|
| **squaredEuclidean** | Any real | Yes | No | General purpose |
| **kl** | Positive | No | No | Distributions |
| **itakuraSaito** | Positive | No | No | Spectra, scale-invariant |
| **cosine** | Non-zero | Yes | Somewhat | Text, embeddings |
| **l1** | Any real | Yes | Yes | Robust clustering |
| **generalizedI** | Non-negative | No | No | Count data |
| **logistic** | (0, 1) | No | No | Probabilities |

---

## Code Examples

### Standard Clustering (Squared Euclidean)

```scala
// Default — good for most numeric data
val kmeans = new GeneralizedKMeans()
  .setK(5)
  .setDivergence("squaredEuclidean")  // This is the default

val model = kmeans.fit(data)
```

### Topic Model Clustering (KL Divergence)

```scala
// For probability distributions (must sum to 1, all positive)
val kmeans = new GeneralizedKMeans()
  .setK(10)
  .setDivergence("kl")
  .setSmoothing(1e-10)  // Avoid division by zero

val model = kmeans.fit(topicDistributions)
```

### Text Document Clustering (Cosine)

```scala
// For TF-IDF or embedding vectors
val kmeans = new GeneralizedKMeans()
  .setK(20)
  .setDivergence("cosine")

val model = kmeans.fit(tfidfVectors)
```

### Audio/Spectral Clustering (Itakura-Saito)

```scala
// For power spectra, audio features
val kmeans = new GeneralizedKMeans()
  .setK(8)
  .setDivergence("itakuraSaito")
  .setSmoothing(1e-10)

val model = kmeans.fit(spectralFeatures)
```

### Robust Clustering with Outliers (L1)

```scala
// When you have outliers — L1 is more robust
val kmeans = new GeneralizedKMeans()
  .setK(5)
  .setDivergence("l1")

val model = kmeans.fit(noisyData)
```

---

## When to NOT Use Default

**Don't use squaredEuclidean when:**
- Your data are probability distributions → use `kl`
- You have high-dimensional text vectors → use `cosine`
- You have spectral/audio data → use `itakuraSaito`
- You have many outliers → use `l1`

**Don't use kl when:**
- Your data can be negative → use `squaredEuclidean`
- Your data are not normalized → use `generalizedI`
- You need symmetric distance → use `squaredEuclidean` or `cosine`

---

## Data Preprocessing Tips

### For KL Divergence
```scala
// Ensure positive values with smoothing
val smoothed = data.withColumn("features",
  transform(col("features"), x => x + 1e-10))
```

### For Cosine Distance
```scala
// Normalize vectors (optional, but common)
import org.apache.spark.ml.feature.Normalizer
val normalizer = new Normalizer()
  .setInputCol("features")
  .setOutputCol("normalized")
val normalized = normalizer.transform(data)
```

### For Logistic Loss
```scala
// Ensure values in (0, 1) - clip extremes
val clipped = data.withColumn("features",
  transform(col("features"), x => greatest(lit(0.001), least(lit(0.999), x))))
```

---

## Still Not Sure?

**Try multiple and compare:**

```scala
val divergences = Seq("squaredEuclidean", "kl", "cosine", "l1")

val results = divergences.map { div =>
  val model = new GeneralizedKMeans()
    .setK(5)
    .setDivergence(div)
    .fit(data)

  (div, model.summary.wcss, model.summary.silhouette)
}

results.foreach { case (div, wcss, sil) =>
  println(f"$div%-20s WCSS: $wcss%.2f  Silhouette: $sil%.3f")
}
```

Choose the divergence with the best silhouette score for your use case.

---

[Back to How-To Guides](index.html) | [Divergence Reference](../reference/divergences.html)
