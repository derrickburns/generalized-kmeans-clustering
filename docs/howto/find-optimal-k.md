---
title: Find Optimal K
---

# Find Optimal K

How to determine the right number of clusters.

---

## Method 1: Elbow Method

Plot cost (WCSS) vs. k and look for the "elbow" where improvements diminish.

```scala
import com.massivedatascience.clusterer.ml.{GeneralizedKMeans, ClusteringMetrics}

// Compute elbow curve
val elbowData = ClusteringMetrics.elbowCurve(
  data = trainingData,
  minK = 2,
  maxK = 15,
  featuresCol = "features"
)

// Print results
elbowData.foreach { case (k, inertia) =>
  println(f"k=$k%2d  inertia=$inertia%.2f")
}

// Find elbow programmatically (second derivative)
val secondDerivatives = elbowData.sliding(3).map { window =>
  val Seq((k1, i1), (k2, i2), (k3, i3)) = window
  (k2, (i1 - 2*i2 + i3))  // Second derivative
}.toSeq

val optimalK = secondDerivatives.maxBy(_._2)._1
println(s"Suggested k: $optimalK")
```

---

## Method 2: Silhouette Score

Higher silhouette = better-defined clusters.

```scala
import com.massivedatascience.clusterer.ml.ClusteringMetrics

val silhouetteScores = (2 to 10).map { k =>
  val model = new GeneralizedKMeans().setK(k).fit(data)
  val predictions = model.transform(data)
  val metrics = ClusteringMetrics(predictions)
  (k, metrics.approximateSilhouetteScore)
}

silhouetteScores.foreach { case (k, score) =>
  println(f"k=$k%2d  silhouette=$score%.4f")
}

val bestK = silhouetteScores.maxBy(_._2)._1
println(s"Best k by silhouette: $bestK")
```

---

## Method 3: X-Means (Automatic)

Let the algorithm decide using BIC/AIC.

```scala
import com.massivedatascience.clusterer.ml.XMeans

val xmeans = new XMeans()
  .setMinK(2)
  .setMaxK(20)
  .setCriterion("bic")  // or "aic"
  .setDivergence("squaredEuclidean")

val model = xmeans.fit(data)
println(s"X-Means selected k=${model.numClusters}")
```

---

## Method 4: Gap Statistic

Compare WCSS to expected WCSS under null reference distribution.

```scala
import scala.util.Random

def gapStatistic(data: DataFrame, kRange: Range, nRefs: Int = 10): Seq[(Int, Double)] = {
  val spark = data.sparkSession
  import spark.implicits._

  // Get data bounds for uniform reference
  val stats = data.select("features").as[org.apache.spark.ml.linalg.Vector]
    .flatMap(_.toArray).summary()
  val minVal = stats.min
  val maxVal = stats.max

  kRange.map { k =>
    // Actual WCSS
    val model = new GeneralizedKMeans().setK(k).fit(data)
    val actualWCSS = model.computeCost(data)

    // Reference WCSS (average over random uniform data)
    val refWCSS = (1 to nRefs).map { _ =>
      val refData = // Generate uniform random data matching dimensions
        // ... implementation details
      val refModel = new GeneralizedKMeans().setK(k).fit(refData)
      refModel.computeCost(refData)
    }.sum / nRefs

    val gap = math.log(refWCSS) - math.log(actualWCSS)
    (k, gap)
  }
}
```

---

## Comparison

| Method | Pros | Cons |
|--------|------|------|
| **Elbow** | Simple, visual | Subjective elbow point |
| **Silhouette** | Principled metric | O(n²) for exact |
| **X-Means** | Automatic, principled | May not match domain needs |
| **Gap** | Statistical foundation | Computationally expensive |

---

## Recommendations

1. **Start with X-Means** for a quick automatic answer
2. **Validate with silhouette** to confirm cluster quality
3. **Use domain knowledge** — sometimes the "right" k comes from business context
4. **Plot multiple metrics** — agreement across methods gives confidence

---

## Complete Example

```scala
import com.massivedatascience.clusterer.ml._
import org.apache.spark.ml.linalg.Vectors

// Sample data
val data = spark.createDataFrame(
  (1 to 1000).map { i =>
    val cluster = i % 5
    val noise = scala.util.Random.nextGaussian() * 0.5
    Tuple1(Vectors.dense(cluster * 3 + noise, cluster * 3 + noise))
  }
).toDF("features")

// Method 1: Elbow
println("=== Elbow Method ===")
ClusteringMetrics.elbowCurve(data, 2, 10).foreach { case (k, cost) =>
  println(f"k=$k%2d  cost=$cost%.2f")
}

// Method 2: Silhouette
println("\n=== Silhouette ===")
(2 to 10).foreach { k =>
  val model = new GeneralizedKMeans().setK(k).setSeed(42).fit(data)
  val metrics = ClusteringMetrics(model.transform(data))
  println(f"k=$k%2d  silhouette=${metrics.approximateSilhouetteScore}%.4f")
}

// Method 3: X-Means
println("\n=== X-Means ===")
val xmodel = new XMeans().setMinK(2).setMaxK(10).fit(data)
println(s"X-Means chose k=${xmodel.numClusters}")
```

---

[Back to How-To](index.html) | [Home](../)
