/*
 * Licensed to the Massive Data Science and Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Massive Data Science and Derrick R. Burns licenses this file to You under the
 * Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.massivedatascience.clusterer.ml

import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/** Scalable clustering evaluation metrics for Spark.
  *
  * Provides common clustering quality metrics computed in a distributed manner:
  *   - Silhouette score (exact and approximate)
  *   - Cluster sizes and balance metrics
  *   - Inertia (within-cluster sum of squares)
  *   - Elbow method helper
  *
  * ==Usage==
  * {{{
  * val predictions = model.transform(data)
  * val metrics = ClusteringMetrics(predictions, "features", "prediction")
  *
  * println(s"Silhouette: ${metrics.silhouetteScore}")
  * println(s"Cluster sizes: ${metrics.clusterSizes}")
  * }}}
  *
  * ==Scalability==
  * The exact silhouette score requires O(n²) distance computations. For large datasets,
  * use `approximateSilhouetteScore` which samples points or uses centroid-based approximation.
  */
object ClusteringMetrics {

  /** Compute clustering metrics from predictions DataFrame.
    *
    * @param predictions
    *   DataFrame with features and predictions columns
    * @param featuresCol
    *   name of features column
    * @param predictionCol
    *   name of prediction column
    * @return
    *   ClusteringMetrics instance
    */
  def apply(
      predictions: DataFrame,
      featuresCol: String = "features",
      predictionCol: String = "prediction"
  ): ClusteringMetricsResult = {
    val df = predictions.select(
      col(featuresCol).as("features"),
      col(predictionCol).as("prediction")
    )

    // Compute cluster sizes
    val sizes = df
      .groupBy("prediction")
      .count()
      .collect()
      .map(r => (r.getInt(0), r.getLong(1)))
      .toMap

    // Compute centroids
    val centroids = computeCentroids(df)

    // Compute inertia (WCSS)
    val inertia = computeInertia(df, centroids)

    ClusteringMetricsResult(
      predictions = df,
      clusterSizes = sizes,
      centroids = centroids,
      inertia = inertia
    )
  }

  /** Compute cluster centroids. */
  private def computeCentroids(df: DataFrame): Map[Int, Vector] = {
    df.groupBy("prediction")
      .agg(
        collect_list("features").as("points")
      )
      .collect()
      .map { row =>
        val cluster = row.getInt(0)
        // Use scala.collection.Seq to handle both mutable and immutable Seq from Parquet
        val points  = row.getAs[scala.collection.Seq[Vector]](1)
        val dim     = points.head.size
        val sum     = new Array[Double](dim)
        points.foreach { v =>
          val arr = v.toArray
          var i   = 0
          while (i < dim) {
            sum(i) += arr(i)
            i += 1
          }
        }
        val n = points.size.toDouble
        (cluster, Vectors.dense(sum.map(_ / n)))
      }
      .toMap
  }

  /** Compute inertia (within-cluster sum of squares). */
  private def computeInertia(df: DataFrame, centroids: Map[Int, Vector]): Double = {
    val spark       = df.sparkSession
    val centroidsBc = spark.sparkContext.broadcast(centroids)

    import spark.implicits._

    df.map { row =>
      val features  = row.getAs[Vector](0)
      val cluster   = row.getInt(1)
      val centroid  = centroidsBc.value(cluster)
      squaredDistance(features, centroid)
    }.reduce(_ + _)
  }

  /** Squared Euclidean distance between two vectors. */
  private def squaredDistance(a: Vector, b: Vector): Double = {
    val aArr = a.toArray
    val bArr = b.toArray
    var sum  = 0.0
    var i    = 0
    while (i < aArr.length) {
      val diff = aArr(i) - bArr(i)
      sum += diff * diff
      i += 1
    }
    sum
  }

  /** Compute exact silhouette score (O(n²) - use for small datasets).
    *
    * For each point i:
    *   - a(i) = mean distance to other points in same cluster
    *   - b(i) = min over other clusters of mean distance to that cluster
    *   - s(i) = (b(i) - a(i)) / max(a(i), b(i))
    *
    * @param df
    *   DataFrame with features and prediction columns
    * @param maxPoints
    *   maximum points to use (samples if larger)
    * @return
    *   mean silhouette score in [-1, 1]
    */
  def exactSilhouetteScore(df: DataFrame, maxPoints: Int = 10000): Double = {
    // Sample if too large
    val n       = df.count()
    val sampled = if (n > maxPoints) df.sample(maxPoints.toDouble / n) else df

    val points = sampled.collect().map { row =>
      (row.getAs[Vector](0), row.getInt(1))
    }

    if (points.length < 2) return 0.0

    val clusters = points.map(_._2).distinct

    // For single cluster, silhouette is undefined
    if (clusters.length < 2) return 0.0

    // Group points by cluster
    val byCluster = points.groupBy(_._2).map { case (c, pts) => (c, pts.map(_._1)) }

    // Compute silhouette for each point
    val silhouettes = points.map { case (point, cluster) =>
      val sameCluster  = byCluster(cluster).filterNot(_ eq point)
      val otherClusters = byCluster.filterNot(_._1 == cluster)

      // a(i) = mean distance to same cluster
      val a = if (sameCluster.nonEmpty) {
        sameCluster.map(p => math.sqrt(squaredDistance(point, p))).sum / sameCluster.length
      } else 0.0

      // b(i) = min mean distance to other clusters
      val b = if (otherClusters.nonEmpty) {
        otherClusters.values.map { clusterPoints =>
          clusterPoints.map(p => math.sqrt(squaredDistance(point, p))).sum / clusterPoints.length
        }.min
      } else 0.0

      if (math.max(a, b) > 0) (b - a) / math.max(a, b) else 0.0
    }

    silhouettes.sum / silhouettes.length
  }

  /** Compute approximate silhouette score using centroids (O(n×k)).
    *
    * Uses simplified silhouette where b(i) is distance to nearest other centroid
    * instead of mean distance to all points in that cluster.
    *
    * @param df
    *   DataFrame with features and prediction columns
    * @param centroids
    *   cluster centroids
    * @return
    *   approximate silhouette score in [-1, 1]
    */
  def approximateSilhouetteScore(df: DataFrame, centroids: Map[Int, Vector]): Double = {
    val spark       = df.sparkSession
    val centroidsBc = spark.sparkContext.broadcast(centroids)

    import spark.implicits._

    if (centroids.size < 2) return 0.0

    val silhouettes = df.map { row =>
      val features = row.getAs[Vector](0)
      val cluster  = row.getInt(1)
      val centers  = centroidsBc.value

      // a(i) = distance to own centroid
      val a = math.sqrt(squaredDistance(features, centers(cluster)))

      // b(i) = distance to nearest other centroid
      val b = centers
        .filterNot(_._1 == cluster)
        .values
        .map(c => math.sqrt(squaredDistance(features, c)))
        .min

      if (math.max(a, b) > 0) (b - a) / math.max(a, b) else 0.0
    }.collect()

    if (silhouettes.nonEmpty) silhouettes.sum / silhouettes.length else 0.0
  }

  /** Compute elbow curve data for different k values.
    *
    * Fits models for k in range [minK, maxK] and returns (k, inertia) pairs.
    *
    * @param data
    *   training data
    * @param minK
    *   minimum k to try
    * @param maxK
    *   maximum k to try
    * @param featuresCol
    *   features column name
    * @return
    *   sequence of (k, inertia) pairs
    */
  def elbowCurve(
      data: DataFrame,
      minK: Int = 2,
      maxK: Int = 10,
      featuresCol: String = "features"
  ): Seq[(Int, Double)] = {
    (minK to maxK).map { k =>
      val model       = new GeneralizedKMeans().setK(k).setFeaturesCol(featuresCol).fit(data)
      val predictions = model.transform(data)
      val metrics     = ClusteringMetrics(predictions, featuresCol, "prediction")
      (k, metrics.inertia)
    }
  }
}

/** Result of clustering metrics computation.
  *
  * @param predictions
  *   DataFrame with features and predictions
  * @param clusterSizes
  *   map from cluster ID to number of points
  * @param centroids
  *   map from cluster ID to centroid vector
  * @param inertia
  *   within-cluster sum of squares (WCSS)
  */
case class ClusteringMetricsResult(
    predictions: DataFrame,
    clusterSizes: Map[Int, Long],
    centroids: Map[Int, Vector],
    inertia: Double
) {

  /** Number of clusters. */
  def k: Int = clusterSizes.size

  /** Total number of points. */
  def numPoints: Long = clusterSizes.values.sum

  /** Compute exact silhouette score (expensive for large datasets). */
  def silhouetteScore: Double = ClusteringMetrics.exactSilhouetteScore(predictions)

  /** Compute approximate silhouette score (fast, uses centroids). */
  def approximateSilhouetteScore: Double =
    ClusteringMetrics.approximateSilhouetteScore(predictions, centroids)

  /** Cluster size balance ratio (min/max). 1.0 = perfectly balanced. */
  def balanceRatio: Double = {
    if (clusterSizes.isEmpty) 1.0
    else clusterSizes.values.min.toDouble / clusterSizes.values.max
  }

  /** Standard deviation of cluster sizes. */
  def sizeStdDev: Double = {
    if (clusterSizes.isEmpty) 0.0
    else {
      val sizes = clusterSizes.values.map(_.toDouble).toSeq
      val mean  = sizes.sum / sizes.length
      math.sqrt(sizes.map(s => (s - mean) * (s - mean)).sum / sizes.length)
    }
  }

  /** Pretty-print metrics summary. */
  override def toString: String = {
    val sizesStr = clusterSizes.toSeq.sortBy(_._1).map { case (c, n) => s"$c:$n" }.mkString(", ")
    s"""ClusteringMetrics:
       |  Clusters: $k
       |  Points: $numPoints
       |  Inertia (WCSS): ${f"$inertia%.4f"}
       |  Balance ratio: ${f"$balanceRatio%.3f"}
       |  Size std dev: ${f"$sizeStdDev%.2f"}
       |  Cluster sizes: $sizesStr""".stripMargin
  }
}
