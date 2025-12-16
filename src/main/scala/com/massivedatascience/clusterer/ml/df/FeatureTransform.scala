/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
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

package com.massivedatascience.clusterer.ml.df

import com.massivedatascience.linalg.BLAS
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/** Pure, composable feature transformation for clustering.
  *
  * Centralizes the logic that "centers live in transformed space" and makes transform pipelines
  * explicit and testable.
  *
  * Example:
  * {{{
  *   val transform = Log1p.andThen(NormalizeL2())
  *   val transformed = transform(df, "features", "transformed_features")
  *   val centers = kmeans.fit(transformed).clusterCenters
  *   val originalSpaceCenters = centers.map(transform.inverseCenter)
  * }}}
  */
trait FeatureTransform extends Serializable {

  /** Human-readable name of this transform */
  def name: String

  /** Apply transform to a DataFrame column.
    *
    * @param df
    *   input DataFrame
    * @param featuresCol
    *   name of input features column
    * @param outCol
    *   name of output column (can be same as featuresCol for in-place transform)
    * @return
    *   DataFrame with transformed features
    */
  def apply(df: DataFrame, featuresCol: String, outCol: String): DataFrame

  /** Transform a single vector (for testing/single-point prediction).
    *
    * @param v
    *   input vector
    * @return
    *   transformed vector
    */
  def apply(v: Vector): Vector

  /** Inverse transform for cluster centers (for reporting in original space).
    *
    * Note: Not all transforms have exact inverses. This provides best-effort back-projection for
    * visualization.
    *
    * @param center
    *   cluster center in transformed space
    * @return
    *   approximate center in original space
    */
  def inverseCenter(center: Vector): Vector

  /** Compose this transform with another (this then that).
    *
    * @param that
    *   transform to apply after this one
    * @return
    *   composed transform
    */
  def andThen(that: FeatureTransform): FeatureTransform = ComposedTransform(this, that)

  /** Check if this transform is compatible with a given divergence.
    *
    * @param divergence
    *   divergence name (e.g., "kl", "squaredEuclidean")
    * @return
    *   true if compatible
    */
  def compatibleWith(divergence: String): Boolean =
    true // Most transforms work with all divergences
}

/** No-op transform (identity). */
case object NoOpTransform extends FeatureTransform {
  override def name: String = "identity"

  override def apply(df: DataFrame, featuresCol: String, outCol: String): DataFrame = {
    if (featuresCol == outCol) df
    else df.withColumn(outCol, col(featuresCol))
  }

  override def apply(v: Vector): Vector = v

  override def inverseCenter(center: Vector): Vector = center
}

/** Log1p transform: log(1 + x). Safe for non-negative data, commonly used with KL divergence. */
case object Log1pTransform extends FeatureTransform {
  override def name: String = "log1p"

  override def apply(df: DataFrame, featuresCol: String, outCol: String): DataFrame = {
    val log1pUDF = udf((v: Vector) => apply(v))
    df.withColumn(outCol, log1pUDF(col(featuresCol)))
  }

  override def apply(v: Vector): Vector = {
    val arr = v.toArray
    Vectors.dense(arr.map(x => math.log1p(x)))
  }

  override def inverseCenter(center: Vector): Vector = {
    val arr = center.toArray
    Vectors.dense(arr.map(x => math.expm1(x))) // expm1(log1p(x)) = x
  }

  override def compatibleWith(divergence: String): Boolean = {
    // Log1p works well with KL and Euclidean after transformation
    val normalized = divergence.toLowerCase.replaceAll("[\\s-]", "")
    Set("kl", "squaredeuclidean", "euclidean").contains(normalized)
  }
}

/** Epsilon shift: x + epsilon. Used to ensure positive values for divergences requiring them.
  *
  * @param epsilon
  *   small positive constant to add (default: 1e-10)
  */
case class EpsilonShiftTransform(epsilon: Double = 1e-10) extends FeatureTransform {
  require(epsilon > 0.0, s"Epsilon must be positive, got $epsilon")

  override def name: String = s"epsilon_shift($epsilon)"

  override def apply(df: DataFrame, featuresCol: String, outCol: String): DataFrame = {
    val shiftUDF = udf((v: Vector) => apply(v))
    df.withColumn(outCol, shiftUDF(col(featuresCol)))
  }

  override def apply(v: Vector): Vector = {
    val arr = v.toArray
    Vectors.dense(arr.map(_ + epsilon))
  }

  override def inverseCenter(center: Vector): Vector = {
    val arr = center.toArray
    Vectors.dense(arr.map(_ - epsilon))
  }

  override def compatibleWith(divergence: String): Boolean = {
    // Epsilon shift is specifically for divergences requiring positive values
    Set("kl", "generalizedi", "itakurosaito").contains(
      divergence.toLowerCase.replaceAll("[\\s-]", "")
    )
  }
}

/** L2 normalization: x / ||x||_2. Creates unit vectors for spherical k-means (cosine distance).
  *
  * @param minNorm
  *   minimum norm to avoid division by zero (default: 1e-10)
  */
case class NormalizeL2Transform(minNorm: Double = 1e-10) extends FeatureTransform {
  require(minNorm > 0.0, s"minNorm must be positive, got $minNorm")

  override def name: String = "normalize_l2"

  override def apply(df: DataFrame, featuresCol: String, outCol: String): DataFrame = {
    val normalizeUDF = udf((v: Vector) => apply(v))
    df.withColumn(outCol, normalizeUDF(col(featuresCol)))
  }

  override def apply(v: Vector): Vector = {
    // Use vectorized BLAS for norm computation
    val norm2 = BLAS.nrm2(v)
    if (norm2 < minNorm) {
      v // Return original if near-zero norm
    } else {
      BLAS.normalize(v, minNorm)
    }
  }

  override def inverseCenter(center: Vector): Vector = {
    // L2 normalization loses scale information; we can't recover original scale
    // Return the normalized center as-is (it's already in the "natural" space for spherical k-means)
    center
  }

  override def compatibleWith(divergence: String): Boolean = {
    // L2 normalization is specifically for cosine/angular distance (via Euclidean on normalized vectors)
    divergence.toLowerCase match {
      case "squaredeuclidean" | "euclidean" | "cosine" => true
      case _                                           => false
    }
  }
}

/** L1 normalization: x / ||x||_1. Creates vectors that sum to 1 (probability distributions).
  *
  * @param minNorm
  *   minimum norm to avoid division by zero (default: 1e-10)
  */
case class NormalizeL1Transform(minNorm: Double = 1e-10) extends FeatureTransform {
  require(minNorm > 0.0, s"minNorm must be positive, got $minNorm")

  override def name: String = "normalize_l1"

  override def apply(df: DataFrame, featuresCol: String, outCol: String): DataFrame = {
    val normalizeUDF = udf((v: Vector) => apply(v))
    df.withColumn(outCol, normalizeUDF(col(featuresCol)))
  }

  override def apply(v: Vector): Vector = {
    // Use vectorized BLAS for L1 norm computation
    val norm1 = BLAS.asum(v)
    if (norm1 < minNorm) {
      v // Return original if near-zero norm
    } else {
      val result = v.copy
      BLAS.scal(1.0 / norm1, result)
      result
    }
  }

  override def inverseCenter(center: Vector): Vector = {
    // L1 normalization loses scale information
    center
  }

  override def compatibleWith(divergence: String): Boolean = {
    // L1 normalization creates probability distributions, ideal for KL divergence
    Set("kl", "squaredEuclidean", "euclidean").contains(divergence.toLowerCase)
  }
}

/** Standard scaling: (x - mean) / stddev. Centers and scales features.
  *
  * Note: Requires computing statistics from data, so this is typically done via MLlib's
  * StandardScaler. This implementation assumes pre-computed statistics.
  *
  * @param mean
  *   feature means
  * @param stddev
  *   feature standard deviations
  */
case class StandardScalingTransform(mean: Vector, stddev: Vector) extends FeatureTransform {
  require(mean.size == stddev.size, "mean and stddev must have same size")
  require(stddev.toArray.forall(_ > 0.0), "stddev values must be positive")

  override def name: String = "standard_scaling"

  override def apply(df: DataFrame, featuresCol: String, outCol: String): DataFrame = {
    val scaleUDF = udf((v: Vector) => apply(v))
    df.withColumn(outCol, scaleUDF(col(featuresCol)))
  }

  override def apply(v: Vector): Vector = {
    require(v.size == mean.size, s"Vector size ${v.size} doesn't match mean size ${mean.size}")
    val arr       = v.toArray
    val meanArr   = mean.toArray
    val stddevArr = stddev.toArray
    Vectors.dense(arr.zipWithIndex.map { case (x, i) => (x - meanArr(i)) / stddevArr(i) })
  }

  override def inverseCenter(center: Vector): Vector = {
    val arr       = center.toArray
    val meanArr   = mean.toArray
    val stddevArr = stddev.toArray
    Vectors.dense(arr.zipWithIndex.map { case (x, i) => x * stddevArr(i) + meanArr(i) })
  }
}

/** Composed transform: applies transforms sequentially.
  *
  * @param first
  *   first transform to apply
  * @param second
  *   second transform to apply
  */
case class ComposedTransform(first: FeatureTransform, second: FeatureTransform)
    extends FeatureTransform {
  override def name: String = s"${first.name} -> ${second.name}"

  override def apply(df: DataFrame, featuresCol: String, outCol: String): DataFrame = {
    // Use temporary column to avoid conflicts
    val tmpCol = s"${outCol}_tmp_${System.nanoTime()}"
    val df1    = first(df, featuresCol, tmpCol)
    val df2    = second(df1, tmpCol, outCol)
    df2.drop(tmpCol)
  }

  override def apply(v: Vector): Vector = second(first(v))

  override def inverseCenter(center: Vector): Vector =
    first.inverseCenter(second.inverseCenter(center))

  override def compatibleWith(divergence: String): Boolean = {
    first.compatibleWith(divergence) && second.compatibleWith(divergence)
  }
}

/** Factory methods for common transform combinations. */
object FeatureTransform {

  /** No transformation (identity). */
  def identity: FeatureTransform = NoOpTransform

  /** Log1p transform for non-negative data. */
  def log1p: FeatureTransform = Log1pTransform

  /** Epsilon shift for ensuring positive values. */
  def epsilonShift(epsilon: Double = 1e-10): FeatureTransform = EpsilonShiftTransform(epsilon)

  /** L2 normalization for spherical k-means. */
  def normalizeL2(minNorm: Double = 1e-10): FeatureTransform = NormalizeL2Transform(minNorm)

  /** L1 normalization for probability distributions. */
  def normalizeL1(minNorm: Double = 1e-10): FeatureTransform = NormalizeL1Transform(minNorm)

  /** Standard scaling with pre-computed statistics. */
  def standardScale(mean: Vector, stddev: Vector): FeatureTransform =
    StandardScalingTransform(mean, stddev)

  /** Common transform for KL divergence: epsilon shift then log1p. */
  def forKL(epsilon: Double = 1e-10): FeatureTransform = epsilonShift(epsilon).andThen(log1p)

  /** Common transform for spherical k-means: L2 normalization. */
  def forSpherical(minNorm: Double = 1e-10): FeatureTransform = normalizeL2(minNorm)

  /** Parse transform from string name.
    *
    * @param name
    *   transform name (e.g., "log1p", "normalize_l2", "identity")
    * @return
    *   corresponding transform
    */
  def fromString(name: String): FeatureTransform = name.toLowerCase match {
    case "identity" | "none"   => NoOpTransform
    case "log1p"               => Log1pTransform
    case "epsilon_shift"       => EpsilonShiftTransform()
    case "normalize_l2" | "l2" => NormalizeL2Transform()
    case "normalize_l1" | "l1" => NormalizeL1Transform()
    case "kl"                  => forKL()
    case "spherical"           => forSpherical()
    case _                     => throw new IllegalArgumentException(s"Unknown transform: $name")
  }
}
