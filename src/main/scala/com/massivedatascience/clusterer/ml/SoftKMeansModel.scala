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

package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df.{ BregmanKernel, SoftAssignments }
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param.{ ParamMap }
import org.apache.spark.ml.util._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Model produced by Soft K-Means clustering.
  *
  * Contains cluster centers and methods for computing soft and hard assignments.
  *
  * @param uid
  *   Unique identifier
  * @param clusterCenters
  *   Final cluster centers
  * @param betaValue
  *   Inverse temperature parameter
  * @param minMembershipValue
  *   Minimum membership probability
  * @param kernel
  *   Kernel for distance calculations
  */
class SoftKMeansModel(
    override val uid: String,
    val clusterCenters: Array[Vector],
    val betaValue: Double,
    val minMembershipValue: Double,
    private val kernel: BregmanKernel
) extends org.apache.spark.ml.Model[SoftKMeansModel]
    with SoftKMeansParams
    with MLWritable
    with Logging
    with HasTrainingSummary
    with CentroidModelHelpers {

  override def clusterCentersAsVectors: Array[Vector] = clusterCenters

  override def numClusters: Int = clusterCenters.length

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()

    // Add soft assignments
    val withProbabilities = computeSoftAssignments(df)

    // Hard assignment
    val hardAssignUDF = udf { (probs: Vector) => probs.toArray.zipWithIndex.maxBy(_._1)._2 }
    val result        =
      withProbabilities.withColumn($(predictionCol), hardAssignUDF(col($(probabilityCol))))

    // Optional distance column
    if (hasDistanceCol) {
      val distanceUDF = udf { (features: Vector, prediction: Int) =>
        kernel.divergence(features, clusterCenters(prediction))
      }
      result.withColumn($(distanceCol), distanceUDF(col($(featuresCol)), col($(predictionCol))))
    } else result
  }

  /** Compute soft assignment probabilities. */
  private def computeSoftAssignments(df: DataFrame): DataFrame = {
    SoftAssignments.withProbabilities(
      df,
      clusterCenters,
      kernel,
      beta = betaValue,
      minMembership = minMembershipValue,
      featuresCol = $(featuresCol),
      probabilityCol = $(probabilityCol)
    )
  }

  /** Predict cluster (hard). */
  def predict(features: Vector): Int = {
    val distances = clusterCenters.map(center => kernel.divergence(features, center))
    distances.zipWithIndex.minBy(_._1)._2
  }

  /** Predict soft probabilities for a single vector. */
  def predictSoft(features: Vector): Vector = {
    val distances         = clusterCenters.map(center => kernel.divergence(features, center))
    val minDistance       = distances.min
    val unnormalizedProbs = distances.map(dist => math.exp(-betaValue * (dist - minDistance)))
    val totalProb         = unnormalizedProbs.sum
    val probabilities     =
      if (totalProb > 1e-100) unnormalizedProbs.map(_ / totalProb).toArray
      else Array.fill(clusterCenters.length)(1.0 / clusterCenters.length)
    val adjusted          = probabilities.map(p => math.max(p, minMembershipValue))
    val adjustedSum       = adjusted.sum
    Vectors.dense(adjusted.map(_ / adjustedSum))
  }

  /** Hard clustering cost. */
  def computeCost(dataset: Dataset[_]): Double = {
    val df      = dataset.toDF()
    val costUDF = udf { (features: Vector) =>
      val prediction = predict(features)
      kernel.divergence(features, clusterCenters(prediction))
    }
    df.select(costUDF(col($(featuresCol))).as("cost")).agg(sum("cost")).head().getDouble(0)
  }

  /** Soft clustering cost. */
  def computeSoftCost(dataset: Dataset[_]): Double = {
    val df        = dataset.toDF()
    val withProbs = computeSoftAssignments(df)
    val costUDF   = udf { (features: Vector, probs: Vector) =>
      probs.toArray.zipWithIndex.map { case (p, c) =>
        p * kernel.divergence(features, clusterCenters(c))
      }.sum
    }
    withProbs
      .select(costUDF(col($(featuresCol)), col($(probabilityCol))).as("cost"))
      .agg(sum("cost"))
      .head()
      .getDouble(0)
  }

  /** Entropy-based effective number of clusters. */
  def effectiveNumberOfClusters(dataset: Dataset[_]): Double = {
    val df         = dataset.toDF()
    val withProbs  = computeSoftAssignments(df)
    val entropyUDF = udf { (probs: Vector) =>
      val entropy =
        -probs.toArray.map(p => if (p > minMembershipValue) p * math.log(p) else 0.0).sum
      math.exp(entropy)
    }
    withProbs
      .select(entropyUDF(col($(probabilityCol))).as("effectiveClusters"))
      .agg(avg("effectiveClusters"))
      .head()
      .getDouble(0)
  }

  override def copy(extra: ParamMap): SoftKMeansModel = {
    val copied = new SoftKMeansModel(uid, clusterCenters, betaValue, minMembershipValue, kernel)
    copyValues(copied, extra).setParent(parent)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema) // probabilityCol added during transform
  }

  override def write: MLWriter = new SoftKMeansModel.SoftKMeansModelWriter(this)

  // Stored for persistence round-trip
  private[ml] var modelDivergence: String = "squaredEuclidean"
  private[ml] var modelSmoothing: Double  = 1e-10
}

object SoftKMeansModel extends MLReadable[SoftKMeansModel] {

  override def read: MLReader[SoftKMeansModel]     = new SoftKMeansModelReader
  override def load(path: String): SoftKMeansModel = super.load(path)

  private class SoftKMeansModelWriter(instance: SoftKMeansModel) extends MLWriter with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession
      logInfo(s"Saving SoftKMeansModel to $path")

      // (center_id, weight, vector) — weight = 1.0 for soft k-means centers
      val centersData =
        instance.clusterCenters.indices.map(i => (i, 1.0, instance.clusterCenters(i)))

      val centersHash = writeCenters(spark, path, centersData)
      logInfo(s"Centers saved with SHA-256: $centersHash")

      val params: Map[String, Any] = Map(
        "k"              -> instance.numClusters,
        "featuresCol"    -> instance.getOrDefault(instance.featuresCol),
        "predictionCol"  -> instance.getOrDefault(instance.predictionCol),
        "probabilityCol" -> instance.getOrDefault(instance.probabilityCol),
        "beta"           -> instance.betaValue,
        "minMembership"  -> instance.minMembershipValue,
        "divergence"     -> instance.modelDivergence,
        "smoothing"      -> instance.modelSmoothing
      )

      val k   = instance.numClusters
      val dim = instance.clusterCenters.headOption.map(_.size).getOrElse(0)

      implicit val formats          = DefaultFormats
      val metaObj: Map[String, Any] = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "SoftKMeansModel",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "divergence"         -> instance.modelDivergence,
        "k"                  -> k,
        "dim"                -> dim,
        "uid"                -> instance.uid,
        "params"             -> params,
        "centers"            -> Map[String, Any](
          "count"    -> k,
          "ordering" -> "center_id ASC (0..k-1)",
          "storage"  -> "parquet"
        ),
        "checksums"          -> Map[String, String](
          "centersParquetSHA256" -> centersHash
        )
      )

      val json         = Serialization.write(metaObj)(formats)
      val metadataHash = writeMetadata(path, json)
      logInfo(s"Metadata saved with SHA-256: $metadataHash")
      logInfo(s"SoftKMeansModel successfully saved to $path")
    }
  }

  private class SoftKMeansModelReader extends MLReader[SoftKMeansModel] with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): SoftKMeansModel = {
      val spark = sparkSession
      logInfo(s"Loading SoftKMeansModel from $path")

      val metaStr          = readMetadata(path)
      implicit val formats = DefaultFormats
      val metaJ            = JsonMethods.parse(metaStr)

      val layoutVersion = (metaJ \ "layoutVersion").extract[Int]
      val k             = (metaJ \ "k").extract[Int]
      val dim           = (metaJ \ "dim").extract[Int]
      val uid           = (metaJ \ "uid").extract[String]
      val divergence    = (metaJ \ "divergence").extract[String]

      logInfo(
        s"Model metadata: layoutVersion=$layoutVersion, k=$k, dim=$dim, divergence=$divergence"
      )

      val centersDF = readCenters(spark, path)
      val rows      = centersDF.collect()
      validateMetadata(layoutVersion, k, dim, rows.length)

      val centers = rows.sortBy(_.getInt(0)).map(_.getAs[Vector]("vector"))

      val paramsJ       = metaJ \ "params"
      val beta          = (paramsJ \ "beta").extract[Double]
      val minMembership = (paramsJ \ "minMembership").extract[Double]
      val smoothing     = (paramsJ \ "smoothing").extract[Double]

      import com.massivedatascience.clusterer.ml.df._
      val kernel: BregmanKernel = divergence match {
        case "squaredEuclidean"     => new SquaredEuclideanKernel()
        case "kl"                   => new KLDivergenceKernel(smoothing)
        case "itakuraSaito"         => new ItakuraSaitoKernel(smoothing)
        case "generalizedI"         => new GeneralizedIDivergenceKernel(smoothing)
        case "logistic"             => new LogisticLossKernel(smoothing)
        case "l1" | "manhattan"     => new L1Kernel()
        case "spherical" | "cosine" => new SphericalKernel()
        case _                      => new SquaredEuclideanKernel()
      }

      val model = new SoftKMeansModel(uid, centers, beta, minMembership, kernel)
      model.modelDivergence = divergence
      model.modelSmoothing = smoothing

      model.set(model.featuresCol, (paramsJ \ "featuresCol").extract[String])
      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])
      model.set(model.probabilityCol, (paramsJ \ "probabilityCol").extract[String])

      logInfo(s"SoftKMeansModel successfully loaded from $path")
      model
    }
  }
}
