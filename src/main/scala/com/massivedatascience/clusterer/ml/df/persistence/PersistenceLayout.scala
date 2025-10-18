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

package com.massivedatascience.clusterer.ml.df.persistence

import org.apache.spark.sql.{ DataFrame, SparkSession }
import org.apache.spark.ml.linalg.Vector
import java.security.MessageDigest
import java.nio.file.{ Files, Paths, StandardOpenOption }
import java.nio.charset.StandardCharsets

/** Versioned persistence layout for K-Means models.
  *
  * Layout V1 provides a stable, engine-neutral format for saving and loading clustering models
  * across different Spark and Scala versions.
  *
  * On-disk structure:
  * {{{
  * <modelPath>/
  *   metadata.json        # versioned metadata
  *   centers.parquet/     # cluster centers in deterministic order
  *   summary.json         # optional training metrics
  * }}}
  */
object PersistenceLayoutV1 {

  val LayoutVersion: Int = 1

  /** Metadata for a persisted clustering model */
  case class Meta(
      layoutVersion: Int,
      algo: String,
      sparkMLVersion: String,
      scalaBinaryVersion: String,
      divergence: String,
      k: Int,
      dim: Int,
      params: Map[String, Any],
      centers: MetaCenters,
      checksums: MetaChecksums
  )

  case class MetaCenters(count: Int, ordering: String, storage: String)
  case class MetaChecksums(centersParquetSHA256: String, metadataCanonicalSHA256: String)

  /** Optional training summary */
  case class Summary(
      iterations: Int,
      initialization: String,
      events: Seq[IterationEvent],
      strategy: String,
      elapsedMillis: Long
  )

  case class IterationEvent(iter: Int, distortion: Double, moved: Int)

  /** Write cluster centers to Parquet with deterministic ordering.
    *
    * @param spark
    *   SparkSession
    * @param path
    *   Base path for model
    * @param centers
    *   Sequence of (center_id, weight, vector) tuples
    * @return
    *   SHA-256 hash of the centers data
    */
  def writeCenters(
      spark: SparkSession,
      path: String,
      centers: Seq[(Int, Double, Vector)]
  ): String = {
    import spark.implicits._

    // Sort by center_id for deterministic ordering
    val df = centers.sortBy(_._1).toDF("center_id", "weight", "vector")

    // Write as single partition to ensure deterministic file output
    df.coalesce(1).write.mode("overwrite").parquet(s"$path/centers.parquet")

    // Compute deterministic hash
    sha256OfParquet(df)
  }

  /** Compute SHA-256 hash of Parquet data in deterministic order */
  private def sha256OfParquet(df: DataFrame): String = {
    // Order by center_id and convert to JSON for deterministic hashing
    val bytes =
      df.orderBy("center_id").toJSON.collect().mkString("\n").getBytes(StandardCharsets.UTF_8)
    sha256(bytes)
  }

  /** Write metadata JSON to disk.
    *
    * @param path
    *   Base path for model
    * @param json
    *   JSON string
    * @return
    *   SHA-256 hash of the metadata
    */
  def writeMetadata(path: String, json: String): String = {
    val p = Paths.get(s"$path/metadata.json")
    Files.createDirectories(p.getParent)
    Files.write(
      p,
      json.getBytes(StandardCharsets.UTF_8),
      StandardOpenOption.CREATE,
      StandardOpenOption.TRUNCATE_EXISTING
    )
    sha256(json.getBytes(StandardCharsets.UTF_8))
  }

  /** Write optional training summary to disk.
    *
    * @param path
    *   Base path for model
    * @param json
    *   JSON string containing training metrics
    */
  def writeSummary(path: String, json: String): Unit = {
    val p = Paths.get(s"$path/summary.json")
    Files.createDirectories(p.getParent)
    Files.write(
      p,
      json.getBytes(StandardCharsets.UTF_8),
      StandardOpenOption.CREATE,
      StandardOpenOption.TRUNCATE_EXISTING
    )
  }

  /** Read cluster centers from Parquet in deterministic order.
    *
    * @param spark
    *   SparkSession
    * @param path
    *   Base path for model
    * @return
    *   DataFrame with centers ordered by center_id
    */
  def readCenters(spark: SparkSession, path: String): DataFrame = {
    spark.read.parquet(s"$path/centers.parquet").orderBy("center_id")
  }

  /** Read metadata JSON from disk.
    *
    * @param path
    *   Base path for model
    * @return
    *   JSON string
    */
  def readMetadata(path: String): String = {
    val p = Paths.get(s"$path/metadata.json")
    new String(Files.readAllBytes(p), StandardCharsets.UTF_8)
  }

  /** Read optional training summary from disk.
    *
    * @param path
    *   Base path for model
    * @return
    *   JSON string if file exists, None otherwise
    */
  def readSummary(path: String): Option[String] = {
    val p = Paths.get(s"$path/summary.json")
    if (Files.exists(p)) {
      Some(new String(Files.readAllBytes(p), StandardCharsets.UTF_8))
    } else {
      None
    }
  }

  /** Compute SHA-256 hash of byte array */
  private def sha256(bytes: Array[Byte]): String = {
    val md = MessageDigest.getInstance("SHA-256")
    md.digest(bytes).map("%02x".format(_)).mkString
  }

  /** Get Scala binary version (e.g., "2.13" from "2.13.14") */
  def getScalaBinaryVersion: String = {
    util.Properties.versionNumberString.split("\\.").take(2).mkString(".")
  }

  /** Validate that loaded metadata is compatible with current environment.
    *
    * @param layoutVersion
    *   Layout version from loaded metadata
    * @param expectedK
    *   Expected number of clusters
    * @param expectedDim
    *   Expected dimensionality
    * @param actualCentersCount
    *   Actual number of centers loaded
    */
  def validateMetadata(
      layoutVersion: Int,
      expectedK: Int,
      expectedDim: Int,
      actualCentersCount: Int
  ): Unit = {
    require(
      layoutVersion == LayoutVersion,
      s"Incompatible layoutVersion=$layoutVersion (expected $LayoutVersion)"
    )
    require(
      actualCentersCount == expectedK,
      s"Expected k=$expectedK centers; found $actualCentersCount"
    )
  }
}
