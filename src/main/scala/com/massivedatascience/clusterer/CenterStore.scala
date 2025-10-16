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

package com.massivedatascience.clusterer

import com.massivedatascience.linalg.WeightedVector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector, Vectors}

/** Abstraction for storing and managing cluster centers.
  *
  * This trait provides a stable, testable API for center management, replacing direct array manipulation. Key benefits:
  * - Stable ordering guarantees for deterministic behavior
  * - Support for both array and DataFrame representations
  * - Encapsulation of center lifecycle (creation, update, filtering)
  * - Easy to mock for testing
  *
  * Design principles:
  * - Centers are indexed 0 to (count-1) with stable ordering
  * - Immutable API: operations return new CenterStore instances
  * - Support for empty center filtering (common in k-means)
  *
  * Example usage:
  * {{{
  *   val store = CenterStore.fromArray(initialCenters)
  *   val filtered = store.filterNonEmpty()
  *   val df = store.toDataFrame(spark, "cluster_id", "center")
  * }}}
  */
trait CenterStore extends Serializable {

  /** Number of centers currently stored. */
  def count: Int

  /** Get center at given index (0-based).
    *
    * @param index
    *   center index (must be >= 0 and < count)
    * @return
    *   center at index
    * @throws IndexOutOfBoundsException
    *   if index is invalid
    */
  def apply(index: Int): BregmanCenter

  /** Get all centers as an indexed sequence.
    *
    * @return
    *   sequence of all centers in stable order
    */
  def toSeq: IndexedSeq[BregmanCenter]

  /** Get all centers as an array (for compatibility with existing code).
    *
    * @return
    *   array of all centers
    */
  def toArray: Array[BregmanCenter] = toSeq.toArray

  /** Update center at given index.
    *
    * @param index
    *   center index to update
    * @param center
    *   new center value
    * @return
    *   new CenterStore with updated center
    */
  def updated(index: Int, center: BregmanCenter): CenterStore

  /** Filter out empty centers (zero weight).
    *
    * @return
    *   new CenterStore containing only non-empty centers
    */
  def filterNonEmpty(): CenterStore

  /** Check if center at given index is empty (zero weight).
    *
    * @param index
    *   center index
    * @return
    *   true if center has zero weight
    */
  def isEmpty(index: Int): Boolean = apply(index).weight <= 0.0

  /** Convert centers to DataFrame representation.
    *
    * @param spark
    *   Spark session
    * @param idCol
    *   name of cluster ID column
    * @param centerCol
    *   name of center vector column
    * @return
    *   DataFrame with (id, center) rows
    */
  def toDataFrame(spark: org.apache.spark.sql.SparkSession, idCol: String = "cluster_id", centerCol: String = "center"): DataFrame

  /** Map a function over all centers.
    *
    * @param f
    *   function to apply to each center
    * @return
    *   new CenterStore with transformed centers
    */
  def map(f: BregmanCenter => BregmanCenter): CenterStore

  /** Fold over all centers.
    *
    * @param z
    *   initial value
    * @param op
    *   combining function
    * @tparam A
    *   result type
    * @return
    *   accumulated result
    */
  def foldLeft[A](z: A)(op: (A, BregmanCenter) => A): A = toSeq.foldLeft(z)(op)

  /** Check if store is empty.
    *
    * @return
    *   true if no centers stored
    */
  def nonEmpty: Boolean = count > 0
}

/** Simple array-based implementation of CenterStore.
  *
  * This implementation wraps an IndexedSeq of centers and provides stable indexing. It's the default implementation for
  * most use cases.
  *
  * @param centers
  *   indexed sequence of centers in stable order
  */
case class ArrayCenterStore(centers: IndexedSeq[BregmanCenter]) extends CenterStore {

  require(centers != null, "Centers cannot be null")

  override def count: Int = centers.length

  override def apply(index: Int): BregmanCenter = {
    require(index >= 0 && index < count, s"Index $index out of bounds [0, $count)")
    centers(index)
  }

  override def toSeq: IndexedSeq[BregmanCenter] = centers

  override def updated(index: Int, center: BregmanCenter): CenterStore = {
    require(index >= 0 && index < count, s"Index $index out of bounds [0, $count)")
    ArrayCenterStore(centers.updated(index, center))
  }

  override def filterNonEmpty(): CenterStore = {
    ArrayCenterStore(centers.filter(_.weight > 0.0))
  }

  override def toDataFrame(
    spark: org.apache.spark.sql.SparkSession,
    idCol: String = "cluster_id",
    centerCol: String = "center"
  ): DataFrame = {
    import spark.implicits._

    val data = centers.zipWithIndex.map { case (center, idx) =>
      (idx, Vectors.dense(center.inhomogeneous.toArray))
    }

    data.toDF(idCol, centerCol)
  }

  override def map(f: BregmanCenter => BregmanCenter): CenterStore = {
    ArrayCenterStore(centers.map(f))
  }

  override def toString: String = s"ArrayCenterStore(count=$count)"
}

/** Factory methods for creating CenterStore instances. */
object CenterStore {

  /** Create CenterStore from array of centers.
    *
    * @param centers
    *   array of centers
    * @return
    *   new ArrayCenterStore
    */
  def fromArray(centers: Array[BregmanCenter]): CenterStore = {
    ArrayCenterStore(centers.toIndexedSeq)
  }

  /** Create CenterStore from sequence of centers.
    *
    * @param centers
    *   sequence of centers
    * @return
    *   new ArrayCenterStore
    */
  def fromSeq(centers: IndexedSeq[BregmanCenter]): CenterStore = {
    ArrayCenterStore(centers)
  }

  /** Create empty CenterStore. */
  def empty: CenterStore = ArrayCenterStore(IndexedSeq.empty)

  /** Create CenterStore from DataFrame.
    *
    * Expects DataFrame with (cluster_id, center) schema where center is a Vector.
    *
    * @param df
    *   input DataFrame
    * @param ops
    *   Bregman point operations for creating centers
    * @param idCol
    *   name of cluster ID column
    * @param centerCol
    *   name of center vector column
    * @return
    *   new CenterStore with centers from DataFrame
    */
  def fromDataFrame(
    df: DataFrame,
    ops: BregmanPointOps,
    idCol: String = "cluster_id",
    centerCol: String = "center"
  ): CenterStore = {
    import df.sparkSession.implicits._

    val centers = df
      .select(col(idCol), col(centerCol))
      .orderBy(col(idCol))
      .as[(Int, Vector)]
      .collect()
      .map { case (_, vector) =>
        // Create a center with weight 1.0 from the vector
        val weightedVec = WeightedVector.fromInhomogeneousWeighted(vector, 1.0)
        ops.toCenter(weightedVec)
      }

    ArrayCenterStore(centers.toIndexedSeq)
  }
}
