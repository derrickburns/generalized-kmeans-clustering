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

import org.apache.spark.sql.{ Column, DataFrame }
import org.apache.spark.sql.functions._

/** Provides stable row identifiers for DataFrame operations.
  *
  * This trait abstracts row ID generation, enabling different strategies for assigning unique
  * identifiers to rows in a DataFrame. Row IDs are essential for:
  *   - Efficient join operations in assignment strategies (especially squared Euclidean cross-join)
  *   - Tracking point-to-cluster assignments across iterations
  *   - Deterministic behavior in distributed operations
  *
  * Design principles:
  *   - IDs must be unique within a DataFrame
  *   - IDs should be monotonically increasing when possible for performance
  *   - ID generation should be deterministic for reproducibility
  *   - Support for both adding new ID columns and using existing columns
  *
  * Example usage:
  * {{{
  *   val provider = RowIdProvider.monotonic()
  *   val dfWithIds = provider.addRowId(df, "row_id")
  *   // Use row_id for joins and assignments
  * }}}
  */
trait RowIdProvider extends Serializable {

  /** Add a row ID column to a DataFrame.
    *
    * @param df
    *   input DataFrame
    * @param idCol
    *   name for the row ID column
    * @return
    *   DataFrame with row ID column added
    */
  def addRowId(df: DataFrame, idCol: String = "row_id"): DataFrame

  /** Get the column expression for row ID generation.
    *
    * This allows using the row ID expression in select/withColumn operations without materializing
    * an intermediate DataFrame.
    *
    * @return
    *   Column expression that generates row IDs
    */
  def idColumn: Column

  /** Check if a DataFrame already has a row ID column.
    *
    * @param df
    *   DataFrame to check
    * @param idCol
    *   row ID column name
    * @return
    *   true if DataFrame has the row ID column
    */
  def hasRowId(df: DataFrame, idCol: String = "row_id"): Boolean = {
    df.columns.contains(idCol)
  }

  /** Remove row ID column from DataFrame.
    *
    * @param df
    *   input DataFrame
    * @param idCol
    *   row ID column name to remove
    * @return
    *   DataFrame without row ID column
    */
  def removeRowId(df: DataFrame, idCol: String = "row_id"): DataFrame = {
    if (hasRowId(df, idCol)) {
      df.drop(idCol)
    } else {
      df
    }
  }
}

/** Monotonically increasing ID provider using Spark's monotonically_increasing_id().
  *
  * This implementation uses Spark's built-in monotonically_increasing_id() function, which
  * generates unique, monotonically increasing 64-bit integers. The IDs:
  *   - Are unique within a DataFrame
  *   - Are monotonically increasing within partitions
  *   - May have gaps between partitions
  *   - Are deterministic for the same input data and partitioning
  *
  * This is the recommended default for most use cases due to its efficiency and Spark optimization.
  */
case object MonotonicRowIdProvider extends RowIdProvider {

  override def addRowId(df: DataFrame, idCol: String = "row_id"): DataFrame = {
    if (hasRowId(df, idCol)) {
      df
    } else {
      df.withColumn(idCol, idColumn)
    }
  }

  override def idColumn: Column = monotonically_increasing_id()

  override def toString: String = "MonotonicRowIdProvider"
}

/** Row ID provider that uses an existing column as the row ID.
  *
  * This implementation allows reusing an existing column (e.g., a primary key or index column) as
  * the row ID, avoiding the overhead of generating new IDs.
  *
  * @param existingCol
  *   name of existing column to use as row ID
  */
case class ExistingColumnRowIdProvider(existingCol: String) extends RowIdProvider {

  override def addRowId(df: DataFrame, idCol: String = "row_id"): DataFrame = {
    require(df.columns.contains(existingCol), s"DataFrame does not contain column: $existingCol")

    if (idCol == existingCol) {
      // No need to add, already using the right column
      df
    } else if (hasRowId(df, idCol)) {
      // Already has the target column, keep it
      df
    } else {
      // Alias the existing column
      df.withColumn(idCol, col(existingCol))
    }
  }

  override def idColumn: Column = col(existingCol)

  override def toString: String = s"ExistingColumnRowIdProvider($existingCol)"
}

/** Row ID provider using zipWithIndex for sequential 0-based IDs.
  *
  * This implementation uses RDD.zipWithIndex to generate sequential, 0-based row IDs. Unlike
  * monotonically_increasing_id, this produces contiguous IDs starting from 0, but requires a
  * shuffle operation to maintain global ordering.
  *
  * Characteristics:
  *   - IDs are sequential: 0, 1, 2, ..., n-1
  *   - No gaps in ID sequence
  *   - Requires shuffle for global ordering (slower than monotonic IDs)
  *   - Deterministic for the same input data
  *
  * Use this when you need sequential IDs starting from 0, such as for array indexing or when
  * interoperating with systems that expect contiguous IDs.
  */
case object ZipWithIndexRowIdProvider extends RowIdProvider {

  override def addRowId(df: DataFrame, idCol: String = "row_id"): DataFrame = {
    if (hasRowId(df, idCol)) {
      df
    } else {
      val spark = df.sparkSession

      // Convert to RDD, zipWithIndex, and convert back using Row objects
      val schema      = df.schema
      val rowsWithIds = df.rdd.zipWithIndex().map { case (row, idx) =>
        org.apache.spark.sql.Row.fromSeq(idx +: row.toSeq)
      }

      // Create new schema with ID column first
      val newSchema = org.apache.spark.sql.types.StructType(
        org.apache.spark.sql.types
          .StructField(idCol, org.apache.spark.sql.types.LongType, nullable = false) +:
          schema.fields
      )

      spark.createDataFrame(rowsWithIds, newSchema)
    }
  }

  override def idColumn: Column = {
    // This implementation requires access to the RDD, so we can't provide a pure column expression
    // Callers should use addRowId instead
    throw new UnsupportedOperationException(
      "ZipWithIndexRowIdProvider does not support idColumn. Use addRowId instead."
    )
  }

  override def toString: String = "ZipWithIndexRowIdProvider"
}

/** Factory methods for creating RowIdProvider instances. */
object RowIdProvider {

  /** Default row ID provider using monotonically increasing IDs.
    *
    * This is the recommended provider for most use cases.
    */
  def default: RowIdProvider = MonotonicRowIdProvider

  /** Monotonically increasing ID provider.
    *
    * Generates unique, monotonically increasing IDs using Spark's built-in function. IDs are
    * monotonic within partitions but may have gaps between partitions.
    */
  def monotonic(): RowIdProvider = MonotonicRowIdProvider

  /** Row ID provider using an existing column.
    *
    * @param columnName
    *   name of existing column to use as row ID
    */
  def fromColumn(columnName: String): RowIdProvider = ExistingColumnRowIdProvider(columnName)

  /** Sequential row ID provider using zipWithIndex.
    *
    * Generates sequential, 0-based IDs. Slower than monotonic due to shuffle requirement, but
    * produces contiguous IDs.
    */
  def sequential(): RowIdProvider = ZipWithIndexRowIdProvider
}
