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

import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.scalatest.matchers.should.Matchers

class RowIdProviderSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("RowIdProviderSuite")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) {
        spark.stop()
      }
    } finally {
      super.afterAll()
    }
  }

  test("MonotonicRowIdProvider should add row IDs") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      ("a", 1),
      ("b", 2),
      ("c", 3)
    ).toDF("letter", "number")

    val provider = MonotonicRowIdProvider
    val result   = provider.addRowId(df, "id")

    assert(result.columns.contains("id"))
    assert(result.count() == 3)

    // Check that IDs are unique
    val ids = result.select("id").collect().map(_.getLong(0))
    assert(ids.distinct.length == 3)
  }

  test("MonotonicRowIdProvider should not duplicate ID column") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(("a", 1), ("b", 2)).toDF("letter", "number")

    val provider = MonotonicRowIdProvider
    val result1  = provider.addRowId(df, "id")
    val result2  = provider.addRowId(result1, "id") // Add again

    // Should not have duplicate columns
    assert(result2.columns.count(_ == "id") == 1)
  }

  test("MonotonicRowIdProvider should support hasRowId") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(("a", 1), ("b", 2)).toDF("letter", "number")

    val provider = MonotonicRowIdProvider

    assert(!provider.hasRowId(df, "id"))

    val withId = provider.addRowId(df, "id")
    assert(provider.hasRowId(withId, "id"))
  }

  test("MonotonicRowIdProvider should support removeRowId") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(("a", 1), ("b", 2)).toDF("letter", "number")

    val provider = MonotonicRowIdProvider
    val withId   = provider.addRowId(df, "id")
    val removed  = provider.removeRowId(withId, "id")

    assert(!removed.columns.contains("id"))
    assert(removed.columns.toSet == Set("letter", "number"))
  }

  test("MonotonicRowIdProvider idColumn should work in select") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      ("a", 1),
      ("b", 2),
      ("c", 3)
    ).toDF("letter", "number")

    val provider = MonotonicRowIdProvider
    val result   = df.select($"*", provider.idColumn.as("id"))

    assert(result.columns.contains("id"))
    assert(result.count() == 3)

    val ids = result.select("id").collect().map(_.getLong(0))
    assert(ids.distinct.length == 3)
  }

  test("ExistingColumnRowIdProvider should use existing column") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      (100L, "a", 1),
      (200L, "b", 2),
      (300L, "c", 3)
    ).toDF("existing_id", "letter", "number")

    val provider = ExistingColumnRowIdProvider("existing_id")
    val result   = provider.addRowId(df, "row_id")

    assert(result.columns.contains("row_id"))
    assert(result.columns.contains("existing_id"))

    // Verify row_id matches existing_id
    val rows = result.select("existing_id", "row_id").collect()
    rows.foreach { row =>
      assert(row.getLong(0) == row.getLong(1))
    }
  }

  test("ExistingColumnRowIdProvider should handle same column name") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      (100L, "a", 1),
      (200L, "b", 2)
    ).toDF("id", "letter", "number")

    val provider = ExistingColumnRowIdProvider("id")
    val result   = provider.addRowId(df, "id") // Use same name

    // Should not duplicate column
    assert(result.columns.count(_ == "id") == 1)
    assert(result.count() == 2)
  }

  test("ExistingColumnRowIdProvider should throw on missing column") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(("a", 1), ("b", 2)).toDF("letter", "number")

    val provider = ExistingColumnRowIdProvider("missing_col")

    intercept[IllegalArgumentException] {
      provider.addRowId(df, "row_id")
    }
  }

  test("ZipWithIndexRowIdProvider should create sequential IDs") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      ("a", 1),
      ("b", 2),
      ("c", 3),
      ("d", 4)
    ).toDF("letter", "number")

    val provider = ZipWithIndexRowIdProvider
    val result   = provider.addRowId(df, "id")

    assert(result.columns.contains("id"))
    assert(result.count() == 4)

    // Check that IDs are sequential 0, 1, 2, 3
    val ids = result.select("id").collect().map(_.getLong(0)).sorted
    assert(ids.toSeq == Seq(0, 1, 2, 3))
  }

  test("ZipWithIndexRowIdProvider should throw on idColumn") {
    val provider = ZipWithIndexRowIdProvider

    intercept[UnsupportedOperationException] {
      provider.idColumn
    }
  }

  test("ZipWithIndexRowIdProvider should not duplicate ID column") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(("a", 1), ("b", 2)).toDF("letter", "number")

    val provider = ZipWithIndexRowIdProvider
    val result1  = provider.addRowId(df, "id")
    val result2  = provider.addRowId(result1, "id")

    // Should not have duplicate columns
    assert(result2.columns.count(_ == "id") == 1)
  }

  test("RowIdProvider.default should return MonotonicRowIdProvider") {
    val provider = RowIdProvider.default
    assert(provider == MonotonicRowIdProvider)
  }

  test("RowIdProvider.monotonic should create MonotonicRowIdProvider") {
    val provider = RowIdProvider.monotonic()
    assert(provider == MonotonicRowIdProvider)
  }

  test("RowIdProvider.fromColumn should create ExistingColumnRowIdProvider") {
    val provider = RowIdProvider.fromColumn("my_id")
    assert(provider.isInstanceOf[ExistingColumnRowIdProvider])
    assert(provider.asInstanceOf[ExistingColumnRowIdProvider].existingCol == "my_id")
  }

  test("RowIdProvider.sequential should create ZipWithIndexRowIdProvider") {
    val provider = RowIdProvider.sequential()
    assert(provider == ZipWithIndexRowIdProvider)
  }

  test("MonotonicRowIdProvider should preserve existing columns") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      ("a", 1),
      ("b", 2),
      ("c", 3)
    ).toDF("letter", "number")

    val provider = MonotonicRowIdProvider
    val result   = provider.addRowId(df, "id")

    assert(result.columns.contains("letter"))
    assert(result.columns.contains("number"))
    assert(result.columns.contains("id"))
    assert(result.columns.length == 3)
  }

  test("ExistingColumnRowIdProvider idColumn should work in select") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(
      (100L, "a", 1),
      (200L, "b", 2)
    ).toDF("existing_id", "letter", "number")

    val provider = ExistingColumnRowIdProvider("existing_id")
    val result   = df.select($"*", provider.idColumn.as("row_id"))

    assert(result.columns.contains("row_id"))
    val rows = result.select("existing_id", "row_id").collect()
    rows.foreach { row =>
      assert(row.getLong(0) == row.getLong(1))
    }
  }

  test("removeRowId should be no-op if column doesn't exist") {
    val sparkSession = spark
    import sparkSession.implicits._

    val df = Seq(("a", 1), ("b", 2)).toDF("letter", "number")

    val provider = MonotonicRowIdProvider
    val result   = provider.removeRowId(df, "nonexistent")

    assert(result.columns.toSet == df.columns.toSet)
  }
}
