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

class MiniBatchSchedulerSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("MiniBatchSchedulerSuite")
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

  test("FullBatchScheduler should use all data") {
    val scheduler = FullBatchScheduler

    assert(scheduler.name == "fullBatch")
    assert(scheduler.batchFraction(0) == 1.0)
    assert(scheduler.batchFraction(10) == 1.0)
    assert(scheduler.learningRate(0) == 1.0)
    assert(scheduler.learningRate(10) == 1.0)

    val data  = spark.sparkContext.parallelize(1 to 100)
    val batch = scheduler.sampleBatch(data, iteration = 0, seed = 42)

    assert(batch.count() == 100)
  }

  test("FixedMiniBatchScheduler should sample constant fraction") {
    val scheduler = FixedMiniBatchScheduler(fraction = 0.5)

    assert(scheduler.name.contains("fixedMiniBatch"))
    assert(scheduler.batchFraction(0) == 0.5)
    assert(scheduler.batchFraction(10) == 0.5)

    val data  = spark.sparkContext.parallelize(1 to 1000)
    val batch = scheduler.sampleBatch(data, iteration = 0, seed = 42)

    // Should sample approximately 50% (allow some variance)
    val count = batch.count()
    assert(count >= 400 && count <= 600, s"Expected ~500, got $count")
  }

  test("FixedMiniBatchScheduler with fraction=1.0 should use all data") {
    val scheduler = FixedMiniBatchScheduler(fraction = 1.0)

    val data  = spark.sparkContext.parallelize(1 to 100)
    val batch = scheduler.sampleBatch(data, iteration = 0, seed = 42)

    assert(batch.count() == 100)
  }

  test("FixedMiniBatchScheduler should validate fraction") {
    intercept[IllegalArgumentException] {
      FixedMiniBatchScheduler(fraction = 0.0)
    }

    intercept[IllegalArgumentException] {
      FixedMiniBatchScheduler(fraction = 1.5)
    }

    intercept[IllegalArgumentException] {
      FixedMiniBatchScheduler(fraction = -0.1)
    }
  }

  test("DecayingMiniBatchScheduler should grow batch size") {
    val scheduler = DecayingMiniBatchScheduler(
      minFraction = 0.1,
      maxFraction = 1.0,
      growthRate = 1.5
    )

    assert(scheduler.batchFraction(0) == 0.1)
    val iter1 = scheduler.batchFraction(1)
    val iter2 = scheduler.batchFraction(2)

    assert(iter1 > 0.1 && iter1 < 1.0)
    assert(iter2 > iter1)
    assert(scheduler.batchFraction(100) == 1.0) // Eventually reaches max
  }

  test("DecayingMiniBatchScheduler should cap at maxFraction") {
    val scheduler = DecayingMiniBatchScheduler(
      minFraction = 0.5,
      maxFraction = 0.8,
      growthRate = 2.0
    )

    assert(scheduler.batchFraction(0) == 0.5)
    assert(scheduler.batchFraction(10) == 0.8) // Capped
  }

  test("DecayingMiniBatchScheduler should validate parameters") {
    intercept[IllegalArgumentException] {
      DecayingMiniBatchScheduler(minFraction = 0.0)
    }

    intercept[IllegalArgumentException] {
      DecayingMiniBatchScheduler(minFraction = 0.8, maxFraction = 0.5) // min > max
    }

    intercept[IllegalArgumentException] {
      DecayingMiniBatchScheduler(minFraction = 0.1, maxFraction = 1.5) // max > 1
    }

    intercept[IllegalArgumentException] {
      DecayingMiniBatchScheduler(minFraction = 0.1, growthRate = 0.9) // growth < 1
    }
  }

  test("LearningRateDecay.Constant should return constant rate") {
    val decay = LearningRateDecay.Constant(0.5)

    assert(decay.name == "constant(0.5)")
    assert(decay.rate(0) == 0.5)
    assert(decay.rate(10) == 0.5)
    assert(decay.rate(100) == 0.5)
  }

  test("LearningRateDecay.Constant should validate rate") {
    intercept[IllegalArgumentException] {
      LearningRateDecay.Constant(0.0)
    }

    intercept[IllegalArgumentException] {
      LearningRateDecay.Constant(1.5)
    }
  }

  test("LearningRateDecay.Inverse should decay inversely") {
    val decay = LearningRateDecay.Inverse(initial = 1.0, decay = 1.0)

    assert(decay.rate(0) == 1.0)
    assert(decay.rate(1) == 0.5)
    assert(decay.rate(9) == 0.1)

    // Should always be positive
    assert(decay.rate(1000) > 0.0)
  }

  test("LearningRateDecay.Inverse should validate parameters") {
    intercept[IllegalArgumentException] {
      LearningRateDecay.Inverse(initial = 0.0)
    }

    intercept[IllegalArgumentException] {
      LearningRateDecay.Inverse(initial = 1.0, decay = 0.0)
    }
  }

  test("LearningRateDecay.Exponential should decay exponentially") {
    val decay = LearningRateDecay.Exponential(initial = 1.0, decay = 0.9)

    assert(decay.rate(0) == 1.0)
    assert(math.abs(decay.rate(1) - 0.9) < 0.001)
    assert(math.abs(decay.rate(2) - 0.81) < 0.001)

    // Should approach zero but never reach it
    assert(decay.rate(100) > 0.0)
    assert(decay.rate(100) < 0.01)
  }

  test("LearningRateDecay.Exponential should validate parameters") {
    intercept[IllegalArgumentException] {
      LearningRateDecay.Exponential(initial = 1.5, decay = 0.9)
    }

    intercept[IllegalArgumentException] {
      LearningRateDecay.Exponential(initial = 1.0, decay = 1.1)
    }
  }

  test("LearningRateDecay.Step should decay in steps") {
    val decay = LearningRateDecay.Step(initial = 1.0, factor = 0.5, stepSize = 10)

    assert(decay.rate(0) == 1.0)
    assert(decay.rate(5) == 1.0)   // Still in first step
    assert(decay.rate(9) == 1.0)
    assert(decay.rate(10) == 0.5)  // First step down
    assert(decay.rate(15) == 0.5)
    assert(decay.rate(20) == 0.25) // Second step down
  }

  test("LearningRateDecay.Step should validate parameters") {
    intercept[IllegalArgumentException] {
      LearningRateDecay.Step(initial = 1.0, factor = 1.1, stepSize = 10) // factor >= 1
    }

    intercept[IllegalArgumentException] {
      LearningRateDecay.Step(initial = 1.0, factor = 0.5, stepSize = 0) // stepSize <= 0
    }
  }

  test("FixedMiniBatchScheduler with custom decay should use it") {
    val scheduler = FixedMiniBatchScheduler(
      fraction = 0.5,
      learningRateDecay = LearningRateDecay.Inverse(1.0)
    )

    assert(scheduler.learningRate(0) == 1.0)
    assert(scheduler.learningRate(1) == 0.5)
  }

  test("MiniBatchScheduler.fromString should parse scheduler names") {
    assert(MiniBatchScheduler.fromString("fullBatch") == FullBatchScheduler)
    assert(MiniBatchScheduler.fromString("fixed").isInstanceOf[FixedMiniBatchScheduler])
    assert(MiniBatchScheduler.fromString("decaying").isInstanceOf[DecayingMiniBatchScheduler])
  }

  test("MiniBatchScheduler.fromString should be case-insensitive") {
    assert(MiniBatchScheduler.fromString("FULLBATCH") == FullBatchScheduler)
    assert(MiniBatchScheduler.fromString("Fixed").isInstanceOf[FixedMiniBatchScheduler])
  }

  test("MiniBatchScheduler.fromString should normalize names") {
    assert(MiniBatchScheduler.fromString("full-batch") == FullBatchScheduler)
    assert(MiniBatchScheduler.fromString("mini_batch").isInstanceOf[FixedMiniBatchScheduler])
  }

  test("MiniBatchScheduler.fromString should throw on unknown scheduler") {
    intercept[IllegalArgumentException] {
      MiniBatchScheduler.fromString("unknown_scheduler")
    }
  }

  test("MiniBatchScheduler factory methods should work") {
    assert(MiniBatchScheduler.fullBatch == FullBatchScheduler)
    assert(MiniBatchScheduler.fixed(0.2).isInstanceOf[FixedMiniBatchScheduler])
    assert(MiniBatchScheduler.decaying().isInstanceOf[DecayingMiniBatchScheduler])
    assert(MiniBatchScheduler.default.isInstanceOf[FixedMiniBatchScheduler])
  }

  test("MiniBatchScheduler should be serializable") {
    val scheduler: MiniBatchScheduler = FixedMiniBatchScheduler(0.5)

    val stream = new java.io.ByteArrayOutputStream()
    val oos    = new java.io.ObjectOutputStream(stream)
    oos.writeObject(scheduler)
    oos.close()

    val bytes = stream.toByteArray
    assert(bytes.nonEmpty)
  }

  test("MiniBatchScheduler with same seed should be deterministic") {
    val scheduler = FixedMiniBatchScheduler(0.5)
    val data      = spark.sparkContext.parallelize(1 to 1000)

    val batch1 = scheduler.sampleBatch(data, iteration = 0, seed = 123).collect().sorted
    val batch2 = scheduler.sampleBatch(data, iteration = 0, seed = 123).collect().sorted

    assert(batch1.toSeq == batch2.toSeq)
  }

  test("MiniBatchScheduler with different iterations should vary") {
    val scheduler = FixedMiniBatchScheduler(0.5)
    val data      = spark.sparkContext.parallelize(1 to 1000)

    val batch0 = scheduler.sampleBatch(data, iteration = 0, seed = 123).collect().sorted
    val batch1 = scheduler.sampleBatch(data, iteration = 1, seed = 123).collect().sorted

    // Different iterations should produce different samples (seed + iteration)
    assert(batch0.toSeq != batch1.toSeq)
  }

  test("DecayingMiniBatchScheduler should sample growing batches") {
    val scheduler = DecayingMiniBatchScheduler(minFraction = 0.1, growthRate = 2.0)
    val data      = spark.sparkContext.parallelize(1 to 1000)

    val batch0 = scheduler.sampleBatch(data, iteration = 0, seed = 42)
    val batch2 = scheduler.sampleBatch(data, iteration = 2, seed = 42)

    val count0 = batch0.count()
    val count2 = batch2.count()

    assert(count2 > count0, s"Expected batch to grow: iter0=$count0, iter2=$count2")
  }

  test("LearningRateDecay should be serializable") {
    val decay: LearningRateDecay = LearningRateDecay.Inverse(1.0)

    val stream = new java.io.ByteArrayOutputStream()
    val oos    = new java.io.ObjectOutputStream(stream)
    oos.writeObject(decay)
    oos.close()

    val bytes = stream.toByteArray
    assert(bytes.nonEmpty)
  }
}
