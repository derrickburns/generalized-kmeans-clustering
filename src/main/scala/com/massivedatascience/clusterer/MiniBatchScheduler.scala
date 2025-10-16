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

import org.apache.spark.rdd.RDD

/** Strategy for sampling data in mini-batch k-means.
  *
  * This trait unifies different batch sampling strategies (full batch, fixed mini-batch, decaying samples) under a
  * single abstraction. It handles:
  * - Batch size determination per iteration
  * - Learning rate schedules for center updates
  * - Sampling strategies for data selection
  *
  * Design principles:
  * - Schedulers are stateless and iteration-aware
  * - Support both deterministic and stochastic sampling
  * - Learning rates can decay over iterations
  * - Easy to add new scheduling strategies
  *
  * Example usage:
  * {{{
  *   val scheduler = MiniBatchScheduler.fixed(fraction = 0.1)
  *   val batch = scheduler.sampleBatch(data, iteration = 0, seed = 42)
  *   val learningRate = scheduler.learningRate(iteration = 0)
  * }}}
  */
trait MiniBatchScheduler extends Serializable {

  /** Human-readable name of this scheduler */
  def name: String

  /** Sample a batch of data for the given iteration.
    *
    * @param data
    *   full dataset
    * @param iteration
    *   current iteration number (0-based)
    * @param seed
    *   random seed for reproducible sampling
    * @tparam T
    *   data point type
    * @return
    *   sampled batch
    */
  def sampleBatch[T](data: RDD[T], iteration: Int, seed: Long): RDD[T]

  /** Get learning rate for the given iteration.
    *
    * The learning rate determines how much new batch statistics affect center updates. Common patterns:
    * - Constant: Always use same rate
    * - Inverse decay: rate = 1 / (1 + iteration)
    * - Exponential decay: rate = initial * decay^iteration
    *
    * @param iteration
    *   current iteration number (0-based)
    * @return
    *   learning rate in [0, 1]
    */
  def learningRate(iteration: Int): Double

  /** Get expected batch size as fraction of full dataset.
    *
    * @param iteration
    *   current iteration number
    * @return
    *   expected fraction in (0, 1]
    */
  def batchFraction(iteration: Int): Double
}

/** Full batch scheduler - use entire dataset each iteration.
  *
  * This is standard (non-mini-batch) k-means. Each iteration uses all data points with learning rate = 1.0.
  *
  * Characteristics:
  * - Batch size: 100% of data
  * - Learning rate: 1.0 (full update)
  * - Best for: Small to medium datasets, highest accuracy
  */
case object FullBatchScheduler extends MiniBatchScheduler {
  override def name: String = "fullBatch"

  override def sampleBatch[T](data: RDD[T], iteration: Int, seed: Long): RDD[T] = {
    data // Use all data
  }

  override def learningRate(iteration: Int): Double = 1.0

  override def batchFraction(iteration: Int): Double = 1.0
}

/** Fixed mini-batch scheduler - use constant fraction of data.
  *
  * Samples a fixed fraction of data each iteration with optional learning rate decay.
  *
  * Characteristics:
  * - Batch size: Constant fraction of data
  * - Learning rate: Can decay over iterations
  * - Best for: Large datasets, trading accuracy for speed
  *
  * @param fraction
  *   fraction of data to sample per iteration (0, 1]
  * @param learningRateDecay
  *   learning rate decay strategy
  */
case class FixedMiniBatchScheduler(
  fraction: Double,
  learningRateDecay: LearningRateDecay = LearningRateDecay.Constant(1.0)
) extends MiniBatchScheduler {

  require(fraction > 0.0 && fraction <= 1.0, s"Batch fraction must be in (0, 1], got $fraction")

  override def name: String = s"fixedMiniBatch(fraction=$fraction, decay=${learningRateDecay.name})"

  override def sampleBatch[T](data: RDD[T], iteration: Int, seed: Long): RDD[T] = {
    if (fraction >= 1.0) {
      data
    } else {
      data.sample(withReplacement = false, fraction, seed + iteration)
    }
  }

  override def learningRate(iteration: Int): Double = {
    learningRateDecay.rate(iteration)
  }

  override def batchFraction(iteration: Int): Double = fraction
}

/** Decaying mini-batch scheduler - increase batch size over iterations.
  *
  * Starts with small batches for fast early iterations, then increases batch size for refinement.
  *
  * Characteristics:
  * - Batch size: Grows from minFraction to maxFraction
  * - Learning rate: Typically decays inversely
  * - Best for: Large datasets with iterative refinement
  *
  * @param minFraction
  *   initial batch fraction
  * @param maxFraction
  *   final batch fraction
  * @param growthRate
  *   how quickly to grow batch size
  * @param learningRateDecay
  *   learning rate decay strategy
  */
case class DecayingMiniBatchScheduler(
  minFraction: Double,
  maxFraction: Double = 1.0,
  growthRate: Double = 1.1,
  learningRateDecay: LearningRateDecay = LearningRateDecay.Inverse(1.0)
) extends MiniBatchScheduler {

  require(minFraction > 0.0 && minFraction <= maxFraction, s"Invalid fractions: min=$minFraction, max=$maxFraction")
  require(maxFraction <= 1.0, s"Max fraction must be <= 1.0, got $maxFraction")
  require(growthRate >= 1.0, s"Growth rate must be >= 1.0, got $growthRate")

  override def name: String = s"decayingMiniBatch(min=$minFraction, max=$maxFraction, growth=$growthRate)"

  override def sampleBatch[T](data: RDD[T], iteration: Int, seed: Long): RDD[T] = {
    val currentFraction = batchFraction(iteration)
    if (currentFraction >= 1.0) {
      data
    } else {
      data.sample(withReplacement = false, currentFraction, seed + iteration)
    }
  }

  override def learningRate(iteration: Int): Double = {
    learningRateDecay.rate(iteration)
  }

  override def batchFraction(iteration: Int): Double = {
    val grown = minFraction * math.pow(growthRate, iteration)
    math.min(grown, maxFraction)
  }
}

/** Learning rate decay strategies */
sealed trait LearningRateDecay extends Serializable {
  def name: String
  def rate(iteration: Int): Double
}

object LearningRateDecay {

  /** Constant learning rate (no decay) */
  case class Constant(rate: Double) extends LearningRateDecay {
    require(rate > 0.0 && rate <= 1.0, s"Learning rate must be in (0, 1], got $rate")

    override def name: String = s"constant($rate)"

    override def rate(iteration: Int): Double = rate
  }

  /** Inverse decay: rate = initial / (1 + iteration * decay) */
  case class Inverse(initial: Double, decay: Double = 1.0) extends LearningRateDecay {
    require(initial > 0.0 && initial <= 1.0, s"Initial rate must be in (0, 1], got $initial")
    require(decay > 0.0, s"Decay must be positive, got $decay")

    override def name: String = s"inverse(initial=$initial, decay=$decay)"

    override def rate(iteration: Int): Double = {
      initial / (1.0 + iteration * decay)
    }
  }

  /** Exponential decay: rate = initial * decay^iteration */
  case class Exponential(initial: Double, decay: Double) extends LearningRateDecay {
    require(initial > 0.0 && initial <= 1.0, s"Initial rate must be in (0, 1], got $initial")
    require(decay > 0.0 && decay <= 1.0, s"Decay must be in (0, 1], got $decay")

    override def name: String = s"exponential(initial=$initial, decay=$decay)"

    override def rate(iteration: Int): Double = {
      initial * math.pow(decay, iteration)
    }
  }

  /** Step decay: reduce rate by factor every N iterations */
  case class Step(initial: Double, factor: Double, stepSize: Int) extends LearningRateDecay {
    require(initial > 0.0 && initial <= 1.0, s"Initial rate must be in (0, 1], got $initial")
    require(factor > 0.0 && factor < 1.0, s"Factor must be in (0, 1), got $factor")
    require(stepSize > 0, s"Step size must be positive, got $stepSize")

    override def name: String = s"step(initial=$initial, factor=$factor, stepSize=$stepSize)"

    override def rate(iteration: Int): Double = {
      initial * math.pow(factor, (iteration / stepSize).toDouble)
    }
  }
}

/** Factory methods for creating mini-batch schedulers */
object MiniBatchScheduler {

  /** Full batch scheduler (standard k-means) */
  def fullBatch: MiniBatchScheduler = FullBatchScheduler

  /** Fixed mini-batch scheduler
    *
    * @param fraction
    *   fraction of data per iteration
    * @param learningRateDecay
    *   learning rate decay strategy
    */
  def fixed(
    fraction: Double,
    learningRateDecay: LearningRateDecay = LearningRateDecay.Constant(1.0)
  ): MiniBatchScheduler = {
    FixedMiniBatchScheduler(fraction, learningRateDecay)
  }

  /** Decaying mini-batch scheduler
    *
    * @param minFraction
    *   starting batch fraction
    * @param maxFraction
    *   ending batch fraction
    * @param growthRate
    *   batch growth rate per iteration
    */
  def decaying(
    minFraction: Double = 0.01,
    maxFraction: Double = 1.0,
    growthRate: Double = 1.1
  ): MiniBatchScheduler = {
    DecayingMiniBatchScheduler(minFraction, maxFraction, growthRate)
  }

  /** Default scheduler for mini-batch k-means (10% fixed) */
  def default: MiniBatchScheduler = fixed(0.1)

  /** Parse scheduler from string name
    *
    * @param name
    *   scheduler name
    * @return
    *   mini-batch scheduler
    */
  def fromString(name: String): MiniBatchScheduler = {
    val normalized = name.toLowerCase.replaceAll("[\\s-_]", "")
    normalized match {
      case "fullbatch" | "full" => FullBatchScheduler
      case "fixed" | "minibatch" => FixedMiniBatchScheduler(0.1)
      case "decaying" | "growing" => DecayingMiniBatchScheduler(0.01)
      case _ =>
        throw new IllegalArgumentException(
          s"Unknown scheduler: $name. Supported: fullBatch, fixed, decaying"
        )
    }
  }
}
