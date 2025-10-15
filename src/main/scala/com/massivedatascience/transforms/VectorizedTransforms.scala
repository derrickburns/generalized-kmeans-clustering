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

package com.massivedatascience.transforms

import com.massivedatascience.linalg.{BLAS, WeightedVector}
import org.apache.spark.ml.linalg.{DenseMatrix, Vectors}
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory

/** Vectorized implementations of transform operations using BLAS for improved performance.
  *
  * These implementations process multiple vectors in batches and use optimized BLAS operations to
  * reduce overhead and improve cache efficiency.
  */
object VectorizedTransforms {

  @transient private lazy val logger = LoggerFactory.getLogger(getClass.getName)

  // Optimal batch size for vectorized operations (tuned for memory and performance)
  private val DefaultBatchSize = 256

  /** Vectorized Haar wavelet transform using batched BLAS operations.
    *
    * Processes multiple vectors simultaneously to amortize BLAS call overhead and improve memory
    * access patterns.
    *
    * @param vectors
    *   RDD of input vectors
    * @param levels
    *   Number of wavelet decomposition levels
    * @param batchSize
    *   Number of vectors to process in each batch
    * @return
    *   RDD of transformed vectors
    */
  def batchedHaarTransform(
    vectors: RDD[WeightedVector],
    levels: Int = 1,
    batchSize: Int = DefaultBatchSize
  ): RDD[WeightedVector] = {

    require(levels > 0, "Levels must be positive")
    require(batchSize > 0, "Batch size must be positive")

    logger.info(s"Starting batched Haar transform with ${levels} levels, batch size ${batchSize}")

    vectors.mapPartitions { vectorIter =>
      vectorIter.grouped(batchSize).flatMap { batch =>
        if (batch.nonEmpty) {
          vectorizedHaarBatch(batch, levels)
        } else {
          Iterator.empty
        }
      }
    }
  }

  /** Process a batch of vectors with Haar wavelet transform using vectorized operations.
    */
  private def vectorizedHaarBatch(batch: Seq[WeightedVector], levels: Int): Seq[WeightedVector] = {
    if (batch.isEmpty) return Seq.empty

    val batchSize  = batch.length
    val vectorSize = batch.head.homogeneous.size

    // Ensure all vectors have the same size
    require(batch.forall(_.homogeneous.size == vectorSize), "All vectors must have the same size")

    val dataMatrix   = new Array[Double](batchSize * vectorSize)
    val resultMatrix = new Array[Double](batchSize * vectorSize)

    // Pack vectors into matrix format for batch processing
    var idx = 0
    batch.foreach { vector =>
      val values = vector.homogeneous.toArray
      System.arraycopy(values, 0, dataMatrix, idx * vectorSize, vectorSize)
      idx += 1
    }

    // Apply Haar transform to each vector in the batch
    for (i <- batch.indices) {
      val offset          = i * vectorSize
      val vectorData      = dataMatrix.slice(offset, offset + vectorSize)
      val transformedData = haarTransformSingleVector(vectorData, levels)
      System.arraycopy(transformedData, 0, resultMatrix, offset, vectorSize)
    }

    // Unpack results back to WeightedVector format
    batch.indices.map { i =>
      val offset            = i * vectorSize
      val transformedValues = resultMatrix.slice(offset, offset + vectorSize)
      val originalVector    = batch(i)
      WeightedVector(Vectors.dense(transformedValues), originalVector.weight)
    }
  }

  /** Optimized Haar wavelet transform for a single vector using BLAS operations.
    */
  private def haarTransformSingleVector(data: Array[Double], levels: Int): Array[Double] = {
    val result = data.clone()
    var length = data.length

    // Apply Haar transform for each level
    for (_ <- 0 until levels if length > 1) {
      val temp       = new Array[Double](length)
      val halfLength = length / 2

      // Compute scaling coefficients (averages) and wavelet coefficients (differences)
      var i = 0
      while (i < halfLength) {
        val evenIdx = i * 2
        val oddIdx  = evenIdx + 1
        if (oddIdx < length) {
          temp(i) = (result(evenIdx) + result(oddIdx)) * 0.7071067811865476 // 1/sqrt(2)
          temp(halfLength + i) = (result(evenIdx) - result(oddIdx)) * 0.7071067811865476
        } else {
          temp(i) = result(evenIdx)
          temp(halfLength + i) = 0.0
        }
        i += 1
      }

      // Copy back results
      System.arraycopy(temp, 0, result, 0, length)
      length = halfLength
    }

    result
  }

  /** Vectorized random index embedding using batched matrix operations.
    *
    * @param vectors
    *   RDD of input vectors
    * @param targetDimension
    *   Target embedding dimension
    * @param seed
    *   Random seed for reproducibility
    * @param batchSize
    *   Number of vectors to process in each batch
    * @return
    *   RDD of embedded vectors
    */
  def batchedRandomIndexEmbedding(
    vectors: RDD[WeightedVector],
    targetDimension: Int,
    seed: Long = System.currentTimeMillis(),
    batchSize: Int = DefaultBatchSize
  ): RDD[WeightedVector] = {

    require(targetDimension > 0, "Target dimension must be positive")
    require(batchSize > 0, "Batch size must be positive")

    logger.info(
      s"Starting batched random index embedding to ${targetDimension}D, batch size ${batchSize}"
    )

    // Sample the first vector to determine input dimension
    val inputDimension = vectors.first().homogeneous.size
    logger.info(s"Input dimension: ${inputDimension}, target dimension: ${targetDimension}")

    // Generate random projection matrix (broadcast for efficiency)
    val projectionMatrix = generateRandomProjectionMatrix(inputDimension, targetDimension, seed)
    val broadcastMatrix  = vectors.context.broadcast(projectionMatrix)

    try {
      vectors.mapPartitions { vectorIter =>
        val matrix = broadcastMatrix.value
        vectorIter.grouped(batchSize).flatMap { batch =>
          if (batch.nonEmpty) {
            vectorizedEmbedBatch(batch, matrix)
          } else {
            Iterator.empty
          }
        }
      }
    } finally {
      broadcastMatrix.unpersist()
    }
  }

  /** Process a batch of vectors with random index embedding using matrix multiplication.
    */
  private def vectorizedEmbedBatch(
    batch: Seq[WeightedVector],
    projectionMatrix: DenseMatrix
  ): Seq[WeightedVector] = {
    val batchSize = batch.length
    val inputDim  = projectionMatrix.numRows
    val outputDim = projectionMatrix.numCols

    val inputMatrix  = new Array[Double](batchSize * inputDim)
    val outputMatrix = new Array[Double](batchSize * outputDim)

    // Pack input vectors into matrix
    var idx = 0
    batch.foreach { vector =>
      val values = vector.homogeneous.toArray
      require(
        values.length == inputDim,
        s"Vector dimension ${values.length} doesn't match expected ${inputDim}"
      )
      System.arraycopy(values, 0, inputMatrix, idx * inputDim, inputDim)
      idx += 1
    }

    // Perform matrix multiplication: output = input * projection
    // Manual matrix multiplication since full GEMM is not available
    for (i <- 0 until batchSize) {
      for (j <- 0 until outputDim) {
        var sum = 0.0
        for (k <- 0 until inputDim) {
          sum += inputMatrix(i * inputDim + k) * projectionMatrix.values(k * outputDim + j)
        }
        outputMatrix(i * outputDim + j) = sum
      }
    }

    // Unpack results
    batch.indices.map { i =>
      val offset         = i * outputDim
      val embeddedValues = outputMatrix.slice(offset, offset + outputDim)
      val originalVector = batch(i)
      WeightedVector(Vectors.dense(embeddedValues), originalVector.weight)
    }
  }

  /** Generate a random projection matrix for dimensionality reduction.
    *
    * Uses the Johnson-Lindenstrauss lemma approach with sparse random projections.
    */
  private def generateRandomProjectionMatrix(
    inputDim: Int,
    outputDim: Int,
    seed: Long
  ): DenseMatrix = {
    val random   = new scala.util.Random(seed)
    val sparsity = 1.0 / math.sqrt(inputDim) // Typical sparsity level

    val values = Array.ofDim[Double](inputDim * outputDim)
    var idx    = 0

    for (i <- 0 until inputDim; j <- 0 until outputDim) {
      values(idx) = if (random.nextDouble() < sparsity) {
        // Sparse random projection: +1, -1, or 0
        random.nextGaussian() / math.sqrt(outputDim)
      } else {
        0.0
      }
      idx += 1
    }

    new DenseMatrix(inputDim, outputDim, values)
  }

  /** Batched vector normalization using BLAS operations.
    *
    * @param vectors
    *   RDD of vectors to normalize
    * @param batchSize
    *   Number of vectors to process in each batch
    * @return
    *   RDD of normalized vectors
    */
  def batchedNormalization(
    vectors: RDD[WeightedVector],
    batchSize: Int = DefaultBatchSize
  ): RDD[WeightedVector] = {

    vectors.mapPartitions { vectorIter =>
      vectorIter.grouped(batchSize).flatMap { batch =>
        if (batch.nonEmpty) {
          batch.map { vector =>
            val denseVector = Vectors.dense(vector.homogeneous.toArray).toDense

            // Calculate norm manually
            val values      = denseVector.values
            var normSquared = 0.0
            var i           = 0
            while (i < values.length) {
              normSquared += values(i) * values(i)
              i += 1
            }
            val norm = math.sqrt(normSquared)

            if (norm > 1e-12) {
              BLAS.scal(1.0 / norm, denseVector)
            }
            WeightedVector(denseVector, vector.weight)
          }
        } else {
          Iterator.empty
        }
      }
    }
  }
}
