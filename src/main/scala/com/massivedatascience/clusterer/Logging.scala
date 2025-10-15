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

import org.slf4j.{Logger, LoggerFactory}

/** Trait providing standardized logging functionality for clusterers.
  *
  * This trait eliminates the need for each class to initialize its own logger and provides helper
  * methods for common logging patterns.
  */
trait Logging extends Serializable {

  /** Lazy logger instance initialized with the class name. Marked @transient to avoid serialization
    * issues.
    */
  @transient protected lazy val logger: Logger = LoggerFactory.getLogger(getClass.getName)

  /** Log the start of a clustering operation.
    */
  protected def logClusteringStart(algorithmName: String, k: Int, maxIterations: Int): Unit = {
    logger.info(s"Starting $algorithmName with k=$k, maxIterations=$maxIterations")
  }

  /** Log the start of a clustering operation with custom parameters.
    */
  protected def logClusteringStart(algorithmName: String, params: String): Unit = {
    logger.info(s"Starting $algorithmName: $params")
  }

  /** Log iteration progress.
    */
  protected def logIteration(iteration: Int, distortion: Double, movedPoints: Int): Unit = {
    logger.info(f"Iteration $iteration: distortion=$distortion%.4f, moved=$movedPoints points")
  }

  /** Log iteration progress with custom message.
    */
  protected def logIteration(iteration: Int, message: String): Unit = {
    logger.info(s"Iteration $iteration: $message")
  }

  /** Log convergence.
    */
  protected def logConvergence(totalIterations: Int, finalDistortion: Double): Unit = {
    logger.info(
      f"Converged after $totalIterations iterations, final distortion=$finalDistortion%.4f"
    )
  }

  /** Log convergence with custom message.
    */
  protected def logConvergence(message: String): Unit = {
    logger.info(s"Converged: $message")
  }

  /** Log clustering completion with results.
    */
  protected def logClusteringComplete(
    algorithmName: String,
    k: Int,
    iterations: Int,
    distortion: Double
  ): Unit = {
    logger.info(
      f"$algorithmName completed: k=$k, iterations=$iterations, distortion=$distortion%.4f"
    )
  }

  /** Log clustering completion with custom results.
    */
  protected def logClusteringComplete(algorithmName: String, results: String): Unit = {
    logger.info(s"$algorithmName completed: $results")
  }

  /** Log configuration details.
    */
  protected def logConfig(configName: String, details: String): Unit = {
    logger.info(s"$configName: $details")
  }

  /** Log data statistics.
    */
  protected def logDataStats(numPoints: Long, numDimensions: Int): Unit = {
    logger.info(s"Data: n=$numPoints points, d=$numDimensions dimensions")
  }

  /** Log warning message.
    */
  protected def logWarning(message: String): Unit = {
    logger.warn(message)
  }

  /** Log debug message.
    */
  protected def logDebug(message: String): Unit = {
    if (logger.isDebugEnabled) {
      logger.debug(message)
    }
  }
}
