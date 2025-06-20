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

package com.massivedatascience.clusterer.coreset

import com.massivedatascience.clusterer.BregmanPoint
import com.massivedatascience.linalg.WeightedVector
import org.apache.spark.ml.linalg.Vector

/**
 * A weighted point for core-set representation.
 * 
 * Represents a data point with an importance weight that reflects how many
 * original points this weighted point represents in the core-set approximation.
 * 
 * @param point The underlying Bregman point
 * @param importance The importance weight (typically 1/sampling_probability)
 * @param sensitivity The sensitivity score used for sampling
 */
case class WeightedPoint(
    point: BregmanPoint,
    importance: Double,
    sensitivity: Double) extends Serializable {
  
  require(importance > 0.0, s"Importance weight must be positive, got: $importance")
  require(sensitivity >= 0.0, s"Sensitivity must be non-negative, got: $sensitivity")
  
  /**
   * Get the effective weight of this point in the core-set.
   * This combines the original point weight with the importance weight.
   */
  def effectiveWeight: Double = point.weight * importance
  
  /**
   * Convert to a standard weighted vector for compatibility.
   */
  def toWeightedVector: WeightedVector = {
    WeightedVector.fromInhomogeneousWeighted(point.inhomogeneous, effectiveWeight)
  }
  
  /**
   * Create a new WeightedPoint with updated importance.
   */
  def withImportance(newImportance: Double): WeightedPoint = {
    copy(importance = newImportance)
  }
  
  /**
   * Create a new WeightedPoint with updated sensitivity.
   */
  def withSensitivity(newSensitivity: Double): WeightedPoint = {
    copy(sensitivity = newSensitivity)
  }
}

object WeightedPoint {
  
  /**
   * Create a WeightedPoint from a BregmanPoint with unit importance.
   */
  def apply(point: BregmanPoint): WeightedPoint = {
    WeightedPoint(point, importance = 1.0, sensitivity = 1.0)
  }
  
  /**
   * Create a WeightedPoint from a BregmanPoint with specified importance.
   */
  def apply(point: BregmanPoint, importance: Double): WeightedPoint = {
    WeightedPoint(point, importance, sensitivity = importance)
  }
  
  /**
   * Create a weighted point from vector components.
   */
  def apply(vector: Vector, weight: Double, f: Double, importance: Double): WeightedPoint = {
    val point = BregmanPoint(WeightedVector.fromInhomogeneousWeighted(vector, weight), f)
    WeightedPoint(point, importance, sensitivity = importance)
  }
  
  /**
   * Convert a sequence of BregmanPoints to WeightedPoints with unit importance.
   */
  def fromBregmanPoints(points: Seq[BregmanPoint]): Seq[WeightedPoint] = {
    points.map(apply(_))
  }
  
  /**
   * Extract BregmanPoints from WeightedPoints (losing importance information).
   */
  def toBregmanPoints(weightedPoints: Seq[WeightedPoint]): Seq[BregmanPoint] = {
    weightedPoints.map(_.point)
  }
}