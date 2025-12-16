/*
 * Licensed to the Massive Data Science and Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Massive Data Science and Derrick R. Burns licenses this file to You under the
 * Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
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

package com.massivedatascience.divergence

import com.massivedatascience.clusterer.ml.df.BregmanKernel
import com.massivedatascience.linalg.BLAS._
import org.apache.spark.ml.linalg.Vector

/** Adapters to bridge between BregmanFunction and existing API interfaces.
  *
  * This enables gradual migration to the unified BregmanFunction trait while
  * maintaining backwards compatibility with:
  *   - BregmanKernel (DataFrame API)
  *   - BregmanDivergence (RDD API)
  */
object BregmanFunctionAdapter {

  /** Wrap a BregmanFunction to implement the BregmanKernel interface.
    *
    * This allows DataFrame API code to use the unified BregmanFunction implementations.
    *
    * @param func
    *   the BregmanFunction to wrap
    * @return
    *   a BregmanKernel that delegates to the function
    */
  def asKernel(func: BregmanFunction): BregmanKernel = new BregmanKernel {
    override def grad(x: Vector): Vector                    = func.gradF(x)
    override def invGrad(theta: Vector): Vector             = func.invGradF(theta)
    override def divergence(x: Vector, mu: Vector): Double  = func.divergence(x, mu)
    override def validate(x: Vector): Boolean               = func.validate(x)
    override def name: String                               = func.name
    override def supportsExpressionOptimization: Boolean    = func.supportsExpressionOptimization
  }

  /** Wrap a BregmanFunction to implement the legacy BregmanDivergence interface.
    *
    * This allows RDD API code to use the unified BregmanFunction implementations.
    *
    * @param func
    *   the BregmanFunction to wrap
    * @return
    *   a BregmanDivergence that delegates to the function
    */
  def asDivergence(func: BregmanFunction): BregmanDivergence = new BregmanDivergence {
    override def convex(v: Vector): Double = func.F(v)

    override def gradientOfConvex(v: Vector): Vector = func.gradF(v)

    override def convexHomogeneous(v: Vector, w: Double): Double = {
      if (w == 0.0) {
        // Handle zero weight gracefully for Euclidean
        if (func.supportsExpressionOptimization) 0.0
        else throw new IllegalArgumentException("Weight must be nonzero")
      } else {
        val scaled = v.copy
        scal(1.0 / w, scaled)
        func.F(scaled)
      }
    }

    override def gradientOfConvexHomogeneous(v: Vector, w: Double): Vector = {
      require(w != 0.0, "Weight must be nonzero for gradient computation")
      val scaled = v.copy
      scal(1.0 / w, scaled)
      func.gradF(scaled)
    }
  }

  /** Wrap a BregmanKernel to implement the BregmanFunction interface.
    *
    * This allows existing BregmanKernel implementations to be used where
    * BregmanFunction is expected.
    *
    * @param kernel
    *   the BregmanKernel to wrap
    * @return
    *   a BregmanFunction that delegates to the kernel
    */
  def fromKernel(kernel: BregmanKernel): BregmanFunction = new BregmanFunction {
    override def F(x: Vector): Double = {
      // Compute F(x) from divergence: F(x) = D(x, 0) when grad(0) = 0
      // This is an approximation - for accurate F, use native BregmanFunction
      0.0 // Not directly available from kernel interface
    }

    override def gradF(x: Vector): Vector                     = kernel.grad(x)
    override def invGradF(theta: Vector): Vector              = kernel.invGrad(theta)
    override def divergence(x: Vector, y: Vector): Double     = kernel.divergence(x, y)
    override def validate(x: Vector): Boolean                 = kernel.validate(x)
    override def name: String                                 = kernel.name
    override def supportsExpressionOptimization: Boolean      = kernel.supportsExpressionOptimization
  }

  /** Wrap a BregmanDivergence to implement the BregmanFunction interface.
    *
    * This allows existing BregmanDivergence implementations to be used where
    * BregmanFunction is expected.
    *
    * @param div
    *   the BregmanDivergence to wrap
    * @return
    *   a BregmanFunction that delegates to the divergence
    */
  def fromDivergence(div: BregmanDivergence): BregmanFunction = new BregmanFunction {
    override def F(x: Vector): Double                    = div.convex(x)
    override def gradF(x: Vector): Vector                = div.gradientOfConvex(x)
    override def invGradF(theta: Vector): Vector         = {
      // BregmanDivergence doesn't have invGrad - approximate via Newton's method
      // or return identity (will be handled by specific implementations)
      theta
    }
    override def divergence(x: Vector, y: Vector): Double = {
      // Use default divergence formula from F and gradF
      val fx      = F(x)
      val fy      = F(y)
      val gradY   = gradF(y)
      val xArr    = x.toArray
      val yArr    = y.toArray
      val gradArr = gradY.toArray

      var dot = 0.0
      var i   = 0
      while (i < xArr.length) {
        dot += gradArr(i) * (xArr(i) - yArr(i))
        i += 1
      }

      fx - fy - dot
    }
    override def validate(x: Vector): Boolean              = true // BregmanDivergence doesn't have validate
    override def name: String                              = div.getClass.getSimpleName
    override def supportsExpressionOptimization: Boolean   = {
      div == SquaredEuclideanDistanceDivergence
    }
  }
}
