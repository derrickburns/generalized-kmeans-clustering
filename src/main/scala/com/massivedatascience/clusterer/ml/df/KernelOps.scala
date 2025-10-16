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

package com.massivedatascience.clusterer.ml.df

/** Typeclass defining kernel (divergence) capabilities and optimization hints.
  *
  * This typeclass replaces string-based kernel switching with compile-time type-safe kernel properties. It provides:
  * - Kernel capabilities (e.g., supports squared Euclidean fast path)
  * - Valid feature transforms for each kernel
  * - Optimization hints (broadcast thresholds, assignment strategies)
  * - Compile-time safety for kernel operations
  *
  * Design principles:
  * - Capabilities are boolean flags, not runtime checks
  * - Transform compatibility is explicit, not inferred
  * - Optimization hints guide auto-selection
  * - Each kernel has exactly one KernelOps instance
  *
  * Example usage:
  * {{{
  *   val kernel = KernelOps.squaredEuclidean
  *   if (kernel.supportsSEFastPath) {
  *     // Use optimized squared Euclidean assignment
  *   }
  *   if (kernel.isTransformSafe(transform)) {
  *     // Safe to apply transform
  *   }
  * }}}
  */
trait KernelOps extends Serializable {

  /** Human-readable kernel name (e.g., "squaredEuclidean", "kl") */
  def name: String

  /** Whether this kernel supports the squared Euclidean fast path.
    *
    * The SE fast path uses cross-join with broadcast centers for efficient distance computation. Only valid for
    * kernels where distance can be decomposed as: d(x,c) = ||x||^2 + ||c||^2 - 2*x'c
    */
  def supportsSEFastPath: Boolean

  /** Whether this kernel requires strictly positive features.
    *
    * Kernels like KL divergence are only defined for positive values and will produce NaN/Infinity for non-positive
    * inputs.
    */
  def requiresPositiveFeatures: Boolean

  /** Whether this kernel is symmetric: d(x, y) = d(y, x) */
  def isSymmetric: Boolean

  /** Recommended broadcast threshold for centers (number of centers).
    *
    * Below this threshold, broadcast centers for assignment. Above, use other strategies.
    */
  def defaultBroadcastThreshold: Int

  /** Check if a feature transform is safe for this kernel.
    *
    * @param transform
    *   the transform to check
    * @return
    *   true if transform is compatible with kernel
    */
  def isTransformSafe(transform: FeatureTransform): Boolean = {
    transform.compatibleWith(name)
  }

  /** Recommended assignment strategy for this kernel given number of centers.
    *
    * @param numCenters
    *   number of cluster centers
    * @return
    *   recommended assignment plan
    */
  def recommendedAssignment(numCenters: Int): String = {
    if (supportsSEFastPath && numCenters < defaultBroadcastThreshold) {
      "crossJoin"
    } else {
      "rddMap"
    }
  }

  /** Whether this kernel benefits from feature normalization.
    *
    * Some kernels (e.g., cosine via Euclidean) require normalization, others benefit from it.
    */
  def benefitsFromNormalization: Boolean = false
}

/** Squared Euclidean distance kernel: d(x,y) = ||x - y||^2 */
case object SquaredEuclideanDescriptor extends KernelOps {
  override def name: String = "squaredEuclidean"

  override def supportsSEFastPath: Boolean = true

  override def requiresPositiveFeatures: Boolean = false

  override def isSymmetric: Boolean = true

  override def defaultBroadcastThreshold: Int = 1000

  override def isTransformSafe(transform: FeatureTransform): Boolean = {
    // SE works with most transforms
    true
  }
}

/** Kullback-Leibler divergence kernel: d(x||y) = sum(x_i * log(x_i / y_i)) */
case object KLDivergenceDescriptor extends KernelOps {
  override def name: String = "kl"

  override def supportsSEFastPath: Boolean = false

  override def requiresPositiveFeatures: Boolean = true

  override def isSymmetric: Boolean = false

  override def defaultBroadcastThreshold: Int = 500

  override def isTransformSafe(transform: FeatureTransform): Boolean = {
    transform.compatibleWith(name)
  }
}

/** Generalized I-divergence kernel (unnormalized KL) */
case object GeneralizedIDivergenceDescriptor extends KernelOps {
  override def name: String = "generalizedI"

  override def supportsSEFastPath: Boolean = false

  override def requiresPositiveFeatures: Boolean = true

  override def isSymmetric: Boolean = false

  override def defaultBroadcastThreshold: Int = 500

  override def isTransformSafe(transform: FeatureTransform): Boolean = {
    transform.compatibleWith(name)
  }
}

/** Itakura-Saito divergence kernel */
case object ItakuraSaitoDescriptor extends KernelOps {
  override def name: String = "itakuraSaito"

  override def supportsSEFastPath: Boolean = false

  override def requiresPositiveFeatures: Boolean = true

  override def isSymmetric: Boolean = false

  override def defaultBroadcastThreshold: Int = 500

  override def isTransformSafe(transform: FeatureTransform): Boolean = {
    transform.compatibleWith(name)
  }
}

/** Logistic loss divergence kernel */
case object LogisticLossDescriptor extends KernelOps {
  override def name: String = "logisticLoss"

  override def supportsSEFastPath: Boolean = false

  override def requiresPositiveFeatures: Boolean = false

  override def isSymmetric: Boolean = false

  override def defaultBroadcastThreshold: Int = 500
}

/** Cosine distance kernel (implemented via normalized Euclidean) */
case object CosineDistanceDescriptor extends KernelOps {
  override def name: String = "cosine"

  override def supportsSEFastPath: Boolean = true // After normalization

  override def requiresPositiveFeatures: Boolean = false

  override def isSymmetric: Boolean = true

  override def defaultBroadcastThreshold: Int = 1000

  override def benefitsFromNormalization: Boolean = true

  override def isTransformSafe(transform: FeatureTransform): Boolean = {
    // Cosine requires L2 normalization
    transform match {
      case _: NormalizeL2Transform => true
      case _: NoOpTransform.type   => false // Missing normalization
      case _                       => false
    }
  }
}

/** Manhattan (L1) distance kernel */
case object ManhattanDistanceDescriptor extends KernelOps {
  override def name: String = "manhattan"

  override def supportsSEFastPath: Boolean = false

  override def requiresPositiveFeatures: Boolean = false

  override def isSymmetric: Boolean = true

  override def defaultBroadcastThreshold: Int = 800
}

/** Factory methods and kernel registry */
object KernelOps {

  /** Get kernel ops by name.
    *
    * @param kernelName
    *   kernel name (case-insensitive, normalized)
    * @return
    *   KernelOps instance
    * @throws IllegalArgumentException
    *   if kernel name is unknown
    */
  def forName(kernelName: String): KernelOps = {
    val normalized = kernelName.toLowerCase.replaceAll("[\\s-_]", "")
    normalized match {
      case "squaredeuclidean" | "euclidean" | "se" => SquaredEuclideanDescriptor
      case "kl" | "kullbackleibler"                 => KLDivergenceDescriptor
      case "generalizedi" | "gi"                    => GeneralizedIDivergenceDescriptor
      case "itakurasaito" | "is"                   => ItakuraSaitoDescriptor
      case "logisticloss"                           => LogisticLossDescriptor
      case "cosine"                                 => CosineDistanceDescriptor
      case "manhattan" | "l1"                       => ManhattanDistanceDescriptor
      case _ =>
        throw new IllegalArgumentException(
          s"Unknown kernel: $kernelName. Supported: squaredEuclidean, kl, generalizedI, itakuraSaito, logisticLoss, cosine, manhattan"
        )
    }
  }

  /** All available kernel ops */
  def all: Seq[KernelOps] = Seq(
    SquaredEuclideanDescriptor,
    KLDivergenceDescriptor,
    GeneralizedIDivergenceDescriptor,
    ItakuraSaitoDescriptor,
    LogisticLossDescriptor,
    CosineDistanceDescriptor,
    ManhattanDistanceDescriptor
  )

  /** Squared Euclidean kernel (default) */
  def squaredEuclidean: KernelOps = SquaredEuclideanDescriptor

  /** KL divergence kernel */
  def kl: KernelOps = KLDivergenceDescriptor

  /** Generalized I-divergence kernel */
  def generalizedI: KernelOps = GeneralizedIDivergenceDescriptor

  /** Itakura-Saito kernel */
  def itakuraSaito: KernelOps = ItakuraSaitoDescriptor

  /** Logistic loss kernel */
  def logisticLoss: KernelOps = LogisticLossDescriptor

  /** Cosine distance kernel */
  def cosine: KernelOps = CosineDistanceDescriptor

  /** Manhattan distance kernel */
  def manhattan: KernelOps = ManhattanDistanceDescriptor

  /** Validate kernel/transform compatibility.
    *
    * @param kernel
    *   the kernel
    * @param transform
    *   the transform
    * @throws IllegalArgumentException
    *   if incompatible
    */
  def validateCompatibility(kernel: KernelOps, transform: FeatureTransform): Unit = {
    require(
      kernel.isTransformSafe(transform),
      s"Transform '${transform.name}' is not compatible with kernel '${kernel.name}'"
    )
  }

  /** Recommend feature transform for a kernel.
    *
    * @param kernel
    *   the kernel
    * @return
    *   recommended transform (may be identity)
    */
  def recommendedTransform(kernel: KernelOps): FeatureTransform = {
    kernel match {
      case KLDivergenceDescriptor | GeneralizedIDivergenceDescriptor =>
        FeatureTransform.forKL()
      case CosineDistanceDescriptor =>
        FeatureTransform.forSpherical()
      case _ =>
        FeatureTransform.identity
    }
  }
}
