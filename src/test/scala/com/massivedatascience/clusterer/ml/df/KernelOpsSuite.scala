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

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class KernelOpsSuite extends AnyFunSuite with Matchers {

  test("SquaredEuclideanDescriptor should have correct properties") {
    val kernel = SquaredEuclideanDescriptor

    assert(kernel.name == "squaredEuclidean")
    assert(kernel.supportsSEFastPath)
    assert(!kernel.requiresPositiveFeatures)
    assert(kernel.isSymmetric)
    assert(kernel.defaultBroadcastThreshold == 1000)
    assert(!kernel.benefitsFromNormalization)
  }

  test("KLDivergenceDescriptor should have correct properties") {
    val kernel = KLDivergenceDescriptor

    assert(kernel.name == "kl")
    assert(!kernel.supportsSEFastPath)
    assert(kernel.requiresPositiveFeatures)
    assert(!kernel.isSymmetric)
    assert(kernel.defaultBroadcastThreshold == 500)
  }

  test("GeneralizedIDivergenceDescriptor should have correct properties") {
    val kernel = GeneralizedIDivergenceDescriptor

    assert(kernel.name == "generalizedI")
    assert(!kernel.supportsSEFastPath)
    assert(kernel.requiresPositiveFeatures)
    assert(!kernel.isSymmetric)
  }

  test("ItakuraSaitoDescriptor should have correct properties") {
    val kernel = ItakuraSaitoDescriptor

    assert(kernel.name == "itakuraSaito")
    assert(!kernel.supportsSEFastPath)
    assert(kernel.requiresPositiveFeatures)
    assert(!kernel.isSymmetric)
  }

  test("LogisticLossDescriptor should have correct properties") {
    val kernel = LogisticLossDescriptor

    assert(kernel.name == "logisticLoss")
    assert(!kernel.supportsSEFastPath)
    assert(!kernel.requiresPositiveFeatures)
    assert(!kernel.isSymmetric)
  }

  test("CosineDistanceDescriptor should have correct properties") {
    val kernel = CosineDistanceDescriptor

    assert(kernel.name == "cosine")
    assert(kernel.supportsSEFastPath) // After normalization
    assert(!kernel.requiresPositiveFeatures)
    assert(kernel.isSymmetric)
    assert(kernel.benefitsFromNormalization)
  }

  test("ManhattanDistanceDescriptor should have correct properties") {
    val kernel = ManhattanDistanceDescriptor

    assert(kernel.name == "manhattan")
    assert(!kernel.supportsSEFastPath)
    assert(!kernel.requiresPositiveFeatures)
    assert(kernel.isSymmetric)
  }

  test("KernelOps.forName should parse kernel names") {
    assert(KernelOps.forName("squaredEuclidean") == SquaredEuclideanDescriptor)
    assert(KernelOps.forName("euclidean") == SquaredEuclideanDescriptor)
    assert(KernelOps.forName("se") == SquaredEuclideanDescriptor)
    assert(KernelOps.forName("kl") == KLDivergenceDescriptor)
    assert(KernelOps.forName("kullbackleibler") == KLDivergenceDescriptor)
    assert(KernelOps.forName("generalizedI") == GeneralizedIDivergenceDescriptor)
    assert(KernelOps.forName("itakuraSaito") == ItakuraSaitoDescriptor)
    assert(KernelOps.forName("cosine") == CosineDistanceDescriptor)
    assert(KernelOps.forName("manhattan") == ManhattanDistanceDescriptor)
  }

  test("KernelOps.forName should be case-insensitive") {
    assert(KernelOps.forName("SQUAREDEUCLIDEAN") == SquaredEuclideanDescriptor)
    assert(KernelOps.forName("KL") == KLDivergenceDescriptor)
    assert(KernelOps.forName("Cosine") == CosineDistanceDescriptor)
  }

  test("KernelOps.forName should normalize spaces and hyphens") {
    assert(KernelOps.forName("squared-euclidean") == SquaredEuclideanDescriptor)
    assert(KernelOps.forName("squared euclidean") == SquaredEuclideanDescriptor)
  }

  test("KernelOps.forName should throw on unknown kernel") {
    intercept[IllegalArgumentException] {
      KernelOps.forName("unknown_kernel")
    }
  }

  test("KernelOps.all should return all kernels") {
    val kernels = KernelOps.all

    assert(kernels.size == 7)
    assert(kernels.contains(SquaredEuclideanDescriptor))
    assert(kernels.contains(KLDivergenceDescriptor))
    assert(kernels.contains(GeneralizedIDivergenceDescriptor))
    assert(kernels.contains(ItakuraSaitoDescriptor))
    assert(kernels.contains(LogisticLossDescriptor))
    assert(kernels.contains(CosineDistanceDescriptor))
    assert(kernels.contains(ManhattanDistanceDescriptor))
  }

  test("KernelOps factory methods should work") {
    assert(KernelOps.squaredEuclidean == SquaredEuclideanDescriptor)
    assert(KernelOps.kl == KLDivergenceDescriptor)
    assert(KernelOps.generalizedI == GeneralizedIDivergenceDescriptor)
    assert(KernelOps.itakuraSaito == ItakuraSaitoDescriptor)
    assert(KernelOps.logisticLoss == LogisticLossDescriptor)
    assert(KernelOps.cosine == CosineDistanceDescriptor)
    assert(KernelOps.manhattan == ManhattanDistanceDescriptor)
  }

  test("SquaredEuclideanDescriptor should accept all transforms") {
    val kernel = SquaredEuclideanDescriptor

    assert(kernel.isTransformSafe(FeatureTransform.identity))
    assert(kernel.isTransformSafe(FeatureTransform.log1p))
    assert(kernel.isTransformSafe(FeatureTransform.normalizeL2()))
    assert(kernel.isTransformSafe(FeatureTransform.normalizeL1()))
  }

  test("KLDivergenceDescriptor should validate transform compatibility") {
    val kernel = KLDivergenceDescriptor

    assert(kernel.isTransformSafe(FeatureTransform.forKL()))
    assert(kernel.isTransformSafe(FeatureTransform.log1p))
    assert(kernel.isTransformSafe(FeatureTransform.epsilonShift()))
    // L2 normalization not compatible with KL
    assert(!kernel.isTransformSafe(FeatureTransform.normalizeL2()))
  }

  test("CosineDistanceDescriptor should require L2 normalization") {
    val kernel = CosineDistanceDescriptor

    assert(kernel.isTransformSafe(FeatureTransform.normalizeL2()))
    assert(!kernel.isTransformSafe(FeatureTransform.identity))
    assert(!kernel.isTransformSafe(FeatureTransform.log1p))
  }

  test("recommendedAssignment should use crossJoin for SE with small k") {
    val kernel = SquaredEuclideanDescriptor

    assert(kernel.recommendedAssignment(50) == "crossJoin")
    assert(kernel.recommendedAssignment(500) == "crossJoin")
  }

  test("recommendedAssignment should use rddMap for SE with large k") {
    val kernel = SquaredEuclideanDescriptor

    assert(kernel.recommendedAssignment(1500) == "rddMap")
  }

  test("recommendedAssignment should use rddMap for non-SE kernels") {
    assert(KLDivergenceDescriptor.recommendedAssignment(50) == "rddMap")
    assert(GeneralizedIDivergenceDescriptor.recommendedAssignment(50) == "rddMap")
    assert(ItakuraSaitoDescriptor.recommendedAssignment(50) == "rddMap")
  }

  test("validateCompatibility should pass for compatible pairs") {
    KernelOps.validateCompatibility(SquaredEuclideanDescriptor, FeatureTransform.identity)
    KernelOps.validateCompatibility(KLDivergenceDescriptor, FeatureTransform.forKL())
    KernelOps.validateCompatibility(CosineDistanceDescriptor, FeatureTransform.normalizeL2())
  }

  test("validateCompatibility should throw for incompatible pairs") {
    intercept[IllegalArgumentException] {
      KernelOps.validateCompatibility(CosineDistanceDescriptor, FeatureTransform.identity)
    }
  }

  test("recommendedTransform should return appropriate transforms") {
    assert(KernelOps.recommendedTransform(SquaredEuclideanDescriptor) == FeatureTransform.identity)
    assert(KernelOps.recommendedTransform(KLDivergenceDescriptor).name.contains("epsilon_shift"))
    assert(KernelOps.recommendedTransform(CosineDistanceDescriptor) == FeatureTransform.normalizeL2())
  }

  test("KernelOps should be serializable") {
    val kernel: KernelOps = SquaredEuclideanDescriptor

    val stream = new java.io.ByteArrayOutputStream()
    val oos    = new java.io.ObjectOutputStream(stream)
    oos.writeObject(kernel)
    oos.close()

    val bytes = stream.toByteArray
    assert(bytes.nonEmpty)
  }

  test("Pattern matching on KernelOps should work") {
    val kernel: KernelOps = KLDivergenceDescriptor

    val result = kernel match {
      case SquaredEuclideanDescriptor => "SE"
      case KLDivergenceDescriptor     => "KL"
      case _                      => "Other"
    }

    assert(result == "KL")
  }

  test("All kernels should have unique names") {
    val names = KernelOps.all.map(_.name)
    assert(names.size == names.distinct.size)
  }

  test("Positive-feature kernels should be marked correctly") {
    val positiveKernels = KernelOps.all.filter(_.requiresPositiveFeatures)

    assert(positiveKernels.contains(KLDivergenceDescriptor))
    assert(positiveKernels.contains(GeneralizedIDivergenceDescriptor))
    assert(positiveKernels.contains(ItakuraSaitoDescriptor))
    assert(!positiveKernels.contains(SquaredEuclideanDescriptor))
    assert(!positiveKernels.contains(CosineDistanceDescriptor))
  }

  test("Symmetric kernels should be marked correctly") {
    val symmetricKernels = KernelOps.all.filter(_.isSymmetric)

    assert(symmetricKernels.contains(SquaredEuclideanDescriptor))
    assert(symmetricKernels.contains(CosineDistanceDescriptor))
    assert(symmetricKernels.contains(ManhattanDistanceDescriptor))
    assert(!symmetricKernels.contains(KLDivergenceDescriptor))
    assert(!symmetricKernels.contains(GeneralizedIDivergenceDescriptor))
  }
}
