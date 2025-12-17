package com.massivedatascience.clusterer.ml.df

/** Bregman kernel implementations for clustering algorithms.
  *
  * This package contains kernel implementations for different Bregman divergences:
  *
  * ==Factory==
  *
  *   - [[kernels.KernelFactory]]: Unified factory for dense/sparse kernel selection
  *
  * ==Dense Kernels==
  *
  *   - [[kernels.SquaredEuclideanKernel]]: Standard k-means (L2 squared)
  *   - [[kernels.KLDivergenceKernel]]: Kullback-Leibler divergence
  *   - [[kernels.ItakuraSaitoKernel]]: Itakura-Saito divergence
  *   - [[kernels.GeneralizedIDivergenceKernel]]: Generalized I-divergence
  *   - [[kernels.LogisticLossKernel]]: Logistic loss
  *   - [[kernels.L1Kernel]]: Manhattan distance (K-Medians)
  *   - [[kernels.SphericalKernel]]: Cosine similarity (Spherical K-Means)
  *
  * ==Sparse-Optimized Kernels==
  *
  *   - [[kernels.SparseSEKernel]]: Sparse Squared Euclidean
  *   - [[kernels.SparseKLKernel]]: Sparse KL Divergence
  *   - [[kernels.SparseL1Kernel]]: Sparse L1/Manhattan
  *   - [[kernels.SparseSphericalKernel]]: Sparse Cosine/Spherical
  *
  * ==Usage==
  *
  * {{{
  * // Create kernel via factory
  * val kernel = KernelFactory.create("squaredEuclidean", sparse = false)
  *
  * // Auto-select based on data sparsity
  * val sparseKernel = KernelFactory.forSparsity("kl", sparsityRatio = 0.1)
  * }}}
  */
package object kernels {
  // All types are defined in their respective files
  // KernelFactory provides the main API for kernel creation
}
