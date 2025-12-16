package com.massivedatascience.clusterer.ml.df

/** Re-exports kernel types from the kernels subpackage for backward compatibility.
  *
  * New code should import directly from the kernels package:
  * {{{
  * import com.massivedatascience.clusterer.ml.df.kernels._
  * }}}
  */
object BregmanKernels {
  type BregmanKernel = kernels.BregmanKernel
  type SquaredEuclideanKernel = kernels.SquaredEuclideanKernel
  type KLDivergenceKernel = kernels.KLDivergenceKernel
  type ItakuraSaitoKernel = kernels.ItakuraSaitoKernel
  type GeneralizedIDivergenceKernel = kernels.GeneralizedIDivergenceKernel
  type LogisticLossKernel = kernels.LogisticLossKernel
  type L1Kernel = kernels.L1Kernel
  type SphericalKernel = kernels.SphericalKernel
}
