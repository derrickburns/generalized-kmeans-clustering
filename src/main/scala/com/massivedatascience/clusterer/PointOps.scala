package com.massivedatascience.clusterer

object PointOps {

  val RELATIVE_ENTROPY = "DENSE_KL_DIVERGENCE"
  val DISCRETE_KL = "DISCRETE_DENSE_KL_DIVERGENCE"
  val SPARSE_SMOOTHED_KL = "SPARSE_SMOOTHED_KL_DIVERGENCE"
  val DISCRETE_SMOOTHED_KL = "DISCRETE_DENSE_SMOOTHED_KL_DIVERGENCE"
  val SIMPLEX_SMOOTHED_KL = "SIMPLEX_SMOOTHED_KL"
  val EUCLIDEAN = "EUCLIDEAN"
  val LOGISTIC_LOSS = "LOGISTIC_LOSS"
  val GENERALIZED_I = "GENERALIZED_I_DIVERGENCE"

  def apply(distanceFunction: String): BregmanPointOps = {
    distanceFunction match {
      case EUCLIDEAN => SquaredEuclideanPointOps
      case RELATIVE_ENTROPY => DenseKLPointOps
      case DISCRETE_KL => DiscreteKLPointOps
      case SIMPLEX_SMOOTHED_KL => DiscreteSimplexSmoothedKLPointOps
      case DISCRETE_SMOOTHED_KL => DiscreteSmoothedKLPointOps
      case SPARSE_SMOOTHED_KL => SparseRealKLPointOps
      case LOGISTIC_LOSS => LogisticLossPointOps
      case GENERALIZED_I => GeneralizedIPointOps
      case _ => throw new RuntimeException(s"unknown distance function $distanceFunction")
    }
  }
}
