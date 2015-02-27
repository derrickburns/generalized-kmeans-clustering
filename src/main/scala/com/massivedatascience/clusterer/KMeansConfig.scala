package com.massivedatascience.clusterer


trait KMeansConfig extends Serializable {
  //for stochastic sampling, the percentage of points to update on each round
  val updateRate: Double

  // maxRoundsToBackfill maximum number of rounds to try to fill empty clusters
  val maxRoundsToBackfill: Int

  // fractionOfPointsToWeigh the fraction of the points to use in the weighting in KMeans++
  val fractionOfPointsToWeigh: Double

  val addOnly: Boolean

}

object DefaultKMeansConfig extends KMeansConfig {
  //for stochastic sampling, the percentage of points to update on each round
  val updateRate: Double = 1.0

  // maxRoundsToBackfill maximum number of rounds to try to fill empty clusters
  val maxRoundsToBackfill: Int = 0

  // fractionOfPointsToWeigh the fraction of the points to use in the weighting in KMeans++
  val fractionOfPointsToWeigh: Double = 0.10

  val addOnly = true
}
