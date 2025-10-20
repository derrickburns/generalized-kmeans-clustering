package com.massivedatascience.clusterer.ml

import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._

/** Parameters for CoresetKMeans clustering algorithm.
  *
  * Core-set based K-Means uses importance sampling to create a small weighted subset of the data,
  * clusters on that subset, then optionally refines on the full data. This provides significant
  * speedup for large datasets with minimal quality loss.
  */
private[ml] trait CoresetKMeansParams
    extends GeneralizedKMeansParams
    with HasMaxIter
    with HasSeed
    with HasTol {

  /** Target size of the core-set (number of sampled points). Default: 1000.
    *
    * Larger core-sets provide better approximation but take longer to cluster. Smaller core-sets
    * are faster but may lose quality. Rule of thumb: 10-100x the number of clusters.
    */
  final val coresetSize: IntParam = new IntParam(
    this,
    "coresetSize",
    "Target size of the core-set (must be > 0)",
    ParamValidators.gt(0)
  )

  /** Gets the target core-set size. */
  def getCoresetSize: Int = $(coresetSize)

  /** Approximation quality parameter (epsilon). Default: 0.1.
    *
    * Smaller epsilon means better approximation but larger theoretical core-set size. Typical
    * values: 0.05 (high quality) to 0.2 (fast). Theoretical core-set size is O(k * log(k) /
    * epsilon^2).
    */
  final val epsilon: DoubleParam = new DoubleParam(
    this,
    "epsilon",
    "Approximation quality parameter (must be in (0, 1))",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = false)
  )

  /** Gets the epsilon parameter. */
  def getEpsilon: Double = $(epsilon)

  /** Sensitivity computation strategy. Default: "hybrid".
    *
    * Options:
    *   - "uniform": All points have equal probability (fastest, least accurate)
    *   - "distance": Sample based on distance to nearest center (accurate, moderate cost)
    *   - "density": Sample based on local density (good for non-uniform distributions)
    *   - "hybrid": Combination of distance and density (recommended)
    */
  final val sensitivityStrategy: Param[String] = new Param[String](
    this,
    "sensitivityStrategy",
    "Sensitivity computation strategy",
    (value: String) => Set("uniform", "distance", "density", "hybrid").contains(value.toLowerCase)
  )

  /** Gets the sensitivity strategy. */
  def getSensitivityStrategy: String = $(sensitivityStrategy)

  /** Weight given to distance component in hybrid sensitivity (0.0 to 1.0). Default: 0.5.
    *
    * Only used when sensitivityStrategy = "hybrid". Higher values emphasize outliers, lower values
    * emphasize dense regions.
    */
  final val distanceWeight: DoubleParam = new DoubleParam(
    this,
    "distanceWeight",
    "Weight for distance component in hybrid sensitivity (must be in [0, 1])",
    ParamValidators.inRange(0.0, 1.0)
  )

  /** Gets the distance weight for hybrid sensitivity. */
  def getDistanceWeight: Double = $(distanceWeight)

  /** Number of refinement iterations on full data after core-set clustering. Default: 3.
    *
    * Refinement improves quality by running Lloyd's iterations on the full dataset using core-set
    * centers as initialization. Set to 0 to disable refinement (fastest, may lose quality).
    */
  final val refinementIterations: IntParam = new IntParam(
    this,
    "refinementIterations",
    "Number of refinement iterations on full data (must be >= 0)",
    ParamValidators.gtEq(0)
  )

  /** Gets the number of refinement iterations. */
  def getRefinementIterations: Int = $(refinementIterations)

  /** Whether to enable refinement on full data after core-set clustering. Default: true.
    *
    * Disabling refinement makes clustering faster but may reduce quality. Generally recommended to
    * keep enabled with refinementIterations >= 1.
    */
  final val enableRefinement: BooleanParam = new BooleanParam(
    this,
    "enableRefinement",
    "Enable refinement on full data after core-set clustering"
  )

  /** Gets whether refinement is enabled. */
  def getEnableRefinement: Boolean = $(enableRefinement)

  /** Minimum sampling probability to avoid extreme importance weights. Default: 1e-6.
    *
    * Points with very low sensitivity would get very high importance weights if sampled. This
    * parameter prevents extreme weights by ensuring minimum sampling probability.
    */
  final val minSamplingProb: DoubleParam = new DoubleParam(
    this,
    "minSamplingProb",
    "Minimum sampling probability (must be in (0, 1])",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true)
  )

  /** Gets the minimum sampling probability. */
  def getMinSamplingProb: Double = $(minSamplingProb)

  /** Maximum allowed importance weight for core-set points. Default: 1e6.
    *
    * Very high importance weights can cause numerical instability. This parameter caps the maximum
    * weight any core-set point can have.
    */
  final val maxWeight: DoubleParam = new DoubleParam(
    this,
    "maxWeight",
    "Maximum importance weight for core-set points (must be > 1)",
    ParamValidators.gt(1.0)
  )

  /** Gets the maximum weight. */
  def getMaxWeight: Double = $(maxWeight)

  /** Number of sample centers to use for distance-based sensitivity. Default: 10.
    *
    * Only used when sensitivityStrategy = "distance" or "hybrid". More sample centers give better
    * sensitivity estimates but take longer to compute.
    */
  final val numSampleCenters: IntParam = new IntParam(
    this,
    "numSampleCenters",
    "Number of sample centers for distance-based sensitivity (must be > 0)",
    ParamValidators.gt(0)
  )

  /** Gets the number of sample centers. */
  def getNumSampleCenters: Int = $(numSampleCenters)

  setDefault(
    coresetSize          -> 1000,
    epsilon              -> 0.1,
    sensitivityStrategy  -> "hybrid",
    distanceWeight       -> 0.5,
    refinementIterations -> 3,
    enableRefinement     -> true,
    minSamplingProb      -> 1e-6,
    maxWeight            -> 1e6,
    numSampleCenters     -> 10,
    maxIter              -> 50,
    tol                  -> 1e-6,
    seed                 -> System.currentTimeMillis()
  )

  /** Validates that parameters are consistent. */
  protected def validateCoresetParams(): Unit = {
    require($(coresetSize) > 0, s"coresetSize must be > 0, got: ${$(coresetSize)}")
    require(
      $(epsilon) > 0.0 && $(epsilon) < 1.0,
      s"epsilon must be in (0, 1), got: ${$(epsilon)}"
    )
    require(
      $(refinementIterations) >= 0,
      s"refinementIterations must be >= 0, got: ${$(refinementIterations)}"
    )
    require(
      $(minSamplingProb) > 0.0 && $(minSamplingProb) <= 1.0,
      s"minSamplingProb must be in (0, 1], got: ${$(minSamplingProb)}"
    )
    require($(maxWeight) > 1.0, s"maxWeight must be > 1, got: ${$(maxWeight)}")
    require(
      $(numSampleCenters) > 0,
      s"numSampleCenters must be > 0, got: ${$(numSampleCenters)}"
    )
  }
}
