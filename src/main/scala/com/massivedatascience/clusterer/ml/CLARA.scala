package com.massivedatascience.clusterer.ml

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.Dataset

/** Parameters for CLARA (Clustering Large Applications). */
trait CLARAParams extends KMedoidsParams {

  /** Number of samples to draw from the dataset. Each sample is clustered independently and the
    * best result is selected.
    */
  final val numSamples = new IntParam(
    this,
    "numSamples",
    "Number of samples to draw",
    ParamValidators.gt(0)
  )

  def getNumSamples: Int = $(numSamples)

  /** Sample size for each sample. Default: 40 + 2*k (as recommended in the original CLARA paper)
    * Set to -1 for automatic sizing (40 + 2*k)
    */
  final val sampleSize = new IntParam(
    this,
    "sampleSize",
    "Sample size for each sample (-1 for auto)",
    (value: Int) => value == -1 || value > 0
  )

  def getSampleSize: Int = $(sampleSize)

  setDefault(
    numSamples -> 5,
    sampleSize -> -1 // -1 means auto (40 + 2*k)
  )
}

/** CLARA (Clustering Large Applications) - Sampling-based K-Medoids for large datasets.
  *
  * CLARA is a more scalable version of PAM that works on large datasets by:
  *   1. Drawing multiple samples from the full dataset 2. Running PAM on each sample 3. For each
  *      sample result, computing the cost on the full dataset 4. Selecting the medoids with the
  *      lowest total cost
  *
  * Time Complexity: O(numSamples * k(s-k)²) where s is the sample size
  *
  * CLARA is recommended when:
  *   - Dataset has > 10,000 points
  *   - PAM is too slow due to O(k(n-k)²) complexity
  *   - Good approximation to PAM is acceptable
  *
  * Example usage:
  * {{{
  * val clara = new CLARA()
  *   .setK(3)
  *   .setNumSamples(10)
  *   .setSampleSize(100)
  *   .setMaxIter(20)
  *   .setDistanceFunction("manhattan")
  *
  * val model = clara.fit(largeDataset)
  * val predictions = model.transform(largeDataset)
  * }}}
  *
  * @param uid
  *   unique identifier
  */
class CLARA(override val uid: String)
    extends KMedoids(uid)
    with CLARAParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("clara"))

  override def fit(dataset: Dataset[_]): KMedoidsModel = {
    transformSchema(dataset.schema, logging = true)

    val df   = dataset.toDF()
    val data = df
      .select($(featuresCol))
      .rdd
      .map { row =>
        row.getAs
      }
      .collect()

    val n           = data.length
    val numClusters = $(k)

    // Determine sample size
    val actualSampleSize = if ($(sampleSize) == -1) {
      math.min(40 + 2 * numClusters, n)
    } else {
      math.min($(sampleSize), n)
    }

    logInfo(s"CLARA with k=$numClusters, numSamples=${$(numSamples)}, sampleSize=$actualSampleSize")
    logInfo(s"Dataset size: $n points")

    if (actualSampleSize >= n * 0.9) {
      logWarning(
        s"Sample size ($actualSampleSize) is >= 90% of dataset ($n). Consider using PAM instead of CLARA."
      )
    }

    // Create distance function
    val distFn = createDistanceFunction($(distanceFunction))

    var bestMedoidIndices: Array[Int] = null
    var bestCost                      = Double.PositiveInfinity

    // Try multiple samples
    val rng = new scala.util.Random($(seed))

    (0 until $(numSamples)).foreach { sampleIdx =>
      logInfo(s"Processing sample ${sampleIdx + 1}/${$(numSamples)}")

      // Draw random sample
      val sampleIndices = rng.shuffle((0 until n).toList).take(actualSampleSize).toArray
      val sample        = sampleIndices.map(data)

      // Run PAM on sample
      val sampleMedoidIndices      = buildPhase(sample, numClusters, distFn, $(seed) + sampleIdx)
      val finalSampleMedoidIndices = swapPhase(sample, sampleMedoidIndices, $(maxIter), distFn)

      // Map sample medoid indices back to original dataset indices
      val originalMedoidIndices = finalSampleMedoidIndices.map(sampleIndices)

      // Compute cost on FULL dataset
      var totalCost = 0.0
      data.foreach { point =>
        val minDist = originalMedoidIndices.map(medIdx => distFn(point, data(medIdx))).min
        totalCost += minDist
      }

      logInfo(f"Sample ${sampleIdx + 1} cost on full dataset: $totalCost%.4f")

      // Keep best result
      if (totalCost < bestCost) {
        bestCost = totalCost
        bestMedoidIndices = originalMedoidIndices
        logInfo(f"New best cost: $bestCost%.4f")
      }
    }

    logInfo(f"CLARA completed. Best cost: $bestCost%.4f")
    logInfo(s"Best medoid indices: ${bestMedoidIndices.mkString(", ")}")

    // Create model
    val medoidVectors = bestMedoidIndices.map(data)
    new KMedoidsModel(uid, medoidVectors, bestMedoidIndices, $(distanceFunction)).setParent(this)
  }

  // Parameter setters
  def setNumSamples(value: Int): this.type = set(numSamples, value)
  def setSampleSize(value: Int): this.type = set(sampleSize, value)

  override def copy(extra: ParamMap): CLARA = defaultCopy(extra)
}

object CLARA extends DefaultParamsReadable[CLARA] {
  override def load(path: String): CLARA = super.load(path)
}
