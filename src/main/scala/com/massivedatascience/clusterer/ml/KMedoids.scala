package com.massivedatascience.clusterer.ml

import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

/** Parameters for K-Medoids clustering.
  */
trait KMedoidsParams
    extends Params
    with HasFeaturesCol
    with HasPredictionCol
    with HasMaxIter
    with HasSeed {

  /** Number of clusters (k). Must be > 1.
    */
  final val k = new IntParam(this, "k", "Number of clusters", ParamValidators.gt(1))

  def getK: Int = $(k)

  /** Distance function name. Supported values:
    *   - "euclidean" (default)
    *   - "manhattan" (L1)
    *   - "cosine"
    *   - "custom" (requires custom distance function)
    */
  final val distanceFunction = new Param[String](
    this,
    "distanceFunction",
    "Distance function",
    ParamValidators.inArray(Array("euclidean", "manhattan", "cosine"))
  )

  def getDistanceFunction: String = $(distanceFunction)

  setDefault(
    k                -> 2,
    featuresCol      -> "features",
    predictionCol    -> "prediction",
    maxIter          -> 20,
    distanceFunction -> "euclidean"
  )

  /** Validate and transform input schema.
    */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    // Add prediction column
    val outputSchema = schema.add($(predictionCol), org.apache.spark.sql.types.IntegerType)
    outputSchema
  }
}

/** K-Medoids clustering using PAM (Partitioning Around Medoids) algorithm.
  *
  * K-Medoids is a clustering algorithm that uses actual data points as cluster centers (called
  * medoids) instead of computed centroids like K-Means. This makes it:
  *   - More robust to outliers
  *   - More interpretable (medoids are real data points)
  *   - Works with any distance function (not just Euclidean)
  *   - More computationally expensive than K-Means
  *
  * The algorithm consists of two phases:
  *   1. BUILD: Greedily select k initial medoids to minimize total cost 2. SWAP: Iteratively swap
  *      medoids with non-medoids to minimize cost
  *
  * Time Complexity: O(k(n-k)²) per iteration
  *
  * Example usage:
  * {{{
  * val kmedoids = new KMedoids()
  *   .setK(3)
  *   .setMaxIter(20)
  *   .setDistanceFunction("manhattan")
  *
  * val model = kmedoids.fit(data)
  * val predictions = model.transform(data)
  * }}}
  *
  * @param uid
  *   unique identifier
  */
class KMedoids(override val uid: String)
    extends Estimator[KMedoidsModel]
    with KMedoidsParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("kmedoids"))

  override def fit(dataset: Dataset[_]): KMedoidsModel = {
    transformSchema(dataset.schema, logging = true)

    val df   = dataset.toDF()
    val data = df
      .select($(featuresCol))
      .rdd
      .map { row =>
        row.getAs[Vector](0)
      }
      .collect()

    logInfo(
      s"K-Medoids with k=${$(k)}, maxIter=${$(maxIter)}, distanceFunction=${$(distanceFunction)}"
    )
    logInfo(s"Dataset size: ${data.length} points")

    // Create distance function
    val distFn = createDistanceFunction($(distanceFunction))

    // BUILD phase: Initialize medoids
    val initialMedoidIndices = buildPhase(data, $(k), distFn, $(seed))
    logInfo(s"BUILD phase completed. Initial medoids: ${initialMedoidIndices.mkString(", ")}")

    // SWAP phase: Iteratively improve medoids
    val finalMedoidIndices = swapPhase(data, initialMedoidIndices, $(maxIter), distFn)
    logInfo(s"SWAP phase completed. Final medoids: ${finalMedoidIndices.mkString(", ")}")

    // Create model
    val medoidVectors = finalMedoidIndices.map(data)
    new KMedoidsModel(uid, medoidVectors, finalMedoidIndices, $(distanceFunction)).setParent(this)
  }

  /** BUILD phase: Greedily select k medoids to minimize total cost.
    *
    * Algorithm:
    *   1. Select first medoid as the point with minimum sum of distances to all other points 2. For
    *      i = 2 to k:
    *      - For each non-medoid point o:
    *        - Compute cost reduction if o becomes the next medoid
    *      - Select point with maximum cost reduction
    *
    * @param data
    *   all data points
    * @param k
    *   number of medoids
    * @param distFn
    *   distance function
    * @param seed
    *   random seed
    * @return
    *   indices of selected medoids
    */
  protected def buildPhase(
      data: Array[Vector],
      k: Int,
      distFn: (Vector, Vector) => Double,
      seed: Long
  ): Array[Int] = {
    val n             = data.length
    val medoidIndices = new Array[Int](k)
    val isMedoid      = new Array[Boolean](n)

    // Step 1: Select first medoid (point with minimum total distance to all others)
    var minCost     = Double.PositiveInfinity
    var firstMedoid = 0

    (0 until n).foreach { i =>
      var totalDist = 0.0
      (0 until n).foreach { j =>
        if (i != j) {
          totalDist += distFn(data(i), data(j))
        }
      }

      if (totalDist < minCost) {
        minCost = totalDist
        firstMedoid = i
      }
    }

    medoidIndices(0) = firstMedoid
    isMedoid(firstMedoid) = true
    logInfo(s"First medoid: $firstMedoid (cost: $minCost)")

    // Step 2: Greedily select remaining k-1 medoids
    (1 until k).foreach { medoidCount =>
      var maxGain       = Double.NegativeInfinity
      var bestCandidate = -1

      // For each non-medoid point
      (0 until n).foreach { candidate =>
        if (!isMedoid(candidate)) {
          // Compute cost reduction if this point becomes a medoid
          var gain = 0.0

          (0 until n).foreach { j =>
            if (!isMedoid(j) && j != candidate) {
              // Current distance: minimum distance to existing medoids
              val currentDist = (0 until medoidCount).map { m =>
                distFn(data(j), data(medoidIndices(m)))
              }.min

              // New distance: minimum of (current distance, distance to new candidate)
              val newDist = math.min(currentDist, distFn(data(j), data(candidate)))

              // Gain is the reduction in distance
              gain += (currentDist - newDist)
            }
          }

          if (gain > maxGain) {
            maxGain = gain
            bestCandidate = candidate
          }
        }
      }

      medoidIndices(medoidCount) = bestCandidate
      isMedoid(bestCandidate) = true
      logInfo(s"Selected medoid ${medoidCount + 1}: $bestCandidate (gain: $maxGain)")
    }

    medoidIndices
  }

  /** SWAP phase: Iteratively swap medoids with non-medoids to minimize cost.
    *
    * Algorithm:
    *   1. For each medoid m and each non-medoid point o:
    *      - Compute cost change if m is swapped with o 2. Perform the swap with the largest cost
    *        reduction 3. Repeat until no improvement is possible or maxIter is reached
    *
    * @param data
    *   all data points
    * @param initialMedoidIndices
    *   initial medoid indices from BUILD phase
    * @param maxIter
    *   maximum number of iterations
    * @param distFn
    *   distance function
    * @return
    *   final medoid indices
    */
  protected def swapPhase(
      data: Array[Vector],
      initialMedoidIndices: Array[Int],
      maxIter: Int,
      distFn: (Vector, Vector) => Double
  ): Array[Int] = {
    val n             = data.length
    val k             = initialMedoidIndices.length
    val medoidIndices = initialMedoidIndices.clone()
    val isMedoid      = new Array[Boolean](n)
    medoidIndices.foreach(i => isMedoid(i) = true)

    var improved  = true
    var iteration = 0

    while (improved && iteration < maxIter) {
      improved = false
      var bestSwapGain        = 0.0
      var bestMedoidToSwap    = -1
      var bestNonMedoidToSwap = -1

      // For each medoid
      (0 until k).foreach { medoidIdx =>
        val medoid = medoidIndices(medoidIdx)

        // For each non-medoid
        (0 until n).foreach { nonMedoid =>
          if (!isMedoid(nonMedoid)) {
            // Compute cost change if we swap this medoid with this non-medoid
            var totalCostChange = 0.0

            (0 until n).foreach { j =>
              if (!isMedoid(j) && j != nonMedoid) {
                // Current distance: minimum distance to current medoids
                val currentDist = medoidIndices.map(m => distFn(data(j), data(m))).min

                // Distance to current medoid being considered for swap
                val distToSwappedMedoid = distFn(data(j), data(medoid))

                // Distance to potential new medoid
                val distToNewMedoid = distFn(data(j), data(nonMedoid))

                // If current closest medoid is the one being swapped
                if (math.abs(currentDist - distToSwappedMedoid) < 1e-10) {
                  // New distance is minimum of (second closest current medoid, new medoid)
                  val secondClosest =
                    medoidIndices.filter(_ != medoid).map(m => distFn(data(j), data(m))).min
                  val newDist       = math.min(secondClosest, distToNewMedoid)
                  totalCostChange += (newDist - currentDist)
                } else {
                  // New distance is minimum of (current distance, new medoid)
                  val newDist = math.min(currentDist, distToNewMedoid)
                  totalCostChange += (newDist - currentDist)
                }
              }
            }

            // Negative cost change means improvement
            if (totalCostChange < bestSwapGain) {
              bestSwapGain = totalCostChange
              bestMedoidToSwap = medoidIdx
              bestNonMedoidToSwap = nonMedoid
            }
          }
        }
      }

      // Perform the best swap if it improves the clustering
      if (bestSwapGain < -1e-10) {
        val oldMedoid = medoidIndices(bestMedoidToSwap)
        medoidIndices(bestMedoidToSwap) = bestNonMedoidToSwap
        isMedoid(oldMedoid) = false
        isMedoid(bestNonMedoidToSwap) = true
        improved = true
        iteration += 1

        logInfo(
          f"Iteration $iteration: Swapped medoid $oldMedoid with $bestNonMedoidToSwap (cost reduction: ${-bestSwapGain}%.4f)"
        )
      }
    }

    if (iteration == 0) {
      logInfo("SWAP phase: No swaps performed (already optimal)")
    } else {
      logInfo(s"SWAP phase converged after $iteration iterations")
    }

    medoidIndices
  }

  /** Create distance function from name.
    */
  protected def createDistanceFunction(name: String): (Vector, Vector) => Double = {
    name match {
      case "euclidean" =>
        (v1: Vector, v2: Vector) => {
          var sum = 0.0
          var i   = 0
          while (i < v1.size) {
            val d = v1(i) - v2(i)
            sum += d * d
            i += 1
          }
          math.sqrt(sum)
        }

      case "manhattan" =>
        (v1: Vector, v2: Vector) => {
          var sum = 0.0
          var i   = 0
          while (i < v1.size) {
            sum += math.abs(v1(i) - v2(i))
            i += 1
          }
          sum
        }

      case "cosine" =>
        (v1: Vector, v2: Vector) => {
          var dot   = 0.0
          var norm1 = 0.0
          var norm2 = 0.0
          var i     = 0
          while (i < v1.size) {
            dot += v1(i) * v2(i)
            norm1 += v1(i) * v1(i)
            norm2 += v2(i) * v2(i)
            i += 1
          }
          1.0 - (dot / (math.sqrt(norm1) * math.sqrt(norm2)))
        }

      case _ =>
        logWarning(s"Unknown distance function $name, using Euclidean")
        (v1: Vector, v2: Vector) => {
          var sum = 0.0
          var i   = 0
          while (i < v1.size) {
            val d = v1(i) - v2(i)
            sum += d * d
            i += 1
          }
          math.sqrt(sum)
        }
    }
  }

  // Parameter setters
  def setK(value: Int): this.type                   = set(k, value)
  def setMaxIter(value: Int): this.type             = set(maxIter, value)
  def setDistanceFunction(value: String): this.type = set(distanceFunction, value)
  def setSeed(value: Long): this.type               = set(seed, value)
  def setFeaturesCol(value: String): this.type      = set(featuresCol, value)
  def setPredictionCol(value: String): this.type    = set(predictionCol, value)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): KMedoids = defaultCopy(extra)
}

object KMedoids extends DefaultParamsReadable[KMedoids] {
  override def load(path: String): KMedoids = super.load(path)
}

/** Model produced by K-Medoids clustering.
  *
  * @param uid
  *   unique identifier
  * @param medoids
  *   cluster medoid vectors
  * @param medoidIndices
  *   indices of medoids in the original dataset (if available)
  * @param distanceFunctionName
  *   name of the distance function used
  */
class KMedoidsModel(
    override val uid: String,
    val medoids: Array[Vector],
    val medoidIndices: Array[Int],
    val distanceFunctionName: String
) extends org.apache.spark.ml.Model[KMedoidsModel]
    with KMedoidsParams
    with org.apache.spark.ml.util.MLWritable
    with Logging {

  /** Number of clusters.
    */
  def numClusters: Int = medoids.length

  /** Dimensionality of features.
    */
  def numFeatures: Int = medoids.headOption.map(_.size).getOrElse(0)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val df = dataset.toDF()

    // Create distance function
    val distFn = createDistanceFunction(distanceFunctionName)

    // Broadcast medoids
    val bcMedoids = df.sparkSession.sparkContext.broadcast(medoids)

    // UDF to find nearest medoid
    val predictUDF = udf { (features: Vector) =>
      val meds    = bcMedoids.value
      var minDist = Double.PositiveInfinity
      var minIdx  = 0
      var i       = 0
      while (i < meds.length) {
        val dist = distFn(features, meds(i))
        if (dist < minDist) {
          minDist = dist
          minIdx = i
        }
        i += 1
      }
      minIdx
    }

    // Add prediction column
    df.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  /** Compute total cost (sum of distances from points to their assigned medoids).
    */
  def computeCost(dataset: Dataset[_]): Double = {
    val df        = dataset.toDF()
    val distFn    = createDistanceFunction(distanceFunctionName)
    val bcMedoids = df.sparkSession.sparkContext.broadcast(medoids)

    df.select($(featuresCol))
      .rdd
      .map { row =>
        val features = row.getAs[Vector](0)
        val meds     = bcMedoids.value
        meds.map(m => distFn(features, m)).min
      }
      .sum()
  }

  /** Create distance function from name.
    */
  private def createDistanceFunction(name: String): (Vector, Vector) => Double = {
    name match {
      case "euclidean" =>
        (v1: Vector, v2: Vector) => {
          var sum = 0.0
          var i   = 0
          while (i < v1.size) {
            val d = v1(i) - v2(i)
            sum += d * d
            i += 1
          }
          math.sqrt(sum)
        }

      case "manhattan" =>
        (v1: Vector, v2: Vector) => {
          var sum = 0.0
          var i   = 0
          while (i < v1.size) {
            sum += math.abs(v1(i) - v2(i))
            i += 1
          }
          sum
        }

      case "cosine" =>
        (v1: Vector, v2: Vector) => {
          var dot   = 0.0
          var norm1 = 0.0
          var norm2 = 0.0
          var i     = 0
          while (i < v1.size) {
            dot += v1(i) * v2(i)
            norm1 += v1(i) * v1(i)
            norm2 += v2(i) * v2(i)
            i += 1
          }
          1.0 - (dot / (math.sqrt(norm1) * math.sqrt(norm2)))
        }

      case _ =>
        (v1: Vector, v2: Vector) => {
          var sum = 0.0
          var i   = 0
          while (i < v1.size) {
            val d = v1(i) - v2(i)
            sum += d * d
            i += 1
          }
          math.sqrt(sum)
        }
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): KMedoidsModel = {
    val copied =
      new KMedoidsModel(uid, medoids.map(_.copy), medoidIndices.clone(), distanceFunctionName)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: org.apache.spark.ml.util.MLWriter =
    new KMedoidsModel.KMedoidsModelWriter(this)
}

object KMedoidsModel extends org.apache.spark.ml.util.MLReadable[KMedoidsModel] {

  override def read: org.apache.spark.ml.util.MLReader[KMedoidsModel] = new KMedoidsModelReader

  override def load(path: String): KMedoidsModel = super.load(path)

  private class KMedoidsModelWriter(instance: KMedoidsModel)
      extends org.apache.spark.ml.util.MLWriter
      with Logging {

    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession

      logInfo(s"Saving KMedoidsModel to $path")

      // Prepare centers data: (center_id, weight, vector)
      // For K-Medoids, medoids have uniform weight (1.0)
      val centersData = instance.medoids.indices.map { i =>
        val weight = 1.0
        val vector = instance.medoids(i)
        (i, weight, vector)
      }

      // Write centers with deterministic ordering
      val centersHash = writeCenters(spark, path, centersData)
      logInfo(s"Centers saved with SHA-256: $centersHash")

      // Collect all model parameters (explicitly typed to avoid Any inference)
      val params: Map[String, Any] = Map(
        "k"                -> instance.getOrDefault(instance.k),
        "featuresCol"      -> instance.getOrDefault(instance.featuresCol),
        "predictionCol"    -> instance.getOrDefault(instance.predictionCol),
        "maxIter"          -> instance.getOrDefault(instance.maxIter),
        "seed"             -> instance.getOrDefault(instance.seed),
        "distanceFunction" -> instance.distanceFunctionName,
        "medoidIndices"    -> instance.medoidIndices.toSeq // Store original indices
      )

      val k   = instance.numClusters
      val dim = instance.numFeatures

      // Build metadata object (explicitly typed to avoid Any inference)
      implicit val formats = DefaultFormats
      val metaObj: Map[String, Any] = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "KMedoidsModel",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "divergence"         -> instance.distanceFunctionName, // Use distance function as divergence
        "k"                  -> k,
        "dim"                -> dim,
        "uid"                -> instance.uid,
        "params"             -> params,
        "centers"            -> Map[String, Any](
          "count"    -> k,
          "ordering" -> "center_id ASC (0..k-1)",
          "storage"  -> "parquet"
        ),
        "checksums"          -> Map[String, String](
          "centersParquetSHA256" -> centersHash
        )
      )

      // Serialize to JSON
      val json = Serialization.write(metaObj)(formats)

      // Write metadata
      val metadataHash = writeMetadata(path, json)
      logInfo(s"Metadata saved with SHA-256: $metadataHash")
      logInfo(s"KMedoidsModel successfully saved to $path")
    }
  }

  private class KMedoidsModelReader
      extends org.apache.spark.ml.util.MLReader[KMedoidsModel]
      with Logging {

    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): KMedoidsModel = {
      val spark = sparkSession

      logInfo(s"Loading KMedoidsModel from $path")

      // Read metadata
      val metaStr          = readMetadata(path)
      implicit val formats = DefaultFormats
      val metaJ            = JsonMethods.parse(metaStr)

      // Extract and validate layout version
      val layoutVersion    = (metaJ \ "layoutVersion").extract[Int]
      val k                = (metaJ \ "k").extract[Int]
      val dim              = (metaJ \ "dim").extract[Int]
      val uid              = (metaJ \ "uid").extract[String]
      val distanceFunction = (metaJ \ "divergence").extract[String]

      logInfo(
        s"Model metadata: layoutVersion=$layoutVersion, k=$k, dim=$dim, distanceFunction=$distanceFunction"
      )

      // Read centers
      val centersDF = readCenters(spark, path)
      val rows      = centersDF.collect()

      // Validate metadata
      validateMetadata(layoutVersion, k, dim, rows.length)

      // Extract medoids (sorted by center_id)
      val medoids = rows.sortBy(_.getInt(0)).map { row =>
        row.getAs[Vector]("vector")
      }

      // Extract parameters
      val paramsJ       = metaJ \ "params"
      val medoidIndices = (paramsJ \ "medoidIndices").extract[Seq[Int]].toArray

      // Reconstruct model
      val model = new KMedoidsModel(uid, medoids, medoidIndices, distanceFunction)

      // Set parameters
      model.set(model.k, (paramsJ \ "k").extract[Int])
      model.set(model.featuresCol, (paramsJ \ "featuresCol").extract[String])
      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])
      model.set(model.maxIter, (paramsJ \ "maxIter").extract[Int])
      model.set(model.seed, (paramsJ \ "seed").extract[Long])

      logInfo(s"KMedoidsModel successfully loaded from $path")
      model
    }
  }
}

/** Parameters for CLARA (Clustering Large Applications).
  */
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
        row.getAs[Vector](0)
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
