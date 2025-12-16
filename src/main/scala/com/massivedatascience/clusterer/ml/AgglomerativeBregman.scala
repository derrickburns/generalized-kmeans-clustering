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

package com.massivedatascience.clusterer.ml

import com.massivedatascience.clusterer.ml.df.BregmanKernel
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{ Estimator, Model }
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{
  DefaultParamsReadable,
  DefaultParamsWritable,
  Identifiable,
  MLReadable,
  MLReader,
  MLWritable,
  MLWriter
}
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

import scala.collection.mutable

/** Parameters for Agglomerative Bregman clustering.
  */
trait AgglomerativeBregmanParams
    extends Params
    with HasFeaturesCol
    with HasPredictionCol
    with HasSeed {

  /** Target number of clusters. */
  final val numClusters: IntParam = new IntParam(
    this,
    "numClusters",
    "Target number of clusters",
    ParamValidators.gt(0)
  )
  def getNumClusters: Int         = $(numClusters)

  /** Distance threshold for merging (alternative to numClusters). If set > 0, clustering stops when
    * min merge distance exceeds threshold.
    */
  final val distanceThreshold: DoubleParam = new DoubleParam(
    this,
    "distanceThreshold",
    "Distance threshold for merging",
    ParamValidators.gtEq(0.0)
  )
  def getDistanceThreshold: Double         = $(distanceThreshold)

  /** Linkage criterion: "single", "complete", "average", "ward", "centroid".
    *   - single: min distance between any two points
    *   - complete: max distance between any two points
    *   - average: average distance between all pairs
    *   - ward: minimize within-cluster variance increase (SE only)
    *   - centroid: distance between cluster centroids
    */
  final val linkage: Param[String] = new Param[String](
    this,
    "linkage",
    "Linkage criterion",
    ParamValidators.inArray(Array("single", "complete", "average", "ward", "centroid"))
  )
  def getLinkage: String           = $(linkage)

  /** Bregman divergence: "squaredEuclidean", "kl", etc. */
  final val divergence: Param[String] = new Param[String](
    this,
    "divergence",
    "Bregman divergence type"
  )
  def getDivergence: String           = $(divergence)

  /** Smoothing parameter for divergences with domain constraints. */
  final val smoothing: DoubleParam = new DoubleParam(
    this,
    "smoothing",
    "Smoothing parameter",
    ParamValidators.gt(0.0)
  )
  def getSmoothing: Double         = $(smoothing)

  setDefault(
    numClusters       -> 2,
    distanceThreshold -> 0.0,
    linkage           -> "average",
    divergence        -> "squaredEuclidean",
    smoothing         -> 1e-10,
    featuresCol       -> "features",
    predictionCol     -> "prediction"
  )
}

/** Agglomerative (bottom-up) hierarchical clustering with Bregman divergences.
  *
  * Starts with each point as its own cluster and iteratively merges the closest pair of clusters
  * until the desired number is reached.
  *
  * ==Algorithm==
  *
  *   1. Initialize: Each point is a singleton cluster 2. Compute pairwise distances/divergences
  *      between all clusters 3. Find and merge the closest pair 4. Update distances to the merged
  *      cluster 5. Repeat until numClusters reached or distanceThreshold exceeded
  *
  * ==Linkage Criteria==
  *
  *   - '''single:''' d(A,B) = min_{a∈A, b∈B} D(a,b) - Chaining tendency
  *   - '''complete:''' d(A,B) = max_{a∈A, b∈B} D(a,b) - Compact clusters
  *   - '''average:''' d(A,B) = 1/(|A||B|) Σ D(a,b) - Balance
  *   - '''ward:''' Minimize variance increase - Balanced sizes (SE only)
  *   - '''centroid:''' d(A,B) = D(μ_A, μ_B) - Fast for Bregman
  *
  * ==Example Usage==
  *
  * {{{
  * val hac = new AgglomerativeBregman()
  *   .setNumClusters(5)
  *   .setLinkage("average")
  *   .setDivergence("squaredEuclidean")
  *
  * val model = hac.fit(data)
  * val predictions = model.transform(data)
  *
  * // Get dendrogram for visualization
  * val dendrogram = model.dendrogram
  * }}}
  *
  * ==Scalability Note==
  *
  * Standard agglomerative clustering has O(n³) or O(n²log n) complexity. This implementation is
  * suitable for datasets up to ~10,000 points. For larger datasets, consider [[BisectingKMeans]]
  * (top-down approach).
  *
  * @see
  *   [[BisectingKMeans]] for top-down hierarchical clustering
  */
class AgglomerativeBregman(override val uid: String)
    extends Estimator[AgglomerativeBregmanModel]
    with AgglomerativeBregmanParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("agglomerative"))

  // Parameter setters
  def setNumClusters(value: Int): this.type          = set(numClusters, value)
  def setDistanceThreshold(value: Double): this.type = set(distanceThreshold, value)
  def setLinkage(value: String): this.type           = set(linkage, value)
  def setDivergence(value: String): this.type        = set(divergence, value)
  def setSmoothing(value: Double): this.type         = set(smoothing, value)
  def setFeaturesCol(value: String): this.type       = set(featuresCol, value)
  def setPredictionCol(value: String): this.type     = set(predictionCol, value)
  def setSeed(value: Long): this.type                = set(seed, value)

  override def fit(dataset: Dataset[_]): AgglomerativeBregmanModel = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()

    logInfo(
      s"Starting Agglomerative Bregman: numClusters=${$(numClusters)}, " +
        s"linkage=${$(linkage)}, divergence=${$(divergence)}"
    )

    df.cache()
    try {
      val startTime = System.currentTimeMillis()

      // Collect data points
      val points = df.select($(featuresCol)).collect().map(_.getAs[Vector](0))
      val n      = points.length

      if (n > 10000) {
        logWarning(s"Large dataset ($n points). Agglomerative clustering has O(n²) complexity.")
      }

      val kernel = createKernel()

      // Run agglomerative clustering
      val (assignments, dendrogram, mergeDistances) = runAgglomerative(points, kernel)

      val elapsed = System.currentTimeMillis() - startTime
      logInfo(s"Agglomerative clustering completed in ${elapsed}ms")

      // Compute final cluster centers
      val centers = computeCenters(points, assignments, kernel)

      val model = new AgglomerativeBregmanModel(
        uid,
        centers.map(Vectors.dense),
        dendrogram,
        mergeDistances
      )
      model.modelDivergence = $(divergence)
      model.modelSmoothing = $(smoothing)
      model.modelLinkage = $(linkage)
      model.kernel = kernel
      copyValues(model.setParent(this))

      // Training summary
      model.trainingSummary = Some(
        TrainingSummary(
          algorithm = "AgglomerativeBregman",
          k = $(numClusters),
          effectiveK = centers.length,
          dim = points.headOption.map(_.size).getOrElse(0),
          numPoints = n,
          iterations = n - $(numClusters),
          converged = true,
          distortionHistory = mergeDistances,
          movementHistory = Array.empty,
          assignmentStrategy = $(linkage),
          divergence = $(divergence),
          elapsedMillis = elapsed
        )
      )

      model
    } finally {
      df.unpersist()
    }
  }

  /** Run the agglomerative clustering algorithm. */
  private def runAgglomerative(
      points: Array[Vector],
      kernel: BregmanKernel
  ): (Array[Int], Array[MergeStep], Array[Double]) = {
    val n = points.length

    // Union-Find structure for tracking clusters
    val parent = Array.tabulate(n)(identity)
    val rank   = Array.fill(n)(0)
    val sizes  = Array.fill(n)(1)

    def find(x: Int): Int = {
      if (parent(x) != x) parent(x) = find(parent(x))
      parent(x)
    }

    def union(x: Int, y: Int): Int = {
      val px            = find(x)
      val py            = find(y)
      if (px == py) return px
      val (root, child) = if (rank(px) < rank(py)) (py, px) else (px, py)
      parent(child) = root
      sizes(root) += sizes(child)
      if (rank(px) == rank(py)) rank(root) += 1
      root
    }

    // Track cluster members for linkage computation
    val clusterMembers = mutable.Map[Int, mutable.Set[Int]]()
    for (i <- 0 until n) clusterMembers(i) = mutable.Set(i)

    // Compute initial pairwise distances
    val distanceMatrix = mutable.Map[(Int, Int), Double]()
    for (i <- 0 until n; j <- i + 1 until n) {
      val dist = computeLinkageDistance(
        clusterMembers(i).toSet,
        clusterMembers(j).toSet,
        points,
        kernel
      )
      distanceMatrix((i, j)) = dist
    }

    // Priority queue of cluster pairs by distance
    val pq = mutable.PriorityQueue[(Double, Int, Int)]()(Ordering.by(-_._1))
    for (((i, j), d) <- distanceMatrix) {
      pq.enqueue((d, i, j))
    }

    val dendrogram     = mutable.ArrayBuffer[MergeStep]()
    val mergeDistances = mutable.ArrayBuffer[Double]()
    var numActive      = n

    while (numActive > $(numClusters) && pq.nonEmpty) {
      val (dist, i, j) = pq.dequeue()

      // Check if this pair is still valid (neither merged)
      val pi = find(i)
      val pj = find(j)
      if (pi != pj) {
        // Check distance threshold
        if ($(distanceThreshold) > 0 && dist > $(distanceThreshold)) {
          // Stop merging
          numActive = 0 // Force exit
        } else {
          // Merge clusters
          val merged     = union(pi, pj)
          val oldCluster = if (merged == pi) pj else pi

          // Update cluster members
          clusterMembers(merged) ++= clusterMembers(oldCluster)
          clusterMembers.remove(oldCluster)

          // Record merge
          dendrogram += MergeStep(pi, pj, merged, dist)
          mergeDistances += dist

          // Compute distances to merged cluster
          for ((cid, members) <- clusterMembers if cid != merged) {
            val newDist = computeLinkageDistance(
              clusterMembers(merged).toSet,
              members.toSet,
              points,
              kernel
            )
            pq.enqueue((newDist, math.min(cid, merged), math.max(cid, merged)))
          }

          numActive -= 1
          logDebug(f"Merged clusters $pi and $pj at distance $dist%.4f, $numActive active")
        }
      }
    }

    // Create final assignments
    val assignments = Array.tabulate(n)(i => find(i))

    // Relabel to 0..k-1
    val uniqueLabels     = assignments.distinct.sorted
    val labelMap         = uniqueLabels.zipWithIndex.toMap
    val finalAssignments = assignments.map(labelMap)

    (finalAssignments, dendrogram.toArray, mergeDistances.toArray)
  }

  /** Compute linkage distance between two clusters. */
  private def computeLinkageDistance(
      clusterA: Set[Int],
      clusterB: Set[Int],
      points: Array[Vector],
      kernel: BregmanKernel
  ): Double = {
    $(linkage) match {
      case "single" =>
        var minDist = Double.MaxValue
        for (i <- clusterA; j <- clusterB) {
          val d = kernel.divergence(points(i), points(j))
          if (d < minDist) minDist = d
        }
        minDist

      case "complete" =>
        var maxDist = Double.MinValue
        for (i <- clusterA; j <- clusterB) {
          val d = kernel.divergence(points(i), points(j))
          if (d > maxDist) maxDist = d
        }
        maxDist

      case "average" =>
        var sum = 0.0
        for (i <- clusterA; j <- clusterB) {
          sum += kernel.divergence(points(i), points(j))
        }
        sum / (clusterA.size * clusterB.size)

      case "centroid" =>
        val centroidA = computeCentroid(clusterA, points, kernel)
        val centroidB = computeCentroid(clusterB, points, kernel)
        kernel.divergence(centroidA, centroidB)

      case "ward" =>
        // Ward's method: increase in total variance
        val centroidA = computeCentroid(clusterA, points, kernel)
        val centroidB = computeCentroid(clusterB, points, kernel)

        // ESS increase = |A||B|/(|A|+|B|) * ||μ_A - μ_B||²
        val nA   = clusterA.size.toDouble
        val nB   = clusterB.size.toDouble
        val dist = kernel.divergence(centroidA, centroidB)
        (nA * nB / (nA + nB)) * dist

      case _ =>
        throw new IllegalArgumentException(s"Unknown linkage: ${$(linkage)}")
    }
  }

  /** Compute centroid of a cluster using gradient mean for Bregman. */
  private def computeCentroid(
      cluster: Set[Int],
      points: Array[Vector],
      kernel: BregmanKernel
  ): Vector = {
    if (cluster.isEmpty) return Vectors.dense(Array.empty[Double])

    val dim     = points.head.size
    val gradSum = Array.fill(dim)(0.0)

    for (i <- cluster) {
      val grad = kernel.grad(points(i)).toArray
      for (d <- 0 until dim) gradSum(d) += grad(d)
    }

    val meanGrad = gradSum.map(_ / cluster.size)
    kernel.invGrad(Vectors.dense(meanGrad))
  }

  /** Compute final cluster centers. */
  private def computeCenters(
      points: Array[Vector],
      assignments: Array[Int],
      kernel: BregmanKernel
  ): Array[Array[Double]] = {
    val k       = assignments.max + 1
    val centers = Array.ofDim[Array[Double]](k)

    for (c <- 0 until k) {
      val clusterPoints = assignments.zipWithIndex.filter(_._1 == c).map(_._2).toSet
      centers(c) = computeCentroid(clusterPoints, points, kernel).toArray
    }

    centers
  }

  private def createKernel(): BregmanKernel = {
    BregmanKernel.create($(divergence), $(smoothing))
  }

  override def copy(extra: ParamMap): AgglomerativeBregman = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    require(
      schema($(featuresCol)).dataType.typeName == "vector",
      s"Features column must be Vector type"
    )
    schema
  }
}

object AgglomerativeBregman extends DefaultParamsReadable[AgglomerativeBregman] {
  override def load(path: String): AgglomerativeBregman = super.load(path)
}

/** Record of a single merge step in the dendrogram.
  *
  * @param cluster1
  *   first merged cluster ID
  * @param cluster2
  *   second merged cluster ID
  * @param merged
  *   resulting cluster ID
  * @param distance
  *   merge distance
  */
case class MergeStep(cluster1: Int, cluster2: Int, merged: Int, distance: Double)

/** Model from Agglomerative Bregman clustering.
  */
class AgglomerativeBregmanModel(
    override val uid: String,
    val clusterCenters: Array[Vector],
    val dendrogram: Array[MergeStep],
    val mergeDistances: Array[Double]
) extends Model[AgglomerativeBregmanModel]
    with AgglomerativeBregmanParams
    with MLWritable
    with Logging
    with HasTrainingSummary {

  private[ml] var modelDivergence: String = "squaredEuclidean"
  private[ml] var modelSmoothing: Double  = 1e-10
  private[ml] var modelLinkage: String    = "average"
  private[ml] var kernel: BregmanKernel   = _

  /** Cluster centers as vectors for downstream consumers/tests. */
  def clusterCentersAsVectors: Array[Vector] = clusterCenters

  /** Number of clusters. */
  def k: Int = clusterCenters.length

  /** Optional access without throwing. */
  def summaryOption: Option[TrainingSummary] = trainingSummary

  /** Get dendrogram for visualization. */
  def getDendrogram: Array[MergeStep] = dendrogram

  /** Get merge distances (useful for elbow method). */
  def getMergeDistances: Array[Double] = mergeDistances

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()

    if (kernel == null) {
      kernel = BregmanKernel.create(modelDivergence, modelSmoothing)
    }

    val bcKernel  = df.sparkSession.sparkContext.broadcast(kernel)
    val bcCenters = df.sparkSession.sparkContext.broadcast(clusterCenters)

    val predictUDF = udf { (features: Vector) =>
      var bestCluster = 0
      var bestDist    = Double.MaxValue

      for (c <- bcCenters.value.indices) {
        val dist = bcKernel.value.divergence(features, bcCenters.value(c))
        if (dist < bestDist) {
          bestDist = dist
          bestCluster = c
        }
      }

      bestCluster
    }

    df.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  override def copy(extra: ParamMap): AgglomerativeBregmanModel = {
    val copied = new AgglomerativeBregmanModel(uid, clusterCenters, dendrogram, mergeDistances)
    copyValues(copied, extra).setParent(parent)
    copied.modelDivergence = this.modelDivergence
    copied.modelSmoothing = this.modelSmoothing
    copied.modelLinkage = this.modelLinkage
    copied.kernel = this.kernel
    copied.trainingSummary = this.trainingSummary
    copied
  }

  override def transformSchema(schema: StructType): StructType = schema

  override def write: MLWriter = new AgglomerativeBregmanModel.AgglomerativeBregmanModelWriter(this)
}

object AgglomerativeBregmanModel extends MLReadable[AgglomerativeBregmanModel] {

  override def read: MLReader[AgglomerativeBregmanModel] = new AgglomerativeBregmanModelReader

  private class AgglomerativeBregmanModelWriter(instance: AgglomerativeBregmanModel)
      extends MLWriter
      with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession
      logInfo(s"Saving AgglomerativeBregmanModel to $path")

      // Save centers using standard format (center_id, weight=1.0, vector)
      val centersData = instance.clusterCenters.indices.map { i =>
        (i, 1.0, instance.clusterCenters(i))
      }
      val centersHash = writeCenters(spark, path, centersData)

      // Save dendrogram separately
      val dendrogramData = instance.dendrogram.zipWithIndex.map { case (m, i) =>
        (i, m.cluster1, m.cluster2, m.merged, m.distance)
      }.toSeq
      spark
        .createDataFrame(dendrogramData)
        .toDF("id", "cluster1", "cluster2", "merged", "distance")
        .write
        .parquet(s"$path/dendrogram")

      val params: Map[String, Any] = Map(
        "k"             -> instance.k,
        "featuresCol"   -> instance.getOrDefault(instance.featuresCol),
        "predictionCol" -> instance.getOrDefault(instance.predictionCol),
        "divergence"    -> instance.modelDivergence,
        "smoothing"     -> instance.modelSmoothing,
        "linkage"       -> instance.modelLinkage
      )

      val k   = instance.k
      val dim = instance.clusterCenters.headOption.map(_.size).getOrElse(0)

      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaObj: Map[String, Any]             = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "AgglomerativeBregmanModel",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "divergence"         -> instance.modelDivergence,
        "linkage"            -> instance.modelLinkage,
        "k"                  -> k,
        "dim"                -> dim,
        "uid"                -> instance.uid,
        "params"             -> params,
        "centers"            -> Map[String, Any](
          "count"    -> k,
          "ordering" -> "center_id ASC (0..k-1)",
          "storage"  -> "parquet"
        ),
        "dendrogram"         -> Map[String, Any](
          "count"   -> instance.dendrogram.length,
          "storage" -> "parquet"
        ),
        "checksums"          -> Map[String, String](
          "centersParquetSHA256" -> centersHash
        )
      )

      val json = Serialization.write(metaObj)
      writeMetadata(path, json)
      logInfo(s"AgglomerativeBregmanModel saved to $path")
    }
  }

  private class AgglomerativeBregmanModelReader
      extends MLReader[AgglomerativeBregmanModel]
      with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): AgglomerativeBregmanModel = {
      val spark = sparkSession
      logInfo(s"Loading AgglomerativeBregmanModel from $path")

      val metaStr                               = readMetadata(path)
      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaJ                                 = JsonMethods.parse(metaStr)

      val layoutVersion = (metaJ \ "layoutVersion").extract[Int]
      val k             = (metaJ \ "k").extract[Int]
      val dim           = (metaJ \ "dim").extract[Int]
      val uid           = (metaJ \ "uid").extract[String]

      val centersDF = readCenters(spark, path)
      val rows      = centersDF.collect()
      validateMetadata(layoutVersion, k, dim, rows.length)

      val centers = rows.sortBy(_.getInt(0)).map(_.getAs[Vector]("vector"))

      val dendrogram = spark.read
        .parquet(s"$path/dendrogram")
        .orderBy("id")
        .collect()
        .map(r => MergeStep(r.getInt(1), r.getInt(2), r.getInt(3), r.getDouble(4)))

      val mergeDistances = dendrogram.map(_.distance)

      val paramsJ    = metaJ \ "params"
      val divergence = (paramsJ \ "divergence").extract[String]
      val smoothing  = (paramsJ \ "smoothing").extract[Double]
      val linkage    = (paramsJ \ "linkage").extract[String]

      val model = new AgglomerativeBregmanModel(uid, centers, dendrogram, mergeDistances)
      model.modelDivergence = divergence
      model.modelSmoothing = smoothing
      model.modelLinkage = linkage

      model.set(model.featuresCol, (paramsJ \ "featuresCol").extract[String])
      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])

      logInfo(s"AgglomerativeBregmanModel loaded from $path")
      model
    }
  }
}
