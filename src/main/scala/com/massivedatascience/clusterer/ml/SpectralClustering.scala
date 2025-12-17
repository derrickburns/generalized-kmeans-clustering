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

import com.massivedatascience.clusterer.ml.df.SpectralGraph
import com.massivedatascience.clusterer.ml.df.kernels.MercerKernel
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{ Estimator, Model }
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{ DataFrame, Dataset }
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ IntegerType, StructType }

/** Parameters for Spectral Clustering. */
trait SpectralClusteringParams
    extends Params
    with HasFeaturesCol
    with HasPredictionCol
    with HasMaxIter
    with HasTol
    with HasSeed {

  /** Number of clusters. Default: 2. */
  final val k   = new IntParam(this, "k", "Number of clusters", ParamValidators.gt(1))
  def getK: Int = $(k)

  /** Type of Mercer kernel for affinity computation.
    *
    * Supported: "rbf" (Gaussian), "laplacian", "polynomial", "linear"
    *
    * Default: "rbf"
    */
  final val kernelType      = new Param[String](this, "kernelType", "Mercer kernel type")
  def getKernelType: String = $(kernelType)

  /** Gamma parameter for RBF/Laplacian kernels.
    *
    * For RBF: k(x,y) = exp(-γ ||x-y||²) Common choices: 1/dim, 1/median_distance, "auto"
    *
    * Default: 1.0
    */
  final val gamma      = new DoubleParam(this, "gamma", "Kernel bandwidth", ParamValidators.gt(0.0))
  def getGamma: Double = $(gamma)

  /** Affinity matrix construction type.
    *
    * Supported: "full", "knn", "epsilon"
    *
    * Default: "full"
    */
  final val affinityType      = new Param[String](this, "affinityType", "Affinity construction type")
  def getAffinityType: String = $(affinityType)

  /** Number of nearest neighbors for k-NN affinity.
    *
    * Only used when affinityType = "knn". Default: 10
    */
  final val numNeighbors   = new IntParam(this, "numNeighbors", "k for k-NN", ParamValidators.gt(0))
  def getNumNeighbors: Int = $(numNeighbors)

  /** Epsilon radius for ε-neighborhood affinity.
    *
    * Only used when affinityType = "epsilon". Default: 1.0
    */
  final val epsilon      =
    new DoubleParam(this, "epsilon", "Neighborhood radius", ParamValidators.gt(0.0))
  def getEpsilon: Double = $(epsilon)

  /** Graph Laplacian type.
    *
    * Supported: "symmetric" (Ng-Jordan-Weiss), "unnormalized", "randomWalk"
    *
    * Default: "symmetric"
    */
  final val laplacianType      = new Param[String](this, "laplacianType", "Laplacian type")
  def getLaplacianType: String = $(laplacianType)

  /** Whether to use Nyström approximation for large datasets.
    *
    * When true, uses a subset of landmark points to approximate eigenvectors. Recommended for n >
    * 5000.
    *
    * Default: false
    */
  final val useNystrom       = new BooleanParam(this, "useNystrom", "Use Nystrom approximation")
  def getUseNystrom: Boolean = $(useNystrom)

  /** Number of landmark points for Nyström approximation.
    *
    * Only used when useNystrom = true. Default: 200
    */
  final val numLandmarks   =
    new IntParam(this, "numLandmarks", "Landmarks for Nystrom", ParamValidators.gt(0))
  def getNumLandmarks: Int = $(numLandmarks)

  /** Degree parameter for polynomial kernel.
    *
    * Only used when kernelType = "polynomial". Default: 3
    */
  final val degree   = new IntParam(this, "degree", "Polynomial degree", ParamValidators.gt(0))
  def getDegree: Int = $(degree)

  setDefault(
    k             -> 2,
    kernelType    -> "rbf",
    gamma         -> 1.0,
    affinityType  -> "full",
    numNeighbors  -> 10,
    epsilon       -> 1.0,
    laplacianType -> "symmetric",
    useNystrom    -> false,
    numLandmarks  -> 200,
    degree        -> 3,
    maxIter       -> 100,
    tol           -> 1e-6,
    seed          -> 42L
  )

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    require(
      schema.fieldNames.contains($(featuresCol)),
      s"Features column '${$(featuresCol)}' not found"
    )
    schema.add($(predictionCol), IntegerType, nullable = false)
  }
}

/** Spectral Clustering using graph Laplacian eigenvectors.
  *
  * Spectral clustering transforms data into a spectral embedding space using eigenvectors of a
  * graph Laplacian, then applies k-means in this space. This captures non-linear cluster structure.
  *
  * ==Algorithm==
  *
  *   1. Construct affinity matrix W using a Mercer kernel (RBF is most common)
  *   1. Compute graph Laplacian L (typically symmetric normalized)
  *   1. Find k smallest eigenvectors of L (excluding the trivial constant eigenvector)
  *   1. Form embedding matrix U with eigenvectors as columns
  *   1. Row-normalize U (for symmetric Laplacian)
  *   1. Run k-means on rows of U
  *
  * ==Laplacian Types==
  *
  *   - '''Symmetric (Ng-Jordan-Weiss):''' L_sym = I - D^(-1/2) W D^(-1/2)
  *   - '''Unnormalized (Shi-Malik):''' L = D - W
  *   - '''Random Walk:''' L_rw = I - D^(-1) W
  *
  * ==Affinity Construction==
  *
  *   - '''Full:''' Compute k(x_i, x_j) for all pairs (O(n²))
  *   - '''k-NN:''' Only connect k-nearest neighbors (sparser, faster)
  *   - '''ε-neighborhood:''' Only connect points within distance ε
  *
  * ==Scalability==
  *
  * For large n, use `useNystrom = true` which approximates the embedding using a subset of landmark
  * points. This reduces complexity from O(n³) to O(nm²) where m is the number of landmarks.
  *
  * ==Example Usage==
  *
  * {{{
  * val spectral = new SpectralClustering()
  *   .setK(3)
  *   .setKernelType("rbf")
  *   .setGamma(1.0)
  *   .setLaplacianType("symmetric")
  *   .setAffinityType("full")
  *
  * val model = spectral.fit(dataset)
  * val predictions = model.transform(dataset)
  * }}}
  *
  * @see
  *   Ng, Jordan, Weiss (2002). On Spectral Clustering: Analysis and an Algorithm.
  * @see
  *   von Luxburg (2007). A Tutorial on Spectral Clustering.
  * @param uid
  *   Unique identifier
  */
class SpectralClustering(override val uid: String)
    extends Estimator[SpectralClusteringModel]
    with SpectralClusteringParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("spectral"))

  // Parameter setters
  def setK(value: Int): this.type                = set(k, value)
  def setKernelType(value: String): this.type    = set(kernelType, value)
  def setGamma(value: Double): this.type         = set(gamma, value)
  def setAffinityType(value: String): this.type  = set(affinityType, value)
  def setNumNeighbors(value: Int): this.type     = set(numNeighbors, value)
  def setEpsilon(value: Double): this.type       = set(epsilon, value)
  def setLaplacianType(value: String): this.type = set(laplacianType, value)
  def setUseNystrom(value: Boolean): this.type   = set(useNystrom, value)
  def setNumLandmarks(value: Int): this.type     = set(numLandmarks, value)
  def setDegree(value: Int): this.type           = set(degree, value)
  def setFeaturesCol(value: String): this.type   = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setMaxIter(value: Int): this.type          = set(maxIter, value)
  def setTol(value: Double): this.type           = set(tol, value)
  def setSeed(value: Long): this.type            = set(seed, value)

  override def fit(dataset: Dataset[_]): SpectralClusteringModel = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()
    logInfo(
      s"Starting Spectral Clustering: k=${$(k)}, kernel=${$(kernelType)}, laplacian=${$(laplacianType)}"
    )

    df.cache()
    try {
      // Collect points
      val points = df.select($(featuresCol)).collect().map(_.getAs[Vector](0))
      val n      = points.length
      logInfo(s"Data: $n points")

      // Create kernel
      val kernel = createKernel()

      // Compute spectral embedding
      val embedding = if ($(useNystrom) || n > 5000) {
        logInfo(s"Using Nyström approximation with ${$(numLandmarks)} landmarks")
        SpectralGraph.nystromApproximation(
          points,
          kernel,
          $(numLandmarks),
          $(k),
          $(seed)
        )
      } else {
        computeFullEmbedding(points, kernel)
      }

      logInfo(s"Computed ${embedding.length} spectral embeddings of dimension ${$(k)}")

      // Run k-means on embedding
      val (centers, assignments) = runKMeansOnEmbedding(embedding)

      logInfo(s"Spectral clustering complete with ${$(k)} clusters")

      val model = new SpectralClusteringModel(uid, centers, embedding)
      copyValues(model.setParent(this))
    } finally {
      df.unpersist()
    }
  }

  /** Create the Mercer kernel based on parameters. */
  private def createKernel(): MercerKernel = {
    MercerKernel.create(
      $(kernelType),
      gamma = $(gamma),
      degree = $(degree)
    )
  }

  /** Compute full spectral embedding (for smaller datasets). */
  private def computeFullEmbedding(points: Array[Vector], kernel: MercerKernel): Array[Vector] = {
    logInfo(s"Building ${$(affinityType)} affinity matrix")
    val affinity = SpectralGraph.buildAffinity(
      points,
      kernel,
      $(affinityType),
      $(numNeighbors),
      $(epsilon)
    )

    logInfo(s"Computing ${$(laplacianType)} Laplacian")
    val laplacian = SpectralGraph.computeLaplacian(affinity, $(laplacianType))

    logInfo(s"Computing ${$(k)} smallest eigenvectors")
    val (eigenvalues, eigenvectors) = SpectralGraph.computeSmallestEigenvectors(
      laplacian,
      $(k) + 1, // +1 because first eigenvector is constant
      maxIter = $(maxIter).toInt,
      tol = $(tol),
      seed = $(seed)
    )

    logInfo(f"Eigenvalues: ${eigenvalues.take(5).map(v => f"$v%.4f").mkString(", ")}...")

    // Skip first eigenvector (constant for connected graphs)
    val selectedEigenvectors = if (eigenvalues.length > $(k)) {
      eigenvectors.drop(1).take($(k))
    } else {
      eigenvectors.take($(k))
    }

    SpectralGraph.buildEmbedding(selectedEigenvectors, $(laplacianType))
  }

  /** Run k-means clustering on the spectral embedding. */
  private def runKMeansOnEmbedding(
      embedding: Array[Vector]
  ): (Array[Vector], Array[Int]) = {
    val kVal   = $(k)
    val maxIt  = $(maxIter)
    val tolVal = $(tol)
    val rng    = new scala.util.Random($(seed))

    // Initialize centers using k-means++
    val centers = kMeansPlusPlusInit(embedding, kVal, rng)

    var currentCenters = centers
    var iter           = 0
    var converged      = false

    while (iter < maxIt && !converged) {
      iter += 1

      // Assign points to nearest centers
      val assignments = embedding.map { point =>
        currentCenters.zipWithIndex.minBy { case (center, _) =>
          squaredEuclidean(point, center)
        }._2
      }

      // Update centers
      val newCenters = (0 until kVal).map { c =>
        val clusterPoints = embedding.zip(assignments).filter(_._2 == c).map(_._1).toSeq
        if (clusterPoints.nonEmpty) {
          computeCentroid(clusterPoints)
        } else {
          currentCenters(c)
        }
      }.toArray

      // Check convergence
      val maxMovement = currentCenters
        .zip(newCenters)
        .map { case (old, newC) =>
          squaredEuclidean(old, newC)
        }
        .max

      if (maxMovement < tolVal * tolVal) {
        converged = true
      }

      currentCenters = newCenters
    }

    val finalAssignments = embedding.map { point =>
      currentCenters.zipWithIndex.minBy { case (center, _) =>
        squaredEuclidean(point, center)
      }._2
    }

    (currentCenters, finalAssignments)
  }

  /** K-means++ initialization. */
  private def kMeansPlusPlusInit(
      points: Array[Vector],
      k: Int,
      rng: scala.util.Random
  ): Array[Vector] = {
    val n       = points.length
    val centers = new Array[Vector](k)

    // First center: random
    centers(0) = points(rng.nextInt(n))

    // Remaining centers: probability proportional to squared distance
    for (i <- 1 until k) {
      val distances  = points.map { p =>
        (0 until i).map(j => squaredEuclidean(p, centers(j))).min
      }
      val totalDist  = distances.sum
      val threshold  = rng.nextDouble() * totalDist
      var selected   = 0
      var cumulative = 0.0

      while (selected < n - 1 && cumulative < threshold) {
        cumulative += distances(selected)
        if (cumulative < threshold) selected += 1
      }

      centers(i) = points(selected)
    }

    centers
  }

  /** Squared Euclidean distance. */
  private def squaredEuclidean(a: Vector, b: Vector): Double = {
    val aArr = a.toArray
    val bArr = b.toArray
    var sum  = 0.0
    var i    = 0
    while (i < aArr.length) {
      val diff = aArr(i) - bArr(i)
      sum += diff * diff
      i += 1
    }
    sum
  }

  /** Compute centroid of points. */
  private def computeCentroid(points: Seq[Vector]): Vector = {
    val n   = points.length
    val dim = points.head.size
    val sum = new Array[Double](dim)

    points.foreach { p =>
      val arr = p.toArray
      var i   = 0
      while (i < dim) {
        sum(i) += arr(i)
        i += 1
      }
    }

    Vectors.dense(sum.map(_ / n))
  }

  override def copy(extra: ParamMap): SpectralClustering = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object SpectralClustering extends DefaultParamsReadable[SpectralClustering] {
  override def load(path: String): SpectralClustering = super.load(path)
}

/** Model produced by Spectral Clustering.
  *
  * Contains the cluster centers in the spectral embedding space and can transform new data.
  */
class SpectralClusteringModel(
    override val uid: String,
    val clusterCenters: Array[Vector],
    val trainingEmbedding: Array[Vector]
) extends Model[SpectralClusteringModel]
    with SpectralClusteringParams
    with MLWritable
    with Logging {

  /** Number of clusters. */
  def numClusters: Int = clusterCenters.length

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val df = dataset.toDF()

    // For spectral clustering, we need to embed new points
    // This requires computing affinity with training points - complex for out-of-sample
    // For simplicity, we use nearest neighbor in original space mapped to training labels

    val trainingEmbed = trainingEmbedding
    val centers       = clusterCenters

    val predictUDF = udf { (features: Vector) =>
      // Find nearest training point and use its embedding
      // This is a simple approximation for out-of-sample extension
      var minDist = Double.MaxValue
      var nearest = 0
      var i       = 0
      while (i < trainingEmbed.length) {
        val dist = squaredEuclideanStatic(features, trainingEmbed(i))
        if (dist < minDist) {
          minDist = dist
          nearest = i
        }
        i += 1
      }

      // Assign to nearest cluster center
      val embedding      = trainingEmbed(nearest)
      var minClusterDist = Double.MaxValue
      var cluster        = 0
      var c              = 0
      while (c < centers.length) {
        val dist = squaredEuclideanStatic(embedding, centers(c))
        if (dist < minClusterDist) {
          minClusterDist = dist
          cluster = c
        }
        c += 1
      }
      cluster
    }

    df.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  private def squaredEuclideanStatic(a: Vector, b: Vector): Double = {
    val aArr = a.toArray
    val bArr = b.toArray
    var sum  = 0.0
    var i    = 0
    while (i < aArr.length) {
      val diff = aArr(i) - bArr(i)
      sum += diff * diff
      i += 1
    }
    sum
  }

  override def copy(extra: ParamMap): SpectralClusteringModel = {
    val copied = new SpectralClusteringModel(uid, clusterCenters, trainingEmbedding)
    copyValues(copied, extra).setParent(parent)
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new SpectralClusteringModel.SpectralModelWriter(this)
}

object SpectralClusteringModel extends MLReadable[SpectralClusteringModel] {

  override def read: MLReader[SpectralClusteringModel] = new SpectralModelReader

  override def load(path: String): SpectralClusteringModel = super.load(path)

  private class SpectralModelWriter(instance: SpectralClusteringModel)
      extends MLWriter
      with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.Serialization

    override protected def saveImpl(path: String): Unit = {
      val spark = sparkSession
      logInfo(s"Saving SpectralClusteringModel to $path")

      // Save centers and embeddings
      import spark.implicits._
      val centersPath   = new Path(path, "centers").toString
      val embeddingPath = new Path(path, "embedding").toString

      val centersData = instance.clusterCenters.zipWithIndex.map { case (v, i) =>
        (i, v.toArray.toSeq)
      }
      spark.createDataFrame(centersData).toDF("id", "vector").write.parquet(centersPath)

      val embeddingData = instance.trainingEmbedding.zipWithIndex.map { case (v, i) =>
        (i, v.toArray.toSeq)
      }
      spark.createDataFrame(embeddingData).toDF("id", "vector").write.parquet(embeddingPath)

      val params: Map[String, Any] = Map(
        "k"             -> instance.getOrDefault(instance.k),
        "kernelType"    -> instance.getOrDefault(instance.kernelType),
        "gamma"         -> instance.getOrDefault(instance.gamma),
        "affinityType"  -> instance.getOrDefault(instance.affinityType),
        "laplacianType" -> instance.getOrDefault(instance.laplacianType),
        "featuresCol"   -> instance.getOrDefault(instance.featuresCol),
        "predictionCol" -> instance.getOrDefault(instance.predictionCol)
      )

      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaObj: Map[String, Any]             = Map(
        "layoutVersion"      -> LayoutVersion,
        "algo"               -> "SpectralClusteringModel",
        "sparkMLVersion"     -> org.apache.spark.SPARK_VERSION,
        "scalaBinaryVersion" -> getScalaBinaryVersion,
        "k"                  -> instance.numClusters,
        "uid"                -> instance.uid,
        "params"             -> params
      )

      val json = Serialization.write(metaObj)
      writeMetadata(path, json)
      logInfo(s"SpectralClusteringModel saved to $path")
    }
  }

  private class SpectralModelReader extends MLReader[SpectralClusteringModel] with Logging {
    import com.massivedatascience.clusterer.ml.df.persistence.PersistenceLayoutV1._
    import org.json4s.DefaultFormats
    import org.json4s.jackson.JsonMethods

    override def load(path: String): SpectralClusteringModel = {
      val spark = sparkSession
      logInfo(s"Loading SpectralClusteringModel from $path")

      val metaStr                               = readMetadata(path)
      implicit val formats: DefaultFormats.type = DefaultFormats
      val metaJ                                 = JsonMethods.parse(metaStr)

      val uid = (metaJ \ "uid").extract[String]
      val k   = (metaJ \ "k").extract[Int]

      val centersPath   = new Path(path, "centers").toString
      val embeddingPath = new Path(path, "embedding").toString

      val centersDF = spark.read.parquet(centersPath)
      val centers   = centersDF.orderBy("id").collect().map { row =>
        val seq = row.getAs[scala.collection.Seq[Double]](1)
        Vectors.dense(seq.toArray)
      }

      val embeddingDF = spark.read.parquet(embeddingPath)
      val embedding   = embeddingDF.orderBy("id").collect().map { row =>
        val seq = row.getAs[scala.collection.Seq[Double]](1)
        Vectors.dense(seq.toArray)
      }

      val paramsJ = metaJ \ "params"

      val model = new SpectralClusteringModel(uid, centers, embedding)
      model.set(model.k, k)
      model.set(model.kernelType, (paramsJ \ "kernelType").extract[String])
      model.set(model.gamma, (paramsJ \ "gamma").extract[Double])
      model.set(model.affinityType, (paramsJ \ "affinityType").extract[String])
      model.set(model.laplacianType, (paramsJ \ "laplacianType").extract[String])
      model.set(model.featuresCol, (paramsJ \ "featuresCol").extract[String])
      model.set(model.predictionCol, (paramsJ \ "predictionCol").extract[String])

      logInfo(s"SpectralClusteringModel loaded from $path")
      model
    }
  }
}
