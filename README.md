Generalized K-Means Clustering
=============================

This project generalizes the Spark MLLIB K-Means (v1.1.0) clusterer to support clustering of sparse
or dense data, in low or high dimension, using distances defined by
[Bregman divergences](http://www.cs.utexas.edu/users/inderjit/public_papers/bregmanclustering_jmlr.pdf) and
[generalized symmetrized Bregman Divergences] (http://www-users.cs.umn.edu/~banerjee/papers/13/bregman-metric.pdf).
This is in contrast to the standard Spark implementation that only supports dense, low-dimensional data
using the squared Euclidean distance function.

Be aware that work on this project is ongoing.  Parts of this project are being integrated into
upcoming releases of the Spark MLLIB clusterer.


### Batch Clusterer Usage

The simplest way to call the batch clusterer is to use the ```KMeans.train``` method, which
will return an instance of the ```KMeansModel``` object.

For dense data in a low dimension space using the squared Euclidean distance function,
one may simply call KMeans.train with the data and the desired number of clusters:

```scala
  import com.com.massivedatascience.clusterer.KMeans
  import org.apache.spark.mllib.linalg.Vector

  val model : KMeansModel = KMeans.train(data: RDD[Vector], k: Int)
```

The full signature of the ```KMeans.train``` method is:

```scala
  package com.massivedatascience.clusterer

  object KMeans {
  /**
   *
   * Train a K-Means model using Lloyd's algorithm.
   *
   *
   * @param data input data
   * @param k  number of clusters desired
   * @param maxIterations maximum number of iterations of Lloyd's algorithm
   * @param runs number of parallel clusterings to run
   * @param mode initialization algorithm to use
   * @param initializationSteps number of steps of the initialization algorithm
   * @param distanceFunctionNames the distance functions to use
   * @param kMeansImplName which k-means implementation to use
   * @param embeddingNames sequence of embeddings to use, from lowest dimension to greatest
   * @return K-Means model
   */
  def train(
    data: RDD[Vector],
    k: Int,
    maxIterations: Int = 20,
    runs: Int = 1,
    mode: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionNames: Seq[String] = Seq(PointOps.EUCLIDEAN),
    kMeansImplName: String = COLUMN_TRACKING,
    embeddingNames: List[String] = List(Embeddings.IDENTITY_EMBEDDING))
  : KMeansModel = { ???
}
```

For high dimensional data, one way wish to embed the data into a lower dimension before clustering to
reduce running time.

For time series data,
[the Haar Transform](http://www.cs.gmu.edu/~jessica/publications/ikmeans_sdm_workshop03.pdf)
has been used successfully to reduce running time while maintaining or improving quality.

For high-dimensional sparse data,
[random indexing](http://en.wikipedia.org/wiki/Random_indexing)
can be used to map the data into a low dimensional dense space.

One may also perform clustering recursively, using lower dimensional clustering to derive initial
conditions for higher dimensional clustering.

```KMeans.train``` and ```KMeans.trainViaSubsampling``` also support recursive clustering.
The former applies a list of embeddings to the input data,
while the latter applies the same embedding iteratively on the data.


```scala
  package com.massivedatascience.clusterer

  object KMeans {

  /**
   *
   * Train a K-Means model by recursively sub-sampling the data via the provided embedding.
   *
   * @param data input data
   * @param k  number of clusters desired
   * @param maxIterations maximum number of iterations of Lloyd's algorithm
   * @param runs number of parallel clusterings to run
   * @param initializerName initialization algorithm to use
   * @param initializationSteps number of steps of the initialization algorithm
   * @param distanceFunctionName the distance functions to use
   * @param clustererName which k-means implementation to use
   * @param embeddingName embedding to use recursively
   * @param depth number of times to recurse
   * @return K-Means model
   */
  def trainViaSubsampling(
    data: RDD[WeightedVector],
    k: Int,
    maxIterations: Int = 20,
    runs: Int = 1,
    initializerName: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionName: String = PointOps.EUCLIDEAN,
    clustererName: String = COLUMN_TRACKING,
    embeddingName: String = Embeddings.HAAR_EMBEDDING,
    depth: Int = 2)
  : (KMeansModel, KMeansResults) = ???
}
```

### Batch Clusterer Examples

Here are examples of using K-Means recursively.

```scala
object RecursiveKMeans {

  import KMeans._

  def sparseTrain(raw: RDD[Vector], k: Int): KMeansModel = {
    KMeans.train(raw, k,
      embeddingNames = List(Embeddings.LOW_DIMENSIONAL_RI, Embeddings.MEDIUM_DIMENSIONAL_RI, Embeddings.HIGH_DIMENSIONAL_RI))
  }

  def timeSeriesTrain(raw: RDD[Vector], k: Int): KMeansModel = {
    val dim = raw.first().toArray.length
    require(dim > 0)
    val maxDepth = Math.floor(Math.log(dim) / Math.log(2.0)).toInt
    val target = Math.max(maxDepth - 4, 0)
    KMeans.trainViaSubsampling(raw, k, depth = target)
  }
}
```

At minimum, you must provide the RDD of ```Vector```s to cluster and the number of clusters you
desire. The method will return a ```KMeansModel``` of the clustering.

### K-Means Model

The value returned from a K-Means clustering is the ```KMeansModel```.

```scala
class KMeansModel(pointOps: BregmanPointOps, centers: Array[BregmanCenter])
  extends Serializable {

  /** The number of clusters. **/
  lazy val k: Int = ???

  /**
   Returns the cluster centers.  N.B. These are in the embedded space where the clustering
   takes place, which may be different from the space of the input vectors!
   */
  lazy val clusterCenters: Array[Vector] = ???

  /** Returns the cluster index that a given point belongs to. */
  def predict(point: Vector): Int = ???

  /** Returns the closest cluster index and distance to that cluster. */
  def predictClusterAndDistance(point: Vector): (Int, Double) = ???

  /** Maps given points to their cluster indices. */
  def predict(points: RDD[Vector]): RDD[Int] = ???

  /** Maps given points to their cluster indices. */
  def predict(points: JavaRDD[Vector]): JavaRDD[java.lang.Integer] = ???

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  def computeCost(data: RDD[Vector]): Double = ???
}
```

### Distance Functions

The Spark MLLIB clusterer is good at one thing: clustering low-medium dimensional data using
squared Euclidean distance as the metric into a fixed number of clusters.

However, there are many interesting distance functions other than Euclidean distance.
It is far from trivial to adapt the Spark MLLIB clusterer to these other distance functions. In fact, recent
modification to the Spark implementation have made it even more difficult.

This project decouples the distance function from the clusterer implementation, allowing the
end-user the opportunity to define an alternative distance function in just a few lines of code.

The most general class of distance functions that work with the K-Means algorithm are called Bregman divergences.
This project implements several Bregman divergences, including the squared Euclidean distance,
the [Kullback-Leibler divergence](http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence),
the logistic loss divergence, the Itakura-Saito divergence, and the generalized I-divergence.

Computing distance for a given divergence and other distance-functions specific operations needed
for the implementation of the clusterer are provided by the ```PointOps``` trait.  Several
implementations of this trait are provided.

When selecting a distance function, consider the domain of the input data.  For example, frequency
data is integral. Similarity of frequencies or distributions are best performed using the
Kullback-Leibler divergence.

| Name (```PointOps._```)            | Space | Divergence              | Input   |
|----------------------------------|-------|-------------------------|---------|
| ```EUCLIDEAN```                  | R^d   |Euclidean                |         |
| ```RELATIVE_ENTROPY```           | R+^d  |Kullback-Leibler         | Dense   |
| ```DISCRETE_KL```                | N+^d  |Kullback-Leibler         | Dense   |
| ```DISCRETE_SMOOTHED_KL```       | N^d   |Kullback-Leibler         | Dense   |
| ```SPARSE_SMOOTHED_KL```         | R+^d  |Kullback-Leibler         | Sparse  |
| ```LOGISTIC_LOSS```              | R     |Logistic Loss            |         |
| ```GENERALIZED_I```              | R     |Generalized I-divergence |         |


### Initialization/seeding algorithm

This clusterer separates the initialization step (the seeding of the initial clusters) from the main clusterer.
This allows for new initialization methods beyond the standard "random" and "K-Means ||" algorithms,
including initialization methods that have different numbers of initial clusters.

There are two pre-defined seeding algorithms.

| Name (```KMeans._```)            | Algorithm                         |
|----------------------------------|-----------------------------------|
| ```RANDOM```                  | Random selection of initial k centers |
| ```K_MEANS_PARALLEL```           | [K-Means Parallel](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf) |

You may provide alternative seeding algorithms using the lower level interface as shown in ```KMeans.train```.

### Dimensionality Reduction via Embeddings

One can embed points into a lower dimensional spaces before clustering in order to speed the
computation.


| Name (```Embeddings._```)         | Algorithm                                                   |
|-------------------------------|-------------------------------------------------------------|
| ```IDENTITY_EMBEDDING```      | Identity                                                    |
| ```HAAR_EMBEDDING```          | [Haar Transform](http://www.cs.gmu.edu/~jessica/publications/ikmeans_sdm_workshop03.pdf) |
| ```LOW_DIMENSIONAL_RI```      | [Random Indexing](https://www.sics.se/~mange/papers/RI_intro.pdf) with dimension 64 and epsilon = 0.1 |
| ```MEDIUM_DIMENSIONAL_RI```   | Random Indexing with dimension 256 and epsilon = 0.1        |
| ```HIGH_DIMENSIONAL_RI```     | Random Indexing with dimension 1024 and epsilon = 0.1       |
| ```SYMMETRIZING_KL_EMBEDDING```     | Symmetrizing KL Embedding       |


### K-Means Implementations

There are three implementations of the K-Means algorithm. Use ```SIMPLE```.  The others
are experimental for performance testing.

| Name (```KMeans._```)            | Algorithm                         |
|----------------------------------|-----------------------------------|
| ```SIMPLE```                  | recomputes closest assignments each iteration |
| ```TRACKING```           |  clusterer tracks last assignments in combined point/assignmentRDD |
| ```COLUMN_TRACKING```           |  clusterer tracks last assignments in separate RDDs |



### Other Differences with Spark MLLIB 1.2 K-Means Clusterer

There are several other differences with this clusterer and the Spark K-Means clusterer.

#### Variable number of clusters

This clusterer may produce fewer than `k` clusters when `k` are requested.  This may sound like a
problem, but your data may not cluster into `k` clusters!
The Spark implementation duplicates cluster centers, resulting in useless computation.
The ```COLUMN_TRACKING``` implementation replenishes empty clusters with
new clusters so that the desired number of clusters is almost always provided.

#### Faster K-Means || implementation

This clusterer uses the K-Means clustering step in the [K-Means ||
initialization](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf) process.
This is much faster, since all cores are utilized versus just one.

Additionally, this implementation performs the implementation in time quadratic in the number of
clusters, whereas the Spark implementation takes time cubic in the number of clusters.

#### Sparse Data

This clusterer works on dense and sparce data.  However, for best performance, we recommend that
you convert youd sparse data into dense data before clustering.
In high dimensions (say > 1024), it is recommended that you embed your sparse data into a lower
dimensional dense space using random indexing.

### Scalability and Testing

This clusterer has been used to cluster millions of points in 700+ dimensional space using an
information theoretic distance function (Kullback-Leibler).

### Internals

#### Bregman Divergences

Underlying ```PointOps``` are the supporting Bregman divergences. The ```BregmanDivergence``` trait
 encapsulates the Bregman Divergence definition.

```scala
trait BregmanDivergence {

  /**
   * F is any convex function.
   *
   * @param v input
   * @return F(v)
   */
  def F(v: Vector): Double

  /**
   * Gradient of F
   *
   * @param v input
   * @return  gradient of F when at v
   */
  def gradF(v: Vector): Vector

  /**
   * F applied to homogeneous coordinates.
   *
   * @param v input
   * @param w weight
   * @return  F(v/w)
   */
  def F(v: Vector, w: Double): Double

  /**
   * Gradient of F, applied to homogeneous coordinates
   * @param v input
   * @param w weight
   * @return  gradient(v/w)
   */
  def gradF(v: Vector, w: Double): Vector
}
```

Several Bregman Divergences are provided:

```scala
/**
 * The squared Euclidean distance function is defined on points in R**n
 *
 * http://en.wikipedia.org/wiki/Euclidean_distance
 */
object SquaredEuclideanDistanceDivergence extends BregmanDivergence

/**
 * The Kullback-Leibler divergence is defined on points on a simplex in R+ ** n
 *
 * If we know that the points are on the simplex, then we may simplify the implementation
 * of KL divergence.  This trait implements that simplification.
 *
 * http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
 *
 */
object RealKLSimplexDivergence extends BregmanDivergence

/**
 * The Kullback-Leibler divergence is defined on points on a simplex in N+ ** n
 *
 * If we know that the points are on the simplex, then we may simplify the implementation
 * of KL divergence.  This trait implements that simplification.
 *
 * http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
 *
 */
object NaturalKLSimplexDivergence extends BregmanDivergence

/**
 * The generalized Kullback-Leibler divergence is defined on points on R+ ** n
 *
 * http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
 *
 */
object RealKLDivergence extends BregmanDivergence

/**
 * The generalized Kullback-Leibler divergence is defined on points on N+ ** n
 *
 * http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
 *
 */
object NaturalKLDivergence extends BregmanDivergence

/**
 * The generalized I-Divergence is defined on points in R**n
 */
object GeneralizedIDivergence extends BregmanDivergence

/**
 * The Logistic loss divergence is defined on points in (0.0,1.0)
 *
 * Logistic loss is the same as KL Divergence with the embedding into R**2
 *
 *    x => (x, 1.0 - x)
 */
object LogisticLossDivergence extends BregmanDivergence

/**
 * The Itakura-Saito Divergence is defined on points in R+ ** n
 *
 * http://en.wikipedia.org/wiki/Itakura%E2%80%93Saito_distance
 */

object ItakuraSaitoDivergence extends BregmanDivergence

```

#### Distance Functions

This clusterer abstracts the distance function, as described above, making it extensible.

The key is to create three new abstractions: point, cluster center, and centroid.  The base
implementation constructs centroids incrementally, then converts them to cluster centers.
The initialization of the cluster centers converts
points to cluster centers.  These abstractions are easy to understand and easy to implement.

```PointOps``` implement fast method for computing distances, taking advantage of the
characteristics of the data to define the fastest methods for evaluating Bregman divergences.

```scala
trait BregmanPointOps  {
  def distance(p: BregmanPoint, c: BregmanCenter): Double = ???
  def toCenter(v: WeightedVector): BregmanCenter = ???
  def toPoint(v: WeightedVector): BregmanPoint =  ???
  def centerMoved(v: BregmanPoint, w: BregmanCenter): Boolean = ???
}
```
Pull requests offering additional distance functions (http://en.wikipedia.org/wiki/Bregman_divergence)
are welcome.








