Generalized K-Means Clustering
=============================

Release of this code is imminent.

This project generalizes the Spark MLLIB Batch K-Means (v1.1.0) clusterer
and the Spark MLLIB Streaming K-Means (v1.2.0) clusterer to support
* sparse or dense data
* parallel runs on non-equal target number of clusters
* low or high dimensional data
* using distances defined by [Bregman divergences](http://mark.reid.name/blog/meet-the-bregman-divergences.html)
* using all data points or a randomly sampled sub-set of the data points (a.k.a. mini-batches)
* backfilling empty clusters.

This is in contrast to the standard Spark implementation that only supports dense, low-dimensional data
using the squared Euclidean distance function.


### Introduction

The goal K-Means clustering is to produce a model of the clusters of a set of points that satisfies
certain optimality constraints. That model is called a K-Means model. It is fundamentally a set
of points and a function that defines the distance from an arbitrary point to a cluster center.

The K-Means algorithm computes a K-Means model using an iterative algorithm known as
[Lloyd's algorithm](http://en.wikipedia.org/wiki/Lloyd%27s_algorithm).
Each iteration of Lloyd's algorithm assigns a set of points to clusters, then updates the cluster
centers to acknowledge the assignment of the points to the cluster.

The update of clusters is a form of averaging.  Newly added points are averaged into the cluster
while (optionally) reassigned points are removed from their prior clusters.

While one can assign a point to a cluster using any distance function, Lloyd's algorithm only
converges for a certain set of distance functions called [Bregman divergences](http://www.cs.utexas.edu/users/inderjit/public_papers/bregmanclustering_jmlr.pdf). Bregman divergences
must define two methods, ```F```  to evaluate a function on a point and ```gradF``` to evaluate the
gradient of the function on a points.

```scala
trait BregmanDivergence {
  def F(v: Vector): Double

  def gradF(v: Vector): Vector
}
```

For example, by defining ```F``` to be the vector norm (i.e. the sum of the squares of the
coordinates), one gets a distance function that equals the square of the well known Euclidean
distance. We name it the ```SquaredEuclideanDistanceDivergence```.

For efficient repeated computation of distance between a fixed set of points and varying cluster
centers, is it convenient to pre-compute certain information and associate that information with
the point or the cluster center.  We call the classes that represent those enriched points ```BregmanPoint```s.
We call the classes that represent those enriched cluster centers ```BregmanCenter```s.  Users
of this package do not construct instances of these objects directly.

```scala
trait BregmanPoint

trait BregmanCenter
```

Internally, we enrich a Bregman divergence with a set of commonly used operations.
The enriched trait is the ```BregmanPointOps```.

```scala
trait BregmanPointOps  {
  type P = BregmanPoint
  type C = BregmanCenter

  val divergence: BregmanDivergence

  def toPoint(v: WeightedVector): P

  def toCenter(v: WeightedVector): C

  def centerMoved(v: P, w: C): Boolean

  def findClosest(centers: IndexedSeq[C], point: P): (Int, Double)

  def findClosestCluster(centers: IndexedSeq[C], point: P): Int

  def distortion(data: RDD[P], centers: IndexedSeq[C])

  def pointCost(centers: IndexedSeq[C], point: P): Double

  def distance(p: BregmanPoint, c: BregmanCenter): Double
}
```

The instance of ```BregmanPointOps``` that supports the ```SquaredEuclideanDistanceDivergence``` is
the ```SquaredEuclideanPointOps```.


With these definitions, we define our realization of a k-means model, ```KMeansModel``` which
we enrich with operations to find closest clusters to a point and to compute distances:

```scala
trait KMeansModel {

  val pointOps: BregmanPointOps

  def centers: IndexedSeq[BregmanCenter]


  def predict(point: Vector): Int

  def predictClusterAndDistance(point: Vector): (Int, Double)

  def predict(points: RDD[Vector]): RDD[Int]

  def predict(points: JavaRDD[Vector]): JavaRDD[java.lang.Integer]

  def computeCost(data: RDD[Vector]): Double


  def predictWeighted(point: WeightedVector): Int

  def predictClusterAndDistanceWeighted(point: WeightedVector): (Int, Double)

  def predictWeighted(points: RDD[WeightedVector]): RDD[Int]

  def computeCostWeighted(data: RDD[WeightedVector]): Double


  def predictBregman(point: BregmanPoint): Int

  def predictClusterAndDistanceBregman(point: BregmanPoint): (Int, Double)

  def predictBregman(points: RDD[BregmanPoint]): RDD[Int]

  def computeCostBregman(data: RDD[BregmanPoint): Double
}
```

### Provided Distance Functions and Divergences

In addition to the squared Euclidean distance function, this implementation provides several
other very useful distance functions. When selecting a distance function, consider the domain of
the input data.  For example, frequency
data is integral. Similarity of frequencies or distributions are best performed using the
Kullback-Leibler divergence.

| Name (```BregmanPointOps._```)   | Space | Divergence              | Input   |  Object |
|----------------------------------|-------|-------------------------|---------|---------|
| ```EUCLIDEAN```                  | R^d   |Euclidean                |         |  ```SquaredEuclideanPointOps```  |
| ```RELATIVE_ENTROPY```           | R+^d  |[Kullback-Leibler](http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)         | Dense   | ```DenseKLPointOps```    |
| ```DISCRETE_KL```                | N+^d  |Kullback-Leibler         | Dense   |  ```DiscreteKLPointOps```     |
| ```DISCRETE_SMOOTHED_KL```       | N^d   |Kullback-Leibler         | Dense   |  ```DiscreteSmoothedKLPointOps```   |
| ```SPARSE_SMOOTHED_KL```         | R+^d  |Kullback-Leibler         | Sparse  |  ```SparseRealKLPointOps```    |
| ```LOGISTIC_LOSS```              | R     |Logistic Loss            |         |   ```LogisticLossPointOps```   |
| ```GENERALIZED_I```              | R     |Generalized I-divergence |         |   ```GeneralizedIPointOps```   |


### Batch Clusterer Usage

A ```KMeansModel``` can be constructed from any set of cluster centers and distance function.
However, the more interesting models satisfy an optimality constraint.  If we sum the distances
from the points in a given set to their closest cluster centers, we get a number called the
"distortion" or "cost". A K-Means Model is locally optimal with respect to a set of points
if each cluster center is determined by the mean of the points assigned to that cluster.
Computing such a ```KMeansModel``` given a set of points is called "training" the model on those
points.


The simplest way to train a ```KMeansModel``` on a fixed set of points is to use the ```KMeans.train```
method.

For dense data in a low dimension space using the squared Euclidean distance function,
one may simply call ```KMeans.train``` with the data and the desired number of clusters:

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
    mode: String = KMeansInitializer.K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionNames: Seq[String] = Seq(BregmanPointOps.EUCLIDEAN),
    kMeansImplName: String = MultiKMeansClusterer.COLUMN_TRACKING,
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
    initializerName: String =  KMeansInitializer.K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionName: String = BregmanPointOps.EUCLIDEAN,
    clustererName: String = MultiKMeansClusterer.COLUMN_TRACKING,
    embeddingName: String = Embeddings.HAAR_EMBEDDING,
    depth: Int = 2)
  : KMeansModel= ???
}
```


#### Initialization/seeding algorithm

This clusterer separates the initialization step (the seeding of the initial clusters) from the main clusterer.
This allows for new initialization methods beyond the standard "random" and "K-Means ||" algorithms,
including initialization methods that have different numbers of initial clusters.

There are two pre-defined seeding algorithms.

| Name (``` KMeansInitializer._```)            | Algorithm                         |
|----------------------------------|-----------------------------------|
| ```RANDOM```                  | Random selection of initial k centers |
| ```K_MEANS_PARALLEL```           | a 5 step [K-Means Parallel implementation](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf) |

You may provide alternative seeding algorithms using the lower level interface as shown in ```KMeans.train```.

#### Dimensionality Reduction via Embeddings

One can embed points into a lower dimensional spaces before clustering in order to speed the
computation.


| Name (```Embeddings._```)         | Algorithm                                                   |
|-------------------------------|-------------------------------------------------------------|
| ```IDENTITY_EMBEDDING```      | Identity                                                    |
| ```HAAR_EMBEDDING```          | [Haar Transform](http://www.cs.gmu.edu/~jessica/publications/ikmeans_sdm_workshop03.pdf) |
| ```LOW_DIMENSIONAL_RI```      | [Random Indexing](https://www.sics.se/~mange/papers/RI_intro.pdf) with dimension 64 and epsilon = 0.1 |
| ```MEDIUM_DIMENSIONAL_RI```   | Random Indexing with dimension 256 and epsilon = 0.1        |
| ```HIGH_DIMENSIONAL_RI```     | Random Indexing with dimension 1024 and epsilon = 0.1       |
| ```SYMMETRIZING_KL_EMBEDDING```     | [Symmetrizing KL Embedding](http://www-users.cs.umn.edu/~banerjee/papers/13/bregman-metric.pdf)       |


#### K-Means Clusterer Implementations

There are standard varieties of K-Means Clusterer.

| Name (```MultiKMeansClusterer._```)            | Algorithm                         |
|----------------------------------|-----------------------------------|
| ```SIMPLE```             | standard clusterer that recomputes all centers and point assignments on each round |
| ```COLUMN_TRACKING```    | high performance variant of SIMPLE that performs less work on later rounds  |
| ```MINI_BATCH_10```      | mini-batch clusterer that samples 10% of the data on round to update centroids |


#### Examples

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

### Customizing the Components

One may define new ```BregmanDivergence```s, new ```BregmanPointOps``` and new ways of building
```KMeansModel```s using various constructors provided on companion objects.

####  Custom Bregman Divergences

In addition to the pre-defined Bregman divergences, one may construct new instances of
```BregmanDivergence``` using the companion object.
```scala
object BregmanDivergence {

  /**
   * Create a Bregman Divergence from
   * @param f any continuously-differentiable real-valued and strictly
   *          convex function defined on a closed convex set in R^^N
   * @param gradientF the gradient of f
   * @return a Bregman Divergence on that function
   */
  def apply(f: (Vector) => Double, gradientF: (Vector) => Vector): BregmanDivergence = ???
}
```

#### Custom Bregman Point Ops

One may access the pre-defined ```BregmanPointOps```  or one may
construct new instances of ```BregmanPointOps``` using the companion object.

```scala
object BregmanPointOps {
  def apply(distanceFunction: String): BregmanPointOps = ???

  def apply(d: BregmanDivergence): BregmanPointOps = ???

  def apply(d: BregmanDivergence, factor: Double): BregmanPointOps = ???
}
```

#### Custom K-Means Models

Training a K-Means model from a set of points using ```KMeans.train``` is one way to create a
```KMeansModel```.  However,
there are many others that are useful.  The ```KMeansModel``` companion object provides a number
of these constructors.


```scala
object KMeansModel {

  /**
   * Create a K-means model from given cluster centers and weights
   *
   * @param ops distance function
   * @param centers initial cluster centers in homogeneous coordinates
   * @param weights initial cluster weights
   * @return  k-means model
   */
  def fromVectorsAndWeights(
    ops: BregmanPointOps,
    centers: IndexedSeq[Vector],
    weights: IndexedSeq[Double]) = ???

  /**
   * Create a K-means model from given weighted vectors
   *
   * @param ops distance function
   * @param centers initial cluster centers as weighted vectors
   * @return  k-means model
   */
  def fromWeightedVectors[T <: WeightedVector : ClassTag](
    ops: BregmanPointOps,
    centers: IndexedSeq[T]) = ???

  /**
   * Create a K-means model by selecting a set of k points at random
   *
   * @param ops distance function
   * @param k number of centers desired
   * @param dim dimension of space
   * @param weight initial weight of points
   * @param seed random number seed
   * @return  k-means model
   */
  def usingRandomGenerator(ops: BregmanPointOps,
    k: Int,
    dim: Int,
    weight: Double,
    seed: Long = XORShiftRandom.random.nextLong()) = ???

  /**
   * Create a K-Means model using the KMeans++ algorithm on an initial set of candidate centers
   *
   * @param ops distance function
   * @param data initial candidate centers
   * @param weights initial weights
   * @param k number of clusters desired
   * @param perRound number of candidates to add per round
   * @param numPreselected initial sub-sequence of candidates to always select
   * @param seed random number seed
   * @return  k-means model
   */
  def fromCenters[T <: WeightedVector : ClassTag](
    ops: BregmanPointOps,
    data: IndexedSeq[T],
    weights: IndexedSeq[Double],
    k: Int,
    perRound: Int,
    numPreselected: Int,
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = ???

  /**
   * Create a K-Means Model from a streaming k-means model.
   *
   * @param streamingKMeansModel mutable streaming model
   * @return immutable k-means model
   */
  def fromStreamingModel(streamingKMeansModel: StreamingKMeansModel): KMeansModel = ???

  /**
   * Create a K-Means Model from a set of assignments of points to clusters
   *
   * @param ops distance function
   * @param points initial bregman points
   * @param assignments assignments of points to clusters
   * @return
   */
  def fromAssignments[T <: WeightedVector : ClassTag](
    ops: BregmanPointOps,
    points: RDD[T],
    assignments: RDD[Int]): KMeansModel = ???

  /**
   * Create a K-Means Model using K-Means || algorithm from an RDD of Bregman points.
   *
   * @param ops distance function
   * @param data initial points
   * @param k  number of cluster centers desired
   * @param numSteps number of iterations of k-Means ||
   * @param sampleRate fractions of points to use in weighting clusters
   * @param seed random number seed
   * @return  k-means model
   */
  def usingKMeansParallel[T <: WeightedVector : ClassTag](
    ops: BregmanPointOps,
    data: RDD[T],
    k: Int,
    numSteps: Int = 2,
    sampleRate: Double = 1.0,
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = ???

  /**
   * Construct a K-Means model using the Lloyd's algorithm given a set of initial
   * K-Means models.
   *
   * @param ops distance function
   * @param data points to fit
   * @param initialModels  initial k-means models
   * @param clusterer k-means clusterer to use
   * @param seed random number seed
   * @return  the best K-means model found
   */
  def usingLloyds[T <: WeightedVector : ClassTag](
    ops: BregmanPointOps,
    data: RDD[T],
    initialModels: Seq[KMeansModel],
    clusterer: MultiKMeansClusterer = new ColumnTrackingKMeans(),
    seed: Long = XORShiftRandom.random.nextLong()): KMeansModel = ???
}

```


### Other Differences with Spark MLLIB 1.2 K-Means Clusterer

There are several other differences with this clusterer and the Spark K-Means clusterer.

#### Variable number of clusters in parallel runs

The Spark MLLIB 1.2 cluster can perform multiple clusterings in parallel, each targetting the
same number of desired clusters.  In our implementation, the initial state of cluster
centers (i.e. the initial ```KMeansModel```s used) determine the target number of clusters.
Consequently, one can cluster for different values of number of clusters simultaneously.

#### Sparse Data

This clusterer works on dense and sparse data.  However, for best performance, we recommend that
you convert your sparse data into dense data before clustering.
In high dimensions (say > 1024), it is recommended that you embed your sparse data into a lower
dimensional dense space using random indexing.

### Cluster Backfilling
The standard implementation of Lloyd's algorithm suffers from the problem that cluster centers
can vanish and not be replaced.  Our ```COLUMN_TRACKING``` implementation allows one to backfill
empty clusters using the K Means || algorithm.

### Scalability and Testing

This clusterer has been used to cluster millions of points in 700+ dimensional space using an
information theoretic distance function (Kullback-Leibler).
