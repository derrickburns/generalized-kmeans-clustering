Generalized K-Means Clustering
=============================

This project generalizes the Spark MLLIB Batch K-Means (v1.1.0) clusterer
and the Spark MLLIB Streaming K-Means (v1.2.0) clusterer to support clustering of sparse
or dense data, in low or high dimension, using distances defined by
[Bregman divergences](http://www.cs.utexas.edu/users/inderjit/public_papers/bregmanclustering_jmlr.pdf) and
[generalized symmetrized Bregman Divergences] (http://www-users.cs.umn.edu/~banerjee/papers/13/bregman-metric.pdf).
This is in contrast to the standard Spark implementation that only supports dense, low-dimensional data
using the squared Euclidean distance function.

Be aware that work on this project is ongoing.  Parts of this project are being integrated into
upcoming releases of the Spark MLLIB clusterer.

### Introduction

The goal K-Means clustering is to produce a model of the clusters of a set of points that satisfies
certain optimality constraints. That model is called a K-Means model. It is fundamentally a set
of points and a function that defines the distance from an arbitrary point to a cluster center.

The K-Means algorithm computes a K-Means model using an iterative algorithm known as Lloyd's algorithm.
Each iteration of Lloyd's algorithm assigns a set of points to clusters, then updates the cluster
centers to acknowledge the assignment of the points to the cluster.

The update of clusters is a form of averaging.  Newly added points are averaged into the cluster
while (optionally) reassigned points are removed from their prior clusters.

While one can assign a point to a cluster using any distance function, Lloyd's algorithm only
converges for a certain set of distance functions called Bregman divergences. Bregman divergences
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
we enrich with operations to find closest clusters to a point:

```scala
trait KMeansModel {

  val pointOps: BregmanPointOps

  def centers: IndexedSeq[BregmanCenter]

  def predictWeighted(point: WeightedVector): Int

  def predictClusterAndDistanceWeighted(point: WeightedVector): (Int, Double)

  def predictWeighted(points: RDD[WeightedVector]): RDD[Int]

  def computeCostWeighted(data: RDD[WeightedVector]): Double

  def predict(point: Vector): Int

  def predictClusterAndDistance(point: Vector): (Int, Double)

  def predict(points: RDD[Vector]): RDD[Int]

  def predict(points: JavaRDD[Vector]): JavaRDD[java.lang.Integer]

  def computeCost(data: RDD[Vector]): Double
}
```

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
    mode: String = K_MEANS_PARALLEL,
    initializationSteps: Int = 5,
    distanceFunctionNames: Seq[String] = Seq(BregmanPointOps.EUCLIDEAN),
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
    distanceFunctionName: String = BregmanPointOps.EUCLIDEAN,
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

### Distance Functions

In addition to the squared Euclidean distance function, this implementation provides several
other very useful distance functions. When selecting a distance function, consider the domain of
the input data.  For example, frequency
data is integral. Similarity of frequencies or distributions are best performed using the
Kullback-Leibler divergence.

| Name (```BregmanPointOps._```)   | Space | Divergence              | Input   |  Object |
|----------------------------------|-------|-------------------------|---------|---------|
| ```EUCLIDEAN```                  | R^d   |Euclidean                |         |  SquaredEuclideanPointOps  |
| ```RELATIVE_ENTROPY```           | R+^d  |Kullback-Leibler         | Dense   | DenseKLPointOps    |
| ```DISCRETE_KL```                | N+^d  |Kullback-Leibler         | Dense   |  DiscreteKLPointOps     |
| ```DISCRETE_SMOOTHED_KL```       | N^d   |Kullback-Leibler         | Dense   |  DiscreteSmoothedKLPointOps   |
| ```SPARSE_SMOOTHED_KL```         | R+^d  |Kullback-Leibler         | Sparse  |  SparseRealKLPointOps    |
| ```LOGISTIC_LOSS```              | R     |Logistic Loss            |         |   LogisticLossPointOps   |
| ```GENERALIZED_I```              | R     |Generalized I-divergence |         |   GeneralizedIPointOps   |


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

There are three implementations of the Lloyd's algorithm. Use ```SIMPLE```.  The others
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








