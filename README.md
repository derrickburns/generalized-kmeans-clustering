Generalized K-Means Clustering
=============================

This project generalizes the Spark MLLIB Batch K-Means (v1.1.0) clusterer
and the Spark MLLIB Streaming K-Means (v1.2.0) clusterer.   Most practical variants of
K-means clustering are implemented or can be implemented with this package, including:

* [clustering using general distance functions (Bregman divergences)](http://www.cs.utexas.edu/users/inderjit/public_papers/bregmanclustering_jmlr.pdf)
* [clustering large numbers of points using mini-batches](http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)
* [clustering high dimensional Euclidean data](http://www.ida.liu.se/~arnjo/papers/pakdd-ws-11.pdf)
* [clustering high dimensional time series data](http://www.cs.gmu.edu/~jessica/publications/ikmeans_sdm_workshop03.pdf)
* [clustering using symmetrized Bregman divergences](http://snowbird.djvuzone.org/2009/abstracts/127.pdf)
* [clustering via bisection](http://www.siam.org/meetings/sdm01/pdf/sdm01_05.pdf)
* [clustering with near-optimality](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf)
* [clustering streaming data](http://papers.nips.cc/paper/3812-streaming-k-means-approximation.pdf)

If you find a novel variant of k-means clustering that is provably superior in some manner,
implement it using the package and send a pull request along with the paper analyzing the variant!

This code has been tested on data sets of tens of millions of points in a 700+ dimensional space
using a variety of distance functions. Thanks to the excellent core Spark implementation, it rocks!

Table of Contents
=================

* [Generalized K-Means Clustering](#generalized-k-means-clustering)
    * [Getting Started](#getting-started)
      * [SBT](#sbt)
      * [Maven](#maven)
    * [Introduction](#introduction)
      * [Bregman Divergences](#bregman-divergences)
      * [Compute Bregman Distances Efficiently using <code>BregmanPoint</code>s  and <code>BregmanCenter</code>s](#compute-bregman-distances-efficiently-using-bregmanpoints--and-bregmancenters)
      * [Representing K-Means Models](#representing-k-means-models)
      * [Constructing K-Means Models using Clusterers](#constructing-k-means-models-using-clusterers)
    * [Constructing K-Means Models via Lloyd's Algorithm](#constructing-k-means-models-via-lloyds-algorithm)
      * [Constructing K-Means Models on <code>WeightedVector</code>s](#constructing-k-means-models-on-weightedvectors)
      * [Constructing K-Means Models Iteratively](#constructing-k-means-models-iteratively)
      * [Seeding the Set of Cluster Centers](#seeding-the-set-of-cluster-centers)
      * [Iterative Clustering](#iterative-clustering)
    * [Creating a Custom K-means Clusterer](#creating-a-custom-k-means-clusterer)
      * [Custom <code>BregmanDivergence</code>](#custom-bregmandivergence)
      * [Custom <code>BregmanPointOps</code>](#custom-bregmanpointops)
      * [Custom <code>Embedding</code>](#custom-embeddings)
    * [Creating K-Means Models using the <code>KMeansModel</code> Helper Object](#creating-k-means-models-using-the-kmeansmodel-helper-object)
    * [Other Differences with Spark MLLIB 1.2 K-Means Clusterer](#other-differences-with-spark-mllib-12-k-means-clusterer)
      * [Variable number of clusters in parallel runs](#variable-number-of-clusters-in-parallel-runs)
      * [Sparse Data](#sparse-data)
      * [Cluster Backfilling](#cluster-backfilling)


### Getting Started

The massivedatascience-clusterer project is built for both Scala 2.10.x and 2.11.x against Spark v1.2.0.


#### SBT

<a href='https://bintray.com/derrickburns/maven/massivedatascience-clusterer/view?source=watch' alt='Get automatic notifications about new "massivedatascience-clusterer" versions'><img src='https://www.bintray.com/docs/images/bintray_badge_color.png'></a>

If you are using SBT, simply add the following to your `build.sbt` file:

```scala

resolvers += Resolver.bintrayRepo("derrickburns", "maven")

libraryDependencies ++= Seq(
  "com.massivedatascience" %% "massivedatascience-clusterer" % "x.y.z"
)
```

#### Maven


```xml
<dependency>
  <groupId>com.massivedatascience</groupId>
  <artifactId>massivedatascience-clusterer_2.10</artifactId>
  <version>x.y.z</version>
</dependency>

<dependency>
  <groupId>com.massivedatascience</groupId>
  <artifactId>massivedatascience-clusterer_2.11</artifactId>
  <version>x.y.z</version>
</dependency>



<repositories>
    <repository>
        <id>bintray</id>
        <name>bintray</name>
        <url>http://dl.bintray.com/derrickburns/maven/</url>
    </repository>
</repositories>
```


### Introduction

The goal of K-Means clustering is to produce a set of clusters of a set of points that satisfies
certain optimality constraints. That model is called a K-Means model. It is fundamentally a set
of points and a function that defines the distance from an arbitrary point to a cluster center.

The K-Means algorithm computes a K-Means model using an iterative algorithm known as
[Lloyd's algorithm](http://en.wikipedia.org/wiki/Lloyd%27s_algorithm).
Each iteration of Lloyd's algorithm assigns a set of points to clusters, then updates the cluster
centers to acknowledge the assignment of the points to the cluster.

The update of clusters is a form of averaging.  Newly added points are averaged into the cluster
while (optionally) reassigned points are removed from their prior clusters.


#### Bregman Divergences

While one can assign a point to a cluster using any distance function, Lloyd's algorithm only
converges for a certain set of distance functions called [Bregman divergences](http://www.cs.utexas.edu/users/inderjit/public_papers/bregmanclustering_jmlr.pdf). Bregman divergences
must define two methods, ```convex```  to evaluate a function on a point and ```gradientOfConvex``` to evaluate the
gradient of the function on a point.

```scala
package com.massivedatascience.divergence

trait BregmanDivergence {
  def convex(v: Vector): Double

  def gradientOfConvex(v: Vector): Vector
}

```

For example, by defining ```convex``` to be the vector norm (i.e. the sum of the squares of the
coordinates), one gets a distance function that equals the square of the well known Euclidean
distance. We name it the ```SquaredEuclideanDistanceDivergence```.

In addition to the squared Euclidean distance function, this implementation provides several
other very useful distance functions.   The provided ```BregmanDivergence```s may be accessed by
supplying the name of the desired object to the apply method of the companion object.


| Name   | Space | Divergence              | Input   |
|--------|-------|-------------------------|---------|
| ```SquaredEuclideanDistanceDivergence```                  | R^d   |Squared Euclidean        |         |
| ```RealKullbackLeiblerSimplexDivergence```           | R+^d  |[Kullback-Leibler](http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)         | Dense   |
| ```NaturalKLSimplexDivergence```                | N+^d  |Kullback-Leibler         | Dense   |
| ```RealKLDivergence```       | R^d   |Kullback-Leibler         | Dense   |
| ```NaturalKLDivergence```       | N^d   |Kullback-Leibler         | Dense   |
| ```ItakuraSaitoDivergence```         | R+^d  |Kullback-Leibler         | Sparse  |
| ```LogisticLossDivergence```              | R     |Logistic Loss            |         |
| ```GeneralizedIDivergence```     | R     |Generalized I |         |

When selecting a distance function, consider the domain of
the input data.  For example, frequency
data is integral. Similarity of frequencies or distributions are best performed using the
Kullback-Leibler divergence.


#### Compute Bregman Distances Efficiently using ```BregmanPoint```s  and ```BregmanCenter```s

For efficient repeated computation of distance between a fixed set of points and varying cluster
centers, is it convenient to pre-compute certain information and associate that information with
the point or the cluster center.  The class that represent an enriched point is ```BregmanPoint```.
The class that represent the enriched cluster center is ```BregmanCenter```.  Users
of this package do not construct instances of these objects directly.

```scala
package com.massivedatascience.divergence

trait BregmanPoint

trait BregmanCenter
```


We enrich a Bregman divergence with a set of commonly used operations, including factory
methods ```toPoint``` and ```toCenter``` to construct instances of the aforementioned ```BregmanPoint```
and ```BregmanCenter```.

The enriched trait is the ```BregmanPointOps```.

```scala
package com.massivedatascience.clusterer

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

object BregmanPointOps {

  def apply(distanceFunction: String): BregmanPointOps = ???

}
```


One may construct instances of ```BregmanPointOps``` using the companion object.

| Name   | Divergence     |
|--------|----------------|
| ```EUCLIDEAN```                  |Squared Euclidean        |
| ```RELATIVE_ENTROPY```           |[Kullback-Leibler](http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)         |
| ```DISCRETE_KL```                |Kullback-Leibler         |
| ```DISCRETE_SMOOTHED_KL```       |Kullback-Leibler         |
| ```SPARSE_SMOOTHED_KL```         |Kullback-Leibler         |
| ```LOGISTIC_LOSS```              |Logistic Loss            |
| ```GENERALIZED_I```              |Generalized I |
| ```ITAKURA_SAITO```              |[Itakura-Saito](http://en.wikipedia.org/wiki/Itakura%E2%80%93Saito_distance) |

#### Representing K-Means Models

With these definitions, we define our realization of a k-means model, ```KMeansModel```, which
we enrich with operations to find closest clusters to a point and to compute distances:

```scala
package com.massivedatascience.clusterer

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

#### Constructing K-Means Models using Clusterers

One may construct K-Means models using one of the provided clusterers that implement Lloyd's algorithm.

```scala
trait MultiKMeansClusterer extends Serializable with Logging {
  def cluster(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centers: Seq[IndexedSeq[BregmanCenter]]): Seq[(Double, IndexedSeq[BregmanCenter])]

  def best(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centers: Seq[IndexedSeq[BregmanCenter]]): (Double, IndexedSeq[BregmanCenter]) = {
    cluster(maxIterations, pointOps, data, centers).minBy(_._1)
  }
}

object MultiKMeansClusterer {
  def apply(clustererName: String): MultiKMeansClusterer = ???
}
```

The ```COLUMN_TRACKING``` algorithm tracks the assignments of points to clusters and the distance of
points to their assigned cluster.  In later iterations of Lloyd's algorithm, this information can
be used to reduce the number of distance calculations needed to accurately reassign points.  This
is a novel implementation.

The ```MINI_BATCH_10``` algorithm implements the [mini-batch algorithm](http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf).
This implementation should be used when the number of points is much larger than the dimension of the data and the
number of clusters desired.

The ```RESEED``` algorithm fills empty clusters with newly seeded cluster centers
in an effort to reach the target number of desired clusters.

Objects implementing these algorithms may be constructed using the ```apply``` method of the
companion object ```MultiKMeansClusterer```.


| Name            | Algorithm                         |
|----------------------------------|-----------------------------------|
| ```COLUMN_TRACKING```    | high performance implementation that performs less work on later rounds  |
| ```MINI_BATCH_10```      | a mini-batch clusterer that samples 10% of the data each round to update centroids |
| ```RESEED```             | a clusterer that re-seeds empty clusters |


### Constructing K-Means Models via Lloyd's Algorithm

A ```KMeansModel``` can be constructed from any set of cluster centers and distance function.
However, the more interesting models satisfy an optimality constraint.  If we sum the distances
from the points in a given set to their closest cluster centers, we get a number called the
"distortion" or "cost". A K-Means Model is locally optimal with respect to a set of points
if each cluster center is determined by the mean of the points assigned to that cluster.
Computing such a ```KMeansModel``` given a set of points is called "training" the model on those
points.

The simplest way to train a ```KMeansModel``` on a fixed set of points is to use the ```KMeans.train```
method.  This method is most similar in style to the one provided by the Spark 1.2.0 K-Means clusterer.

For dense data in a low dimension space using the squared Euclidean distance function,
one may simply call ```KMeans.train``` with the data and the desired number of clusters:

```scala
import com.com.massivedatascience.clusterer
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
   * @param data input data
   * @param k  number of clusters desired
   * @param maxIterations maximum number of iterations of Lloyd's algorithm
   * @param runs number of parallel clusterings to run
   * @param mode initialization algorithm to use
   * @param distanceFunctionNames the distance functions to use
   * @param clustererName which k-means implementation to use
   * @param embeddingNames sequence of embeddings to use, from lowest dimension to greatest
   * @return K-Means model
   */
  def train(
    data: RDD[Vector],
    k: Int,
    maxIterations: Int = KMeans.defaultMaxIterations,
    runs: Int = KMeans.defaultNumRuns,
    mode: String = KMeansSelector.K_MEANS_PARALLEL,
    distanceFunctionNames: Seq[String] = Seq(BregmanPointOps.EUCLIDEAN),
    clustererName: String = MultiKMeansClusterer.COLUMN_TRACKING,
    embeddingNames: List[String] = List(Embedding.IDENTITY_EMBEDDING)): KMeansModel = ???
}
```

Many of these parameters will be familiar to anyone who is familiar with the Spark 1.1 clusterer.

Similar to the Spark clusterer, we support data provided as ```Vectors```, a request for a number
```k``` of clusters desired, a limit ```maxIterations``` on the number of iterations of Lloyd's
algorithm, and the number of parallel ```runs``` of the clusterer.

We also offer different initialization ```mode```s.  But
unlike the Spark clusterer, we do not support setting the number of initialization steps for the
mode at this level of the interface.

The ```K-Means.train``` helper methods allows one to name a sequence of embeddings.
Several embeddings are provided that may be constructed using the ```apply``` method
of the companion object ```Embedding```.


| Name         | Algorithm                                                   |
|-------------------------------|-------------------------------------------------------------|
| ```IDENTITY_EMBEDDING```      | Identity                                                    |
| ```HAAR_EMBEDDING```          | [Haar Transform](http://www.cs.gmu.edu/~jessica/publications/ikmeans_sdm_workshop03.pdf) |
| ```LOW_DIMENSIONAL_RI```      | [Random Indexing](https://www.sics.se/~mange/papers/RI_intro.pdf) with dimension 64 and epsilon = 0.1 |
| ```MEDIUM_DIMENSIONAL_RI```   | Random Indexing with dimension 256 and epsilon = 0.1        |
| ```HIGH_DIMENSIONAL_RI```     | Random Indexing with dimension 1024 and epsilon = 0.1       |
| ```SYMMETRIZING_KL_EMBEDDING```     | [Symmetrizing KL Embedding](http://www-users.cs.umn.edu/~banerjee/papers/13/bregman-metric.pdf)       |

Different distance functions may be used for each embedding. There must be exactly one
distance function per embedding provided.

#### Constructing K-Means Models on ```WeightedVector```s

Often, data points that are clustered have varying significance, i.e. they are weighted.
This clusterer operates on weighted vectors.   Use these ```WeightedVector``` companion object to construct weighted vectors.

```scala
package com.massivedatascience.linalg

trait WeightedVector extends Serializable {
  def weight: Double

  def inhomogeneous: Vector

  def homogeneous: Vector

  def size: Int = homogeneous.size
}

object WeightedVector {

  def apply(v: Vector): WeightedVector = ???

  def apply(v: Array[Double]): WeightedVector = ???

  def apply(v: Vector, weight: Double): WeightedVector = ???

  def apply(v: Array[Double], weight: Double): WeightedVector = ???

  def fromInhomogeneousWeighted(v: Array[Double], weight: Double): WeightedVector = ???

  def fromInhomogeneousWeighted(v: Vector, weight: Double): WeightedVector = ???
}
```

Indeed, the ```KMeans.train``` helper translates the parameters into a call to the underlying
```KMeans.trainWeighted``` method.

```scala
package com.massivedatascience.clusterer

object KMeans {
  /**
   *
   * Train a K-Means model using Lloyd's algorithm on WeightedVectors
   *
   * @param data input data
   * @param runConfig run configuration
   * @param pointOps the distance functions to use
   * @param initializer initialization algorithm to use
   * @param embeddings sequence of embeddings to use, from lowest dimension to greatest
   * @param clusterer which k-means implementation to use
   * @return K-Means model
   */

  def trainWeighted(
    runConfig: RunConfig,
    data: RDD[WeightedVector],
    initializer: KMeansSelector,
    pointOps: Seq[BregmanPointOps],
    embeddings: Seq[Embedding],
    clusterer: MultiKMeansClusterer): KMeansModel = ???
  }
}
```

The ```KMeans.trainWeighted``` method ultimately makes various calls to the underlying
```KMeans.simpleTrain``` method, which clusters the provided ```BregmanPoint```s using
the provided ```BregmanPointOps``` and the provided ```KMeansSelector``` with the provided ```MultiKMeansClusterer```.


```scala
package com.massivedatascience.clusterer

object KMeans {
  /**
   *
   * @param runConfig run configuration
   * @param data input data
   * @param pointOps the distance functions to use
   * @param initializer initialization algorithm to use
   * @param clusterer which k-means implementation to use
   * @return K-Means model
   */
  def simpleTrain(
    runConfig: RunConfig,
    data: RDD[BregmanPoint],
    pointOps: BregmanPointOps,
    initializer: KMeansSelector,
    clusterer: MultiKMeansClusterer): KMeansModel = ???
    }
}
```

#### Constructing K-Means Models Iteratively

If multiple embeddings are provided, the ```KMeans.train``` method actually performs the embeddings
and trains on the embedded data sets iteratively.

For example, for high dimensional data, one way wish to embed the data into a lower dimension before clustering to
reduce running time.

For time series data,
[the Haar Transform](http://www.cs.gmu.edu/~jessica/publications/ikmeans_sdm_workshop03.pdf)
has been used successfully to reduce running time while maintaining or improving quality.

For high-dimensional sparse data,
[random indexing](http://en.wikipedia.org/wiki/Random_indexing)
can be used to map the data into a low dimensional dense space.

One may also perform clustering recursively, using lower dimensional clustering to derive initial
conditions for higher dimensional clustering.

Should you wish to train a model iteratively on data sets derived maps of a shared original data
set, you may use ```KMeans.iterativelyTrain```.


```scala
package com.massivedatascience.clusterer

object KMeans {
  /**
   * Train on a series of data sets, where the data sets were derived from the same
   * original data set via embeddings. Use the cluster assignments of one stage to
   * initialize the clusters of the next stage.
   *
   * @param runConfig run configuration
   * @param dataSets  input data sets to use
   * @param initializer  initialization algorithm to use
   * @param pointOps distance function
   * @param clusterer  clustering implementation to use
   * @return
   */
  def iterativelyTrain(
    runConfig: RunConfig,
    pointOps: Seq[BregmanPointOps],
    dataSets: Seq[RDD[BregmanPoint]],
    initializer: KMeansSelector,
    clusterer: MultiKMeansClusterer): KMeansModel = ???

```

#### Seeding the Set of Cluster Centers

Any K-Means model may be used as seed value to Lloyd's algorithm. In fact, our clusterers accept
multiple seed sets. The ```K-Means.train``` helper methods allows one to name an initialization
method.

Two algorithms are implemented that produce viable seed sets.
They may be constructed by using the ```apply``` method
of the companion object```KMeansSelector```".

| Name            | Algorithm                         |
|----------------------------------|-----------------------------------|
| ```RANDOM```             | Random selection of initial k centers |
| ```K_MEANS_PARALLEL```   | a 5 step [K-Means Parallel implementation](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf) |

Under the covers, these initializers implement the ```KMeansSelector``` trait

```scala
package com.massivedatascience.clusterer

trait KMeansSelector extends Serializable {
  def init(
    ops: BregmanPointOps,
    d: RDD[BregmanPoint],
    numClusters: Int,
    initialInfo: Option[(Seq[IndexedSeq[BregmanCenter]], Seq[RDD[Double]])] = None,
    runs: Int,
    seed: Long): Seq[IndexedSeq[BregmanCenter]]
}

object KMeansSelector {
  def apply(name: String): KMeansSelector = ???
}
```

#### Iterative Clustering

K-means clustering can be performed iteratively using different embeddings of the data.  For example,
with high-dimensional time series data, it may be advantageous to:

* Down-sample the data via the Haar transform (aka averaging)
* Solve the K-means clustering problem on the down-sampled data
* Assign the downsampled points to clusters.
* Create a new KMeansModel using the assignments on the original data
* Solve the K-Means clustering on the KMeansModel so constructed

This technique has been named the ["Anytime" Algorithm](http://www.cs.gmu.edu/~jessica/publications/ikmeans_sdm_workshop03.pdf).

The ```com.massivedatascience.clusterer.KMeans``` helper method provides a method, ```timeSeriesTrain```
that embeds the data iteratively.

```scala
package com.massivedatascience.clusterer

object KMeans {

  def timeSeriesTrain(
    runConfig: RunConfig,
    data: RDD[WeightedVector],
    initializer: KMeansSelector,
    pointOps: BregmanPointOps,
    clusterer: MultiKMeansClusterer,
    embedding: Embedding = Embedding(HAAR_EMBEDDING)): KMeansModel = ???
  }
}
```

High dimensional data can be clustered directly, but the cost is proportional to the dimension.  If
the divergence of interest is squared Euclidean distance, one can using
[Random Indexing](http://en.wikipedia.org/wiki/Random_indexing) to
down-sample the data while preserving distances between clusters, with high probability.

The ```com.massivedatascience.clusterer.KMeans``` helper method provides a method, ```sparseTrain```
that embeds into various dimensions using random indexing.

```scala
package com.massivedatascience.clusterer

object KMeans {

  def sparseTrain(raw: RDD[Vector], k: Int): KMeansModel = {
    train(raw, k,
      embeddingNames = List(Embedding.LOW_DIMENSIONAL_RI, Embedding.MEDIUM_DIMENSIONAL_RI,
        Embedding.HIGH_DIMENSIONAL_RI))
  }
}
```

### Creating a Custom K-means Clusterer

There are many ways to create your our custom K-means clusterer from these components.


#### Custom ```BregmanDivergence```

You may create your own custom ```BregmanDivergence``` given a suitable continuously-differentiable
real-valued and strictly convex function defined on a closed convex set in R^^N using the
```apply``` method of the companion object. Send a pull request to have it added
the the package.

```scala
package com.massivedatascience.divergence

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

#### Custom ```BregmanPointOps```

You may create your own custom ```BregmanPointsOps```
from your own implementation of the ```BregmanDivergence``` trait given a ```BregmanDivergence```
using the ```apply``` method of the companion object. Send a pull request to have it added
the the package.


```scala
package com.massivedatascience.clusterer

object BregmanPointOps {

  def apply(d: BregmanDivergence): BregmanPointOps = ???

  def apply(d: BregmanDivergence, factor: Double): BregmanPointOps = ???
}
```

#### Custom ```Embedding```

Perhaps you have a dimensionality reduction method that is not provided by one of the standard
embeddings.  You may create your own embedding.

For example, If the number of clusters desired is small, but the dimension is high, one may also use the method
of [Random Projections](http://www.cs.toronto.edu/~zouzias/downloads/papers/NIPS2010_kmeans.pdf).
At present, no embedding is provided for random projections, but, hey, I have to leave something for
you to do!  Send a pull request!!!


### Creating K-Means Models using the ```KMeansModel``` Helper Object

Training a K-Means model from a set of points using ```KMeans.train``` is one way to create a
```KMeansModel```.  However,
there are many others that are useful.  The ```KMeansModel``` companion object provides a number
of these constructors.


```scala
package com.massivedatascience.clusterer

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
