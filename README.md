Generalized K-Means Clustering
=============================

This project generalizes the Spark MLLIB K-Means (v1.1.0) clusterer to support clustering of sparse
or dense data using distances defined by
[Bregman divergences](http://www.cs.utexas.edu/users/inderjit/public_papers/bregmanclustering_jmlr.pdf) and
[generalized symmetrized Bregman Divergences] (http://www-users.cs.umn.edu/~banerjee/papers/13/bregman-metric.pdf).


### Usage

The simplest way to call the clusterer is to use the ```KMeans.train``` method.

```scala
  package com.massivedatascience.clusterer

  object KMeans {
     /**
      *
      * @param raw input data
      * @param k  number of clusters desired
      * @param maxIterations maximum number of iterations of Lloyd's algorithm
      * @param runs number of parallel clusterings to run
      * @param mode initialization algorithm to use
      * @param initializationSteps number of steps of the initialization algorithm
      * @param distanceFunction the distance functions to use
      * @return (distortion, K-Means model)
      */
     def train(
       raw: RDD[Vector],
       k: Int,
       maxIterations: Int = 20,
       runs: Int = 1,
       mode: String = K_MEANS_PARALLEL,
       initializationSteps: Int = 5,
       distanceFunction: String = EUCLIDEAN)
     : KMeansModel
   }
```

At minimum, you must provide the RDD of ```Vector```s to cluster and the number of clusters you
desire. The method will return a ```KMeansModel``` of the clustering.

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

The Spark MLLIB clusterer is good at one thing: clustering data using Euclidean distance as the metric into
a fixed number of clusters.  However, there are many interesting distance functions other than Euclidean distance.
It is far from trivial to adapt the Spark MLLIB clusterer to these other distance functions. In fact, recent
modification to the Spark implementation have made it even more difficult.

This project decouples the distance function from the clusterer implementation, allowing the end-user the opportunity
to define an alternative distance function in just a few lines of code.

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


| Name (```KMeans._```)            | Space | Divergence              | Input   | Dimension |
|----------------------------------|-------|-------------------------|---------|-----------|
| ```EUCLIDEAN```                  | R^d   |Euclidean                | Dense   |  low      |
| ```LOW_DIMENSIONAL_RI```         | R^d   |Euclidean                | Either  | medium    |
| ```MEDIUM_DIMENSIONAL_RI```      | R^d   |Euclidean                | Either  | high      |
| ```HIGH_DIMENSIONAL_RI```        | R^d   |Euclidean                | Either  | very high |
| ```SPARSE_EUCLIDEAN```           | R^d   |Euclidean                | Sparse  | low       |
| ```RELATIVE_ENTROPY```           | R+^d  |Kullback-Leibler         | Dense   | low       |
| ```DISCRETE_KL```                | N+^d  |Kullback-Leibler         | Dense   | low       |
| ```DISCRETE_SMOOTHED_KL```       | N^d   |Kullback-Leibler         | Dense   | low       |
| ```SPARSE_SMOOTHED_KL```         | R+^d  |Kullback-Leibler         | Sparse  | low       |
| ```GENERALIZED_SYMMETRIZED_KL``` | R+^d  |Kullback-Leibler         | Dense   | low       |
| ```LOGISTIC_LOSS```              | R     |Logistic Loss            |         | 1         |
| ```GENERALIZED_I```              | R     |Generalized I-divergence |         | 1         |


### Initialization/seeding algorithm

This clusterer separates the initialization step (the seeding of the initial clusters) from the main clusterer.
This allows for new initialization methods beyond the standard "random" and "K-Means ||" algorithms,
including initialization methods that have different numbers of initial clusters.

There are two pre-defined seeding algorithms.

| Name (```KMeans._```)            | Algorithm                         |
|----------------------------------|-----------------------------------|
| ```RANDOM```                  | Random selection of initial k centers |
| ```K_MEANS_PARALLEL```           | [K-Means Parallel](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf)              |

You may provide alternative seeding algorithms using the lower level interface as shown in ```KMeans.train```.

### Other Differences with Spark MLLIB 1.2 K-Means Clusterer

There are several other differences with this clusterer and the Spark K-Means clusterer.

#### Variable number of clusters

This clusterer may produce fewer than `k` clusters when `k` are requested.  This may sound like a problem, but your data may not cluster into `k` clusters!
The Spark implementation duplicates cluster centers, resulting in useless computation.  This implementation
tracks the number of cluster centers. 

#### Faster K-Means || implementation

This clusterer uses the K-Means clustering step in the [K-Means || initialization](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf) process.
This is much faster, since all cores are utilized versus just one.

Additionally, this implementation performs the implementation in time quadratic in the number of clusters, whereas the Spark implementation takes time cubic in the number of clusters.

#### Sparse Data

This clusterer works well on sparse input data of high dimension.  Note, some distance functions are not defined on
sparse data (i.e. Kullback-Leibler).  However, one can approximate those distance functions to
achieve similar results.  This implementation provides such approximations.

### Scalability and Testing

This clusterer has been used to cluster millions of points in 700+ dimensional space using an information theoretic distance
function (Kullback-Leibler).

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

The key is to create three new abstractions: point, cluster center, and centroid.  The base implementation constructs
centroids incrementally, then converts them to cluster centers.  The initialization of the cluster centers converts
points to cluster centers.  These abstractions are easy to understand and easy to implement.

```PointOps``` implement fast method for computing distances, taking advantage of the
characteristics of the data to define the fastest methods for evaluating Bregman divergences.

```scala
class BregmanPointOps(val divergence: BregmanDivergence, val clusterFactory: ClusterFactory)
  extends PointOps[BregmanPoint, BregmanCenter]
  with ClusterFactory {

  val weightThreshold = 1e-4
  val distanceThreshold = 1e-8
  def embed(v:Vector) : Vector = ???
  def distance(p: BregmanPoint, c: BregmanCenter): Double = ???
  def homogeneousToPoint(h: Vector, weight: Double): BregmanPoint = ???
  def inhomogeneousToPoint(inh: Vector, weight: Double): BregmanPoint = ???
  def toCenter(v: WeightedVector): BregmanCenter = ???
  def toPoint(v: WeightedVector): BregmanPoint =  ???
  def centerMoved(v: BregmanPoint, w: BregmanCenter): Boolean = ???
}
```
Pull requests offering additional distance functions (http://en.wikipedia.org/wiki/Bregman_divergence)
are welcome.

#### Embeddings

```PointOps``` also provide a means to embed points into a different space before clustering.
Embedding is used to provide means to cluster data using generalized symmetrized Bregman
divergences.

Embedding is also used to support clustering of sparse data which can
be quite inefficient to cluster directly.


```scala

  trait Embedding extends Serializable {
    def embed(v: Vector): Vector
  }
```

Two embedding are provided.  One uses [random indexing](https://www.sics.se/~mange/papers/RI_intro.pdf)
to reduce the dimension of the data while the other simply converts vectors to isomorphic
dense vectors. Note that random indexing may result in vectors with negative values, which may
not be used in divergences that require non-negative values.


```scala
  class RandomIndexEmbedding(dim: Int, on: Int) extends Embedding
  object DirectEmbedding extends Embedding
```

Embedding objects are not used directly in the main interface.  Rather, they are used in the
construction of instances of ```PointOps```. See  the definition of ```RandomIndexedSquaredEuclideanPointOps```
for an example.






