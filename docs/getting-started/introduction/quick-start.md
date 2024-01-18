# Quick Start

The simplest way to train a `KMeansModel` on a fixed set of points is to use the `KMeans.train` method. This method is most similar in style to the one provided by the Spark 1.2.0 K-Means clusterer.

For dense data in a low dimension space using the squared Euclidean distance function, one may simply call `KMeans.train` with the data and the desired number of clusters:

```scala
import com.com.massivedatascience.clusterer
import org.apache.spark.ml.linalg.Vector

val model : KMeansModel = KMeans.train(data: RDD[Vector], k: Int)
```

The full signature of the `KMeans.train` method is:

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

Similar to the Spark clusterer, we support data provided as `Vectors`, a request for a number `k` of clusters desired, a limit `maxIterations` on the number of iterations of Lloyd's algorithm, and the number of parallel `runs` of the clusterer.

We also offer different initialization `mode`s. But unlike the Spark clusterer, we do not support setting the number of initialization steps for the mode at this level of the interface.

The `K-Means.train` helper methods allows one to name a sequence of embeddings. Several embeddings are provided that may be constructed using the `apply` method of the companion object `Embedding`.

Different distance functions may be used for each embedding. There must be exactly one distance function per embedding provided.

Indeed, the `KMeans.train` helper translates the parameters into a call to the underlying `KMeans.trainWeighted` method.

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

The `KMeans.trainWeighted` method ultimately makes various calls to the underlying `KMeans.simpleTrain` method, which clusters the provided `BregmanPoint`s using the provided `BregmanPointOps` and the provided `KMeansSelector` with the provided `MultiKMeansClusterer`.

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
