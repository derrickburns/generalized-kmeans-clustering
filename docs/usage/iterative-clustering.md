# Iterative Clustering

K-means clustering can be performed iteratively using different embeddings of the data. For example, with high-dimensional time series data, it may be advantageous to:

* Down-sample the data via the Haar transform (aka averaging)
* Solve the K-means clustering problem on the down-sampled data
* Assign the downsampled points to clusters.
* Create a new KMeansModel using the assignments on the original data
* Solve the K-Means clustering on the KMeansModel so constructed

This technique has been named the ["Anytime" Algorithm](http://www.cs.gmu.edu/\~jessica/publications/ikmeans\_sdm\_workshop03.pdf).

The `com.massivedatascience.clusterer.KMeans` helper method provides a method, `timeSeriesTrain` that embeds the data iteratively.

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

High dimensional data can be clustered directly, but the cost is proportional to the dimension. If the divergence of interest is squared Euclidean distance, one can using [Random Indexing](http://en.wikipedia.org/wiki/Random\_indexing) to down-sample the data while preserving distances between clusters, with high probability.

The `com.massivedatascience.clusterer.KMeans` helper method provides a method, `sparseTrain` that embeds into various dimensions using random indexing.

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



If multiple embeddings are provided, the `KMeans.train` method actually performs the embeddings and trains on the embedded data sets iteratively.

For example, for high dimensional data, one way wish to embed the data into a lower dimension before clustering to reduce running time.

For time series data, [the Haar Transform](http://www.cs.gmu.edu/\~jessica/publications/ikmeans\_sdm\_workshop03.pdf) has been used successfully to reduce running time while maintaining or improving quality.

For high-dimensional sparse data, [random indexing](http://en.wikipedia.org/wiki/Random\_indexing) can be used to map the data into a low dimensional dense space.

One may also perform clustering recursively, using lower dimensional clustering to derive initial conditions for higher dimensional clustering.

Should you wish to train a model iteratively on data sets derived maps of a shared original data set, you may use `KMeans.iterativelyTrain`.

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
