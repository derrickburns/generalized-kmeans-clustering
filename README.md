Generalized K-Means Clustering
=============================

This project generalizes the Spark MLLIB K-Means (v1.1.0) clusterer to support clustering of sparse
or dense data using distances defined by
[Bregman divergences](http://www.cs.utexas.edu/users/inderjit/public_papers/bregmanclustering_jmlr.pdf) and
[generalized symmetrized Bregman Divergences] (http://www-users.cs.umn.edu/~banerjee/papers/13/bregman-metric.pdf).


### General Distance Function 

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

Generally speaking, Bregman divergences are not metrics. However, one may take any Bregman divergence
and transform it into a related Bregman divergence that is also a metric. To demonstrate this, we
also implement a distance function that is a symmetric version of the Kullback-Leibler divergence
that is also a metric.

Several distance functions are predefined:
```scala
  object KMeans {
    val RELATIVE_ENTROPY = "DENSE_KL_DIVERGENCE"
    val DISCRETE_KL = "DISCRETE_DENSE_KL_DIVERGENCE"
    val SPARSE_SMOOTHED_KL = "SPARSE_SMOOTHED_KL_DIVERGENCE"
    val DISCRETE_SMOOTHED_KL = "DISCRETE_DENSE_SMOOTHED_KL_DIVERGENCE"
    val GENERALIZED_SYMMETRIZED_KL = "GENERALIZED_SYMMETRIZED_KL"
    val EUCLIDEAN = "DENSE_EUCLIDEAN"
    val SPARSE_EUCLIDEAN = "SPARSE_EUCLIDEAN"
    val LOGISTIC_LOSS = "LOGISTIC_LOSS"
    val GENERALIZED_I = "GENERALIZED_I_DIVERGENCE"
  }
```

Pull requests offering additional distance functions (http://en.wikipedia.org/wiki/Bregman_divergence) are welcome.

### Variable number of clusters

The second major deviation between this implementation and the Spark implementation is that this clusterer may produce
fewer than `k` clusters when `k` are requested.  This may sound like a problem, but your data may not cluster into `k` clusters!
The Spark implementation duplicates cluster centers, resulting in useless computation.  This implementation
tracks the number of cluster centers. 

### Plugable seeding algorithm

The third major difference between this implementation and the Spark implementation is that this clusterer
separates the initialization step (the seeding of the initial clusters) from the main clusterer.
This allows for new initialization methods beyond the standard "random" and "K-Means ||" algorithms,
including initialization methods that have different numbers of initial clusters.

### Faster K-Means || implementation  

The fourth major difference between this implementation and the Spark implementation is that this clusterer
uses the K-Means clustering step in the [K-Means || initialization](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf) process.
This is much faster, since all cores are utilized versus just one.

Additionally, this implementation performs the implementation in time quadratic in the number of cluster, whereas the Spark implementation takes time cubic in the number of clusters.

### Sparse Data

The fifth major difference between this implementation and the Spark implementation is that this clusterer
works well on sparse input data of high dimension.  Note, some distance functions are not defined on
sparse data (i.e. Kullback-Leibler).  However, one can approximate those distance functions to
achieve similar results.  This implementation provides such approximations.

### Internals

The key is to create three new abstractions: point, cluster center, and centroid.  The base implementation constructs
centroids incrementally, then converts them to cluster centers.  The initialization of the cluster centers converts
points to cluster centers.  These abstractions are easy to understand and easy to implement.

### Scalability and Testing

This clusterer has been used to cluster millions of points in 700+ dimensional space using an information theoretic distance
function (Kullback-Leibler). 




