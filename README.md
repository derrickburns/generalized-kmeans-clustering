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

This project decouples the metric from the clusterer implementation, allowing the end-user the opportunity
to define a custom distance function in just a few lines of code.  We demonstrate this by implementing several
Bregman divergences, including the squared Euclidean distance, the [Kullback-Leibler divergence](http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence),
the logistic loss divergence, the Itakura-Saito divergence, and the generalized I-divergence. We also implement a distance function
that is a symmetric version of the Kullback-Leibler divergence that is also a metric.
Pull requests offering additional distance functions (http://en.wikipedia.org/wiki/Bregman_divergence) are welcome.

The key is to create three new abstractions: point, cluster center, and centroid.  The base implementation constructs
centroids incrementally, then converts them to cluster centers.  The initialization of the cluster centers converts
points to cluster centers.  These abstractions are easy to understand and easy to implement.

### Variable number of clusters

The second major deviation between this implementation and the Spark implementation is that this clusterer may produce
fewer than `k` clusters when `k` are requested.  This may sound like a problem, but your data may not cluster into `k` clusters!
The Spark implementation duplicates cluster centers, resulting in useless computation.  This implementation
tracks the number of cluster centers. 

### Plugable seeding algorithm

The third major difference between this implementation and the Spark implementation is that this clusterer
separates the initialization step (the seeding of the initial clusters) from the main clusterer.
This allows for new initialization methods, including initialization methods that have different numbers of initial clusters.

### Faster K-Means || implementation  

The fourth major difference between this implementation and the Spark implementation is that this clusterer
uses the K-Means clustering step in the [K-Means || initialization](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf) process.  This is much faster, since all cores
are utilized versus just one.

Additionally, this implementation performs the implementation in time quadratic in the number of cluster, whereas the Spark implementation takes time cubic in the number of clusters.

### Scalability and Testing

This clusterer has been used to cluster millions of points in 700+ dimensional space using an information theoretic distance
function (Kullback-Leibler). 




