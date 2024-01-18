Generalized K-Means Clustering
=============================

This project generalizes the Spark MLLIB Batch K-Means (v1.1.0) clusterer
and the Spark MLLIB Streaming K-Means (v1.2.0) clusterer.   Most practical variants of
K-means clustering are implemented or can be implemented with this package, including:

* [clustering using general distance functions (Bregman divergences)](http://www.cs.utexas.edu/users/inderjit/public_papers/bregmanclustering_jmlr.pdf)
* [clustering large numbers of points using mini-batches](https://arxiv.org/abs/1108.1351)
* [clustering high dimensional Euclidean data](http://www.ida.liu.se/~arnjo/papers/pakdd-ws-11.pdf)
* [clustering high dimensional time series data](http://www.cs.gmu.edu/~jessica/publications/ikmeans_sdm_workshop03.pdf)
* [clustering using symmetrized Bregman divergences](https://people.clas.ufl.edu/yun/files/article-8-1.pdf)
* [clustering via bisection](http://www.siam.org/meetings/sdm01/pdf/sdm01_05.pdf)
* [clustering with near-optimality](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf)
* [clustering streaming data](http://papers.nips.cc/paper/3812-streaming-k-means-approximation.pdf)

If you find a novel variant of k-means clustering that is provably superior in some manner,
implement it using the package and send a pull request along with the paper analyzing the variant!

This code has been tested on data sets of tens of millions of points in a 700+ dimensional space
using a variety of distance functions. Thanks to the excellent core Spark implementation, it rocks!
