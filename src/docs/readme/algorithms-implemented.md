# Algorithms Implemented

Most practical variants of K-means clustering are implemented or can be implemented with this package.

* [Clustering with Bregman Divergences](https://www.cs.utexas.edu/users/inderjit/public\_papers/bregmanclustering\_jmlr.pdf) - observes that Lloyd's algorithms converges for distance functions defined by Bregman Divergences
* [Fast k-means algorithm clustering](https://arxiv.org/abs/1108.1351) - uses a 2-step iterative algorithm to cluster a subset of the data and then the full set
* [A Random Indexing Approach for Web User Clustering and Web Prefetching](https://www.ida.liu.se/\~arnjo82/papers/pakdd-ws-11.pdf) - uses random indexing to lower the dimension of high dimensional data
* [A Wavelet-Based Anytime Algorithm for K-Means Clustering of Time Series](https://cs.gmu.edu/\~jessica/publications/ikmeans\_sdm\_workshop03.pdf) - uses the Haar Transform to embed time series data before clustering
* [Metrics Defined By Bregman Divergences](https://people.clas.ufl.edu/yun/files/article-8-1.pdf) - shows metrics can can make use of the triangle inequality to speed up clustering
* [On the performance of bisecting K-means and PDDP](https://archive.siam.org/meetings/sdm01/pdf/sdm01\_05.pdf) - a recursive subdivision algorithm
* [Scalable K-Means++](https://theory.stanford.edu/\~sergei/papers/vldb12-kmpar.pdf) - a provably good initial set of cluster centers
* [Streaming k-means approximation](https://proceedings.neurips.cc/paper\_files/paper/2009/file/4f16c818875d9fcb6867c7bdc89be7eb-Paper.pdf) - a mini-batch algorithm suitable for online data sets

If you find a novel variant of k-means clustering that is provably superior in some manner, implement it using the package and send a pull request along with the paper analyzing the variant!\
\
Here are some newer algorithms that are worth investigating:

* [Fast and Provably Good Seedings for k-Means](https://proceedings.neurips.cc/paper/2016/file/d67d8ab4f4c10bf22aa353e27879133c-Paper.pdf) - even better seeding&#x20;
* [An efficient approximation to the K-means clustering for Massive Data](https://bird.bcamath.org/bitstream/handle/20.500.11824/797/RPKM\_reviewed.pdf) - a recursive subdivision algorithm&#x20;
* [Scalable K-Means by Ranked Retrieval](https://www.researchgate.net/profile/Andrei-Broder/publication/261959598\_Scalable\_K-Means\_by\_ranked\_retrieval/links/59a4ad4fa6fdcc773a374964/Scalable-K-Means-by-ranked-retrieval.pdf) - a novel inversion of the k-means algorithm with dramatic speedups on large data sets\
