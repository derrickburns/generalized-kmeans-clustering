generalized-kmeans-clustering
=============================

This project generalizes the Spark MLLIB K-Means clusterer to support arbitrary distance functions.

The Spark MLLIB clusterer is good at one thing: clustering data using Euclidean distance as the metric into
a fixed number of clusters.  However, there are many interesting distance functions other than Euclidean distance.
It is far from trivial to adapt the Spark MLLIB clusterer to these other distance functions. In fact, recent
modification to the Spark implementation have made it even more difficult.

This project decouples the metric from the clusterer implementation, allowing the end-user the opportunity
to define a custom distance function in just a few lines of code.  We demonstrate this by implementing the 
the Euclidean distance in two very different ways.  

The key is to create three new abstractions: point, cluster center, and centroid.  The base implementation constructs
centroids incrementally, then converts them to cluster centers.  The initialization of the cluster centers converts
points to cluster centers.  These abstractions are easy to understand and easy to implement.




