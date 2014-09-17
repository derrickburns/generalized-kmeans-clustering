Generalized K-Means Clustering
=============================

This project generalizes the Spark MLLIB K-Means (v1.1.0) clusterer to support arbitrary distance functions.  
For backward compatibility, the KMeans object provides an interface that is consistent with the 
object of the same name in the Spark implementation.

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

The second major deviation between this implementation and the Spark implementation is that this clusterer may produce
fewer than k clusters when k are requested.  This may sound like a problem, but your data may not cluster into k clusters!
The Spark implementation duplicates cluster centers, resulting in useless computation.  This implementation
tracks the number of cluster centers. 

The third major difference between this implementation and the Spark implementation is that this clusterer
separates the initialization step from the main clusterer.  This allows for new initialization methods, including 
initialization methods that have different numbers of initial clusters.

The fourth major difference between this implementation and the Spark implementation is that this clusterer
uses the K-Means clustering step in the K-Means parallel initialization process.  This is much faster, since all cores
are utilized versus just one.

The fifth major difference between this implementation and the Spark implementation is that this clusterer allows 
one to identify which points are in which clusters efficiently.  In many applications this is necessary. 

This clusterer has been used to cluster millions of points in 700+ dimensional space using an information theoretic distance
function (Kullback-Leibler). 




