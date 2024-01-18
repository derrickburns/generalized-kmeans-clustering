# Introduction

The goal of K-Means clustering is to produce a set of clusters of a set of points that satisfies certain optimality constraints. That model is called a **K-Means model** \[`trait KMeansModel]`. It is fundamentally a set of points and a function that defines the distance from an arbitrary point to a cluster center.

The K-Means algorithm computes a K-Means model using an iterative algorithm known as [Lloyd's algorithm](http://en.wikipedia.org/wiki/Lloyd's\_algorithm). Each iteration of Lloyd's algorithm assigns a set of points to clusters, then updates the cluster centers to acknowledge the assignment of the points to the cluster.

The update of clusters is a form of averaging. Newly added points are averaged into the cluster while (optionally) reassigned points are removed from their prior clusters.

A K-Means Model can be constructed from any set of cluster centers and distance function. However, the more interesting models satisfy an optimality constraint. If we sum the distances from the points in a given set to their closest cluster centers, we get a number called the "distortion" or "cost".&#x20;

A K-Means Model is locally optimal with respect to a set of points if each cluster center is determined by the mean of the points assigned to that cluster. Computing such a `KMeansModel` given a set of points is called "training" the model on those points.

