# Constructing K-Means Models using Clusterers

Clusterers implment LLoyd's algorithm to find  optimal clusterings by implementing the `MultiKMeansClusterer` trait.

<pre><code><strong>trait MultiKMeansClusterer extends Serializable with Logging {
</strong>  def cluster(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centers: Seq[IndexedSeq[BregmanCenter]]): Seq[(Double, IndexedSeq[BregmanCenter])]

  def best(
    maxIterations: Int,
    pointOps: BregmanPointOps,
    data: RDD[BregmanPoint],
    centers: Seq[IndexedSeq[BregmanCenter]]): (Double, IndexedSeq[BregmanCenter]) = {
    cluster(maxIterations, pointOps, data, centers).minBy(_._1)
  }
}
</code></pre>

We offer several different clusterers.

| Name                                   | Algorithm                                                                          |
| -------------------------------------- | ---------------------------------------------------------------------------------- |
| MultiKMeansClusterer.`COLUMN_TRACKING` | high performance implementation that performs less work on later rounds            |
| MultiKMeansClusterer.`MINI_BATCH_10`   | a mini-batch clusterer that samples 10% of the data each round to update centroids |
| MultiKMeansClusterer.`RESEED`          | a clusterer that re-seeds empty clusters                                           |

The `COLUMN_TRACKING` algorithm tracks the assignments of points to clusters and the distance of points to their assigned cluster. In later iterations of Lloyd's algorithm, this information can be used to reduce the number of distance calculations needed to accurately reassign points. This is a novel implementation.

The `MINI_BATCH_10` algorithm implements the [mini-batch algorithm](http://www.eecs.tufts.edu/\~dsculley/papers/fastkmeans.pdf). This implementation should be used when the number of points is much larger than the dimension of the data and the number of clusters desired.

The `RESEED` algorithm fills empty clusters with newly seeded cluster centers in an effort to reach the target number of desired clusters.\
\
You may use the `apply` method of the companion object `MultiKMeansClusterer` to create a cluster.

<pre class="language-scala"><code class="lang-scala"><strong>package com.massivedatascience.clusterer
</strong>
object MultiKMeansClusterer {
  def apply(clustererName: String): MultiKMeansClusterer = ???
}
</code></pre>

