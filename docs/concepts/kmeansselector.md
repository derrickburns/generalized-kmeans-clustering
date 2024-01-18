# KMeansSelector

Any K-Means model may be used as seed value to Lloyd's algorithm. In fact, our clusterers accept multiple seed sets. The `K-Means.train` helper methods allows one to name an initialization method.

Two algorithms are implemented that produce viable seed sets. They may be constructed by using the `apply` method of the companion object`KMeansSelector`.

Initializers are implemented with the `KMeansSelector` trait.

```scala
package com.massivedatascience.clusterer

trait KMeansSelector extends Serializable {
  def init(
    ops: BregmanPointOps,
    d: RDD[BregmanPoint],
    numClusters: Int,
    initialInfo: Option[(Seq[IndexedSeq[BregmanCenter]], Seq[RDD[Double]])] = None,
    runs: Int,
    seed: Long): Seq[IndexedSeq[BregmanCenter]]
}
```
