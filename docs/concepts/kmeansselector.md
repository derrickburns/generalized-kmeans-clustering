# KMeansSelector

The initial selection of cluster centers is called the initialization step. We abstract implementations of the initialization step with the `KMeansSelector` trait.

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
