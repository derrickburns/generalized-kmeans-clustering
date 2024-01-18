# MultiKMeansClusterer

One may construct K-Means models using one of the provided clusterers that implement Lloyd's algorithm.

```scala
trait MultiKMeansClusterer extends Serializable with Logging {
  def cluster(
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
```
