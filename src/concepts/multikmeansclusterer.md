# MultiKMeansClusterer

Lloyd's algorithm is simple to describe, but in practice different implementations are possible that can yield dramatically different running times depending on the data being clusters. We abstract the clusterer using the `MultiKMeansClusterer` trait.

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
