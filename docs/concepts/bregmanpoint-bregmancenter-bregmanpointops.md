# BregmanPoint, BregmanCenter, BregmanPointOps

For efficient repeated computation of distance between a fixed set of points and varying cluster centers, is it convenient to pre-compute certain information and associate that information with the point or the cluster center. The class that represent an enriched point is `BregmanPoint`. The class that represent the enriched cluster center is `BregmanCenter`. Users of this package do not construct instances of these objects directly.

```scala
package com.massivedatascience.divergence

trait BregmanPoint

trait BregmanCenter
```

We enrich a Bregman divergence with a set of commonly used operations, including factory methods `toPoint` and `toCenter` to construct instances of the aforementioned `BregmanPoint` and `BregmanCenter`.

The enriched trait is the `BregmanPointOps`.

```scala
package com.massivedatascience.clusterer

trait BregmanPointOps  {
  type P = BregmanPoint
  type C = BregmanCenter

  val divergence: BregmanDivergence

  def toPoint(v: WeightedVector): P

  def toCenter(v: WeightedVector): C

  def centerMoved(v: P, w: C): Boolean

  def findClosest(centers: IndexedSeq[C], point: P): (Int, Double)

  def findClosestCluster(centers: IndexedSeq[C], point: P): Int

  def distortion(data: RDD[P], centers: IndexedSeq[C])

  def pointCost(centers: IndexedSeq[C], point: P): Double

  def distance(p: BregmanPoint, c: BregmanCenter): Double
}
```
