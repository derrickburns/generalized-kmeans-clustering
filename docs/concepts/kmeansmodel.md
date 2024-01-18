# KMeansModel

We define our realization of a k-means model, `KMeansModel`, which we enrich with operations to find closest clusters to a point and to compute distances:

```scala
package com.massivedatascience.clusterer

trait KMeansModel {

  val pointOps: BregmanPointOps

  def centers: IndexedSeq[BregmanCenter]


  def predict(point: Vector): Int

  def predictClusterAndDistance(point: Vector): (Int, Double)

  def predict(points: RDD[Vector]): RDD[Int]

  def predict(points: JavaRDD[Vector]): JavaRDD[java.lang.Integer]

  def computeCost(data: RDD[Vector]): Double


  def predictWeighted(point: WeightedVector): Int

  def predictClusterAndDistanceWeighted(point: WeightedVector): (Int, Double)

  def predictWeighted(points: RDD[WeightedVector]): RDD[Int]

  def computeCostWeighted(data: RDD[WeightedVector]): Double


  def predictBregman(point: BregmanPoint): Int

  def predictClusterAndDistanceBregman(point: BregmanPoint): (Int, Double)

  def predictBregman(points: RDD[BregmanPoint]): RDD[Int]

  def computeCostBregman(data: RDD[BregmanPoint): Double
}
```
