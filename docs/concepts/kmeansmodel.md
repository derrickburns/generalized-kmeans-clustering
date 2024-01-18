# KMeansModel

A K-means model is a set of cluster centers.  We abstract the K-means model with the `KMeansModel` trait with methods to map an arbitrary point (viz. `Vector`, `WeightedVector`, or `BregmanPoint`) to the nearest cluster center and to compute the cost/distance to that center.&#x20;

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
