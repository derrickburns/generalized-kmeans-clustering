# Bregman Divergence

While one can assign a point to a cluster using any distance function, Lloyd's algorithm only converges for a certain set of distance functions called [Bregman divergences](http://www.cs.utexas.edu/users/inderjit/public\_papers/bregmanclustering\_jmlr.pdf). Bregman divergences must define two methods, `convex` to evaluate a function on a point and `gradientOfConvex` to evaluate the gradient of the function on a point.

```scala
package com.massivedatascience.divergence

trait BregmanDivergence {
  def convex(v: Vector): Double
  def gradientOfConvex(v: Vector): Vector
}
```

For example, by defining `convex` to be the squared vector norm (i.e. the sum of the squares of the coordinates), one gets a distance function that equals the square of the well known Euclidean distance.&#x20;
