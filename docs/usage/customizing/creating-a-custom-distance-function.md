# Creating a Custom Distance Function

To create a custom distance function from a Bregman Divergence, you must provide two functions: a suitable continuously-differentiable real-valued and strictly convex function defined on a closed convex set in R^^N. With these, you may use the `apply` method on the `BregmanDivergence` companion object to create a `BregmanDivergence`.

```scala
package com.massivedatascience.divergence

object BregmanDivergence {

  /**
   * Create a Bregman Divergence from
   * @param f any continuously-differentiable real-valued and strictly
   *          convex function defined on a closed convex set in R^^N
   * @param gradientF the gradient of f
   * @return a Bregman Divergence on that function
   */
  def apply(f: (Vector) => Double, gradientF: (Vector) => Vector): BregmanDivergence = ???
}
```

Finally, from your `BregmanDivergence`, you may create an instance of the distance function by using the `apply` method of the `BregmanPointOps` companion object.&#x20;

```scala
package com.massivedatascience.clusterer

object BregmanPointOps {

  def apply(d: BregmanDivergence): BregmanPointOps = ???

  def apply(d: BregmanDivergence, factor: Double): BregmanPointOps = ???
}
```

