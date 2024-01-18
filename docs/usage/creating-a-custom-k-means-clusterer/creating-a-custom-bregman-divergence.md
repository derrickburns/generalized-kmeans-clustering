# Creating a Custom Bregman Divergence

**Custom `BregmanDivergence`**

You may create your own custom `BregmanDivergence` given a suitable continuously-differentiable real-valued and strictly convex function defined on a closed convex set in R^^N using the `apply` method of the companion object. Send a pull request to have it added the the package.

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

**Custom `BregmanPointOps`**

You may create your own custom `BregmanPointsOps` from your own implementation of the `BregmanDivergence` trait given a `BregmanDivergence` using the `apply` method of the companion object. Send a pull request to have it added the the package.

```scala
package com.massivedatascience.clusterer

object BregmanPointOps {

  def apply(d: BregmanDivergence): BregmanPointOps = ???

  def apply(d: BregmanDivergence, factor: Double): BregmanPointOps = ???
}
```

