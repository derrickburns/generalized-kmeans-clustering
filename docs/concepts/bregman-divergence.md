# Bregman Divergence

While one can assign a point to a cluster using any distance function, Lloyd's algorithm only converges for a certain set of distance functions called [Bregman divergences](http://www.cs.utexas.edu/users/inderjit/public\_papers/bregmanclustering\_jmlr.pdf). Bregman divergences must define two methods, `convex` to evaluate a function on a point and `gradientOfConvex` to evaluate the gradient of the function on a point.

```scala
package com.massivedatascience.divergence

trait BregmanDivergence {
  def convex(v: Vector): Double

  def gradientOfConvex(v: Vector): Vector
}

```

For example, by defining `convex` to be the squared vector norm (i.e. the sum of the squares of the coordinates), one gets a distance function that equals the square of the well known Euclidean distance. We name it the `SquaredEuclideanDistanceDivergence`.

These distance functions are provided by the package. &#x20;

| Name                                   | Space                | Divergence                                                                            | Input  |
| -------------------------------------- | -------------------- | ------------------------------------------------------------------------------------- | ------ |
| `SquaredEuclideanDistanceDivergence`   | $\mathbb{R}^d$       | Squared Euclidean                                                                     |        |
| `RealKullbackLeiblerSimplexDivergence` | $\mathbb{R}^d\_{>0}$ | [Kullback-Leibler](http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler\_divergence) | Dense  |
| `NaturalKLSimplexDivergence`           | $\mathbb{N}^d\_{>0}$ | Kullback-Leibler                                                                      | Dense  |
| `RealKLDivergence`                     | $\mathbb{R}^d$       | Kullback-Leibler                                                                      | Dense  |
| `NaturalKLDivergence`                  | $\mathbb{N}^d$       | Kullback-Leibler                                                                      | Dense  |
| `ItakuraSaitoDivergence`               | $\mathbb{R}^d\_{>0}$ | Kullback-Leibler                                                                      | Sparse |
| `LogisticLossDivergence`               | $\mathbb{R}$         | Logistic Loss                                                                         |        |
| `GeneralizedIDivergence`               | $\mathbb{R}$         | Generalized I                                                                         |        |

When selecting a distance function, consider the domain of the input data. For example, frequency data is integral. Similarity of frequencies or distributions are best performed using the Kullback-Leibler divergence.
