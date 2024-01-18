# Seeding the Set of Cluster Centers

Any K-Means model may be used as seed value to Lloyd's algorithm. In fact, our clusterers accept multiple seed sets. The `K-Means.train` helper methods allows one to name an initialization method.

Two algorithms are implemented that produce viable seed sets. They may be constructed by using the `apply` method of the companion object`KMeansSelector`".

| Name                              | Algorithm                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------- |
| KMeansSelector.`RANDOM`           | Random selection of initial k centers                                                                   |
| KMeansSelector.`K_MEANS_PARALLEL` | a 5 step [K-Means Parallel implementation](http://theory.stanford.edu/\~sergei/papers/vldb12-kmpar.pdf) |

You may create a KMeansSelector using the apply method of the KMeansSelector companion object.

```scala
package com.massivedatascience.clusterer

object KMeansSelector {
  def apply(name: String): KMeansSelector = ???
}
```
