# Distance Functions

Lloyd's algorithm converges for the class of distance functions called Bregman Divergences.  We provide a number of Bregman Divergences.

<table><thead><tr><th width="449">Name</th><th>Divergence</th></tr></thead><tbody><tr><td><code>BregmanDivergence.EUCLIDEAN</code></td><td>Squared Euclidean</td></tr><tr><td><code>BregmanDivergence.RELATIVE_ENTROPY</code></td><td><a href="http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback-Leibler</a></td></tr><tr><td><code>BregmanDivergence.DISCRETE_KL</code></td><td>Kullback-Leibler</td></tr><tr><td><code>BregmanDivergence.DISCRETE_SMOOTHED_KL</code></td><td>Kullback-Leibler</td></tr><tr><td><code>BregmanDivergence.SPARSE_SMOOTHED_KL</code></td><td>Kullback-Leibler</td></tr><tr><td><code>BregmanDivergence.LOGISTIC_LOSS</code></td><td>Logistic Loss</td></tr><tr><td><code>BregmanDivergence.GENERALIZED_I</code></td><td>Generalized I</td></tr><tr><td><code>BregmanDivergence.ITAKURA_SAITO</code></td><td><a href="http://en.wikipedia.org/wiki/Itakura%E2%80%93Saito_distance">Itakura-Saito</a></td></tr></tbody></table>

You may construct instances of `BregmanDivergence` using the `BregmanDivergence`companion object.

```scala
package com.massivedatascience.divergence

object BregmanDivergence {
  def apply(name: String): BregmanDivergence = ???
}
```

From this, one may construct a distance function using the `BregmanPointOps` companion function.

From your `BregmanDivergence`, you may create an instance of the distance function by using the `apply` method of the `BregmanPointOps` companion object.&#x20;

```scala
package com.massivedatascience.clusterer

object BregmanPointOps {
  def apply(d: BregmanDivergence): BregmanPointOps = ???
}
```
