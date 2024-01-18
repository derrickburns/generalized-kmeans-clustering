# Using an Embedding

Often raw data must be embedded in a different space before clustering. We provide several common embeddings. You may also [create your own](creating-a-custom-k-means-clusterer/).

<table><thead><tr><th width="400">Name</th><th>Algorithm</th></tr></thead><tbody><tr><td>Embedding.<code>IDENTITY_EMBEDDING</code></td><td>Identity</td></tr><tr><td>Embedding.<code>HAAR_EMBEDDING</code></td><td><a href="http://www.cs.gmu.edu/~jessica/publications/ikmeans_sdm_workshop03.pdf">Haar Transform</a></td></tr><tr><td>Embedding.<code>LOW_DIMENSIONAL_RI</code></td><td><a href="https://en.wikipedia.org/wiki/Random_indexing">Random Indexing</a> with dimension 64 and epsilon = 0.1</td></tr><tr><td>Embedding.<code>MEDIUM_DIMENSIONAL_RI</code></td><td>Random Indexing with dimension 256 and epsilon = 0.1</td></tr><tr><td>Embedding.<code>HIGH_DIMENSIONAL_RI</code></td><td>Random Indexing with dimension 1024 and epsilon = 0.1</td></tr><tr><td>Embedding.<code>SYMMETRIZING_KL_EMBEDDING</code></td><td><a href="http://www-users.cs.umn.edu/~banerjee/papers/13/bregman-metric.pdf">Symmetrizing KL Embedding</a></td></tr></tbody></table>

You may create an embedding using the apply method of the companion object.

```scala
package com.massivedatascience.transforms

object Embedding {
   def apply(embeddingName: String): Embedding = ???
}
```
