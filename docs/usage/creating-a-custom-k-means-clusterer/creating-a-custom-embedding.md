---
description: How to create a custom embedding
---

# Creating a Custom Embedding

Perhaps you have a dimensionality reduction method that is not provided by one of the standard embeddings. You may create your own embedding by implementing the `Embedding` trait.

<pre class="language-scala"><code class="lang-scala"><strong>package com.massivedatascience.transforms
</strong><strong>
</strong><strong>trait Embedding extends Serializable {
</strong>  /**
   * Tranform a weighted vector into another space
   * @param v the weighted vector
   * @return the transformed vector
   */
  def embed(v: WeightedVector): WeightedVector = WeightedVector(embed(v.homogeneous), v.weight)
  def embed(v: Vector): Vector
  def embed(v: VectorIterator): Vector
}
</code></pre>

For example, If the number of clusters desired is small, but the dimension is high, one may also use the method of [Random Projections](http://www.cs.toronto.edu/\~zouzias/downloads/papers/NIPS2010\_kmeans.pdf). At present, no embedding is provided for random projections, but, hey, I have to leave something for you to do!&#x20;
