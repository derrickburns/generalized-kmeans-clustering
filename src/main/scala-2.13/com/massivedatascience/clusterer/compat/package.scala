package com.massivedatascience.clusterer

import scala.language.implicitConversions

// Scala 2.13: Provide .par extension method via compat package
package object compat {
  implicit class ParOps[A, CC[X] <: Iterable[X]](private val coll: CC[A]) extends AnyVal {
    @inline def par: scala.collection.parallel.ParIterable[A] =
      scala.collection.parallel.CollectionConverters.IterableIsParallelizable(coll).par
  }
}
