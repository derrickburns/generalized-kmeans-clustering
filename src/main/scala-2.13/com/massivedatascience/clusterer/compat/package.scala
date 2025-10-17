package com.massivedatascience.clusterer

// Scala 2.13: Re-export CollectionConverters implicits for .par support
package object compat {
  implicit def asParIterable[A](iterable: Iterable[A]): scala.collection.parallel.CollectionConverters.IterableIsParallelizable[A] =
    scala.collection.parallel.CollectionConverters.IterableIsParallelizable(iterable)

  implicit def asParSeq[A](seq: scala.collection.Seq[A]): scala.collection.parallel.CollectionConverters.SeqIsParallelizable[A] =
    scala.collection.parallel.CollectionConverters.SeqIsParallelizable(seq)

  implicit def asParMap[K, V](map: scala.collection.Map[K, V]): scala.collection.parallel.CollectionConverters.MapIsParallelizable[K, V] =
    scala.collection.parallel.CollectionConverters.MapIsParallelizable(map)
}
