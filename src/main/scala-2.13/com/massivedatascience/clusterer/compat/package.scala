package com.massivedatascience.clusterer

// Scala 2.13: Re-export CollectionConverters for .par support
package object compat extends scala.collection.parallel.CollectionConverters.type
