package com.massivedatascience.clusterer

// Scala 2.13: Re-export CollectionConverters implicits for .par support
package object compat extends scala.collection.parallel.CollectionConverters
