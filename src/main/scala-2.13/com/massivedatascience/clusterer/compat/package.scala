package com.massivedatascience.clusterer

import scala.collection.parallel.CollectionConverters

// Scala 2.13: Re-export CollectionConverters implicits for .par support
package object compat extends CollectionConverters
