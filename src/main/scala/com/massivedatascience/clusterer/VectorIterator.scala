package com.massivedatascience.clusterer

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}


trait VectorIterator {
  def hasNext: Boolean

  def forward(): Unit

  def index: Int

  def value: Double
}

class SparseVectorIterator(x: SparseVector) extends VectorIterator {
  var i = 0

  def hasNext: Boolean = i < x.indices.length

  def forward(): Unit = i = i + 1

  def index: Int = x.indices(i)

  def value: Double = x.indices(i)
}

class DenseVectorIterator(x: DenseVector) extends VectorIterator {
  var i = 0

  def hasNext: Boolean = i < x.values.length

  def forward(): Unit = i = i + 1

  def index: Int = i

  def value: Double = x.values(i)
}

class NegativeSparseVectorIterator(x: SparseVector) extends VectorIterator {
  var i = 0

  def hasNext: Boolean = i < x.indices.length

  def forward(): Unit = i = i + 1

  def index: Int = x.indices(i)

  def value: Double = -x.indices(i)
}

class NegativeDenseVectorIterator(x: DenseVector) extends VectorIterator {
  var i = 0

  def hasNext: Boolean = i < x.values.length

  def forward(): Unit = i = i + 1

  def index: Int = i

  def value: Double = -x.values(i)
}