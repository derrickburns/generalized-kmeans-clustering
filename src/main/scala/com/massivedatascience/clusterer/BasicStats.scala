package com.massivedatascience.clusterer

trait BasicStats {
  def getMovement: Double

  def getNonEmptyClusters: Int

  def getEmptyClusters: Int

  def getRound: Int
}