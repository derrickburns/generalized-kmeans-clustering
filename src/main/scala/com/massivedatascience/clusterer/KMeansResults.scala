package com.massivedatascience.clusterer

import org.apache.spark.rdd.RDD

case class KMeansResults(distortion: Double, assignments: RDD[(Int, Double)])
