package com.massivedatascience.clusterer

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD


object ResamplingRandomIndexingTrainer {
  val embeddings = List( new RandomIndexEmbedding(32, 0.05), new RandomIndexEmbedding(128, 0.05), new RandomIndexEmbedding(512, 0.05), new RandomIndexEmbedding(2048, 0.05))

  def train(
    raw: RDD[Vector],
    k: Int,
    maxIterations: Int,
    runs: Int,
    initializer: KMeansInitializer)(kMeans: MultiKMeansClusterer = new MultiKMeans(30) )
  : KMeansModel = {

    KMeans.trainViaResampling(DenseSquaredEuclideanPointOps)(raw, initializer, kMeans, embeddings)._2
  }
}
