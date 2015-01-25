package com.massivedatascience.clusterer

import com.massivedatascience.clusterer.util.HaarWavelet
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.rdd.RDD

/**
 * Implements the "Wavelet-based Anytime Algorithm" of
 *
 * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.12.6944
 * http://www.cs.gmu.edu/~jessica/publications/ikmeans_sdm_workshop03.pdf
 *
 */

object WaveletKmeans {

  def train(pointOps: BregmanPointOps, maxIterations: Int = 30)(
    raw: RDD[Vector],
    baseInitializer: KMeansInitializer,
    embedding : Embedding = HaarEmbedding,
    depth: Int = 0,
    initializationSteps: Int = 5,
    kMeans: MultiKMeansClusterer = new MultiKMeans(pointOps, maxIterations)
    ) : (Double, KMeansModel) = {


    def recurse(data: RDD[Vector], remaining: Int) : (Double, KMeansModel) = {
      val initializer = if (remaining > 0) {
        val downData = data.map{embedding.embed}
        downData.cache()
        val (downCost, model) = recurse(downData, remaining - 1)
        val assignments = model.predict(downData)
        downData.unpersist(blocking = false)
        new SampleInitializer(pointOps, assignments)
      } else {
        baseInitializer
      }

      val (points,centerArray) = initializer.init(data)
      val (cost, centers)  = kMeans.cluster(points, centerArray)
      (cost, new KMeansModel(pointOps, centers))
    }

    recurse(raw, depth)
  }

  object HaarEmbedding extends Embedding {
    def embed(raw: Vector): Vector = Vectors.dense(HaarWavelet.average(raw.toArray))
  }
}
