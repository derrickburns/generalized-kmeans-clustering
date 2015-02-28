package com.massivedatascience.transforms

object Embeddings {
  val IDENTITY_EMBEDDING = "IDENTITY"
  val HAAR_EMBEDDING = "HAAR"
  val SYMMETRIZING_KL_EMBEDDING = "SYMMETRIZING_KL_EMBEDDING"
  val LOW_DIMENSIONAL_RI = "LOW_DIMENSIONAL_RI"
  val MEDIUM_DIMENSIONAL_RI = "MEDIUM_DIMENSIONAL_RI"
  val HIGH_DIMENSIONAL_RI = "HIGH_DIMENSIONAL_RI"


  def apply(embeddingName: String): Embedding = {
    embeddingName match {
      case IDENTITY_EMBEDDING => IdentityEmbedding
      case LOW_DIMENSIONAL_RI => new RandomIndexEmbedding(64, 0.01)
      case MEDIUM_DIMENSIONAL_RI => new RandomIndexEmbedding(256, 0.01)
      case HIGH_DIMENSIONAL_RI => new RandomIndexEmbedding(1024, 0.01)
      case HAAR_EMBEDDING => HaarEmbedding
      case SYMMETRIZING_KL_EMBEDDING => SymmetrizingKLEmbedding
      case _ => throw new RuntimeException(s"unknown embedding name $embeddingName")
    }
  }
}
