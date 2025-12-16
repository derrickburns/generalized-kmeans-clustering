package com.massivedatascience.clusterer.ml

import java.nio.file.Files

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Persistence round-trips for clustering models using the shared model helpers. */
class ExtendedPersistenceSuite extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  private val spark: SparkSession = SparkSession
    .builder()
    .master("local[2]")
    .appName("ExtendedPersistenceSuite")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  import spark.implicits._

  override def afterAll(): Unit = {
    try {
      spark.stop()
    } finally {
      super.afterAll()
    }
  }

  private def tinyDF() = {
    Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(9.0, 9.0)),
      Tuple1(Vectors.dense(10.0, 10.0))
    ).toDF("features")
  }

  private def tinyPositiveDF() = {
    Seq(
      Tuple1(Vectors.dense(0.1, 0.2)),
      Tuple1(Vectors.dense(1.0, 1.1)),
      Tuple1(Vectors.dense(9.0, 9.1)),
      Tuple1(Vectors.dense(10.0, 10.1))
    ).toDF("features")
  }

  private def withTempDir(prefix: String)(f: String => Unit): Unit = {
    val dir = Files.createTempDirectory(prefix).toFile
    try {
      f(dir.getCanonicalPath)
    } finally {
      if (dir.exists()) {
        dir.listFiles().foreach(_.delete())
        dir.delete()
      }
    }
  }

  test("SoftKMeans save/load round-trip preserves centers and columns") {
    val df = tinyDF()
    val model = new SoftKMeans()
      .setK(2)
      .setBeta(1.0)
      .setSeed(1234L)
      .setProbabilityCol("probs")
      .fit(df)

    withTempDir("softkmeans-persist") { path =>
      model.write.overwrite().save(path)
      val loaded = SoftKMeansModel.load(path)

      loaded.numClusters shouldBe model.numClusters
      loaded.clusterCentersAsVectors.map(_.toArray).zip(model.clusterCentersAsVectors.map(_.toArray)).foreach {
        case (l, r) => l should contain theSameElementsInOrderAs r
      }
      loaded.hasSummary shouldBe false

      val pred = loaded.transform(df)
      pred.columns should contain("probs")
      pred.count() shouldBe df.count()
    }
  }

  test("KernelKMeans save/load round-trip with linear kernel") {
    val df = tinyDF()
    val model = new KernelKMeans()
      .setK(2)
      .setKernelType("linear")
      .setMaxIter(5)
      .setSeed(42L)
      .fit(df)

    withTempDir("kernelkmeans-persist") { path =>
      model.write.overwrite().save(path)
      val loaded = KernelKMeansModel.load(path)

      loaded.numSupportVectors shouldBe model.numSupportVectors
      loaded.clusterSizes.sum shouldBe df.count()
      loaded.summaryOption shouldBe empty

      val pred = loaded.transform(df)
      pred.count() shouldBe df.count()
    }
  }

  test("AgglomerativeBregman save/load round-trip") {
    val df = tinyDF()
    val model = new AgglomerativeBregman()
      .setNumClusters(2)
      .setDivergence("squaredEuclidean")
      .setLinkage("average")
      .fit(df)

    withTempDir("agglo-persist") { path =>
      model.write.overwrite().save(path)
      val loaded = AgglomerativeBregmanModel.load(path)

      loaded.k shouldBe model.k
      loaded.clusterCentersAsVectors.length shouldBe model.clusterCentersAsVectors.length
      loaded.summaryOption shouldBe empty

      val pred = loaded.transform(df)
      pred.count() shouldBe df.count()
    }
  }

  test("BregmanMixtureModel save/load round-trip") {
    val df = tinyPositiveDF()
    val model = new BregmanMixture()
      .setK(2)
      .setDivergence("squaredEuclidean")
      .setSeed(99L)
      .fit(df)

    withTempDir("bmm-persist") { path =>
      model.write.overwrite().save(path)
      val loaded = BregmanMixtureModelInstance.load(path)

      loaded.numComponents shouldBe model.numComponents
      loaded.componentMeans.length shouldBe model.componentMeans.length
      loaded.summaryOption shouldBe empty

      val pred = loaded.transform(df)
      pred.count() shouldBe df.count()
      pred.columns should contain(model.getProbabilityCol)
    }
  }

  test("StreamingKMeans save/load round-trip") {
    val df = tinyDF()
    val model = new StreamingKMeans()
      .setK(2)
      .setDecayFactor(0.8)
      .setSeed(7L)
      .fit(df)

    withTempDir("streamingkmeans-persist") { path =>
      model.write.overwrite().save(path)
      val loaded = StreamingKMeansModel.load(path)

      loaded.numClusters shouldBe model.numClusters
      loaded.clusterCentersAsVectors.length shouldBe model.clusterCentersAsVectors.length
      loaded.hasSummary shouldBe false

      val pred = loaded.transform(df)
      pred.count() shouldBe df.count()
    }
  }
}
