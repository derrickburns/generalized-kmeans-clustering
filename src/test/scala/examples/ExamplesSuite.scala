package examples

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import java.io.{ ByteArrayOutputStream, PrintStream }
import java.nio.file.{ Files, Paths }

/** Test suite that ensures all documented examples compile and run successfully with assertions
  * passing.
  *
  * This suite is invoked by CI to keep documentation executable and trustworthy.
  */
class ExamplesSuite extends AnyFunSuite with Matchers {

  /** Helper to capture output and verify example ran successfully */
  private def runExample(name: String, example: => Unit): Unit = {
    val outputStream = new ByteArrayOutputStream()
    val printStream  = new PrintStream(outputStream)
    val originalOut  = System.out
    val originalErr  = System.err

    try {
      System.setOut(printStream)
      System.setErr(printStream)

      example

      val output = outputStream.toString
      info(s"$name output: ${output.take(200)}")

      // Verify no exceptions were thrown (example completed)
      output should not include "Exception"
      output should not include "Error"
      output should not include "assertion failed"

    } catch {
      case e: Throwable =>
        fail(s"$name failed with exception: ${e.getMessage}\n${e.getStackTrace.mkString("\n")}")
    } finally {
      System.setOut(originalOut)
      System.setErr(originalErr)
      printStream.close()
    }
  }

  test("BisectingExample runs successfully with assertions") {
    runExample("BisectingExample", BisectingExample.main(Array.empty))
  }

  test("SoftKMeansExample runs successfully with assertions") {
    runExample("SoftKMeansExample", SoftKMeansExample.main(Array.empty))
  }

  test("XMeansExample runs successfully with assertions") {
    runExample("XMeansExample", XMeansExample.main(Array.empty))
  }

  test("PersistenceRoundTrip save/load cycle works") {
    val tmpDir = Files.createTempDirectory("gkm-test-persistence")
    try {
      val savePath = tmpDir.resolve("model").toString

      // Save
      runExample(
        "PersistenceRoundTrip save",
        PersistenceRoundTrip.main(Array("save", savePath))
      )

      // Verify model was saved
      Files.exists(Paths.get(savePath)) shouldBe true
      Files.isDirectory(Paths.get(savePath)) shouldBe true

      // Load
      runExample(
        "PersistenceRoundTrip load",
        PersistenceRoundTrip.main(Array("load", savePath))
      )

    } finally {
      // Cleanup
      import scala.util.Try
      Try {
        def deleteRecursively(path: java.nio.file.Path): Unit = {
          if (Files.isDirectory(path)) {
            Files.list(path).forEach(deleteRecursively)
          }
          Files.deleteIfExists(path)
        }
        deleteRecursively(tmpDir)
      }
    }
  }

  test("PersistenceRoundTripKMedoids save/load cycle works") {
    val tmpDir = Files.createTempDirectory("gkm-test-kmedoids")
    try {
      val savePath = tmpDir.resolve("kmedoids").toString

      // Save
      runExample(
        "PersistenceRoundTripKMedoids save",
        PersistenceRoundTripKMedoids.main(Array("save", savePath))
      )

      // Verify model was saved
      Files.exists(Paths.get(savePath)) shouldBe true
      Files.isDirectory(Paths.get(savePath)) shouldBe true

      // Load
      runExample(
        "PersistenceRoundTripKMedoids load",
        PersistenceRoundTripKMedoids.main(Array("load", savePath))
      )

    } finally {
      // Cleanup
      import scala.util.Try
      Try {
        def deleteRecursively(path: java.nio.file.Path): Unit = {
          if (Files.isDirectory(path)) {
            Files.list(path).forEach(deleteRecursively)
          }
          Files.deleteIfExists(path)
        }
        deleteRecursively(tmpDir)
      }
    }
  }

  test("PersistenceRoundTripStreamingKMeans save/load cycle works") {
    val tmpDir = Files.createTempDirectory("gkm-test-streaming")
    try {
      val savePath = tmpDir.resolve("streaming").toString

      // Save
      runExample(
        "PersistenceRoundTripStreamingKMeans save",
        PersistenceRoundTripStreamingKMeans.main(Array("save", savePath))
      )

      // Verify model was saved
      Files.exists(Paths.get(savePath)) shouldBe true
      Files.isDirectory(Paths.get(savePath)) shouldBe true

      // Load
      runExample(
        "PersistenceRoundTripStreamingKMeans load",
        PersistenceRoundTripStreamingKMeans.main(Array("load", savePath))
      )

    } finally {
      // Cleanup
      import scala.util.Try
      Try {
        def deleteRecursively(path: java.nio.file.Path): Unit = {
          if (Files.isDirectory(path)) {
            Files.list(path).forEach(deleteRecursively)
          }
          Files.deleteIfExists(path)
        }
        deleteRecursively(tmpDir)
      }
    }
  }

  test("PersistenceRoundTripSoftKMeans save/load cycle works") {
    val tmpDir = Files.createTempDirectory("gkm-test-softkmeans")
    try {
      val savePath = tmpDir.resolve("softkmeans").toString

      // Save
      runExample(
        "PersistenceRoundTripSoftKMeans save",
        PersistenceRoundTripSoftKMeans.main(Array("save", savePath))
      )

      // Verify model was saved
      Files.exists(Paths.get(savePath)) shouldBe true
      Files.isDirectory(Paths.get(savePath)) shouldBe true

      // Load
      runExample(
        "PersistenceRoundTripSoftKMeans load",
        PersistenceRoundTripSoftKMeans.main(Array("load", savePath))
      )

    } finally {
      // Cleanup
      import scala.util.Try
      Try {
        def deleteRecursively(path: java.nio.file.Path): Unit = {
          if (Files.isDirectory(path)) {
            Files.list(path).forEach(deleteRecursively)
          }
          Files.deleteIfExists(path)
        }
        deleteRecursively(tmpDir)
      }
    }
  }

  test("PersistenceRoundTripCoresetKMeans save/load cycle works") {
    val tmpDir = Files.createTempDirectory("gkm-test-coresetkmeans")
    try {
      val savePath = tmpDir.resolve("coresetkmeans").toString

      // Save
      runExample(
        "PersistenceRoundTripCoresetKMeans save",
        PersistenceRoundTripCoresetKMeans.main(Array("save", savePath))
      )

      // Verify model was saved
      Files.exists(Paths.get(savePath)) shouldBe true
      Files.isDirectory(Paths.get(savePath)) shouldBe true

      // Load
      runExample(
        "PersistenceRoundTripCoresetKMeans load",
        PersistenceRoundTripCoresetKMeans.main(Array("load", savePath))
      )
    } finally {
      // Cleanup
      import scala.util.Try
      Try {
        def deleteRecursively(path: java.nio.file.Path): Unit = {
          if (Files.isDirectory(path)) {
            Files.list(path).forEach(deleteRecursively)
          }
          Files.deleteIfExists(path)
        }
        deleteRecursively(tmpDir)
      }
    }
  }

  test("All examples contain assertions (truth-linked documentation)") {
    // Verify that examples have assertions to ensure they validate correctness
    val exampleDir = Paths.get("src/main/scala/examples")

    if (Files.exists(exampleDir)) {
      val exampleFiles = Files
        .list(exampleDir)
        .filter(_.toString.endsWith(".scala"))
        .toArray
        .map(_.asInstanceOf[java.nio.file.Path])

      exampleFiles should not be empty

      exampleFiles.foreach { file =>
        val content = new String(Files.readAllBytes(file))
        withClue(s"${file.getFileName} should contain assertions: ") {
          content should (include("assert(") or include("require("))
        }
      }
    } else {
      info("Examples directory not found; skipping assertion verification")
    }
  }
}
