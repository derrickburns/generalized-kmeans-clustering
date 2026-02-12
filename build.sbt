    name := "massivedatascience-clusterer"
    organization := "com.massivedatascience"
    organizationName := "Massive Data Science, LLC"
    licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0"))

    // Cross-compilation: Scala 2.13 and 2.12
    scalaVersion := "2.13.14"
    crossScalaVersions := Seq("2.13.14", "2.12.18")

    scalacOptions ++= Seq("-unchecked", "-feature")

    // Dead code detection flags (version-specific)
    // Scala 2.13+ uses -W flags, Scala 2.12 uses -Ywarn flags
    scalacOptions ++= {
      CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n >= 13 =>
          Seq(
            "-Wunused:imports",
            "-Wunused:privates",
            "-Wunused:locals",
            "-Wvalue-discard"
          )
        case Some((2, 12)) =>
          Seq(
            "-Ywarn-unused:imports",
            "-Ywarn-unused:privates",
            "-Ywarn-unused:locals",
            "-Ywarn-value-discard"
          )
        case _ => Seq.empty
      }
    }

    // Conditional fatal warnings (disabled in CI by default, enable with -Dci=true)
    val isCi = sys.props.get("ci").contains("true")
    Compile / compile / scalacOptions ++= {
      val base = Seq("-Xlint", "-deprecation")
      if (isCi) base :+ "-Xfatal-warnings" else base
    }

// Enforce Java 17 runtime/toolchain (Spark/Hadoop incompatibilities on JDK 21+)
val javaSpecVersion = sys.props.getOrElse("java.specification.version", "0")

// Helper to safely parse int for Scala 2.12 (sbt build definition)
def safeToInt(s: String): Option[Int] = try { Some(s.toInt) } catch { case _: NumberFormatException => None }

// Parse Java version: handles both legacy "1.8" format and modern "17" format
val javaMajor = {
  val parts = javaSpecVersion.split('.')
  val first = parts.headOption.flatMap(safeToInt).getOrElse(0)
  // Legacy format: "1.8" means Java 8; modern format: "17" means Java 17
  if (first == 1 && parts.length > 1) safeToInt(parts(1)).getOrElse(0) else first
}

// Relaxed check: Warn instead of fail, to allow Java 8
val _ = if (javaMajor != 17 && javaMajor != 8) {
  println(s"[WARN] JDK detected: $javaSpecVersion. Recommended: Java 17. Java 8 is also supported.")
}

// Adjust javac target based on detected version (fallback to 1.8 if running on 8)
// Note: Java 9+ uses simple version numbers (e.g., "17"), not "17.0"
javacOptions ++= (if (javaMajor <= 8) Seq("-source", "1.8", "-target", "1.8") else Seq("-source", "17", "-target", "17"))
    // ============================================
    // Maven Central Publishing Configuration
    // ============================================
    publishMavenStyle := true
    Test / publishArtifact := false
    pomIncludeRepository := { _ => false }

    // sbt-ci-release 1.11+ publishes to Central Portal by default

    // POM metadata required by Maven Central
    homepage := Some(url("https://github.com/derrickburns/generalized-kmeans-clustering"))
    scmInfo := Some(
      ScmInfo(
        url("https://github.com/derrickburns/generalized-kmeans-clustering"),
        "scm:git@github.com:derrickburns/generalized-kmeans-clustering.git"
      )
    )
    developers := List(
      Developer(
        id = "derrickburns",
        name = "Derrick Burns",
        email = "derrick@massivedatascience.com",
        url = url("https://github.com/derrickburns")
      )
    )
    description := "Generalized K-Means clustering with Bregman divergences for Apache Spark"
    // Test dependencies
    libraryDependencies ++= Seq(
      "org.scalactic" %% "scalactic" % "3.2.19",
      "org.scalatest" %% "scalatest" % "3.2.19" % "test",
      "org.scalacheck" %% "scalacheck" % "1.17.0" % "test"
    )

    // Scala 2.13+ requires parallel collections as a separate dependency
    // Scala 2.12 has parallel collections built-in
    libraryDependencies ++= {
      CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n >= 13 =>
          Seq("org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4")
        case _ =>
          Seq()
      }
    }

    val sparkPackageName = "derrickburns/generalized-kmeans-clustering"
    Test / testOptions += Tests.Argument("-Dlog4j.configurationFile=log4j2.properties")
    Test / fork := true
    Test / javaOptions ++= Seq(
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
      "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
      "-Djava.awt.headless=true",
      "-XX:+IgnoreUnrecognizedVMOptions",
      "-Dlog4j.configurationFile=log4j2.properties",
      "-Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.F2jBLAS",
      "-Dcom.github.fommil.netlib.LAPACK=com.github.fommil.netlib.F2jLAPACK",
      "-Dcom.github.fommil.netlib.ARPACK=com.github.fommil.netlib.F2jARPACK"
    )

// Spark version (override with -Dspark.version=3.5.1)
val sparkVersion = sys.props.getOrElse("spark.version", "3.5.1")

// Spark dependencies (Provided for deployment, available for compile/test)
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-streaming" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",

  // Test dependencies (need Spark at test time)
  "org.apache.spark" %% "spark-core" % sparkVersion % "test",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "test",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "test",
  // spark-testing-base version mapping: Spark 3.4.x -> 3.4.0_1.4.4, Spark 3.5.x -> 3.5.0_1.5.2, Spark 4.0.x -> 4.0.0_2.1.2
  "com.holdenkarau" %% "spark-testing-base" % {
    if (sparkVersion.startsWith("3.4.")) "3.4.0_1.4.4"
    else if (sparkVersion.startsWith("3.5.")) "3.5.0_1.5.2"
    else if (sparkVersion.startsWith("4.0.")) "4.0.0_2.1.2"
    else s"${sparkVersion}_1.4.4"
  } % "test"
)

libraryDependencies ++= Seq(
  "com.github.fommil.netlib" % "core" % "1.1.2"
)

// Enable Scaladoc generation with sbt-unidoc
enablePlugins(ScalaUnidocPlugin)

// Scaladoc settings
Compile / doc / scalacOptions ++= Seq(
  "-doc-title", "Generalized K-Means Clustering API",
  "-doc-version", version.value,
  "-groups",
  "-implicits",
  "-no-link-warnings"
)

// Unidoc settings
ScalaUnidoc / unidoc / unidocProjectFilter := inProjects(ThisProject)
ScalaUnidoc / unidoc / target := baseDirectory.value / "docs" / "api"

ScalaUnidoc / unidoc / scalacOptions ++= Seq(
  "-doc-title", "Generalized K-Means Clustering API",
  "-doc-version", version.value,
  "-groups",
  "-implicits",
  "-no-link-warnings"
)

// SBOM (Software Bill of Materials) and vulnerability scanning
// Run with: sbt dependencyCheck
// Generates: target/dependency-check-report.html and dependency-check-report.json
// The JSON report serves as SBOM
dependencyCheckFormats := Seq("HTML", "JSON")
dependencyCheckAssemblyAnalyzerEnabled := Some(false)
dependencyCheckSuppressionFile := Some(baseDirectory.value / ".dependency-check-suppressions.xml")
