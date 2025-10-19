    name := "massivedatascience-clusterer"
    organization := "com.massivedatascience"
    organizationName := "Massive Data Science, LLC"
    licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0"))

    // Cross-compilation: Scala 2.13 and 2.12
    scalaVersion := "2.13.14"
    crossScalaVersions := Seq("2.13.14", "2.12.18")

    scalacOptions ++= Seq("-unchecked", "-feature")

    // Conditional fatal warnings (disabled in CI by default, enable with -Dci=true)
    val isCi = sys.props.get("ci").contains("true")
    Compile / compile / scalacOptions ++= {
      val base = Seq("-Xlint", "-deprecation")
      if (isCi) base :+ "-Xfatal-warnings" else base
    }

    javacOptions ++= Seq( "-source", "17.0", "-target", "17.0")
    publishMavenStyle := true
    Test / publishArtifact := false
    pomIncludeRepository := { _ => false }
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
  // spark-testing-base version mapping: Spark 3.4.x -> 3.4.0_1.4.4, Spark 3.5.x -> 3.5.0_1.5.2
  "com.holdenkarau" %% "spark-testing-base" % {
    if (sparkVersion.startsWith("3.4.")) "3.4.0_1.4.4"
    else if (sparkVersion.startsWith("3.5.")) "3.5.0_1.5.2"
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
