    name := "massivedatascience-clusterer"
    organization := "com.massivedatascience"
    organizationName := "Massive Data Science, LLC"
    licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0"))

    // Cross-compilation: Scala 2.12 and 2.13
    scalaVersion := "2.12.18"
    crossScalaVersions := Seq("2.12.18", "2.13.14")

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

    val sparkPackageName = "derrickburns/generalized-kmeans-clustering"
    Test / testOptions += Tests.Argument("-Dlog4j.configurationFile=log4j2.properties")
    Test / fork := true
    Test / javaOptions ++= Seq(
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
  "com.holdenkarau" %% "spark-testing-base" % s"${sparkVersion}_2.1.2" % "test"
)

libraryDependencies ++= Seq(
  "com.github.fommil.netlib" % "core" % "1.1.2"
)

// Enable Scaladoc generation with sbt-unidoc
enablePlugins(ScalaUnidocPlugin)

// Scaladoc settings (disable -diagrams to avoid compiler bugs)
Compile / doc / scalacOptions ++= Seq(
  "-doc-title", "Generalized K-Means Clustering API",
  "-doc-version", version.value,
  "-groups",
  "-implicits",
  "-no-link-warnings"
)

// Exclude XORShiftRandom.scala from scaladoc due to Scala 2.12 compiler bug
Compile / doc / sources := {
  val orig = (Compile / doc / sources).value
  orig.filterNot(_.getName == "XORShiftRandom.scala")
}

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
