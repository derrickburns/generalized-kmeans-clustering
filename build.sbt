    name := "massivedatascience-clusterer"
    organization := "com.massivedatascience"
    organizationName := "Massive Data Science, LLC"
    licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0"))
    scalaVersion := "2.12.18"
    scalacOptions ++= Seq("-unchecked", "-feature")
    Compile / compile / scalacOptions ++= Seq("-Xlint", "-Xfatal-warnings", "-deprecation")

    javacOptions ++= Seq( "-source", "17.0", "-target", "17.0")
    publishMavenStyle := true
    Test / publishArtifact := false
    pomIncludeRepository := { _ => false }
    libraryDependencies ++= Seq(
      "org.scalactic" %% "scalactic" % "3.2.18",
      "org.scalatest" %% "scalatest" % "3.2.18" % "test"
    )
    val sparkPackageName = "derrickburns/generalized-kmeans-clustering"
    // sparkComponents += "mllib"
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

val sparkVersion = "3.4.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.holdenkarau" %% "spark-testing-base" % "3.4.0_1.4.7" % "test"
)

libraryDependencies ++= Seq(
  "com.github.fommil.netlib" % "core" % "1.1.2"
)



