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
      "joda-time" % "joda-time" % "2.2",
      "org.joda" % "joda-convert" % "1.6",
      "org.scalactic" %% "scalactic" % "3.2.17",
      "org.scalatest" %% "scalatest" % "3.2.17" % "test"

    )
    val sparkPackageName = "derrickburns/generalized-kmeans-clustering"
    // sparkComponents += "mllib"
    Test / testOptions += Tests.Argument("-Dlog4j.configuration=log4j.properties")

val sparkVersion = "3.4.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.holdenkarau" %% "spark-testing-base" % "3.4.0_1.4.7" % "test"
)

libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2"



