ThisBuild / organization := "com.massivedatascience"
ThisBuild / organizationName := "Massive Data Science, LLC"
ThisBuild / organizationHomepage := Some(url("http://example.com/"))
ThisBuild / versionScheme := Some("semver-spec")


ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/derrickburns/generalized-kmeans-clustering"),
    "scm:git@github.com:derrickburns/generalized-kmeans-clustering.git"
  )
)
ThisBuild / developers := List(
  Developer(
    id = "derrickburns",
    name = "Derrick R. Burns",
    email = "derrickrburns@gmail.com",
    url = url("http://your.url")
  )
)

ThisBuild / description := "Some description about your project."
ThisBuild / licenses := List(
  "Apache 2" -> new URL("http://www.apache.org/licenses/LICENSE-2.0.txt")
)
ThisBuild / homepage := Some(url("https://github.com/derrickburns/generalized-kmeans-clustering"))

// Remove all additional repository other than Maven Central from POM
ThisBuild / pomIncludeRepository := { _ => false }
ThisBuild / publishTo := {
  // For accounts created after Feb 2021:
  // val nexus = "https://s01.oss.sonatype.org/"
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value) Some("snapshots" at nexus + "content/repositories/snapshots")
  else Some("releases" at nexus + "service/local/staging/deploy/maven2")
}
ThisBuild / publishMavenStyle := true

releaseIgnoreUntrackedFiles := true
    
    
    
ThisBuild / name := "massivedatascience-clusterer"
ThisBuild / version := "0.2.0"

ThisBuild / scalaVersion := "2.12.18"
ThisBuild / scalacOptions ++= Seq("-unchecked", "-feature")
Compile / compile / scalacOptions ++= Seq("-Xlint", "-Xfatal-warnings", "-deprecation")

ThisBuild / javacOptions ++= Seq( "-source", "17.0", "-target", "17.0")
Test / publishArtifact := false

ThisBuild / libraryDependencies ++= Seq(
      "joda-time" % "joda-time" % "2.2",
      "org.joda" % "joda-convert" % "1.6",
      "org.scalactic" %% "scalactic" % "3.2.17",
      "org.scalatest" %% "scalatest" % "3.2.17" % "test"

    )
    val sparkPackageName = "derrickburns/generalized-kmeans-clustering"
    // sparkComponents += "mllib"
ThisBuild / Test / testOptions += Tests.Argument("-Dlog4j.configuration=log4j.properties")

val sparkVersion = "3.4.0"

ThisBuild / libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.holdenkarau" %% "spark-testing-base" % "3.4.0_1.4.7" % "test"
)

ThisBuild / libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2"

ThisBuild / publishTo := sonatypePublishToBundle.value
// For all Sonatype accounts created on or after February 2021
ThisBuild / sonatypeCredentialHost := "s01.oss.sonatype.org"

ThisBuild / pgpSigningKey := Some("0xE3F60B02")  // replace with your key ID
