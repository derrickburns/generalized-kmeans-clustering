// Basic facts
name := "massivedatascience-clusterer"

organization := "com.massivedatascience"

organizationName := "Massive Data Science, LLC"

scalaVersion := "2.10.4"

crossScalaVersions := Seq("2.10.4", "2.11.5")

scalacOptions ++= Seq("-deprecation", "-unchecked", "-feature", "-Xfatal-warnings", "-Xlint")

javacOptions ++= Seq(
  "-source", "1.7",
  "-target", "1.7")

publishMavenStyle := true

publishArtifact in Test := false

pomIncludeRepository := { _ => false }

libraryDependencies ++= Seq(
  "joda-time" % "joda-time" % "2.2",
  "org.joda" % "joda-convert" % "1.6",
  // test dependencies
  "org.scalatest" %% "scalatest" % "2.2.1" % "test"
)

scalariformSettings

// site
site.settings

site.includeScaladoc()

ghpages.settings

git.remoteRepo := "git@github.com:derrickburns/generalized-kmeans-clustering.git"

sparkPackageName := "derrickburns/generalized-kmeans-clustering"

sparkVersion := "1.2.0" // the Spark Version your package depends on.

sparkComponents += "mllib" // creates a dependency on spark-mllib.

testOptions in Test += Tests.Argument("-Dlog4j.configuration=log4j.properties")

bintraySettings

licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0"))

import com.typesafe.sbt.SbtGit._

versionWithGit

ScoverageSbtPlugin.ScoverageKeys.coverageExcludedPackages := ".*.clusterer.CachingKMeans;.*.clusterer.SingleKMeans;.*.clusterer.TrackingKMeans;.*.clusterer.DetailedTrackingStats"

ScoverageSbtPlugin.ScoverageKeys.coverageHighlighting := false

// Suggested Wartremover errors to improve inference rules and avoid partial methods which throw
wartremoverErrors ++= Seq(
  Wart.Any,
  Wart.Any2StringAdd,
  Wart.EitherProjectionPartial,
  Wart.OptionPartial,
  Wart.Product,
  Wart.Serializable,
  Wart.ListOps,
  Wart.Nothing
)
