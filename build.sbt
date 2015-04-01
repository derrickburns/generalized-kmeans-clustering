import sbt.Keys._
import com.typesafe.sbt.SbtGit._

lazy val root = (project in file(".")).
  settings(
    name := "massivedatascience-clusterer",
    organization := "com.massivedatascience",
    organizationName := "Massive Data Science, LLC",
    version := "1.0",
    licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0")),
    scalaVersion := "2.11.6",
    crossScalaVersions := Seq("2.10.4", "2.11.6"),
    scalacOptions ++= Seq("-unchecked", "-feature"),
    scalacOptions in (Compile, compile) ++= Seq("-Xlint", "-Xfatal-warnings", "-deprecation"),
    javacOptions ++= Seq( "-source", "1.7", "-target", "1.7"),
    publishMavenStyle := true,
    publishArtifact in Test := false,
    pomIncludeRepository := { _ => false },
    libraryDependencies ++= Seq(
      "joda-time" % "joda-time" % "2.2",
      "org.joda" % "joda-convert" % "1.6",
      "org.scalatest" %% "scalatest" % "2.2.1" % "test"
    ),
    git.remoteRepo := "git@github.com:derrickburns/generalized-kmeans-clustering.git",
    sparkPackageName := "derrickburns/generalized-kmeans-clustering",
    sparkVersion := "1.3.0",
    sparkComponents += "mllib",
    testOptions in Test += Tests.Argument("-Dlog4j.configuration=log4j.properties"),
    ScoverageSbtPlugin.ScoverageKeys.coverageHighlighting := highlight(scalaVersion.value),
    ScoverageSbtPlugin.ScoverageKeys.coverageExcludedPackages := ".*.clusterer.CachingKMeans;" +
      ".*.clusterer.SingleKMeans;" +
      ".*.clusterer.TrackingKMeans;" +
      ".*.clusterer.DetailedTrackingStats;" +
      ".*.clusterer.AssignmentSelector;" +
      ".*.clusterer.MultiKMeans",
    wartremoverErrors in (Compile, compile) ++= Seq(
      Wart.Any,
      Wart.Any2StringAdd,
      Wart.EitherProjectionPartial,
      Wart.OptionPartial,
      Wart.Product,
      Wart.Serializable,
      Wart.ListOps,
      Wart.Nothing
    )
  ).
  settings(scalariformSettings: _*).
  settings(site.settings: _*).
  settings(ghpages.settings: _*).
  settings(bintraySettings: _*).
  settings(versionWithGit: _*).
  settings(site.includeScaladoc(): _*)

def highlight(scalaVersion: String) = {
  CrossVersion.partialVersion(scalaVersion) match {
    case Some((2, scalaMajor)) if scalaMajor < 11  => false
    case _ => true
  }
}
