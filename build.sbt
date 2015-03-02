// Basic facts
name := "massivedatascience-clusterer"

organization := "com.massivedatascience"

scalaVersion := "2.10.4"

crossScalaVersions := Seq("2.10.4", "2.11.4")

scalacOptions ++= Seq("-deprecation", "-unchecked", "-feature")

scalacOptions in (Compile, compile) += "-Xfatal-warnings"

javacOptions ++= Seq(
  "-source", "1.7",
  "-target", "1.7",
  "-bootclasspath", (new File(System.getenv("JAVA_HOME")) / "jre" / "lib" / "rt.jar").toString
)

// Try to future-proof scala jvm targets
scalacOptions += "-target:jvm-1.7"

libraryDependencies ++= Seq(
  "joda-time" % "joda-time" % "2.2",
  "org.joda" % "joda-convert" % "1.6",
  "org.apache.spark" %% "spark-core" % "1.2.0",
  "org.apache.spark" %% "spark-mllib" % "1.2.0",
  // test dependencies
  "org.scalatest" %% "scalatest" % "2.2.1" % "test"
)

scalariformSettings

// site
site.settings

site.includeScaladoc()

ghpages.settings

git.remoteRepo := "git@github.com:derrickburns/generalized-kmeans-clustering.git"
