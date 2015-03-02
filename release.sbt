import sbtrelease.ReleasePlugin._
import ReleaseKeys._
import SonatypeKeys._
import sbtrelease.ReleaseStateTransformations._
import sbtrelease.Utilities._
import com.typesafe.sbt.pgp.PgpKeys._

// Your project orgnization (package name)
organization := "com.massivedatascience"

// Your profile name of the sonatype account. The default is the same with the organization
profileName := "derrickburns"

// Import default settings. This changes `publishTo` settings to use the Sonatype repository
// and add several commands for publishing.
sonatypeSettings

// publishing
publishMavenStyle := true

publishArtifact in Test := false

pomIncludeRepository := { _ => false }

pomExtra := {
    <licenses>
      <license>
        <name>Apache License, Version 2.0</name>
        <url>http://www.apache.org/licenses/LICENSE-2.0.txt</url>
        <distribution>repo</distribution>
      </license>
    </licenses>
    <scm>
      <connection>scm:git:git@github.com:derrickburns/generalized-kmeans-clustering.git</connection>
      <url>http://github.com/derrickburns/generalized-kmeans-clustering.git</url>
      <developerConnection>scm:git:git@github.com:derrickburns/generalized-kmeans-clustering.git
      </developerConnection>
    </scm>
    <developers>
      <developer>
        <id>derrick</id>
        <name>Derrick Burns</name>
        <email>derrick.burns@rincaro.com</email>
        <organization>Massive Data Science, LLC</organization>
        <roles>
          <role>founder</role>
          <role>architect</role>
          <role>developer</role>
        </roles>
        <timezone>-8</timezone>
      </developer>
    </developers>
}

// release
releaseSettings

// bump bugfix on release
nextVersion := { ver => sbtrelease.Version(ver).map(_.bumpBugfix.asSnapshot.string).getOrElse(sbtrelease.versionFormatError) }

// use maven style tag name
tagName <<= (name, version in ThisBuild) map { (n,v) => n + "-" + v }

// sign artifacts

publishArtifactsAction := PgpKeys.publishSigned.value
