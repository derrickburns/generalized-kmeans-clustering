import sbtrelease.ReleasePlugin._
import ReleaseKeys._
import bintray.Keys._

name := "massivedatascience-clusterer"

bintrayPublishSettings

bintrayOrganization in bintray := Some("derrickburns")


// publishing
publishMavenStyle := true

publishArtifact in Test := false

pomIncludeRepository := { _ => false }

pomExtra := {
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
    <url>http://github.com/derrickburns/generalized-kmeans-clustering</url>
}

// release
releaseSettings

// bump bugfix on release
nextVersion := { ver => sbtrelease.Version(ver).map(_.bumpBugfix.asSnapshot.string).getOrElse(sbtrelease.versionFormatError) }

// use maven style tag name
tagName <<= (name, version in ThisBuild) map { (n,v) => n + "-" + v }

// sign artifacts

publishArtifactsAction := PgpKeys.publishSigned.value


