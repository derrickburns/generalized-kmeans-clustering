import sbtrelease.ReleasePlugin._
import ReleaseKeys._
import sbtrelease.ReleaseStateTransformations._
import sbtrelease.Utilities._
import com.typesafe.sbt.pgp.PgpKeys._


// publishing
publishMavenStyle := true

publishTo <<= version { v =>
  val nexus = "https://oss.sonatype.org/"
  if (v.trim.endsWith("SNAPSHOT"))
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases" at nexus + "service/local/staging/deploy/maven2")
}

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
