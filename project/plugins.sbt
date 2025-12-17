// addSbtPlugin("com.github.mwz" % "sbt-dependency-updates" % "0.5.5")

// For OSGI bundles
// addSbtPlugin("com.typesafe.sbt" % "sbt-osgi" % "0.7.0")

// For making releases
// addSbtPlugin("com.github.gseitz" % "sbt-release" % "0.8.5")

// For signing releases
// addSbtPlugin("com.typesafe.sbt" % "sbt-pgp" % "0.8.3")

// For creating the github site
// addSbtPlugin("com.typesafe.sbt" % "sbt-site" % "0.8.1")

// addSbtPlugin("com.typesafe.sbt" % "sbt-ghpages" % "0.5.3")

// addSbtPlugin("com.typesafe.sbt" % "sbt-scalariform" % "1.2.0")

// resolvers += "Spark Package Main Repo" at "https://dl.bintray.com/spark-packages/maven"

// addSbtPlugin("org.spark-packages" % "sbt-spark-package" % "0.1")

// addSbtPlugin("me.lessis" % "bintray-sbt" % "0.2.0")

// resolvers += Classpaths.sbtPluginReleases

// addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.0.4")

// addSbtPlugin("org.scoverage" % "sbt-coveralls" % "1.0.0.BETA1")

// addSbtPlugin("org.brianmckenna" % "sbt-wartremover" % "0.11")

// addSbtPlugin("org.scalastyle"  %% "scalastyle-sbt-plugin" % "0.6.0")

// Modern SBT plugins for code quality and dependency management
addSbtPlugin("org.scalastyle" %% "scalastyle-sbt-plugin" % "1.0.0")
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "2.2.2")

// Resolve scala-xml version conflict between scoverage and scalariform
ThisBuild / libraryDependencySchemes += "org.scala-lang.modules" %% "scala-xml" % VersionScheme.Always

// Publishing plugins
addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % "3.9.21")
addSbtPlugin("com.github.sbt" % "sbt-pgp" % "2.2.1")

// Documentation
addSbtPlugin("com.github.sbt" % "sbt-unidoc" % "0.5.0")

addSbtPlugin("org.scalameta" % "sbt-scalafmt" % "2.5.2")

// SBOM generation (Software Bill of Materials)
// Using dependency-check for vulnerability scanning and SBOM generation
addSbtPlugin("net.vonbuchholtz" % "sbt-dependency-check" % "5.1.0")
