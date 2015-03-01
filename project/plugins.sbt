
// For OSGI bundles
addSbtPlugin("com.typesafe.sbt" % "sbt-osgi" % "0.7.0")

// For making releases
addSbtPlugin("com.github.gseitz" % "sbt-release" % "0.8.5")

// For signing releases
addSbtPlugin("com.typesafe.sbt" % "sbt-pgp" % "0.8.3")

// For creating the github site
addSbtPlugin("com.typesafe.sbt" % "sbt-site" % "0.8.1")

addSbtPlugin("com.typesafe.sbt" % "sbt-ghpages" % "0.5.3")

addSbtPlugin("com.typesafe.sbt" % "sbt-scalariform" % "1.2.0")

resolvers += "jgit-repo" at "http://download.eclipse.org/jgit/maven"

addSbtPlugin("com.typesafe.sbt" % "sbt-git" % "0.6.4")