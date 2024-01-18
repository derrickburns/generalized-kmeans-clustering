// Your profile name of the sonatype account. The default is the same with the organization value
sonatypeProfileName := "com.massivedatascience"

// To sync with Maven central, you need to supply the following information:
publishMavenStyle := true

// Open-source license of your choice
licenses := Seq("APL2" -> url("http://www.apache.org/licenses/LICENSE-2.0.txt"))

// Where is the source code hosted: GitHub or GitLab?
import xerial.sbt.Sonatype._
sonatypeProjectHosting := Some(GitHubHosting("derrickburns", "generalized-kmeans-clustering", "derrickrburns@gmail.com"))

developers := List(
  Developer(id="derrickrburns", name="Derrick Burns", email="derrickrburns@gmail.com", url=url("https://none"))
)
