scalaVersion := "3.3.8"
name := "scala-torch-artifact-consumer"
version := "0.0.0"
publish / skip := true
libraryDependencies += "com.github.kmizu" %% "scala-torch-models" % "0.1.0"
resolvers ++= sys.props.get("scalatorch.repository").toSeq.map(path =>
  "scala-torch-verification" at file(path).toURI.toString)
Compile / run / fork := true
Compile / run / javaOptions += "--enable-native-access=ALL-UNNAMED"
scalacOptions ++= Seq("-release", "25")
