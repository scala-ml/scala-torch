scalaVersion := "3.3.8"
organization := "com.github.kmizu"
version := "0.1.0-SNAPSHOT"

lazy val common = Seq(
  scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-release", "25", "-Yfuture-lazy-vals"),
  libraryDependencies += "org.scalameta" %% "munit" % "1.1.1" % Test,
  Test / fork := true
)

lazy val core = project.settings(common)
lazy val cpu = project.dependsOn(core).settings(common).settings(
  libraryDependencies += "com.github.kmizu" %% "numscala" % "0.5.0"
)
lazy val ffm = project.dependsOn(core).settings(common).settings(
  Test / javaOptions += "--enable-native-access=ALL-UNNAMED",
  Compile / run / fork := true,
  Compile / run / javaOptions += "--enable-native-access=ALL-UNNAMED"
)
lazy val autograd = project.dependsOn(cpu).settings(common)
lazy val tensorio = (project in file("io")).dependsOn(core).settings(common).settings(
  libraryDependencies += "com.fasterxml.jackson.core" % "jackson-core" % "2.20.1"
)
lazy val nn = project.dependsOn(autograd, tensorio).settings(common).settings(
  Test / unmanagedResourceDirectories += baseDirectory.value.getParentFile / "io" / "src" / "test" / "resources"
)
lazy val models = project.dependsOn(nn).settings(common).settings(
  Compile / run / fork := true
)
lazy val root = (project in file(".")).aggregate(core, cpu, ffm, autograd, nn, tensorio, models).settings(
  name := "scala-torch",
  publish / skip := true
)
