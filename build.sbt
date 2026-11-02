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
lazy val nn = project.dependsOn(autograd).settings(common)
lazy val root = (project in file(".")).aggregate(core, cpu, ffm, autograd, nn).settings(
  name := "scala-torch",
  publish / skip := true
)
