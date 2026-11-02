ThisBuild / scalaVersion := "3.3.8"
ThisBuild / organization := "com.github.kmizu"
ThisBuild / version := "0.1.0"
ThisBuild / organizationName := "Kota Mizushima"
ThisBuild / versionScheme := Some("early-semver")
ThisBuild / developers := List(Developer("kmizu", "Kota Mizushima", "kmizu.main@gmail.com", uri("https://github.com/kmizu")))

lazy val common = Seq(
  scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-release", "25", "-Yfuture-lazy-vals"),
  libraryDependencies += "org.scalameta" %% "munit" % "1.1.1" % Test,
  Test / fork := true,
  Test / publishArtifact := false,
  publishMavenStyle := true,
  pomIncludeRepository := { _ => false },
  publishTo := localStaging.value
)

lazy val core = project.settings(common).settings(
  name := "scala-torch-core",
  description := "Checked CPU tensors, layouts and reference operations for scala-torch"
)
lazy val cpu = project.dependsOn(core).settings(common).settings(
  name := "scala-torch-cpu",
  description := "numscala-backed CPU matrix multiplication for scala-torch",
  libraryDependencies += "com.github.kmizu" %% "numscala" % "0.5.0"
)
lazy val ffm = project.dependsOn(core, cpu % "test->compile").settings(common).settings(
  name := "scala-torch-ffm",
  description := "Java 25 FFM CUDA driver bindings and resident operations for scala-torch",
  Test / javaOptions += "--enable-native-access=ALL-UNNAMED",
  Compile / run / fork := true,
  Compile / run / javaOptions += "--enable-native-access=ALL-UNNAMED"
)
lazy val autograd = project.dependsOn(cpu, ffm % "test->compile").settings(common).settings(
  name := "scala-torch-autograd",
  description := "First-order CPU automatic differentiation for scala-torch",
  libraryDependencies += "org.apache.commons" % "commons-numbers-gamma" % "1.3",
  Test / javaOptions += "--enable-native-access=ALL-UNNAMED"
)
lazy val tensorio = (project in file("io")).dependsOn(core).settings(common).settings(
  name := "scala-torch-io",
  description := "Validated single and sharded safetensors loading for scala-torch",
  libraryDependencies += "com.fasterxml.jackson.core" % "jackson-core" % "2.20.1"
)
lazy val nn = project.dependsOn(autograd, tensorio).settings(common).settings(
  name := "scala-torch-nn",
  description := "Derived neural network modules, optimizers and checkpoints for scala-torch",
  Test / unmanagedResourceDirectories += baseDirectory.value.getParentFile / "io" / "src" / "test" / "resources"
)
lazy val models = project.dependsOn(nn, ffm).settings(common).settings(
  name := "scala-torch-models",
  description := "Scala-native Llama inference and training, CLIP text conditioning and diffusion U-Net",
  Compile / run / fork := true,
  Compile / run / javaOptions += "--enable-native-access=ALL-UNNAMED",
  Test / javaOptions += "--enable-native-access=ALL-UNNAMED"
)
lazy val root = (project in file(".")).aggregate(core, cpu, ffm, autograd, nn, tensorio, models).settings(
  name := "scala-torch",
  publish / skip := true
)
