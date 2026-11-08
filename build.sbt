ThisBuild / scalaVersion := "3.3.8"
ThisBuild / organization := "com.github.kmizu"
ThisBuild / version := "0.1.2"
ThisBuild / organizationName := "Kota Mizushima"
ThisBuild / homepage := Some(uri("https://github.com/scala-ml/scala-torch"))
ThisBuild / licenses := Seq("BSD-3-Clause" -> uri("https://opensource.org/licenses/BSD-3-Clause"))
ThisBuild / scmInfo := Some(
  ScmInfo(
    uri("https://github.com/scala-ml/scala-torch"),
    "scm:git:https://github.com/scala-ml/scala-torch.git",
    Some("scm:git:git@github.com:scala-ml/scala-torch.git")
  )
)
ThisBuild / versionScheme := Some("early-semver")
ThisBuild / developers := List(Developer("kmizu", "Kota Mizushima", "kmizu.main@gmail.com", uri("https://github.com/kmizu")))
Global / PgpKeys.useGpg := false
Global / PgpKeys.pgpPassphrase := {
  val f = file(sys.props("user.home")) / ".sbt" / "gpg" / "passphrase.txt"
  if (f.isFile) Some(IO.read(f).trim.toCharArray)
  else sys.env.get("PGP_PASSPHRASE").map(_.toCharArray)
}
ThisBuild / credentials ++= {
  val f = file(sys.props("user.home")) / ".sbt" / "sonatype_central.properties"
  if (f.isFile) {
    val props = new java.util.Properties()
    val in = new java.io.FileInputStream(f)
    try props.load(in) finally in.close()
    val u = Option(props.getProperty("user")).getOrElse("").trim
    val p = Option(props.getProperty("password")).getOrElse("").trim
    if (u.nonEmpty && p.nonEmpty) Seq(
      Credentials("Sonatype Central Repository Manager", "central.sonatype.com", u, p),
      Credentials("Sonatype Nexus Repository Manager", "central.sonatype.com", u, p),
      Credentials("", "central.sonatype.com", u, p)
    ) else Seq.empty
  } else Seq.empty
}

lazy val workerDistribution = taskKey[Unit]("Build a relocatable classpath distribution for the native diffusion worker")

lazy val common = Seq(
  scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-release", "25", "-Yfuture-lazy-vals"),
  libraryDependencies += "org.scalameta" %% "munit" % "1.1.1" % Test,
  Test / fork := true,
  Test / parallelExecution := false,
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
  workerDistribution := {
    val output = target.value / "worker-distribution"
    val lib = output / "lib"
    val converter = fileConverter.value
    val entries = (Compile / fullClasspath).value.map(attributed => converter.toPath(attributed.data).toFile)
    IO.delete(output)
    IO.createDirectory(lib)
    val relativeClasspath = entries.zipWithIndex.map { (source, index) =>
      val isDirectory = source.isDirectory
      val suffix = if isDirectory then ".jar" else ""
      val filename = f"$index%03d-${source.getName}$suffix"
      val destination = lib / filename
      if isDirectory then
        val files = (source ** "*").get().filter(_.isFile).sortBy(_.getAbsolutePath)
        val jar = new java.util.jar.JarOutputStream(new java.io.FileOutputStream(destination))
        try files.foreach { input =>
          val name = source.toPath.relativize(input.toPath).toString.replace('\\', '/')
          jar.putNextEntry(new java.util.jar.JarEntry(name))
          java.nio.file.Files.copy(input.toPath, jar)
          jar.closeEntry()
        }
        finally jar.close()
      else IO.copyFile(source, destination)
      s"lib/$filename"
    }
    IO.write(output / "worker-classpath.txt", relativeClasspath.mkString(java.io.File.pathSeparator) + "\n", java.nio.charset.StandardCharsets.UTF_8)
    IO.write(output / "README.txt", "Native diffusion worker runtime distribution. Keep this directory intact and select worker-classpath.txt in NUI. Use a Java 25 runtime. Select the Diffusers model folder; NUI defaults to its scheduler/scheduler_config.json unless another scheduler JSON is selected.\n", java.nio.charset.StandardCharsets.UTF_8)
    streams.value.log.info(s"Built native worker distribution at ${output.getAbsolutePath}")
  },
  Test / javaOptions += {
    val converter = fileConverter.value
    "-Dscalatorch.worker.classpath=" + (Compile / fullClasspath).value.map(a => converter.toPath(a.data)).mkString(java.io.File.pathSeparator)
  },
  Compile / run / fork := true,
  Compile / run / javaOptions += "--enable-native-access=ALL-UNNAMED",
  Test / javaOptions ++= Seq("--enable-native-access=ALL-UNNAMED", "-Xmx24g")
)
lazy val root = (project in file(".")).aggregate(core, cpu, ffm, autograd, nn, tensorio, models).settings(
  name := "scala-torch",
  publish / skip := true
)
