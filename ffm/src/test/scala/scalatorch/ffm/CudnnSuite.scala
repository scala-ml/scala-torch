package scalatorch.ffm

import java.nio.file.Files

class CudnnSuite extends munit.FunSuite:
  test("only complete cuDNN installations are candidates, in the documented order") {
    assume(System.getProperty("os.name").toLowerCase.startsWith("windows"),"Windows library names")
    val root=Files.createTempDirectory("cudnn-candidates")
    def dir(name: String,files: String*) =
      val d=Files.createDirectories(root.resolve(name));files.foreach(f => Files.write(d.resolve(f),Array[Byte](0)));d
    val complete9=dir("explicit","cudnn64_9.dll","cudnn_ops64_9.dll","cudnn_cnn64_9.dll","cudnn_graph64_9.dll")
    val frontOnly=dir("front-only","cudnn64_9.dll")
    val missingPart=dir("missing-part","cudnn64_9.dll","cudnn_ops64_9.dll","cudnn_graph64_9.dll")
    val complete8=dir("onpath","cudnn64_8.dll","cudnn_ops_infer64_8.dll","cudnn_cnn_infer64_8.dll")
    val toolkit=dir("toolkit/bin","cudnn64_9.dll","cudnn_ops64_9.dll","cudnn_cnn64_9.dll","cudnn_graph64_9.dll")
    val installed=dir("pf/NVIDIA/CUDNN/v9.5/bin/12.6","cudnn64_9.dll","cudnn_ops64_9.dll","cudnn_cnn64_9.dll","cudnn_graph64_9.dll")
    val sep=java.io.File.pathSeparator
    val env=Map("SCALATORCH_CUDNN_PATH"->complete9.toString,"CUDA_PATH"->root.resolve("toolkit").toString,
      "ProgramFiles"->root.resolve("pf").toString,"Path"->s"$frontOnly$sep$missingPart$sep$complete8$sep${root.resolve("absent")}")
    assertEquals(Cudnn.candidates(env),Seq(complete9.resolve("cudnn64_9.dll"),toolkit.resolve("cudnn64_9.dll"),
      installed.resolve("cudnn64_9.dll"),complete8.resolve("cudnn64_8.dll")).map(_.toAbsolutePath.toString))
    assertEquals(Cudnn.candidates(Map("Path"->s"$frontOnly$sep$missingPart")),Seq.empty)
    assertEquals(Cudnn.candidates(Map.empty),Seq.empty)
  }

  test("cuBLAS is looked for on the search path, in the cuDNN folder, in CUDA_PATH and in every Toolkit under Program Files") {
    assume(System.getProperty("os.name").toLowerCase.startsWith("windows"),"Windows library names")
    val root=Files.createTempDirectory("cublas-candidates")
    def dir(name: String,files: String*) =
      val d=Files.createDirectories(root.resolve(name));files.foreach(f => Files.write(d.resolve(f),Array[Byte](0)));d
    val torchLib=dir("torch-lib","cublas64_12.dll","cublasLt64_12.dll")
    val toolkit=dir("toolkit/bin","cublas64_12.dll")
    val older=dir("pf/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin","cublas64_11.dll")
    val newer=dir("pf/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin","cublas64_12.dll")
    dir("pf/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin")
    val bare=Seq("cublas64_13.dll","cublas64_12.dll","cublas64_11.dll")
    val env=Map("SCALATORCH_CUDNN_PATH"->torchLib.toString,"CUDA_PATH"->root.resolve("toolkit").toString,"ProgramFiles"->root.resolve("pf").toString)
    // bare names first (the search path), then absolute paths: explicit folder, CUDA_PATH, newest Toolkit first
    assertEquals(Cublas.candidates(env),bare++Seq(torchLib.resolve("cublas64_12.dll"),toolkit.resolve("cublas64_12.dll"),
      newer.resolve("cublas64_12.dll"),older.resolve("cublas64_11.dll")).map(_.toAbsolutePath.toString))
    assertEquals(Cublas.candidates(Map.empty),bare)
  }
