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
