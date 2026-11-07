package scalatorch.models.worker

import java.nio.file.{Files,Path}

class NativeTinyWeightsSuite extends munit.FunSuite:
  test("official pinned tiny weights download and load") {
    assume(sys.env.get("SDFX_TEST_TINY_DOWNLOAD").contains("1"),"network download test is opt-in")
    val cache=Files.createTempDirectory("tiny-official-download")
    for xl<-Vector(false,true) do
      val dir=NativeTinyWeights.resolve(Map.empty,cache,true,xl,()=>())
      scalatorch.models.TinyAutoencoderEncoder.load(dir)
      scalatorch.models.TinyAutoencoderDecoder.load(dir)
  }
  test("tiny weights honor configured directory, local cache and explicit download permission") {
    val cache=Files.createTempDirectory("tiny-weights-cache")
    var downloads=0
    def download(target: Path,xl: Boolean,cancelled: () => Unit): Unit =
      downloads+=1
      Files.createDirectories(target.getParent)
      Files.write(target,Array[Byte](1,2,3))
    intercept[NativeBackendError](NativeTinyWeights.resolve(Map.empty,cache,false,false,()=>(),download))
    assertEquals(downloads,0)
    val path=NativeTinyWeights.resolve(Map.empty,cache,true,false,()=>(),download)
    assertEquals(downloads,1)
    assertEquals(NativeTinyWeights.resolve(Map.empty,cache,false,false,()=>(),download),path)
    assertEquals(downloads,1)
    val explicit=Files.createTempDirectory("tiny-weights-explicit")
    Files.write(explicit.resolve("diffusion_pytorch_model.safetensors"),Array[Byte](4))
    assertEquals(NativeTinyWeights.resolve(Map("SDFX_SCALATORCH_TAESD_DIR"->explicit.toString),cache,true,false,()=>(),download),explicit)
    assertEquals(downloads,1)
  }
