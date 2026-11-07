package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.{UNetExecution,VaeExecution}
import NativeJson.*

class NativeSd15CkptSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(20,"minutes")
  test("official SD1 checkpoint generates on CUDA without Python") {
    val config=Vector(Path.of(".superpowers/sd15-ckpt-test.properties"),
      Path.of("../.superpowers/sd15-ckpt-test.properties")).find(Files.isRegularFile(_))
    assume(config.nonEmpty,"real SD1 checkpoint and CUDA required")
    val properties=new java.util.Properties()
    scala.util.Using.resource(Files.newInputStream(config.get))(properties.load)
    val checkpoint=Path.of(properties.getProperty("checkpoint"))
    val output=Path.of(properties.getProperty("output"))
    Files.createDirectories(output)
    val backend=new NativeDiffusionBackend(checkpoint,Path.of("-"),
      vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
    val params=Obj(Map("prompt"->Str("a photograph of a red apple on a wooden table"),
      "width"->Number(256),"height"->Number(256),"steps"->Number(8),"seed"->Number(42),
      "cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform")))
    val result=backend.execute(NativeRequest("generate","txt2img",Map("output_dir"->Str(output.toString),"params"->params),None),context)
    backend.execute(NativeRequest("unload","unload",Map.empty,None),context)
    System.gc()
    val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
    val image=ImageIO.read(path.toFile)
    assertEquals((image.getWidth,image.getHeight),(256,256))
  }
