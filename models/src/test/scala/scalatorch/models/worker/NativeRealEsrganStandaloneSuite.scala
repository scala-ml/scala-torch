package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import NativeJson.*

/** Real-weights check of the standalone R-ESRGAN upscale; set SDFX_TEST_ESRGAN (weights), SDFX_TEST_IMAGE and SDFX_TEST_OUT. */
class NativeRealEsrganStandaloneSuite extends munit.FunSuite:
  test("standalone upscale with real RealESRGAN_x4plus weights") {
    val weights=sys.env.get("SDFX_TEST_ESRGAN")
    val image=sys.env.get("SDFX_TEST_IMAGE")
    val out=sys.env.get("SDFX_TEST_OUT")
    assume(weights.isDefined && image.isDefined && out.isDefined,"real-weights gate disabled")
    val root=Files.createTempDirectory("native-real-esrgan")
    val scheduler=root.resolve("scheduler.json"); Files.writeString(scheduler,"{}")
    val output=Path.of(out.get); Files.createDirectories(output)
    val backend=new NativeDiffusionBackend(root,scheduler,unetExecution=scalatorch.models.UNetExecution.fromEnvironment(Map("SDFX_SCALATORCH_UNET_DEVICE"->"cuda","SDFX_SCALATORCH_UNET_MAX_BYTES"->"17179869184")))
    val params=Map[String,NativeJson]("image_path"->Str(image.get),"output_dir"->Str(output.toString),
      "upscaler"->Str("R-ESRGAN 4x+"),"scale"->Number(BigDecimal("2")),
      "upscaler_paths"->Obj(Map("R-ESRGAN 4x+"->Str(weights.get))))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val result=backend.execute(NativeRequest("up","upscale",params,None),context)
    val source=ImageIO.read(Path.of(image.get).toFile)
    val upscaled=ImageIO.read(Path.of(result("path").asInstanceOf[Str].value).toFile)
    assertEquals((upscaled.getWidth,upscaled.getHeight),(source.getWidth*2,source.getHeight*2))
    println(s"[real-esrgan] ${result("path").asInstanceOf[Str].value}")
  }
