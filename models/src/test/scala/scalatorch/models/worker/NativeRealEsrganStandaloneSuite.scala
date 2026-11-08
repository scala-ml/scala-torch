package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import NativeJson.*

/** Real-weights check of the standalone R-ESRGAN upscale; set SDFX_TEST_ESRGAN (weights), SDFX_TEST_IMAGE and SDFX_TEST_OUT.
  * The model runs on the device named by SDFX_SCALATORCH_UNET_DEVICE (cpu when unset: far too slow for large images). */
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

  test("model-free image_tool tools on a real image (gated on SDFX_TEST_IMAGE / SDFX_TEST_OUT)") {
    val image=sys.env.get("SDFX_TEST_IMAGE");val out=sys.env.get("SDFX_TEST_OUT")
    assume(image.isDefined && out.isDefined,"real-image gate disabled")
    val root=Files.createTempDirectory("native-tools-real")
    val scheduler=root.resolve("scheduler.json"); Files.writeString(scheduler,"{}")
    val output=Path.of(out.get).resolve("tools"); Files.createDirectories(output)
    val backend=new NativeDiffusionBackend(root,scheduler)
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    def run(tool: String,extra: (String,NativeJson)*): Unit =
      val params=Map[String,NativeJson]("tool"->Str(tool),"image_path"->Str(image.get),"output_dir"->Str(output.toString))++extra
      val result=backend.execute(NativeRequest(tool,"image_tool",params,None),context)
      println(s"[tool:$tool] $result")
    run("cas_upscale","scale"->Number(BigDecimal("2")),"sharpness"->Number(BigDecimal("0.8")))
    run("color_enhance","strength"->Number(BigDecimal("1")))
    run("pixel_art","pixel_size"->Number(BigDecimal("8")),"colors"->Number(BigDecimal("16")))
    // needs the Depth Anything weights (SDFX_SCALATORCH_ANNOTATOR_DIR)
    if sys.env.contains("SDFX_SCALATORCH_ANNOTATOR_DIR") then run("depth_mask","near"->Number(BigDecimal("0.6")),"far"->Number(BigDecimal("1.0")))
  }
