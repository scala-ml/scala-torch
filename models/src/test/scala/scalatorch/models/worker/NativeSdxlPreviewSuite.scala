package scalatorch.models.worker

import java.nio.file.{Files,Path}
import java.util.Base64
import javax.imageio.ImageIO
import scalatorch.models.{UNetExecution,VaeExecution}
import NativeJson.*

class NativeSdxlPreviewSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(5,"minutes")
  test("SDXL CUDA generation emits a decodable intermediate preview") {
    val config=Vector(Path.of(".superpowers/sdxl-preview-test.properties"),
      Path.of("../.superpowers/sdxl-preview-test.properties")).find(Files.isRegularFile(_))
    assume(config.nonEmpty,"tiny SDXL CUDA fixture required")
    val properties=new java.util.Properties()
    scala.util.Using.resource(Files.newInputStream(config.get))(properties.load)
    val root=Path.of(properties.getProperty("model"))
    val output=Files.createTempDirectory("native-sdxl-preview")
    val backend=new NativeDiffusionBackend(root,root.resolve("scheduler/scheduler_config.json"),
      vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val previews=scala.collection.mutable.ArrayBuffer.empty[Array[Byte]]
    val context=new NativeWorkContext(()=>false,()=>true,(kind,fields)=>
      if kind=="preview" then previews+=Base64.getDecoder.decode(fields("jpeg_b64").asInstanceOf[Str].value))
    backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
    val params=Obj(Map("prompt"->Str("a red apple"),"width"->Number(64),"height"->Number(64),
      "steps"->Number(3),"seed"->Number(42),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform")))
    backend.execute(NativeRequest("generate","txt2img",Map("output_dir"->Str(output.toString),
      "preview_every"->Number(1),"preview_method"->Str("approx"),"params"->params),None),context)
    assertEquals(previews.size,2)
    previews.foreach { bytes =>
      val image=ImageIO.read(new java.io.ByteArrayInputStream(bytes))
      assert(image!=null && image.getWidth==256 && image.getHeight==256)
    }
    val warnings=scala.collection.mutable.ArrayBuffer.empty[String]
    val brokenPreview=new NativeWorkContext(()=>false,()=>true,(kind,fields)=>
      if kind=="preview" then throw new IllegalArgumentException("injected preview failure")
      else if kind=="log" then warnings+=fields("message").asInstanceOf[Str].value)
    val result=backend.execute(NativeRequest("broken-preview","txt2img",Map("output_dir"->Str(output.toString),
      "preview_every"->Number(1),"preview_method"->Str("approx"),"params"->params),None),brokenPreview)
    assertEquals(result("images").asInstanceOf[Arr].values.size,1)
    assert(warnings.exists(_.contains("Preview unavailable")))
  }
  test("SDXL TAESD uses real weights and missing weights fall back without failing generation") {
    val config=Vector(Path.of(".superpowers/sdxl-preview-test.properties"),
      Path.of("../.superpowers/sdxl-preview-test.properties")).find(Files.isRegularFile(_))
    assume(config.nonEmpty,"tiny SDXL CUDA fixture required")
    val properties=new java.util.Properties()
    scala.util.Using.resource(Files.newInputStream(config.get))(properties.load)
    val root=Path.of(properties.getProperty("model"))
    val tiny=root.getParent.resolve("taesdxl")
    assume(Files.isRegularFile(tiny.resolve("diffusion_pytorch_model.safetensors")),"trained TAESDXL required")
    def run(env: Map[String,String],method: String): (Vector[Array[Byte]],Vector[String]) =
      val backend=new NativeDiffusionBackend(root,root.resolve("scheduler/scheduler_config.json"),
        vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true),
        tinyPreviewEnvironment=env)
      val previews=scala.collection.mutable.ArrayBuffer.empty[Array[Byte]]
      val warnings=scala.collection.mutable.ArrayBuffer.empty[String]
      val context=new NativeWorkContext(()=>false,()=>true,(kind,fields)=>
        if kind=="preview" then previews+=Base64.getDecoder.decode(fields("jpeg_b64").asInstanceOf[Str].value)
        else if kind=="log" then warnings+=fields("message").asInstanceOf[Str].value)
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val params=Obj(Map("prompt"->Str("a red apple"),"width"->Number(64),"height"->Number(64),
        "steps"->Number(2),"seed"->Number(42),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform")))
      val output=Files.createTempDirectory("native-taesdxl-preview")
      backend.execute(NativeRequest("generate","txt2img",Map("output_dir"->Str(output.toString),
        "preview_every"->Number(1),"preview_method"->Str(method),"params"->params),None),context)
      (previews.toVector,warnings.toVector)
    val (neural,warnings)=run(Map("SDFX_SCALATORCH_TAESDXL_DIR"->tiny.toString),"taesd")
    assertEquals(neural.size,1)
    assert(!warnings.exists(_.contains("TAESD preview unavailable")))
    val image=ImageIO.read(new java.io.ByteArrayInputStream(neural.head))
    assert(image!=null && image.getWidth==256 && image.getHeight==256)
    val (fallback,missing)=run(Map.empty,"taesd")
    assertEquals(fallback.size,1)
    assert(missing.exists(_.contains("TAESD preview unavailable")))
    val (approx,_)=run(Map.empty,"approx")
    assertEquals(fallback.head.toVector,approx.head.toVector)
    assertNotEquals(neural.head.toVector,approx.head.toVector)
    val broken=Files.createTempDirectory("bad-taesd")
    Files.write(broken.resolve("diffusion_pytorch_model.safetensors"),Array[Byte](1,2,3))
    val (recovered,bad)=run(Map("SDFX_SCALATORCH_TAESDXL_DIR"->broken.toString),"taesd")
    assert(bad.exists(_.contains("TAESD preview unavailable")))
    assertEquals(recovered.head.toVector,approx.head.toVector)
  }
