package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import java.util.concurrent.CancellationException
import javax.imageio.ImageIO
import scalatorch.models.*
import NativeJson.*

class NativeVaeToolsSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(5,"minutes")
  test("VAE Tools does not apply JPEG EXIF orientation") {
    import scalatorch.*
    val dir=Files.createTempDirectory("vae-jpeg-orientation")
    val input=dir.resolve("image.data")
    val source=getClass.getResourceAsStream("/image-input/orientation-6.jpg")
    try Files.copy(source,input) finally source.close()
    var dimensions=(0,0)
    val codec=new NativeVaeToolsCodec:
      def encode(image: BufferedImage,cancelled: () => Unit): Tensor[Float] =
        dimensions=(image.getWidth,image.getHeight)
        Tensor.fromFloats(Array.fill(4)(0f),Shape.from(Seq(1,4,1,1)))
      def decode(latent: Tensor[Float],cancelled: () => Unit)=new BufferedImage(1,1,BufferedImage.TYPE_INT_RGB)
    val params=Map[String,NativeJson]("input_paths"->Arr(Vector(Str(input.toString))),"output_dir"->Str(dir.toString),
      "operation"->Str("encode+decode"),"approximation"->Str("Full"))
    val result=NativeVaeTools.execute(params,codec,new NativeWorkContext(()=>false,()=>true,(_,_)=>()))
    assertEquals(result("results").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("status"),Str("complete"))
    assertEquals(dimensions,(3,2))
  }
  test("Full recovery passes decoder input bits directly to the VAE") {
    import scalatorch.*
    import scalatorch.io.{SafeTensorWrite,SafeTensorsWriter}
    val dir=Files.createTempDirectory("native-vae-recovery")
    val input=dir.resolve("recovery.safetensors")
    val values=Array(0.12345679f,-0f,3.5f,-1.2345f)
    val tensor=Tensor.fromFloats(values,Shape.from(Seq(1,4,1,1)))
    SafeTensorsWriter.write(input,Map("tensor"->SafeTensorWrite.F32(tensor)),
      Map("sdfx_vae_tools"->"{\"coordinate\":\"vae_decoder_input\"}"))
    var bits=Vector.empty[Int]
    val tool=new NativeVaeToolsCodec:
      def encode(image: BufferedImage,cancelled: () => Unit)=throw UnsupportedOperationException()
      def decode(latent: Tensor[Float],cancelled: () => Unit)=throw AssertionError("scaled decode used")
      override def decodeRaw(latent: Tensor[Float],cancelled: () => Unit): BufferedImage =
        bits=latent.toFloatArray.map(java.lang.Float.floatToRawIntBits).toVector
        new BufferedImage(1,1,BufferedImage.TYPE_INT_RGB)
    val params=Map[String,NativeJson]("input_paths"->Arr(Vector(Str(input.toString))),"operation"->Str("decode"),
      "approximation"->Str("Full"),"output_dir"->Str(dir.toString))
    val report=NativeVaeTools.execute(params,tool,new NativeWorkContext(()=>false,()=>true,(_,_)=>()))
    val item=report("results").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assertEquals(item("status"),Str("complete"))
    assertEquals(bits,values.map(java.lang.Float.floatToRawIntBits).toVector)
  }
  test("VAE Tools is routed through the backend and requires a loaded checkpoint") {
    val dir=Files.createTempDirectory("native-vae-route")
    val config=dir.resolve("scheduler.json")
    Files.writeString(config,"{}")
    val backend=new NativeDiffusionBackend(dir,config)
    assert(backend.supported.contains("vae_tools"))
    val error=intercept[NativeBackendError] {
      backend.execute(NativeRequest("vae","vae_tools",Map.empty,None),
        new NativeWorkContext(()=>false,()=>true,(_,_)=>()))
    }
    assertEquals(error.code,"no_model")
  }
  test("TAESD VAE Tools publishes a report and preserves cancellation") {
    val root=Vector(Path.of(".superpowers/models/taesd"),Path.of("../.superpowers/models/taesd"))
      .find(p=>Files.isRegularFile(p.resolve("diffusion_pytorch_model.safetensors")))
    assume(root.nonEmpty,"trained TAESD weights required")
    val tool=new NativeTinyVaeTools(TinyAutoencoderEncoder.load(root.get),TinyAutoencoderDecoder.load(root.get),VaeExecution.Cuda())
    val dir=Files.createTempDirectory("native-vae-tools")
    val input=dir.resolve("input.png")
    ImageIO.write(new BufferedImage(16,16,BufferedImage.TYPE_INT_RGB),"png",input.toFile)
    val params=Map[String,NativeJson]("input_paths"->Arr(Vector(Str(input.toString))),"operation"->Str("encode+decode"),
      "representation"->Str("L"),"approximation"->Str("TAESD"),"output_dir"->Str(dir.toString))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val report=NativeVaeTools.execute(params,tool,context)
    val item=report("results").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assertEquals(item("status"),Str("complete"))
    assert(Files.isRegularFile(Path.of(item("path").asInstanceOf[Str].value)))
    assert(Files.isRegularFile(Path.of(report("report_path").asInstanceOf[Str].value)))
    val provenance=Map[String,NativeJson]("model"->Obj(Map("name"->Str("fixture"))),
      "family"->Str("sd15"),"scaling_factor"->Number(BigDecimal("0.18215")))
    val encoded=NativeVaeTools.execute(params.updated("operation",Str("encode")),tool,context,provenance)
    val encodedItem=encoded("results").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assertEquals(encodedItem("status"),Str("complete"))
    val encodedPng=Path.of(encodedItem("path").asInstanceOf[Str].value)
    val pngMetadata=NativeJson.parse(NativePreprocess.sourceParameters(encodedPng,"VAE Tools").get).asInstanceOf[Obj].fields
    assertEquals(pngMetadata("coordinate"),Str("diffusion_scaled"))
    assertEquals(pngMetadata("operation"),Str("encode"))
    assertEquals(pngMetadata("model"),provenance("model"))
    val previewMetadata=NativeJson.parse(NativePreprocess.sourceParameters(
      Path.of(encodedItem("preview_path").asInstanceOf[Str].value),"VAE Tools").get).asInstanceOf[Obj].fields
    assertEquals(previewMetadata("display_only"),Bool(true))
    val raw=encodedItem("latent_path").asInstanceOf[Str].value
    scala.util.Using.resource(scalatorch.io.SafeTensors.open(Path.of(raw))) { archive =>
      val metadata=NativeJson.parse(archive.metadata("sdfx_vae_tools")).asInstanceOf[Obj].fields
      assertEquals(metadata("coordinate"),Str("diffusion_scaled"))
      assertEquals(metadata("model"),provenance("model"))
      assertEquals(metadata("scaling_factor"),provenance("scaling_factor"))
    }
    val decoded=NativeVaeTools.execute(params.updated("operation",Str("decode"))
      .updated("input_paths",Arr(Vector(Str(raw)))),tool,context)
    val decodedItem=decoded("results").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assertEquals(decodedItem("status"),Str("complete"))
    val decodedImage=ImageIO.read(Path.of(decodedItem("path").asInstanceOf[Str].value).toFile)
    assertEquals((decodedImage.getWidth,decodedImage.getHeight),(16,16))
    val output=dir.resolve("vae-tools")
    val before=Files.list(output);val count=try before.count() finally before.close()
    val denied=new NativeWorkContext(()=>false,()=>false,(_,_)=>())
    intercept[CancellationException](NativeVaeTools.execute(params,tool,denied))
    val after=Files.list(output);try assertEquals(after.count(),count) finally after.close()
    val invalid=params.updated("approximation",Str("not-a-mode"))
    intercept[NativeBackendError](NativeVaeTools.execute(invalid,tool,context))
  }
