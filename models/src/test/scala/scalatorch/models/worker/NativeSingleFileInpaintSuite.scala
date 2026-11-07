package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.{UNetExecution,VaeExecution}
import NativeJson.*

class NativeSingleFileInpaintSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(20,"minutes")
  test("trained nine-channel single-file model inpaints on CUDA") {
    val properties = new java.util.Properties()
    val config = Vector(Path.of(".superpowers/sd15-single-inpaint-test.properties"),
      Path.of("../.superpowers/sd15-single-inpaint-test.properties")).find(Files.isRegularFile(_))
    config.foreach(path => scala.util.Using.resource(Files.newInputStream(path))(properties.load))
    def setting(name: String): Option[String] = sys.env.get(name).orElse(Option(properties.getProperty(name)))
    val checkpoint = setting("SCALATORCH_SD15_INPAINT_SINGLE")
    val source = setting("SCALATORCH_SD15_INPAINT_SOURCE")
    val output = setting("SCALATORCH_SD15_INPAINT_OUTPUT")
    assume(checkpoint.nonEmpty && source.nonEmpty && output.nonEmpty &&
      setting("SCALATORCH_CUDA_TESTS").contains("1"),
      "trained checkpoint, source, output and CUDA required")
    val image = ImageIO.read(Path.of(source.get).toFile)
    require(image != null && image.getWidth == 256 && image.getHeight == 256,"source must be a 256px image")
    val outputDir = Path.of(output.get).toAbsolutePath
    Files.createDirectories(outputDir)
    val mask = new BufferedImage(256,256,BufferedImage.TYPE_BYTE_GRAY)
    for y <- 0 until 256; x <- 0 until 256 do mask.getRaster.setSample(x,y,0,if x < 128 then 0 else 255)
    val maskPath = outputDir.resolve("mask.png")
    assert(ImageIO.write(mask,"png",maskPath.toFile))
    val backend = new NativeDiffusionBackend(Path.of(checkpoint.get),Path.of("-"),
      vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val context = new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
    val params = Obj(Map("prompt"->Str("a photograph of a green apple on a wooden table"),
      "width"->Number(256),"height"->Number(256),"steps"->Number(8),"seed"->Number(42),
      "cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform"),
      "img2img"->Obj(Map("init_image"->Str(source.get),"mask"->Str(maskPath.toString),
        "mask_blur"->Number(0),"inpainting_fill"->Number(1),"denoising_strength"->Number(BigDecimal("0.75"))))))
    val result = backend.execute(NativeRequest("inpaint","img2img",Map("output_dir"->Str(outputDir.toString),"params"->params),None),context)
    val path = Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
    val generated = ImageIO.read(path.toFile)
    assertEquals((generated.getWidth,generated.getHeight),(256,256))
    for y <- 0 until 256; x <- 0 until 128 do assertEquals(generated.getRGB(x,y),image.getRGB(x,y))
    assert((128 until 256).exists(x => (0 until 256).exists(y => generated.getRGB(x,y)!=image.getRGB(x,y))))
    setting("SCALATORCH_SD15_CONTROLNET").foreach { control =>
      val cacheCopy=Option.when(setting("SCALATORCH_CONTROLNET_TEST_INVALIDATION").contains("1")) {
        val path=Files.createTempFile(outputDir,"control-cache-",".safetensors")
        Files.copy(Path.of(control),path,java.nio.file.StandardCopyOption.REPLACE_EXISTING)
        path
      }
      val unit=Obj(Map("enabled"->Bool(true),"model"->Str(cacheCopy.fold(control)(_.toString)),"image"->Str(source.get),
        "preprocessor"->Str("canny"),"weight"->Number(1)))
      val controlled=Obj(params.asInstanceOf[Obj].fields.updated("controlnet",Arr(Vector(unit))))
      if control.toLowerCase(java.util.Locale.ROOT).endsWith(".pth") then
        def outputFiles: Long = scala.util.Using.resource(Files.list(outputDir))(_.count())
        val before=outputFiles
        var readerChecks=0
        val cancelledContext=new NativeWorkContext(() => {
          if Thread.currentThread().getStackTrace.exists(frame => frame.getClassName.contains("PyTorchTensorArchive") &&
              frame.getMethodName.contains("readTensor")) then readerChecks+=1
          readerChecks>=40
        },()=>true,(_,_)=>())
        intercept[java.util.concurrent.CancellationException] {
          backend.execute(NativeRequest("cancel-reading","img2img",
            Map("output_dir"->Str(outputDir.toString),"params"->controlled),None),cancelledContext)
        }
        assertEquals(readerChecks,40)
        assertEquals(outputFiles,before,"cancelled loading must not publish output")
      val output=backend.execute(NativeRequest("controlled","img2img",
        Map("output_dir"->Str(outputDir.toString),"params"->controlled),None),context)
      val controlledPath=Path.of(output("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val controlledImage=ImageIO.read(controlledPath.toFile)
      for y <- 0 until 256; x <- 0 until 128 do assertEquals(controlledImage.getRGB(x,y),image.getRGB(x,y))
      assert((128 until 256).exists(x => (0 until 256).exists(y => controlledImage.getRGB(x,y)!=generated.getRGB(x,y))))
      cacheCopy.foreach { path =>
        try
          Files.write(path,Array[Byte](1,2,3))
          intercept[Exception](backend.execute(NativeRequest("changed","img2img",
            Map("output_dir"->Str(outputDir.toString),"params"->controlled),None),context))
        finally Files.deleteIfExists(path)
      }
      setting("SCALATORCH_SD15_CONTROLNET_REFERENCE").foreach { reference =>
        val referenceUnit=Obj(unit.asInstanceOf[Obj].fields.updated("model",Str(reference)))
        val referenceParams=Obj(params.asInstanceOf[Obj].fields.updated("controlnet",Arr(Vector(referenceUnit))))
        val compared=backend.execute(NativeRequest("reference","img2img",
          Map("output_dir"->Str(outputDir.toString),"params"->referenceParams),None),context)
        val referencePath=Path.of(compared("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        val expected=ImageIO.read(referencePath.toFile)
        assertEquals(controlledImage.getRGB(0,0,256,256,null,0,256).toVector,
          expected.getRGB(0,0,256,256,null,0,256).toVector)
      }
    }
  }
