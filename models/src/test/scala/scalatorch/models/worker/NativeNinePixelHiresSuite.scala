package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.{UNetExecution,VaeExecution}
import NativeJson.*

class NativeNinePixelHiresSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(180,"seconds")
  test("nine-channel txt2img runs full-mask pixel Hires with the same seeded inpaint pipeline") {
    val configured=sys.props.get("scalatorch.nine.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"nine-channel tiny model and CUDA required")
    val root=Path.of(configured.get)
    val dir=Files.createTempDirectory("native-nine-pixel-hires")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
      unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
    val hires=Obj(Map("enabled"->Bool(true),"scale"->Number(2),"upscaler"->Str("Nearest"),
      "second_pass_steps"->Number(2),"denoising_strength"->Number(BigDecimal("0.5"))))
    val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(2),
      "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("Euler"),
      "scheduler"->Str("Uniform"),"hires"->hires,"extra"->Obj(Map("rng"->Str("cpu")))))
    val baseParams=Obj(params.asInstanceOf[Obj].fields-"hires")
    val baseResult=backend.execute(NativeRequest("base","txt2img",Map("output_dir"->Str(dir.toString),"params"->baseParams),None),context)
    val basePath=Path.of(baseResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
    val base=ImageIO.read(basePath.toFile)
    val baseResource=getClass.getResourceAsStream("/nine-contiguous-base.png")
    val baseExpected=try ImageIO.read(baseResource) finally baseResource.close()
    val baseDifferences=for y<-0 until 64; x<-0 until 64; shift<-Vector(0,8,16) yield
      math.abs(((base.getRGB(x,y)>>>shift)&255)-((baseExpected.getRGB(x,y)>>>shift)&255))
    assert(baseDifferences.max<=1 && baseDifferences.sum.toDouble/baseDifferences.size<0.01,
      s"base max=${baseDifferences.max} mean=${baseDifferences.sum.toDouble/baseDifferences.size}")
    val result=backend.execute(NativeRequest("generate","txt2img",Map("output_dir"->Str(dir.toString),"params"->params),None),context)
    val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
    val actual=ImageIO.read(path.toFile)
    val resource=getClass.getResourceAsStream("/nine-contiguous-output.png")
    val expected=try ImageIO.read(resource) finally resource.close()
    assertEquals((actual.getWidth,actual.getHeight),(128,128))
    val differences=for y<-0 until 128; x<-0 until 128; shift<-Vector(0,8,16) yield
      math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
    assert(differences.max<=1 && differences.sum.toDouble/differences.size<0.01,
      s"max=${differences.max} mean=${differences.sum.toDouble/differences.size}")
    val sourceResource=getClass.getResourceAsStream("/nine-contiguous-source.png")
    val sourcePath=dir.resolve("source.png")
    try Files.copy(sourceResource,sourcePath) finally sourceResource.close()
    val imageParams=Obj((params.asInstanceOf[Obj].fields-"hires").updated("width",Number(128)).updated("height",Number(128))
      .updated("steps",Number(4)).updated("img2img",Obj(Map("init_image"->Str(sourcePath.toString),
        "denoising_strength"->Number(BigDecimal("0.5"))))))
    val imageResult=backend.execute(NativeRequest("img2img","img2img",Map("output_dir"->Str(dir.toString),"params"->imageParams),None),context)
    val imagePath=Path.of(imageResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
    val image=ImageIO.read(imagePath.toFile)
    val imageDifferences=for y<-0 until 128; x<-0 until 128; shift<-Vector(0,8,16) yield
      math.abs(((image.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
    assert(imageDifferences.max<=1 && imageDifferences.sum.toDouble/imageDifferences.size<0.01,
      s"img2img max=${imageDifferences.max} mean=${imageDifferences.sum.toDouble/imageDifferences.size}")
  }


  test("nine-channel latent Hires decodes the resized latent before the second inpaint pass") {
    val configured=sys.props.get("scalatorch.nine.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"nine-channel tiny model and CUDA required")
    val dir=Files.createTempDirectory("native-nine-latent-hires")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(Path.of(configured.get),scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
      unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val passes=scala.collection.mutable.ArrayBuffer.empty[String]
    val context=new NativeWorkContext(()=>false,()=>true,(name,fields)=>
      if name=="progress" then passes+=fields("pass").asInstanceOf[Str].value)
    backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
    for (mode,ref,height)<-Vector(("Latent (nearest)","nearest",128),("Latent","bilinear",128),
      ("Latent (nearest)","nearest-crop",72)) do
      passes.clear()
      val hires=Obj(Map("enabled"->Bool(true),"scale"->Number(2),"upscaler"->Str(mode),
        "second_pass_steps"->Number(2),"denoising_strength"->Number(BigDecimal("0.5")),
        "resize_x"->Number(128),"resize_y"->Number(height)))
      val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(2),
        "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("Euler"),
        "scheduler"->Str("Uniform"),"hires"->hires,"extra"->Obj(Map("rng"->Str("cpu")))))
      val result=backend.execute(NativeRequest("generate","txt2img",Map("output_dir"->Str(dir.toString),"params"->params),None),context)
      val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val actual=ImageIO.read(path.toFile)
      val resource=getClass.getResourceAsStream(s"/nine-latent-$ref-output.png")
      val expected=try ImageIO.read(resource) finally resource.close()
      assertEquals((actual.getWidth,actual.getHeight),(128,height))
      val differences=for y<-0 until height; x<-0 until 128; shift<-Vector(0,8,16) yield
        math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
      assert(differences.max<=1 && differences.sum.toDouble/differences.size<0.01,
        s"$mode max=${differences.max} mean=${differences.sum.toDouble/differences.size}")
      assertEquals(passes.toVector,Vector("base","base","hires","hires"))
      assert(result("warnings").asInstanceOf[Arr].values.contains(
        Str("latent hires upscale is not supported for inpainting models; decoding first")))
  }
