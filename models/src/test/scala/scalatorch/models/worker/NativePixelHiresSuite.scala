package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.{UNetExecution,VaeExecution}
import NativeJson.*

class NativePixelHiresSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(180,"seconds")
  test("pixel Hires crops with NUI's eight-pixel alignment") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny model and CUDA required")
    val root=Path.of(configured.get).getParent
    val dir=Files.createTempDirectory("native-pixel-hires-crop")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
      unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
    for mode<-Vector("Nearest","Lanczos") do
      val hires=Obj(Map("enabled"->Bool(true),"scale"->Number(2),"upscaler"->Str(mode),
        "resize_x"->Number(128),"resize_y"->Number(72),"second_pass_steps"->Number(2),
        "denoising_strength"->Number(BigDecimal("0.5"))))
      val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(2),
        "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("Euler"),
        "scheduler"->Str("Uniform"),"hires"->hires,"extra"->Obj(Map("rng"->Str("cpu")))))
      val result=backend.execute(NativeRequest("crop","txt2img",Map("output_dir"->Str(dir.toString),"params"->params),None),context)
      val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val actual=ImageIO.read(path.toFile)
      val resource=getClass.getResourceAsStream(s"/pixel-hires-${mode.toLowerCase(java.util.Locale.ROOT)}-crop-reference.png")
      val expected=try ImageIO.read(resource) finally resource.close()
      assertEquals((actual.getWidth,actual.getHeight),(128,72))
      val errors=for y<-0 until 72; x<-0 until 128; shift<-Vector(0,8,16) yield
        math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
      assert(errors.max<=1 && errors.sum.toDouble/errors.size<0.01,s"$mode max=${errors.max} mean=${errors.sum.toDouble/errors.size}")
  }
  test("pixel Hires matches an independent Diffusers and Pillow reference") {
    val configured=sys.props.get("scalatorch.clip.dir")
    val reference=sys.props.get("scalatorch.pixel.hires.reference")
    assume(configured.nonEmpty,"reference tiny model required")
    val root=Path.of(configured.get).getParent
    val dir=Files.createTempDirectory("native-pixel-hires-reference")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val devices=if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then Vector(true) else Vector(false)
    devices.foreach { cuda =>
      val backend=if cuda then new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
        unetExecution=UNetExecution.Cuda(offloadWeights=true)) else new NativeDiffusionBackend(root,scheduler)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val comparisons=scala.collection.mutable.ArrayBuffer.empty[(String,Int,Double)]
      if reference.nonEmpty then
        val base=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(2),
          "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("DPM++ 2M"),"scheduler"->Str("Karras"),"extra"->Obj(Map("rng"->Str("cpu")))))
        val result=backend.execute(NativeRequest("base","txt2img",Map("output_dir"->Str(dir.toString),"params"->base),None),context)
        val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        val actual=ImageIO.read(path.toFile)
        val expected=ImageIO.read(Path.of(reference.get).resolve("DPM-2M-base.png").toFile)
        val differences=for y<-0 until 64; x<-0 until 64; shift<-Vector(0,8,16) yield
          math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
        println(s"DPM++ base PNG cuda=$cuda max=${differences.max} mean=${differences.sum.toDouble/differences.size}")
        val source=Path.of(reference.get).resolve("DPM-2M-Nearest-source.png")
        val imgParams=Obj(Map("prompt"->Str("a red cat"),"width"->Number(128),"height"->Number(128),"steps"->Number(4),
          "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("DPM++ 2M"),"scheduler"->Str("Karras"),
          "extra"->Obj(Map("rng"->Str("cpu"))),"img2img"->Obj(Map("init_image"->Str(source.toString),
            "denoising_strength"->Number(BigDecimal("0.5")),"resize_mode"->Number(0)))))
        val imageResult=backend.execute(NativeRequest("second","img2img",Map("output_dir"->Str(dir.toString),"params"->imgParams),None),context)
        val imagePath=Path.of(imageResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        val image=ImageIO.read(imagePath.toFile)
        val expectedImage=ImageIO.read(Path.of(reference.get).resolve("DPM-2M-Nearest-result.png").toFile)
        val secondDifferences=for y<-0 until 128; x<-0 until 128; shift<-Vector(0,8,16) yield
          math.abs(((image.getRGB(x,y)>>>shift)&255)-((expectedImage.getRGB(x,y)>>>shift)&255))
        println(s"DPM++ identical-source second pass cuda=$cuda max=${secondDifferences.max} mean=${secondDifferences.sum.toDouble/secondDifferences.size}")
        assert(secondDifferences.max<=1 && secondDifferences.sum.toDouble/secondDifferences.size<0.01)
      val samplers=if sys.env.get("SCALATORCH_PIXEL_HIRES_STAGE_ONLY").contains("1") then Vector.empty else
        Vector(("Euler","Uniform","Euler"),("Euler a","Uniform","Euler-a"),("DPM++ 2M","Karras","DPM-2M"))
      for (sampler,schedule,referenceName)<-samplers; mode<-Vector("Nearest","Lanczos") do
        val hires=Obj(Map("enabled"->Bool(true),"scale"->Number(2),"upscaler"->Str(mode),
          "second_pass_steps"->Number(2),"denoising_strength"->Number(BigDecimal("0.5"))))
        val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(2),
          "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str(sampler),"scheduler"->Str(schedule),"hires"->hires,
          "extra"->Obj(Map("rng"->Str("cpu")))))
        val result=backend.execute(NativeRequest("generate","txt2img",Map("output_dir"->Str(dir.toString),"params"->params),None),context)
        val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        val actual=ImageIO.read(path.toFile)
        val expected=reference match
          case Some(dir) => ImageIO.read(Path.of(dir).resolve(s"$referenceName-$mode-result.png").toFile)
          case None =>
            val prefix=if referenceName=="Euler" then "" else referenceName.toLowerCase(java.util.Locale.ROOT)+"-"
            val resource=getClass.getResourceAsStream(s"/pixel-hires-$prefix${mode.toLowerCase(java.util.Locale.ROOT)}-reference.png")
            try ImageIO.read(resource) finally resource.close()
        val differences=for y<-0 until 128; x<-0 until 128; shift<-Vector(0,8,16) yield
          math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
        val maximum=differences.max
        val mean=differences.sum.toDouble/differences.size
        comparisons+=((s"$sampler $mode cuda=$cuda",maximum,mean))
        println(s"$sampler $mode cuda=$cuda max=$maximum mean=$mean")
      assert(comparisons.forall { case (_,max,mean) => max<=1 && mean<0.01 },
        comparisons.mkString(", "))
    }
  }
  test("pixel hires decodes and re-encodes the base image for Nearest and Lanczos") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny model and CUDA required")
    val root=Path.of(configured.get).getParent
    val dir=Files.createTempDirectory("native-pixel-hires")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
      unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
    def generate(mode: String,scale: Int=1): Vector[Int] =
      val hires=Obj(Map("enabled"->Bool(true),"scale"->Number(scale),"upscaler"->Str(mode),
        "second_pass_steps"->Number(2),"denoising_strength"->Number(BigDecimal("0.5"))))
      val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),
        "steps"->Number(2),"seed"->Number(42),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform"),"hires"->hires))
      val result=backend.execute(NativeRequest("generate","txt2img",Map("output_dir"->Str(dir.toString),"params"->params),None),context)
      val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val image=ImageIO.read(path.toFile)
      assertEquals((image.getWidth,image.getHeight),(64*scale,64*scale))
      image.getRGB(0,0,64*scale,64*scale,null,0,64*scale).toVector
    val nearest=generate("Nearest")
    assertEquals(generate("Lanczos"),nearest,"both pixel upscalers are identity at equal dimensions")
    assertNotEquals(generate("Latent (nearest)"),nearest,"pixel hires must pass through the VAE")
    assertNotEquals(generate("Nearest",2),generate("Lanczos",2),"distinct pixel filters should affect enlarged output")
  }
