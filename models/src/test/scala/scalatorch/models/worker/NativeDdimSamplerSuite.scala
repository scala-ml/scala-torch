package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.{UNetExecution,VaeExecution}
import NativeJson.*

class NativeDdimSamplerSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(180,"seconds")
  test("DDIM sampler matches NUI references for clipping and txt2img/img2img") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny model and CUDA required")
    val root=Path.of(configured.get).getParent
    val dir=Files.createTempDirectory("native-ddim-sampler")
    val source=dir.resolve("source.png")
    val input=this.getClass.getResourceAsStream("/ddim-sampler-source.png")
    try Files.copy(input,source) finally input.close()
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    for clip<-Vector(false,true) do
      val scheduler=dir.resolve(s"scheduler-$clip.json")
      Files.writeString(scheduler,s"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear","clip_sample":$clip}""")
      val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
        unetExecution=UNetExecution.Cuda(offloadWeights=true))
      val listing=backend.execute(NativeRequest("list","list_samplers",Map.empty,None),context)
      assert(listing("samplers").asInstanceOf[Arr].values.exists {
        case Obj(fields)=>fields.get("name").contains(Str("DDIM"))
        case _=>false
      })
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      for (schedule,img)<-Vector(("Uniform",false),("DDIM",false),("Simple",false),("Uniform",true)) do
        val data=Map[String,NativeJson]("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(4),
          "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("DDIM"),
          "scheduler"->Str(schedule),"extra"->Obj(Map("rng"->Str("cpu")))) ++
          Option.when(img)("img2img"->Obj(Map("init_image"->Str(source.toString),"denoising_strength"->Number(BigDecimal("0.75")))))
        val result=backend.execute(NativeRequest("generate",if img then "img2img" else "txt2img",
          Map("output_dir"->Str(dir.toString),"params"->Obj(data)),None),context)
        val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        val actual=ImageIO.read(path.toFile)
        val name=if img then "img2img" else schedule
        val resource=this.getClass.getResourceAsStream(s"/ddim-sampler-$name-${if clip then "True" else "False"}.png")
        val expected=try ImageIO.read(resource) finally resource.close()
        val errors=for y<-0 until 64;x<-0 until 64;shift<-Vector(0,8,16) yield
          math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
        assert(errors.max<=1 && errors.sum.toDouble/errors.size<0.01,
          s"$schedule clip=$clip img=$img max=${errors.max} mean=${errors.sum.toDouble/errors.size}")
  }

  test("DDIM eta=1 follows the PyTorch seeded stochastic stream") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny model and CUDA required")
    val root=Path.of(configured.get).getParent
    val dir=Files.createTempDirectory("native-ddim-eta")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear","clip_sample":false}""")
    val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
      unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
    val source=dir.resolve("source.png");val input=this.getClass.getResourceAsStream("/ddim-sampler-source.png")
    try Files.copy(input,source) finally input.close()
    for img<-Vector(false,true) do
      val data=Map[String,NativeJson]("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(4),
        "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("DDIM"),"scheduler"->Str("Uniform"),
        "eta"->Number(1),"extra"->Obj(Map("rng"->Str("cpu")))) ++
        Option.when(img)("img2img"->Obj(Map("init_image"->Str(source.toString),"denoising_strength"->Number(BigDecimal("0.75")))))
      val result=backend.execute(NativeRequest("eta",if img then "img2img" else "txt2img",Map("output_dir"->Str(dir.toString),"params"->Obj(data)),None),context)
      val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val actual=ImageIO.read(path.toFile)
      val ref=this.getClass.getResourceAsStream(s"/ddim-sampler-eta-${if img then "img2img" else "txt2img"}.png")
      val expected=try ImageIO.read(ref) finally ref.close()
      val errors=for y<-0 until 64;x<-0 until 64;shift<-Vector(0,8,16) yield
        math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
      assert(errors.max<=1 && errors.sum.toDouble/errors.size<0.01,
        s"eta img=$img max=${errors.max} mean=${errors.sum.toDouble/errors.size}")
  }
  test("eta is ignored with a warning for samplers that do not accept it") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny model and CUDA required")
    val root=Path.of(configured.get).getParent
    val dir=Files.createTempDirectory("native-eta-warning")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
    val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(2),
      "seed"->Number(42),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform"),"eta"->Number(1)))
    val result=backend.execute(NativeRequest("eta","txt2img",Map("output_dir"->Str(dir.toString),"params"->params),None),context)
    assert(result("warnings").asInstanceOf[Arr].values.contains(Str("eta is ignored by sampler 'Euler'")))
  }
  test("DDIM sampler supports four- and nine-channel inpainting") {
    val clipRoot=sys.props.get("scalatorch.clip.dir")
    val nineRoot=sys.props.get("scalatorch.nine.dir")
    assume(clipRoot.nonEmpty && nineRoot.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"models and CUDA required")
    val dir=Files.createTempDirectory("native-ddim-inpaint")
    val source=dir.resolve("source.png")
    val input=this.getClass.getResourceAsStream("/ddim-sampler-source.png")
    try Files.copy(input,source) finally input.close()
    val mask=new java.awt.image.BufferedImage(64,64,java.awt.image.BufferedImage.TYPE_BYTE_GRAY)
    for y<-0 until 64;x<-0 until 64 do mask.getRaster.setSample(x,y,0,255)
    val maskPath=dir.resolve("mask.png");ImageIO.write(mask,"png",maskPath.toFile)
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear","clip_sample":false}""")
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    for (channels,root)<-Vector((4,Path.of(clipRoot.get).getParent),(9,Path.of(nineRoot.get))) do
      val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
        unetExecution=UNetExecution.Cuda(offloadWeights=true))
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(4),
        "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("DDIM"),
        "scheduler"->Str("Uniform"),"extra"->Obj(Map("rng"->Str("cpu"))),
        "img2img"->Obj(Map("init_image"->Str(source.toString),"mask"->Str(maskPath.toString),
          "mask_blur"->Number(0),"inpainting_fill"->Number(1),"denoising_strength"->Number(BigDecimal("0.75"))))))
      val result=backend.execute(NativeRequest("inpaint","img2img",Map("output_dir"->Str(dir.toString),"params"->params),None),context)
      val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val actual=ImageIO.read(path.toFile)
      val ref=this.getClass.getResourceAsStream(s"/ddim-sampler-inpaint-$channels.png")
      val expected=try ImageIO.read(ref) finally ref.close()
      val errors=for y<-0 until 64;x<-0 until 64;shift<-Vector(0,8,16) yield
        math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
      assert(errors.max<=1 && errors.sum.toDouble/errors.size<0.01,
        s"$channels-channel max=${errors.max} mean=${errors.sum.toDouble/errors.size}")
  }

  test("DDIM partial masks and seeded eta match independent references") {
    val clipRoot=sys.props.get("scalatorch.clip.dir")
    val nineRoot=sys.props.get("scalatorch.nine.dir")
    assume(clipRoot.nonEmpty && nineRoot.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"models and CUDA required")
    val dir=Files.createTempDirectory("native-ddim-partial")
    def copy(resource: String,name: String): Path =
      val destination=dir.resolve(name);val in=this.getClass.getResourceAsStream(resource)
      try Files.copy(in,destination) finally in.close()
      destination
    val source=copy("/ddim-sampler-source.png","source.png")
    val mask=copy("/ddim-sampler-half-mask.png","mask.png")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear","clip_sample":false}""")
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    for (channels,root)<-Vector((4,Path.of(clipRoot.get).getParent),(9,Path.of(nineRoot.get))) do
      val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
        unetExecution=UNetExecution.Cuda(offloadWeights=true))
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      for eta<-Vector(0,1) do
        val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(4),
          "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("DDIM"),"eta"->Number(eta),
          "scheduler"->Str("Uniform"),"extra"->Obj(Map("rng"->Str("cpu"))),
          "img2img"->Obj(Map("init_image"->Str(source.toString),"mask"->Str(mask.toString),"mask_blur"->Number(0),
            "inpainting_fill"->Number(1),"denoising_strength"->Number(BigDecimal("0.75"))))))
        val result=backend.execute(NativeRequest("partial","img2img",Map("output_dir"->Str(dir.toString),"params"->params),None),context)
        val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        val actual=ImageIO.read(path.toFile)
        val ref=this.getClass.getResourceAsStream(s"/ddim-sampler-half-$channels-eta$eta.png")
        val expected=try ImageIO.read(ref) finally ref.close()
        val errors=for y<-0 until 64;x<-0 until 64;shift<-Vector(0,8,16) yield
          math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
        assert(errors.max<=1 && errors.sum.toDouble/errors.size<0.01,
          s"$channels-channel eta=$eta max=${errors.max} mean=${errors.sum.toDouble/errors.size}")
  }

  test("DDIM sampler supports pixel and nine-channel latent Hires") {
    val clipRoot=sys.props.get("scalatorch.clip.dir")
    val nineRoot=sys.props.get("scalatorch.nine.dir")
    assume(clipRoot.nonEmpty && nineRoot.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"models and CUDA required")
    val dir=Files.createTempDirectory("native-ddim-hires")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear","clip_sample":false}""")
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    for (root,upscaler,ref)<-Vector((Path.of(clipRoot.get).getParent,"Nearest","pixel"),
      (Path.of(nineRoot.get),"Latent (nearest)","nine-latent")) do
      val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
        unetExecution=UNetExecution.Cuda(offloadWeights=true))
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val hires=Obj(Map("enabled"->Bool(true),"scale"->Number(2),"upscaler"->Str(upscaler),
        "second_pass_steps"->Number(2),"denoising_strength"->Number(BigDecimal("0.5"))))
      val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(2),
        "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("DDIM"),
        "scheduler"->Str("Uniform"),"extra"->Obj(Map("rng"->Str("cpu"))),"hires"->hires))
      val result=backend.execute(NativeRequest("hires","txt2img",Map("output_dir"->Str(dir.toString),"params"->params),None),context)
      val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val actual=ImageIO.read(path.toFile)
      val stream=this.getClass.getResourceAsStream(s"/ddim-sampler-$ref-hires.png")
      val expected=try ImageIO.read(stream) finally stream.close()
      assertEquals((actual.getWidth,actual.getHeight),(128,128))
      val errors=for y<-0 until 128;x<-0 until 128;shift<-Vector(0,8,16) yield
        math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
      assert(errors.max<=1 && errors.sum.toDouble/errors.size<0.01,
        s"$upscaler max=${errors.max} mean=${errors.sum.toDouble/errors.size}")
  }
