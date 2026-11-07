package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.{UNetExecution,VaeExecution}
import NativeJson.*

class NativeDdimScheduleSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(180,"seconds")
  test("DDIM schedule generates the NUI reference for each supported sampler") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny model and CUDA required")
    val root=Path.of(configured.get).getParent
    val dir=Files.createTempDirectory("native-ddim")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
      unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val listing=backend.execute(NativeRequest("list","list_samplers",Map.empty,None),context)
    assert(listing("schedulers").asInstanceOf[Arr].values.contains(Str("DDIM")))
    backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
    def generate(sampler: String,schedule: String): (java.awt.image.BufferedImage,Vector[NativeJson]) =
      val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(4),
        "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str(sampler),
        "scheduler"->Str(schedule),"extra"->Obj(Map("rng"->Str("cpu")))))
      val result=backend.execute(NativeRequest("generate","txt2img",Map("output_dir"->Str(dir.toString),"params"->params),None),context)
      val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      (ImageIO.read(path.toFile),result("warnings").asInstanceOf[Arr].values)
    def differences(a: java.awt.image.BufferedImage,b: java.awt.image.BufferedImage): Vector[Int] =
      (for y<-0 until 64; x<-0 until 64; shift<-Vector(0,8,16) yield
        math.abs(((a.getRGB(x,y)>>>shift)&255)-((b.getRGB(x,y)>>>shift)&255))).toVector
    for (sampler,ref)<-Vector(("Euler","euler"),("Euler a","ancestral"),("DPM++ 2M","dpm")) do
      val (actual,warnings)=generate(sampler,"DDIM")
      val resource=this.getClass.getResourceAsStream(s"/ddim-$ref-reference.png")
      val expected=try ImageIO.read(resource) finally resource.close()
      val errors=differences(actual,expected)
      assert(errors.max<=1 && errors.sum.toDouble/errors.size<0.01,
        s"$sampler max=${errors.max} mean=${errors.sum.toDouble/errors.size}")
      assertEquals(warnings,Vector.empty)

    val sevenParams=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(7),
      "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("Euler"),
      "scheduler"->Str("DDIM"),"extra"->Obj(Map("rng"->Str("cpu")))))
    val seven=backend.execute(NativeRequest("seven","txt2img",Map("output_dir"->Str(dir.toString),"params"->sevenParams),None),context)
    val sevenPath=Path.of(seven("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
    val sevenActual=ImageIO.read(sevenPath.toFile)
    val sevenResource=this.getClass.getResourceAsStream("/ddim-seven-reference.png")
    val sevenExpected=try ImageIO.read(sevenResource) finally sevenResource.close()
    val sevenErrors=differences(sevenActual,sevenExpected)
    assert(sevenErrors.max<=1 && sevenErrors.sum.toDouble/sevenErrors.size<0.01,
      s"seven steps max=${sevenErrors.max} mean=${sevenErrors.sum.toDouble/sevenErrors.size}")
    val input=this.getClass.getResourceAsStream("/ddim-img2img-source.png")
    val source=dir.resolve("source.png")
    try Files.copy(input,source) finally input.close()
    val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),"steps"->Number(7),
      "seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),"sampler_name"->Str("Euler"),
      "scheduler"->Str("DDIM"),"extra"->Obj(Map("rng"->Str("cpu"))),
      "img2img"->Obj(Map("init_image"->Str(source.toString),"denoising_strength"->Number(BigDecimal("0.6"))))))
    var count=0
    val progressContext=new NativeWorkContext(()=>false,()=>true,(name,_)=>if name=="progress" then count+=1)
    val result=backend.execute(NativeRequest("img2img","img2img",Map("output_dir"->Str(dir.toString),"params"->params),None),progressContext)
    val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
    val actual=ImageIO.read(path.toFile)
    val reference=this.getClass.getResourceAsStream("/ddim-img2img-reference.png")
    val expected=try ImageIO.read(reference) finally reference.close()
    val errors=differences(actual,expected)
    assertEquals(count,5)
    assert(errors.max<=1 && errors.sum.toDouble/errors.size<0.01,
      s"img2img max=${errors.max} mean=${errors.sum.toDouble/errors.size}")

  }
