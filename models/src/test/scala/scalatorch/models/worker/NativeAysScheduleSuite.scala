package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.{UNetExecution,VaeExecution}
import NativeJson.*

class NativeAysScheduleSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(180,"seconds")
  test("AYS schedule generates reference images and preserves NUI sampler fallbacks") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny model and CUDA required")
    val root=Path.of(configured.get).getParent
    val dir=Files.createTempDirectory("native-ays")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
      unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val listing=backend.execute(NativeRequest("list","list_samplers",Map.empty,None),context)
    assert(listing("schedulers").asInstanceOf[Arr].values.contains(Str("Align Your Steps")))
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
    val (actual,warning)=generate("Euler","ays")
    val resource=this.getClass.getResourceAsStream("/ays-euler-reference.png")
    val expected=try ImageIO.read(resource) finally resource.close()
    val errors=differences(actual,expected)
    assert(errors.max<=1 && errors.sum.toDouble/errors.size<0.01,s"max=${errors.max} mean=${errors.sum.toDouble/errors.size}")
    assertEquals(warning,Vector.empty)
    for (sampler,fallback)<-Vector(("DPM++ 2M","Karras"),("Euler a","Normal")) do
      val (image,warnings)=generate(sampler,"Align Your Steps")
      val (equivalent,_)=generate(sampler,fallback)
      assertEquals(differences(image,equivalent).max,0)
      assert(warnings.exists { case Str(value)=>value.contains("Align Your Steps");case _=>false })
  }
