package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.{UNetExecution,VaeExecution}
import NativeJson.*

class NativeDdimSdxlSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(180,"seconds")
  test("SDXL DDIM txt2img and img2img match independent Diffusers references") {
    val configured=sys.props.get("scalatorch.sdxl.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny SDXL and CUDA required")
    val root=Path.of(configured.get)
    val dir=Files.createTempDirectory("native-ddim-sdxl")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear","clip_sample":false}""")
    val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
      unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
    val source=dir.resolve("source.png");val stream=this.getClass.getResourceAsStream("/ddim-sdxl-source.png")
    try Files.copy(stream,source) finally stream.close()
    for img<-Vector(false,true);eta<-Vector(0,1) do
      val data=Map[String,NativeJson]("prompt"->Str("a red apple"),"negative_prompt"->Str("blur"),"width"->Number(64),
        "height"->Number(64),"steps"->Number(4),"seed"->Number(42),"cfg_scale"->Number(BigDecimal("7.5")),
        "sampler_name"->Str("DDIM"),"scheduler"->Str("Uniform"),"eta"->Number(eta),"extra"->Obj(Map("rng"->Str("cpu")))) ++
        Option.when(img)("img2img"->Obj(Map("init_image"->Str(source.toString),"denoising_strength"->Number(BigDecimal("0.75")))))
      val result=backend.execute(NativeRequest("generate",if img then "img2img" else "txt2img",
        Map("output_dir"->Str(dir.toString),"params"->Obj(data)),None),context)
      val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val actual=ImageIO.read(path.toFile)
      val ref=this.getClass.getResourceAsStream(s"/ddim-sdxl-${if img then "img" else "txt"}-eta$eta.png")
      val expected=try ImageIO.read(ref) finally ref.close()
      val errors=for y<-0 until 64;x<-0 until 64;shift<-Vector(0,8,16) yield
        math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
      println(s"SDXL DDIM img=$img eta=$eta max=${errors.max} mean=${errors.sum.toDouble/errors.size}")
      assert(errors.max<=1 && errors.sum.toDouble/errors.size<0.01,
        s"img=$img eta=$eta max=${errors.max} mean=${errors.sum.toDouble/errors.size}")
  }
